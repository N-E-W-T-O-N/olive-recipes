# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
"""Model builder for sarvamai/sarvam-30b (SarvamMoEForCausalLM).

Architecture summary
--------------------
- 19 decoder layers, hidden_size=4096
- GQA: 64 attention heads / 4 KV heads
- 128 routed experts + 1 always-active shared expert per layer
- top-6 routing (group-limited), aux-loss-free router balancing
- SwiGLU activation (swiglu_fusion=1)
- normalize_routing_weights=True (softmax after top-k selection)
- RoPE with theta=8,000,000
- Closest to DeepSeek-V2 / Qwen3.5-MoE style MoE

Expected HF module attribute paths (trust_remote_code=True):
  layer.self_attn          - GQA attention
  layer.mlp.gate           - Router nn.Linear (no bias), [hidden_size → num_experts]
  layer.mlp.experts[i].gate_proj   - i-th routed expert gate projection
  layer.mlp.experts[i].up_proj     - i-th routed expert up projection
  layer.mlp.experts[i].down_proj   - i-th routed expert down projection
  layer.mlp.shared_experts.gate_proj  - shared expert gate projection
  layer.mlp.shared_experts.up_proj    - shared expert up projection
  layer.mlp.shared_experts.down_proj  - shared expert down projection
"""

import torch

from .base import Model


class SarvamModel(Model):
    """ONNX model builder for sarvamai/sarvam-30b.

    Sarvam-30B is a Llama-style base model with a DeepSeek-V2-style
    Mixture-of-Experts FFN block.  Each decoder layer has:

    * A router that selects the top-6 experts from 128 routed candidates
      (routing weights are normalised via softmax after top-k selection).
    * Packed routed expert weights handled by ORT's ``MoE`` / ``QMoE`` op.
    * A single always-active shared expert whose output is added to the
      routed-expert aggregate (no sigmoid gating).

    The attention side (GQA, RoPE theta=8e6) is inherited from ``Model``.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # ------------------------------------------------------------------
        # Normalise config so the base class can find all the attributes it
        # expects.  The HF sarvam_moe config may expose MoE sizes under
        # different names than what the base class reads.
        # ------------------------------------------------------------------
        if hasattr(config, "moe_intermediate_size") and not hasattr(config, "intermediate_size"):
            # Base class uses intermediate_size for MLP sizing; forward the
            # MoE-specific value so inherited methods don't crash.
            config.intermediate_size = config.moe_intermediate_size

        if hasattr(config, "num_experts") and not hasattr(config, "num_local_experts"):
            # Base class reads num_local_experts → moe_attrs["num_experts"]
            config.num_local_experts = config.num_experts

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # ------------------------------------------------------------------
        # MoE op attributes
        # ------------------------------------------------------------------
        self.moe_attrs["activation_type"] = "swiglu"
        self.moe_attrs["swiglu_fusion"] = 1
        self.moe_attrs["normalize_routing_weights"] = True

        # swiglu_limit is only required by the TRT-RTX EP for QMoE; use
        # +inf (no clamp) when the model config does not specify a value.
        if self.moe_attrs.get("swiglu_limit") is None and ep == "trt-rtx":
            self.moe_attrs["swiglu_limit"] = float("inf")

        # ------------------------------------------------------------------
        # MoE intermediate sizes
        # ------------------------------------------------------------------
        self.moe_intermediate_size = getattr(config, "moe_intermediate_size", self.intermediate_size)
        # Shared expert may have a different (usually larger) intermediate size
        self.shared_expert_intermediate_size = getattr(
            config, "shared_expert_intermediate_size", self.moe_intermediate_size
        )

    # ----------------------------------------------------------------------
    # Layer assembly
    # ----------------------------------------------------------------------

    def _is_moe_layer(self, mlp) -> bool:
        """Return True if this layer's mlp is a sparse MoE block.

        Sarvam-30B is a *hybrid* model: some layers use a dense MLP
        (SarvamMoEMLP) and others use a sparse MoE block
        (SarvamMoESparseMoeBlock).  We distinguish them by checking
        for the presence of the router `gate` attribute.
        """
        return hasattr(mlp, "gate") and hasattr(mlp, "experts")

    def make_layer(self, layer_id, layer):
        """Assemble one Sarvam decoder layer.

        Structure (MoE layers)
        ----------------------
        input_layernorm → attention → post_attention_layernorm → MoE FFN

        Structure (dense layers)
        ------------------------
        input_layernorm → attention → post_attention_layernorm → dense MLP
        """
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.make_attention(layer_id, layer.attention, root_input=self.layernorm_attrs["output_0"])
        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )

        if self._is_moe_layer(layer.mlp):
            self.make_moe(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])
        else:
            # Dense MLP layer — delegate to the base class implementation
            self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    # ----------------------------------------------------------------------
    # MoE subgraph
    # ----------------------------------------------------------------------

    def make_moe(self, layer_id, mlp, root_input):
        """Build the MoE FFN subgraph for one decoder layer.

        Graph sketch
        ------------
        root_input ──┬──────────────────────────────── shared_expert ──┐
                     │                                                   Add → skip_input
                     └── Router ── Reshape ── MoE/QMoE op ────────────┘
        """
        basename = f"/model/layers.{layer_id}/moe"
        op_type = self.moe_attrs["op_type"]
        moe_weight_type = f"{'q' if op_type == 'QMoE' else ''}weight"

        # ------------------------------------------------------------------
        # Router  (SarvamMoEGate: weight is nn.Parameter [num_experts, hidden_size]
        #          plus optional expert_bias [num_experts])
        # ------------------------------------------------------------------
        router_basename = f"{basename}/router/MatMul"
        router_matmul_name = self.make_matmul(mlp.gate, router_basename, root_input)

        # Add expert_bias if present (SarvamMoEGate uses sigmoid + bias routing)
        router_last_name = router_matmul_name
        if hasattr(mlp.gate, "expert_bias") and mlp.gate.expert_bias is not None:
            router_bias_name = f"{basename}/router/Add"
            self.make_add_bias(
                mlp.gate.expert_bias,
                router_bias_name,
                root_input=f"{router_matmul_name}/output_0",
            )
            router_last_name = router_bias_name

        router_reshape_name = f"{basename}/router/Reshape"
        self.make_reshape(
            router_reshape_name,
            [
                f"{router_last_name}/output_0",
                f"/model/constants/INT64/{[-1, self.moe_attrs['num_experts']]}",
            ],
            dtype=self.io_dtype,
            shape=["batch_size * sequence_length", self.moe_attrs["num_experts"]],
        )

        # ------------------------------------------------------------------
        # Routed expert weight initializers
        # ------------------------------------------------------------------
        gate_up_proj_weight = f"model.layers.{layer_id}.moe.experts.gate_up_proj.{moe_weight_type}"
        gate_up_proj_scales = f"model.layers.{layer_id}.moe.experts.gate_up_proj.scales"
        gate_up_proj_bias   = f"model.layers.{layer_id}.moe.experts.gate_up_proj.bias"
        down_proj_weight    = f"model.layers.{layer_id}.moe.experts.down_proj.{moe_weight_type}"
        down_proj_scales    = f"model.layers.{layer_id}.moe.experts.down_proj.scales"
        down_proj_bias      = f"model.layers.{layer_id}.moe.experts.down_proj.bias"

        # Pack individual expert weights into the [num_experts, …] tensors
        # that the MoE / QMoE op expects.
        #
        # For each expert i:
        #   gate_proj.weight : [moe_intermediate_size, hidden_size]
        #   up_proj.weight   : [moe_intermediate_size, hidden_size]
        #   → concat along dim=0 → [2*moe_intermediate_size, hidden_size]
        #
        #   down_proj.weight : [hidden_size, moe_intermediate_size]
        #
        # ORT MoE kernel with swiglu_fusion=1 expects the gate/up dimension
        # to be interleaved as [g0,u0, g1,u1, …] rather than [gate|up].
        # We achieve this by stacking gate and up slices with dim=1 and
        # then reshaping, equivalent to the Qwen35MoE approach.

        experts = mlp.experts  # list or nn.ModuleList of length num_experts

        def _collect_gate_up(experts_):
            """Stack [gate|up] for all experts → [N, 2*I, H], then interleave."""
            gate_list = [e.gate_proj.weight for e in experts_]  # each [I, H]
            up_list   = [e.up_proj.weight   for e in experts_]  # each [I, H]
            # Concatenate along dim=0 for each expert → [2*I, H]
            raw = torch.stack(
                [torch.cat([g, u], dim=0) for g, u in zip(gate_list, up_list)], dim=0
            )  # [N, 2*I, H]
            # Interleave gate and up rows: [g0,u0, g1,u1, …]
            n, two_i, h = raw.shape
            half = two_i // 2
            interleaved = torch.stack(
                [raw[:, :half, :], raw[:, half:, :]], dim=2
            ).reshape(n, two_i, h)
            return interleaved

        def _collect_down(experts_):
            """Stack down_proj for all experts → [N, H, I]."""
            return torch.stack(
                [e.down_proj.weight.T for e in experts_], dim=0
            )  # each weight [H, I] after transpose

        if op_type == "MoE":
            gate_up_packed = _collect_gate_up(experts)   # [N, 2I, H]
            down_packed    = _collect_down(experts)       # [N, H, I]
            self.make_initializer(gate_up_packed, gate_up_proj_weight, to=self.io_dtype)
            self.make_initializer(down_packed,    down_proj_weight,    to=self.io_dtype)
        else:
            # QMoE path: quantise each expert's weights individually
            gate_up_qw_list, gate_up_sc_list = [], []
            down_qw_list,    down_sc_list    = [], []
            gate_up_all = _collect_gate_up(experts)
            down_all    = _collect_down(experts)
            for i in range(self.moe_attrs["num_experts"]):
                qw1, sc1 = self.make_qmoe_weights(gate_up_all[i])
                gate_up_qw_list.append(qw1)
                gate_up_sc_list.append(sc1)
                qw2, sc2 = self.make_qmoe_weights(down_all[i])
                down_qw_list.append(qw2)
                down_sc_list.append(sc2)
            self.make_initializer(
                torch.stack(gate_up_qw_list, dim=0).to(torch.uint8), gate_up_proj_weight
            )
            self.make_initializer(
                torch.stack(down_qw_list, dim=0).to(torch.uint8), down_proj_weight
            )
            self.make_initializer(
                torch.stack(gate_up_sc_list, dim=0), gate_up_proj_scales, to=self.io_dtype
            )
            self.make_initializer(
                torch.stack(down_sc_list, dim=0), down_proj_scales, to=self.io_dtype
            )

        # Zero biases required by the MoE op interface (Sarvam experts are
        # bias-free; the op still expects the tensor to be present).
        num_e = self.moe_attrs["num_experts"]
        self.make_initializer(
            torch.zeros(num_e, 2 * self.moe_intermediate_size), gate_up_proj_bias, to=self.io_dtype
        )
        self.make_initializer(
            torch.zeros(num_e, self.hidden_size), down_proj_bias, to=self.io_dtype
        )

        # ------------------------------------------------------------------
        # MoE / QMoE op
        # ------------------------------------------------------------------
        moe_name = f"{basename}/{op_type}"
        self.make_moe_op(
            moe_name,
            root_input=root_input,
            router_probs=f"{router_reshape_name}/output_0",
            weight1=gate_up_proj_weight,
            scales1=gate_up_proj_scales if op_type == "QMoE" else "",
            bias1=gate_up_proj_bias,
            weight2=down_proj_weight,
            scales2=down_proj_scales if op_type == "QMoE" else "",
            bias2=down_proj_bias,
        )

        # ------------------------------------------------------------------
        # Shared expert (always active, no sigmoid gating)
        # Sarvam uses a single shared expert whose output is added directly
        # to the routed-expert aggregate, following DeepSeek-V2 convention.
        # ------------------------------------------------------------------
        shared_output = self.make_shared_expert(layer_id, mlp.shared_experts, root_input)

        combine_name = f"{basename}/Add"
        self.make_add(
            combine_name,
            [f"{moe_name}/output_0", shared_output],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )

        # Feed the combined output as skip_input to the next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{combine_name}/output_0"

    # ----------------------------------------------------------------------
    # Shared expert helper
    # ----------------------------------------------------------------------

    def make_shared_expert(self, layer_id, shared_expert, root_input):
        """Build the always-active shared SiLU-MLP expert.

        Graph
        -----
        root_input ──┬── gate_proj ── SiLU ──┐
                     └── up_proj    ──────────┴── Mul ── down_proj → output
        """
        basename = f"/model/layers.{layer_id}/shared_expert"

        gate_matmul = self.make_matmul(
            shared_expert.gate_proj, f"{basename}/gate_proj/MatMul", root_input
        )
        up_matmul = self.make_matmul(
            shared_expert.up_proj, f"{basename}/up_proj/MatMul", root_input
        )

        # SiLU = x * sigmoid(x)
        sigmoid_name = f"{basename}/gate_proj/Sigmoid"
        self.make_sigmoid(
            sigmoid_name,
            f"{gate_matmul}/output_0",
            self.io_dtype,
            shape=["batch_size", "sequence_length", self.shared_expert_intermediate_size],
        )

        silu_mul_name = f"{basename}/gate_proj/Mul"
        self.make_mul(
            silu_mul_name,
            [f"{gate_matmul}/output_0", f"{sigmoid_name}/output_0"],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.shared_expert_intermediate_size],
        )

        # Gate × up
        gate_up_mul_name = f"{basename}/Mul"
        self.make_mul(
            gate_up_mul_name,
            [f"{silu_mul_name}/output_0", f"{up_matmul}/output_0"],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.shared_expert_intermediate_size],
        )

        # down_proj
        down_matmul = self.make_matmul(
            shared_expert.down_proj, f"{basename}/down_proj/MatMul", f"{gate_up_mul_name}/output_0"
        )

        return f"{down_matmul}/output_0"
