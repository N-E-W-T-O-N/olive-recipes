---
license: apache-2.0
language:
- en
pipeline_tag: text-to-speech
tags:
- text-to-speech
library_name: transformers
---

## CSM 1B

**2025/05/20** - CSM is availabile natively in [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/csm) 🤗 as of version `4.52.1`

**2025/03/13** - We are releasing the 1B CSM variant. The checkpoint is [hosted on Hugging Face](https://huggingface.co/sesame/csm_1b).

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

A fine-tuned variant of CSM powers the [interactive voice demo](https://www.sesame.com/voicedemo) shown in our [blog post](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice).

A hosted [HuggingFace space](https://huggingface.co/spaces/sesame/csm-1b) is also available for testing audio generation.

## Usage

### Generate a sentence

```python
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor

model_id = "sesame/csm-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs
text = "[0]Hello from Sesame." # `[0]` for speaker id 0
inputs = processor(text, add_special_tokens=True).to(device)

# another equivalent way to prepare the inputs
conversation = [
    {"role": "0", "content": [{"type": "text", "text": "Hello from Sesame."}]},
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(device)

# infer the model
audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, "example_without_context.wav")
```

### CSM sounds best when provided with context

```python
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor
from datasets import load_dataset, Audio

model_id = "sesame/csm-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs
ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
# ensure the audio is 24kHz
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
conversation = []

# 1. context
for text, audio, speaker_id in zip(ds[:4]["text"], ds[:4]["audio"], ds[:4]["speaker_id"]):
    conversation.append(
        {
            "role": f"{speaker_id}",
            "content": [{"type": "text", "text": text}, {"type": "audio", "path": audio["array"]}],
        }
    )

# 2. text prompt
conversation.append({"role": f"{ds[4]['speaker_id']}", "content": [{"type": "text", "text": ds[4]["text"]}]})

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(device)

# infer the model
audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, "example_with_context.wav")
```

---

### Batched Inference 📦

CSM supports batched inference:

<details>

<summary> code snippet </summary>

```python
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor
from datasets import load_dataset, Audio

model_id = "sesame/csm-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs 
ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
# ensure the audio is 24kHz
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
# here a batch with two prompts
conversation = [
    [
        {
            "role": f"{ds[0]['speaker_id']}",
            "content": [
                {"type": "text", "text": ds[0]["text"]},
                {"type": "audio", "path": ds[0]["audio"]["array"]},
            ],
        },
        {
            "role": f"{ds[1]['speaker_id']}",
            "content": [
                {"type": "text", "text": ds[1]["text"]},
            ],
        },
    ],
    [
        {
            "role": f"{ds[0]['speaker_id']}",
            "content": [
                {"type": "text", "text": ds[0]["text"]},
            ],
        }
    ],
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(device)

audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, [f"speech_batch_idx_{i}.wav" for i in range(len(audio))])
```

</details>


### Making The Model Go Brrr 🏎️

CSM supports full-graph compilation with CUDA graphs!

<details>

<summary> code snippet </summary>

```python
import torch
import copy
from transformers import CsmForConditionalGeneration, AutoProcessor
from datasets import load_dataset

model_id = "sesame/csm-1b"
device = "cuda"

# set logs to ensure no recompilation and graph breaks
torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# use static cache, enabling automatically torch compile with fullgraph and reduce-overhead
model.generation_config.max_length = 250 # big enough to avoid recompilation
model.generation_config.max_new_tokens = None # would take precedence over max_length
model.generation_config.cache_implementation = "static"
model.depth_decoder.generation_config.cache_implementation = "static"

# generation kwargs
gen_kwargs = {
    "do_sample": False,
    "depth_decoder_do_sample": False,
    "temperature": 1.0,
    "depth_decoder_temperature": 1.0,
}

# Define a timing decorator
class TimerContext:
    def __init__(self, name="Execution"):
        self.name = name
        self.start_event = None
        self.end_event = None
        
    def __enter__(self):
        # Use CUDA events for more accurate GPU timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event) / 1000.0
        print(f"{self.name} time: {elapsed_time:.4f} seconds")

# prepare the inputs 
ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")

conversation = [
    {
        "role": f"{ds[0]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[0]["text"]},
            {"type": "audio", "path": ds[0]["audio"]["array"]},
        ],
    },
    {
        "role": f"{ds[1]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[1]["text"]},
            {"type": "audio", "path": ds[1]["audio"]["array"]},
        ],
    },
    {
        "role": f"{ds[2]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[2]["text"]},
        ],
    },
]

padded_inputs_1 = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(device)

print("\n" + "="*50)
print("First generation - compiling and recording CUDA graphs...")
with TimerContext("First generation"):
    _ = model.generate(**padded_inputs_1, **gen_kwargs)
print("="*50)

print("\n" + "="*50)
print("Second generation - fast !!!")
with TimerContext("Second generation"):
    _ = model.generate(**padded_inputs_1, **gen_kwargs)
print("="*50)

# now with different inputs
conversation = [
    {
        "role": f"{ds[0]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[2]["text"]},
            {"type": "audio", "path": ds[2]["audio"]["array"]},
        ],
    },
    {
        "role": f"{ds[1]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[3]["text"]},
            {"type": "audio", "path": ds[3]["audio"]["array"]},
        ],
    },
    {
        "role": f"{ds[2]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[4]["text"]},
        ],
    },
]
padded_inputs_2 = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(device)

print("\n" + "="*50)
print("Generation with other inputs!")
with TimerContext("Generation with different inputs"):
    _ = model.generate(**padded_inputs_2, **gen_kwargs)
print("="*50)
```

</details>

### Fine-tuning & training 📉

CSM can be fine-tuned using [Transformers' Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer).

<details>

<summary> code snippet </summary>

```python
from datasets import load_dataset, Audio
from transformers import (
    CsmForConditionalGeneration,
    TrainingArguments,
    CsmProcessor,
    Trainer
)

processor = CsmProcessor.from_pretrained("sesame/csm-1b")
model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b")
model.train()
model.codec_model.eval()

ds = load_dataset("eustlb/dailytalk-conversations-grouped", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

def data_collator(samples):
    conversations = [] 

    for sample in samples:
        concatenated_audio_array = sample["audio"]["array"]
        audio = [concatenated_audio_array[s: e] for s, e in sample["audio_cut_idxs"]]
            
        conversation = []
        for speaker_id, text, audio in zip(sample["speaker_ids"], sample["texts"], audio):
            conversation.append({
                "role": f"{speaker_id}",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "audio", "audio": audio}
                ]
            })
            
        conversations.append(conversation)

    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        output_labels=True,
    )
    return inputs

training_args = TrainingArguments(
    "test-trainer",
    remove_unused_columns=False,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model, 
    training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

trainer.train()
```

</details>

---

## FAQ

**Does this model come with any voices?**

The model open sourced here is a base generation model. It is capable of producing a variety of voices, but it has not been fine-tuned on any specific voice.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general purpose multimodal LLM. It cannot generate text. We suggest using a separate LLM for text generation.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

**Authors**
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.