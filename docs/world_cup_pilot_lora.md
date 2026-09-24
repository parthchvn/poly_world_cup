# First World Cup QLoRA run

Use `datasets/world_cup_2026_pilot_15k` for a small experiment drawn from
multiple matches. Its split manifest describes which fixtures were selected
and the actual counts. The budget counts **assistant decision targets**, not
JSONL lines: one line can contain several successive decisions for one actor.
The published splits are `train.jsonl.gz`, `validation.jsonl.gz`, and
`test.jsonl.gz`. The trainer reads them directly without unpacking; it also
accepts `.jsonl` files and prefers those if both versions are present.
The targets are captured executions, conditional on an execution occurring.
This experiment does not train future trade-versus-no-trade prediction.

Each `messages` conversation supplies market meaning, time-appropriate context,
and the actor's retained earlier decisions. Prior assistant answers remain in
the context for later turns. Separate JSONL lines do not share attention or
memory. Outer `fixture_id`, `actor_id`, `market_id`, `target_count`, and other
metadata are not serialized into model input automatically.

All contracts for a held-out match belong to the same split. Sampling from many
matches does not make a random row split safe. Test matches must remain unused
when selecting training settings. The baseline and adapter should see the same
held-out examples and the same history. A model's release before a match is
useful evidence about temporal separation, not proof that all supplied context
or preprocessing is free of leakage.

## Libraries and output

`scripts/train_market_qlora.py` uses **Transformers Trainer, PEFT and
bitsandbytes**. These libraries provide the training loop, LoRA layers and
quantized model loading. There is no custom implementation of the LoRA
algorithm. TRL's SFTTrainer is another suitable trainer; this entry point uses
Transformers Trainer so it can supply explicit masks for our multi-turn JSON
targets without assuming the model's chat template has assistant-mask support.

The frozen Qwen3.6 text model is loaded in 4-bit NF4. Only its LoRA parameters
are updated. Defaults are rank 16, alpha 32, dropout 0.05, learning rate 0.0001,
one epoch, batch size 1, and gradient accumulation 8. These are starting choices,
not measured optimal settings. GPU memory also depends on sequence length,
token vocabulary, implementation and activations; a GPU size alone does not
guarantee a successful run.

Qwen3.6 shares the Qwen3.5 architecture. The entry point uses the documented
`Qwen3_5ForCausalLM` text-only class, not the vision model. The official template
is rendered with `enable_thinking=False` and `preserve_thinking=True`. With no
reasoning supplied, this keeps a consistent empty thinking wrapper across
assistant turns. Loss covers each assistant's actual JSON and its
`<|im_end|>` token. System/user messages, empty wrappers and padding are masked
with `-100`. The model's ordinary causal mask prevents a prediction from
attending to later turns.

## Run on the existing RunPod GPU

The model folder must already contain the complete model, tokenizer and chat
template at `/workspace/models/Qwen3.6-27B`. Run these commands from the
`poly_world_cup` repository or the extracted pilot bundle in the RunPod terminal.
These files already contain training conversations. Feed them directly to the
new trainer; the earlier raw-row converter is not needed.
The environment needs CUDA PyTorch; a normal CUDA RunPod PyTorch image
provides it. Training dependencies are separate from this repository's
standard-library data collection requirements.

To download just the pilot data and trainer into the persistent workspace:

```bash
cd /workspace
curl -fL https://raw.githubusercontent.com/parthchvn/poly_world_cup/main/downloads/world_cup_15k_qlora.zip -o world_cup_15k_qlora.zip
python -m zipfile -e world_cup_15k_qlora.zip .
cd world_cup_15k_qlora
```

```bash
export HF_HOME=/workspace/hf_cache
export PIP_CACHE_DIR=/workspace/pip_cache
python -m pip install --upgrade transformers peft datasets accelerate bitsandbytes

CUDA_VISIBLE_DEVICES=0 python scripts/train_market_qlora.py --dataset-dir datasets/world_cup_2026_pilot_15k --prepare-only

CUDA_VISIBLE_DEVICES=0 python scripts/train_market_qlora.py --dataset-dir datasets/world_cup_2026_pilot_15k --max-steps 10 --out outputs/world_cup_qlora_smoke
```

The preparation command loads only the tokenizer. It reports actual token
lengths, target counts and loss-token counts. It refuses sequences over 8,192
tokens instead of silently dropping targets. Increase `--max-length` only if
the model and GPU budget support it, or rebuild shorter actor conversations
with explicit earlier context. The 10-step command performs real optimization
on the GPU and evaluates on validation data. It is an installation and memory
check, not a completed experiment.

After the smoke run succeeds, start the full one-epoch pilot in a fresh output
directory:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_market_qlora.py --dataset-dir datasets/world_cup_2026_pilot_15k --out outputs/world_cup_qlora
```

The full command starts from the original model, not the smoke adapter. It uses
validation loss for checkpoint selection and never opens the test split.
It expects one visible CUDA GPU with bfloat16 support and does not automatically
offload the model or shard it across devices. The hybrid attention stack's
kernel availability affects speed and memory; diagnose the first 10 steps on
the actual RunPod machine before committing to the longer run.

On completion, `outputs/world_cup_qlora/adapter/` contains:

- `adapter_model.safetensors`: learned A and B matrices for each adapted layer.
- `adapter_config.json`: rank, target modules and other adapter settings.
- The tokenizer and chat template used for this run.

`outputs/world_cup_qlora/training_metadata.json` records dependency versions,
training settings, hashes of the uncompressed train/validation JSONL content,
token counts and the selected
checkpoint. Keep that alongside the adapter for reproducibility. The saved
adapter is not a standalone model: inference needs the original base weights
as well. It is many small A/B matrix pairs, not one global LoRA matrix.

## Use the saved adapter

On RunPod, after a completed run, the following illustrates generation for the
**first decision** in one held-out conversation. It is an inspection example,
not the full evaluation. Do not inspect test outputs while choosing settings.

```python
import json
import gzip
import torch
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

base_path = "/workspace/models/Qwen3.6-27B"
adapter_path = "outputs/world_cup_qlora/adapter"
tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
base = Qwen3_5ForCausalLM.from_pretrained(
    base_path,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)
model = PeftModel.from_pretrained(base, adapter_path).eval()
with gzip.open("datasets/world_cup_2026_pilot_15k/test.jsonl.gz", "rt") as source:
    record = json.loads(next(source))
target_index = next(i for i, m in enumerate(record["messages"]) if m["role"] == "assistant")
prompt = tokenizer.apply_chat_template(
    record["messages"][:target_index],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
    preserve_thinking=True,
)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=512, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )
print("Predicted:", tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
print("Observed:", record["messages"][target_index]["content"])
```

For the final comparison, evaluate all target turns for both the unchanged
base and saved adapter. At each target use only the preceding messages. Decide
in advance whether earlier **observed** trades are supplied, as intended here,
or the model's own predictions are rolled forward. Those measure different
tasks. Report valid-JSON rate, side/outcome accuracy, and errors for shares and
price, along with results per match. Grouped same-time executions require a
defined matching rule if scoring individual trades. Validation loss alone is
not a measure of trading accuracy or profitability.

The training entry point has not been run on a GPU in this workspace. A saved
dataset or preparation report is not evidence that QLoRA training completed.

## Primary references

- [Qwen3.5/3.6 text-model loading](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Official Qwen3.6 chat template](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/chat_template.jinja)
- [PEFT quantized training](https://huggingface.co/docs/peft/developer_guides/quantization)
- [PEFT adapter checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [TRL SFTTrainer and assistant-only training](https://huggingface.co/docs/trl/sft_trainer)
