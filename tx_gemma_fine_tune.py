import torch
import re
import os
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 5e-4
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
EVAL_STEPS = 20
MAX_SEQ_LENGTH = 512


def format_protein_sequences(text):
    pattern = r'(ENSP\d+):([A-Z]+)'
    matches = re.findall(pattern, text)

    seen = set()
    formatted_entries = []

    for ensp_id, sequence in matches:
        if ensp_id not in seen:
            seen.add(ensp_id)
            formatted_entries.append(
                f"Protein: {ensp_id}\nSequence: {sequence}"
            )

    return "\n\n".join(formatted_entries)


sample_folders = ["sft_input", "sft_input_size_3"]

input_samples = []
output_samples = []

for folder in sample_folders:
    for file in os.listdir(folder):
        input_path = os.path.join(folder, file)

        with open(input_path, "r") as f:
            content = f.read()
            input_samples.append(format_protein_sequences(content))

        output_path = input_path.replace("input", "output")
        with open(output_path, "r") as f:
            output_samples.append(f.read())


# HuggingFace-safe split
input_train, input_val, output_train, output_val = train_test_split(
    input_samples,
    output_samples,
    test_size=0.1,
    random_state=42,
)

train_data = Dataset.from_dict({
    "input_text": input_train,
    "output_text": output_train,
})

val_data = Dataset.from_dict({
    "input_text": input_val,
    "output_text": output_val,
})


def formatting_func(example):
    return f"{example['input_text']}\n{example['output_text']}<eos>"

model_id = "google/txgemma-2b-predict"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Memory optimizations
model.gradient_checkpointing_enable()
model.config.use_cache = False

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)

sft_config = SFTConfig(
    output_dir="txgemma_finetuned",

    num_train_epochs=NUM_TRAIN_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUMULATION,

    learning_rate=LEARNING_RATE,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",

    optim="paged_adamw_8bit",
    bf16=True,
    fp16=False,

    logging_steps=2,

    save_strategy="steps",
    save_steps=200,

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    max_grad_norm=0.3,

    packing=True,
    remove_unused_columns=False,
    label_names=["labels"],

    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_data,
    eval_dataset=val_data,
    formatting_func=formatting_func,
)

trainer.train()

print("Training completed successfully.")