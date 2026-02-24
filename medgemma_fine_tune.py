import os
import pickle
from typing import List, Dict
import re 

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Constants & Config
MODEL_ID = "google/medgemma-4b-it"
OUTPUT_DIR = "medgemma-4b-it-sft-lora-interactome"
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 5e-4
BATCH_SIZE = 1
GRAD_ACCUMULATION = 2
EVAL_STEPS = 15

def format_protein_sequences(text):
    """
    Extracts ENSP protein IDs and sequences from any block of text
    and returns formatted output as:

    Protein: <ENSP_ID>
    Sequence: <SEQUENCE>
    """

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

def build_sample(user_prompt: str, assistant_response: str) -> dict:
    """Constructs SFT data sample with image + conversation messages."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_response}
                ]
            }
        ]
    }


# Utility Functions
def load_samples(file_path: str) -> List[Dict]:
    """Load dataset samples from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def collate_fn(samples: List[Dict], processor: AutoProcessor) -> Dict:
    """
    Data collator for SFT training.
    Prepares prompts and images for the model and handles label masking.
    """
    prompts = []

    for sample in samples:
        prompt = processor.apply_chat_template(
            sample["messages"],
            add_generation_prompt=False,
            tokenise=False
        ).strip()
        prompts.append(prompt)

    batch = processor(text=prompts, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.special_tokens_map["boi_token"]
    )
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100 
    batch["labels"] = labels
    return batch


def check_gpu_bf16_support():
    """Ensure the GPU supports bfloat16 precision."""
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("GPU does not support bfloat16. Use an A100 or H100.")


# Main Script
def main():
    sample_1_input_folder = "sft_input"
    sample_1_output_folder = "sft_output"
    sample_2_input_folder = "sft_input_size_3"
    sample_2_output_folder = "sft_output_size_3"

    input_samples = []
    output_samples = []
    input_files = [os.path.join(sample_1_input_folder,file) for file in os.listdir(sample_1_input_folder)]

    input_files.extend([os.path.join(sample_2_input_folder,file) for file in os.listdir(sample_2_input_folder)])

    for input_file in input_files:
        with open(input_file, "r") as file:
            content = file.read()
            formatted_content = format_protein_sequences(content)
        input_samples.append(formatted_content)
        output_file = input_file.replace("input","output")
        with open(output_file, "r") as file:
            content = file.read()
            output_samples.append(content)

    samples = []
    for input_text, output_text in zip(input_samples, output_samples):
        sample = build_sample(input_text, output_text)
        samples.append(sample)

    train_samples, test_samples = train_test_split(samples, test_size=0.1, random_state=42)

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"

    # Check GPU capabilities
    check_gpu_bf16_support()

    # Model kwargs for bfloat16 and 4-bit quantization
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        ),
    )

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)

    # LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # SFT training configuration
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=2,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        push_to_hub=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        packing=False
    )

    # Trainer initialization
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_samples,
        eval_dataset=test_samples,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=lambda batch: collate_fn(batch, processor),
    )

    # Start training
    trainer.train()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
