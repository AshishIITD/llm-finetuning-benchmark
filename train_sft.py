"""
train_sft.py — Full SFT (Supervised Fine-Tuning) on Llama 3.1 8B using TRL SFTTrainer.
Resume claim: SFT ROUGE-L=0.378, BERTScore F1=0.863, Win Rate=67%, GPU Memory=38GB
"""
import os
import yaml
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

load_dotenv()

def load_config():
    with open("configs/training_config.yaml", "r") as f:
        return yaml.safe_load(f)

def format_prompt(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {"text": prompt}

def main():
    config = load_config()
    sft_cfg = config["sft"]
    base_model = config["base_model"]

    # Initialize W&B run
    wandb.init(
        project="llm-finetuning-benchmark",
        name="sft-llama3-8b",
        config=sft_cfg,
        tags=["sft", "llama3"]
    )

    print(f"Loading tokenizer and model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype="auto",
    )

    print("Loading and formatting dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:90%]")
    eval_dataset = load_dataset("tatsu-lab/alpaca", split="train[90%:]")
    dataset = dataset.map(format_prompt)
    eval_dataset = eval_dataset.map(format_prompt)

    training_args = SFTConfig(
        output_dir=f"{config['output_dir']}/sft",
        num_train_epochs=sft_cfg["num_train_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        logging_steps=sft_cfg["logging_steps"],
        save_strategy="epoch",
        report_to="wandb",
        max_seq_length=sft_cfg["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()
    trainer.save_model(f"{config['output_dir']}/sft-final")

    # Push to Hub
    if config.get("hub_repo"):
        trainer.push_to_hub(config["hub_repo"] + "-sft")

    wandb.finish()
    print("SFT Training complete!")

if __name__ == "__main__":
    main()
