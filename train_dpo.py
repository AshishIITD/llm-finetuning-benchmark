"""
train_dpo.py — DPO (Direct Preference Optimization) on Llama 3.1 8B using TRL DPOTrainer.
Resume claim: DPO achieves highest win rate (73%) vs base model using GPT-4o as judge.
"""
import os
import yaml
import wandb
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from dotenv import load_dotenv

load_dotenv()

def load_config():
    with open("configs/training_config.yaml", "r") as f:
        return yaml.safe_load(f)

def prepare_dpo_dataset(tokenizer):
    """
    DPO requires a dataset with 'prompt', 'chosen', and 'rejected' fields.
    We use the HH-RLHF preference dataset from Anthropic as it has this format.
    """
    raw = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")

    def extract_preference(example):
        return {
            "prompt": "Human: " + example["chosen"].split("Human: ")[1].split("Assistant:")[0].strip(),
            "chosen": "Assistant: " + example["chosen"].split("Assistant:")[-1].strip(),
            "rejected": "Assistant: " + example["rejected"].split("Assistant:")[-1].strip(),
        }

    dataset = raw.map(extract_preference, remove_columns=raw.column_names)
    return dataset.train_test_split(test_size=0.1)

def main():
    config = load_config()
    dpo_cfg = config["dpo"]
    base_model = config["base_model"]

    wandb.init(
        project="llm-finetuning-benchmark",
        name="dpo-llama3-8b",
        config=dpo_cfg,
        tags=["dpo", "rlhf", "llama3", "preference"]
    )

    print(f"Loading tokenizer and model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype="auto",
    )

    # Reference model (frozen copy of the base model for KL divergence)
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype="auto",
    )

    print("Preparing DPO preference dataset...")
    splits = prepare_dpo_dataset(tokenizer)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    training_args = DPOConfig(
        output_dir=f"{config['output_dir']}/dpo",
        beta=dpo_cfg["beta"],
        num_train_epochs=dpo_cfg["num_train_epochs"],
        per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        learning_rate=dpo_cfg["learning_rate"],
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        max_length=dpo_cfg["max_length"],
        max_prompt_length=dpo_cfg["max_prompt_length"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()
    trainer.save_model(f"{config['output_dir']}/dpo-final")

    if config.get("hub_repo"):
        trainer.push_to_hub(config["hub_repo"] + "-dpo")

    wandb.finish()
    print("DPO Training complete!")

if __name__ == "__main__":
    main()
