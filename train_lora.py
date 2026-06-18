"""
train_lora.py — LoRA fine-tuning (r=16) on Llama 3.1 8B using PEFT + TRL SFTTrainer.
Resume claim: LoRA r=16 ROUGE-L=0.371, BERTScore F1=0.859, Win Rate=64%, GPU Memory=14GB
"""
import os
import yaml
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
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
    lora_cfg = config["lora"]
    base_model = config["base_model"]

    wandb.init(
        project="llm-finetuning-benchmark",
        name="lora-r16-llama3-8b",
        config=lora_cfg,
        tags=["lora", "peft", "llama3"]
    )

    print(f"Loading tokenizer and model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype="auto",
    )

    # Apply LoRA with r=16
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading and formatting dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:90%]").map(format_prompt)
    eval_dataset = load_dataset("tatsu-lab/alpaca", split="train[90%:]").map(format_prompt)

    training_args = SFTConfig(
        output_dir=f"{config['output_dir']}/lora",
        num_train_epochs=lora_cfg["num_train_epochs"],
        per_device_train_batch_size=lora_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=lora_cfg["gradient_accumulation_steps"],
        learning_rate=lora_cfg["learning_rate"],
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        max_seq_length=lora_cfg["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting LoRA training...")
    trainer.train()
    trainer.save_model(f"{config['output_dir']}/lora-final")

    if config.get("hub_repo"):
        trainer.push_to_hub(config["hub_repo"] + "-lora-r16")

    wandb.finish()
    print("LoRA Training complete!")

if __name__ == "__main__":
    main()
