"""
LoRA / QLoRA fine-tuning using PEFT + SFTTrainer.
Supports both LoRA (bf16) and QLoRA (4-bit) via config flag.
"""
import yaml
import os
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import wandb
from loguru import logger


def load_config(path: str = "configs/lora_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_qlora_model(cfg: dict):
    """Load model with 4-bit quantization for QLoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    return model


def build_lora_model(cfg: dict):
    """Load model in bf16 for standard LoRA."""
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def apply_lora(model, cfg: dict):
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


def main():
    cfg = load_config()
    use_qlora = cfg["quantization"]["enabled"]
    run_name = cfg["wandb"]["run_name"]
    if use_qlora:
        run_name = run_name.replace("lora", "qlora")

    logger.info(f"Starting {'QLoRA' if use_qlora else 'LoRA'} training: {cfg['model']['base_model']}")

    # W&B
    wandb.init(
        project=cfg["wandb"]["project"],
        name=run_name,
        tags=cfg["wandb"]["tags"] + (["qlora"] if use_qlora else []),
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model
    model = build_qlora_model(cfg) if use_qlora else build_lora_model(cfg)
    model, peft_config = apply_lora(model, cfg)

    # Data
    train_ds = load_from_disk("./data/sft_train")
    eval_ds = load_from_disk("./data/sft_eval")
    logger.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # Training args
    t = cfg["training"]
    training_args = SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_seq_length=t["max_seq_length"],
        bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        report_to="wandb",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save
    out_dir = Path(t["output_dir"])
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"Model saved to {out_dir}")

    if cfg["hub"]["push_to_hub"]:
        trainer.push_to_hub(cfg["hub"]["hub_model_id"])
        logger.info(f"Pushed to HuggingFace Hub: {cfg['hub']['hub_model_id']}")

    wandb.finish()


if __name__ == "__main__":
    main()
