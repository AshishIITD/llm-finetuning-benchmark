"""
Full SFT (Supervised Fine-Tuning) — no LoRA, all weights updated.
Use this as the baseline and as the starting point for DPO.
Requires ~38GB GPU. Use train_lora.py for lower memory options.
"""
import yaml
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import wandb
from loguru import logger


def load_config(path: str = "configs/sft_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    logger.info(f"Starting full SFT: {cfg['model']['base_model']}")

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        tags=cfg["wandb"]["tags"],
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params / 1e9:.2f}B")

    train_ds = load_from_disk("./data/sft_train")
    eval_ds = load_from_disk("./data/sft_eval")

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
        tokenizer=tokenizer,
    )

    trainer.train()
    out_dir = Path(t["output_dir"])
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"SFT model saved to {out_dir}")

    if cfg["hub"]["push_to_hub"]:
        trainer.push_to_hub(cfg["hub"]["hub_model_id"])

    wandb.finish()


if __name__ == "__main__":
    main()
