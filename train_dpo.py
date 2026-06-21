"""
DPO (Direct Preference Optimization) fine-tuning using TRL DPOTrainer.
Starts from an SFT checkpoint and trains on chosen/rejected preference pairs.
"""
import yaml
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
import wandb
from loguru import logger


def load_config(path: str = "configs/dpo_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    logger.info(f"Starting DPO training from: {cfg['model']['base_model']}")

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        tags=cfg["wandb"]["tags"],
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO requires left padding

    # Policy model (to be optimized)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Reference model (frozen copy — DPO KL penalty)
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA to policy model only
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

    # Data
    train_ds = load_from_disk("./data/dpo_train")
    eval_ds = load_from_disk("./data/dpo_eval")
    logger.info(f"DPO Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # DPO Training args
    t = cfg["training"]
    d = cfg["dpo"]
    training_args = DPOConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        beta=d["beta"],
        loss_type=d["loss_type"],
        max_prompt_length=d["max_prompt_length"],
        max_length=d["max_length"],
        label_smoothing=d["label_smoothing"],
        report_to="wandb",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    out_dir = Path(t["output_dir"])
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"DPO model saved to {out_dir}")

    if cfg["hub"]["push_to_hub"]:
        trainer.push_to_hub(cfg["hub"]["hub_model_id"])

    wandb.finish()


if __name__ == "__main__":
    main()
