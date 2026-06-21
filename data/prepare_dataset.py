"""
Dataset preparation for fine-tuning benchmark.
Downloads and formats datasets for SFT and DPO training.
- SFT: instruction-response pairs (Alpaca format)
- DPO: prompt + chosen + rejected pairs
"""
import json
from pathlib import Path
from datasets import load_dataset
from loguru import logger

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


def prepare_sft_dataset():
    """Download and format Alpaca-cleaned for SFT."""
    logger.info("Loading Alpaca-cleaned dataset...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")

    def format_alpaca(example):
        if example["input"]:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
        return {
            "prompt": prompt,
            "response": example["output"],
            "text": prompt + example["output"] + "</s>",
        }

    ds = ds.map(format_alpaca, remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=0.05, seed=42)
    split["train"].save_to_disk(str(DATA_DIR / "sft_train"))
    split["test"].save_to_disk(str(DATA_DIR / "sft_eval"))
    logger.info(f"SFT dataset: {len(split['train'])} train, {len(split['test'])} eval")
    return split


def prepare_dpo_dataset():
    """Download and format HH-RLHF for DPO (chosen/rejected pairs)."""
    logger.info("Loading Anthropic HH-RLHF dataset...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train[:10000]")

    def format_dpo(example):
        return {
            "prompt": example["chosen"].rsplit("\n\nAssistant:", 1)[0] + "\n\nAssistant:",
            "chosen": example["chosen"].rsplit("\n\nAssistant:", 1)[-1].strip(),
            "rejected": example["rejected"].rsplit("\n\nAssistant:", 1)[-1].strip(),
        }

    ds = ds.map(format_dpo, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["prompt"]) < 1024 and len(x["chosen"]) > 10)
    split = ds.train_test_split(test_size=0.05, seed=42)
    split["train"].save_to_disk(str(DATA_DIR / "dpo_train"))
    split["test"].save_to_disk(str(DATA_DIR / "dpo_eval"))
    logger.info(f"DPO dataset: {len(split['train'])} train, {len(split['test'])} eval")
    return split


def prepare_eval_dataset(n_samples: int = 200):
    """Prepare a held-out evaluation set with reference answers."""
    logger.info("Preparing evaluation dataset...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train[-500:]")

    eval_samples = []
    for i, ex in enumerate(ds):
        if i >= n_samples:
            break
        eval_samples.append({
            "id": i,
            "instruction": ex["instruction"],
            "input": ex.get("input", ""),
            "reference_answer": ex["output"],
        })

    with open(DATA_DIR / "eval_set.json", "w") as f:
        json.dump(eval_samples, f, indent=2)
    logger.info(f"Saved {len(eval_samples)} eval samples to data/eval_set.json")
    return eval_samples


if __name__ == "__main__":
    logger.info("=== Preparing datasets for fine-tuning benchmark ===")
    prepare_sft_dataset()
    prepare_dpo_dataset()
    prepare_eval_dataset()
    logger.info("=== All datasets ready ===")
