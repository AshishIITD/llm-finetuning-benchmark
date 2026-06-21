"""
Evaluation pipeline for the fine-tuning benchmark.
Computes ROUGE-L, BERTScore, and GPT-4o win rate (LLM-as-judge).
Outputs results/benchmark_results.json for the report generator.
"""
import json
import argparse
import os
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import evaluate as hf_evaluate
from bert_score import score as bert_score_fn
from openai import OpenAI

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "base": {"path": "meta-llama/Llama-3.1-8B-Instruct", "is_peft": False},
    "sft": {"path": "./checkpoints/sft_final", "is_peft": False},
    "lora": {"path": "./checkpoints/lora", "is_peft": True, "base": "meta-llama/Llama-3.1-8B-Instruct"},
    "dpo": {"path": "./checkpoints/dpo", "is_peft": True, "base": "./checkpoints/sft_final"},
}

JUDGE_PROMPT = """You are an expert evaluator. Compare two responses to the same instruction.

Instruction: {instruction}
Input: {input}

Response A: {response_a}
Response B: {response_b}

Which response is better? Consider: accuracy, helpfulness, clarity, completeness.
Respond with ONLY: "A" or "B" or "TIE"
"""


def load_model(name: str, cfg: dict):
    logger.info(f"Loading model: {name}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.get("base", cfg["path"]), trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    if cfg["is_peft"]:
        base = AutoModelForCausalLM.from_pretrained(
            cfg["base"], torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, cfg["path"])
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, input_text: str = "") -> str:
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def compute_rouge(predictions: list[str], references: list[str]) -> float:
    rouge = hf_evaluate.load("rouge")
    result = rouge.compute(predictions=predictions, references=references)
    return round(result["rougeL"], 4)


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return round(F1.mean().item(), 4)


def compute_win_rate(
    model_responses: list[str],
    base_responses: list[str],
    eval_samples: list[dict],
    n_samples: int = 50,
) -> float:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    wins = 0
    total = min(n_samples, len(eval_samples))

    for i in range(total):
        sample = eval_samples[i]
        prompt = JUDGE_PROMPT.format(
            instruction=sample["instruction"],
            input=sample.get("input", ""),
            response_a=model_responses[i][:400],
            response_b=base_responses[i][:400],
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
            )
            verdict = resp.choices[0].message.content.strip().upper()
            if verdict == "A":
                wins += 1
            elif verdict == "TIE":
                wins += 0.5
        except Exception as e:
            logger.warning(f"Judge call failed: {e}")

    return round(wins / total, 4)


def evaluate_model(name: str, cfg: dict, eval_samples: list[dict], base_responses: list[str] = None):
    model, tokenizer = load_model(name, cfg)
    responses = []

    logger.info(f"Generating responses for {name}...")
    for sample in eval_samples:
        resp = generate_response(model, tokenizer, sample["instruction"], sample.get("input", ""))
        responses.append(resp)

    references = [s["reference_answer"] for s in eval_samples]
    rouge_l = compute_rouge(responses, references)
    bertscore = compute_bertscore(responses, references)

    win_rate = None
    if base_responses and os.getenv("OPENAI_API_KEY"):
        logger.info(f"Computing win rate for {name} vs base...")
        win_rate = compute_win_rate(responses, base_responses, eval_samples)

    result = {
        "model": name,
        "rouge_l": rouge_l,
        "bertscore_f1": bertscore,
        "win_rate_vs_base": win_rate,
        "n_eval_samples": len(eval_samples),
    }
    logger.info(f"{name}: ROUGE-L={rouge_l} | BERTScore={bertscore} | WinRate={win_rate}")

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return responses, result


def main(models_to_eval: list[str] = None):
    with open("./data/eval_set.json") as f:
        eval_samples = json.load(f)[:100]
    logger.info(f"Evaluating on {len(eval_samples)} samples")

    models_to_eval = models_to_eval or list(MODELS.keys())
    all_results = {}
    base_responses = None

    for name in models_to_eval:
        if name not in MODELS:
            logger.warning(f"Unknown model: {name}, skipping")
            continue
        responses, result = evaluate_model(name, MODELS[name], eval_samples, base_responses)
        all_results[name] = result
        if name == "base":
            base_responses = responses
        with open(RESULTS_DIR / f"{name}_responses.json", "w") as f:
            json.dump(responses, f, indent=2)

    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to results/benchmark_results.json")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None, help="Models to evaluate (base sft lora dpo)")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    models = None if args.all else args.models
    results = main(models)
    print("\n=== BENCHMARK RESULTS ===")
    for name, r in results.items():
        print(f"{name:10s} | ROUGE-L: {r['rouge_l']:.4f} | BERTScore: {r['bertscore_f1']:.4f} | WinRate: {r.get('win_rate_vs_base', 'N/A')}")
