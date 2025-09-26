"""Run a prompt through the PLD-enhanced speculative decoder using Llama 3.2 models."""

import argparse
import sys
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pld_engine import PLDKVCacheModel


def load_model_and_tokenizer(model_name: str, dtype: torch.dtype) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    return model, tokenizer


def run_pld_generation(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    gamma: int,
    max_ngram_size: int,
    num_pred_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model, tokenizer = load_model_and_tokenizer(model_name, dtype)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    kv_model = PLDKVCacheModel(model, temperature=temperature, top_k=top_k, top_p=top_p)
    kv_model.vocab_size = model.config.vocab_size

    target_length = input_ids.shape[1] + max_new_tokens
    generated = input_ids
    eos_token_id = tokenizer.eos_token_id

    while generated.shape[1] < target_length:
        generated = kv_model.generate_with_pld(
            generated,
            gamma=gamma,
            max_ngram_size=max_ngram_size,
            num_pred_tokens=num_pred_tokens,
        )
        if eos_token_id is not None and (generated[:, -1] == eos_token_id).all():
            break

    if generated.shape[1] > target_length:
        generated = torch.cat([generated[:, :input_ids.shape[1]], generated[:, input_ids.shape[1]:target_length]], dim=1)

    completion = tokenizer.batch_decode(generated[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return completion.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="PLD inference demo with Llama 3.2 models")
    parser.add_argument("prompt", type=str, nargs="?", default="Explain the benefits of speculative decoding in one paragraph.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--pld-max-ngram-size", type=int, default=3)
    parser.add_argument("--pld-num-pred-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    models = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]

    for name in models:
        print(f"\n=== Running PLD inference with {name} ===")
        try:
            completion = run_pld_generation(
                model_name=name,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                max_ngram_size=args.pld_max_ngram_size,
                num_pred_tokens=args.pld_num_pred_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        except Exception as exc:
            print(f"Failed to generate with {name}: {exc}", file=sys.stderr)
            continue
        print(completion)


if __name__ == "__main__":
    main()
