"""Accelerate-driven demo for the parallel PLD speculative decoder.

Run with two processes, for example:

accelerate launch --num_processes 2 run_parallel_pld_accelerate.py "Your prompt here"
"""

import argparse
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast as broadcast_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pld_engine import parallel_speculative_decoding_with_pld


def build_args(namespace: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        temp=namespace.temperature,
        top_k=namespace.top_k,
        top_p=namespace.top_p,
        gamma=namespace.gamma,
        max_tokens=namespace.max_new_tokens,
        pld_max_ngram_size=namespace.pld_max_ngram_size,
        pld_num_pred_tokens=namespace.pld_num_pred_tokens,
    )


def load_model(name: str, dtype: torch.dtype) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()


def main() -> None:
    accelerator = Accelerator()

    if accelerator.num_processes < 2:
        raise RuntimeError(
            "Parallel PLD demo requires at least two accelerator processes. "
            "Launch with e.g. `accelerate launch --num_processes 2 run_parallel_pld_accelerate.py "
            "\"Your prompt\"`"
        )

    parser = argparse.ArgumentParser(description="Parallel PLD speculative decoding demo")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="Compare the roles of draft and target models in speculative decoding.",
    )
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--pld-max-ngram-size", type=int, default=3)
    parser.add_argument("--pld-num-pred-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draft-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--target-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging for debugging")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.draft_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if accelerator.is_main_process:
        model = load_model(args.draft_model, dtype)
        draft_model = model
        target_model = None
    else:
        model = load_model(args.target_model, dtype)
        draft_model = None
        target_model = model

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded.input_ids
    input_ids = input_ids.to(accelerator.device)
    input_ids = broadcast_tensor(input_ids, from_process=0)

    pl_args = build_args(args)
    num_acc_tokens: list[int] = []

    prefix, draft_inc, target_inc = parallel_speculative_decoding_with_pld(
        input_ids,
        accelerator=accelerator,
        args=pl_args,
        draft_model=draft_model,
        target_model=target_model,
        vocab_size=tokenizer.vocab_size,
        seed=args.seed,
        num_acc_tokens=num_acc_tokens,
        debug=args.debug,
    )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        completion = tokenizer.batch_decode(
            prefix[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        accelerator.print(f"Draft forward passes: {draft_inc}")
        accelerator.print(f"Target forward passes: {target_inc}")
        accelerator.print(f"Accepted token counts: {num_acc_tokens}")
        accelerator.print("\nGenerated text:\n" + completion.strip())


if __name__ == "__main__":
    main()
