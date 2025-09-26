"""Parallel PLD decoding helpers extracted from the engine module."""

from typing import Any, List, Optional, Tuple

import torch
from accelerate import Accelerator

from .kvcache import KVCacheModel
from .pld_model import PLDKVCacheModel, find_candidate_pred_tokens
from .util import max_fn, sample


def parallel_speculative_decoding_with_pld(
    prefix: torch.Tensor,
    *,
    accelerator: Accelerator,
    args: Any,
    draft_model: Optional[torch.nn.Module] = None,
    target_model: Optional[torch.nn.Module] = None,
    vocab_size: int,
    seed: int,
    num_acc_tokens: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, int, int]:
    """Parallel PLD speculative decoding shared by both engine and demos.

    Returns the updated prefix together with draft/target forward counts that were
    consumed in this call. ``num_acc_tokens`` is mutated in-place when provided.
    """

    if num_acc_tokens is None:
        num_acc_tokens = []

    if accelerator.is_main_process:
        if draft_model is None:
            raise ValueError("Draft model is required on the main process")
        model = PLDKVCacheModel(draft_model, args.temp, args.top_k, args.top_p)
        device = draft_model.device
    else:
        if target_model is None:
            raise ValueError("Target model is required on the worker process")
        model = KVCacheModel(target_model, args.temp, args.top_k, args.top_p)
        device = target_model.device

    model.vocab_size = vocab_size

    max_tokens = prefix.shape[1] + args.max_tokens
    max_ngram_size = getattr(args, "pld_max_ngram_size", 3)
    num_pred_tokens = getattr(args, "pld_num_pred_tokens", 10)

    cur_mode = True  # True: Pre-verify, False: Post-verify
    num_acc_token = 0
    previous_tokens = args.gamma
    current_tokens = 0

    draft_forward_times = 0
    target_forward_times = 0

    prefix_out = prefix

    while prefix_out.shape[1] < max_tokens:
        prefix_len = prefix_out.shape[1]
        input_ids = prefix_out.to(device)

        if accelerator.is_main_process:
            x = model.generate_with_pld(
                input_ids,
                args.gamma,
                max_ngram_size,
                num_pred_tokens,
            )
            current_tokens = model.get_actual_tokens_generated()
            prob = model._prob_history[:, prefix_len - previous_tokens - 1 : prefix_len, :vocab_size].to(
                torch.float32
            )
            prob[:, 0, 0] = -1
            prob[:, 0, 1 : previous_tokens + current_tokens] = x[
                :, prefix_len - previous_tokens + 1 : prefix_len + current_tokens
            ]
            if current_tokens * 2 < prob.shape[2]:
                prob[:, 0, current_tokens * 2] = current_tokens
            draft_forward_times += args.gamma
        else:
            x = model.generate(input_ids, 1)
            prob = model._prob_history[:, prefix_len - previous_tokens - 1 : prefix_len, :vocab_size].to(
                torch.float32
            )
            prob = prob.to(device)
            target_forward_times += 1

        accelerator.wait_for_everyone()

        all_prob = accelerator.gather(prob).to(device)
        if all_prob.shape[0] < 2:
            raise RuntimeError(
                "Parallel PLD decoding requires at least two accelerator processes; "
                "got gather output with fewer than two entries."
            )

        draft_ids = all_prob[0, [0], 1 : previous_tokens + current_tokens].int()
        draft_prob = all_prob[[0], 1:, :]
        target_prob = all_prob[[1], 1:, :]

        if cur_mode:
            if previous_tokens + current_tokens == 0:
                continue

            target_pld_candidates = find_candidate_pred_tokens(
                input_ids, max_ngram_size, num_pred_tokens
            )
            target_pld_length = len(target_pld_candidates)

            total_tokens_to_verify = min(len(draft_ids[0]), target_pld_length + 1)
            verified_tokens = 0

            for i in range(total_tokens_to_verify):
                if i >= len(draft_ids[0]):
                    break

                token = draft_ids[0, i]
                torch.manual_seed(seed + prefix_len + i)
                r = torch.rand(1, device=device)

                target_pos = target_prob.shape[1] - total_tokens_to_verify + i
                if 0 <= target_pos < target_prob.shape[1]:
                    if i < target_pld_length:
                        if target_prob[0, target_pos, token] > 0.9:
                            verified_tokens += 1
                        else:
                            break
                    else:
                        draft_pos = i - target_pld_length
                        if 0 <= draft_pos < draft_prob.shape[1]:
                            if r <= target_prob[0, target_pos, token] / draft_prob[0, draft_pos, token]:
                                verified_tokens += 1
                            else:
                                break
                        else:
                            break
                else:
                    break

            if verified_tokens > 0:
                if verified_tokens >= total_tokens_to_verify:
                    cur_mode = False
                    accepted_tokens = draft_ids[:, :verified_tokens]
                    prefix_out = torch.cat((input_ids, accepted_tokens), dim=1)
                    num_acc_token += verified_tokens
                else:
                    accepted_tokens = draft_ids[:, :verified_tokens]
                    prefix_out = torch.cat((input_ids, accepted_tokens), dim=1)
                    num_acc_token += verified_tokens
                    num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0
                    if accelerator.is_main_process:
                        model.rollback(prefix_out.shape[1])
            else:
                t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, 0, :]))
                prefix_out = torch.cat((input_ids, t), dim=1)
                num_acc_tokens.append(num_acc_token)
                num_acc_token = 0
                if accelerator.is_main_process:
                    model.rollback(prefix_len)

        else:
            if previous_tokens + current_tokens == 0:
                continue

            target_pld_candidates = find_candidate_pred_tokens(
                input_ids, max_ngram_size, num_pred_tokens
            )
            target_pld_length = len(target_pld_candidates)

            total_tokens_to_verify = len(draft_ids[0])
            n = total_tokens_to_verify

            for i in range(total_tokens_to_verify):
                token = draft_ids[0, i]
                torch.manual_seed(seed + prefix_len - total_tokens_to_verify + i)
                r = torch.rand(1, device=device)

                target_pos = target_prob.shape[1] - total_tokens_to_verify + i

                if 0 <= target_pos < target_prob.shape[1]:
                    if i < target_pld_length:
                        if target_prob[0, target_pos, token] <= 0.9:
                            n = i
                            break
                    else:
                        draft_pos = i - target_pld_length
                        if 0 <= draft_pos < draft_prob.shape[1]:
                            if r > target_prob[0, target_pos, token] / draft_prob[0, draft_pos, token]:
                                n = i
                                break
                        else:
                            n = i
                            break
                else:
                    n = i
                    break

            if n == total_tokens_to_verify:
                prefix_out = torch.cat((input_ids, draft_ids), dim=1)
                num_acc_token += total_tokens_to_verify
            else:
                assert n < total_tokens_to_verify
                cur_mode = True

                target_reject_pos = target_prob.shape[1] - total_tokens_to_verify + n
                draft_reject_pos = max(0, n - target_pld_length)

                if (
                    0 <= target_reject_pos < target_prob.shape[1]
                    and draft_reject_pos < draft_prob.shape[1]
                ):
                    t = sample(max_fn(target_prob[:, target_reject_pos, :] - draft_prob[:, draft_reject_pos, :]))
                else:
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))

                prefix_out = torch.cat(
                    (
                        input_ids[:, : prefix_len - total_tokens_to_verify + n + 1],
                        t,
                    ),
                    dim=1,
                )
                num_acc_tokens.append(num_acc_token + n)
                num_acc_token = 0
                model.rollback(prefix_len - total_tokens_to_verify + n + 1)

        previous_tokens = current_tokens

    return prefix_out, draft_forward_times, target_forward_times
