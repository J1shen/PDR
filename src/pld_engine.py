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
    debug: bool = False,
) -> Tuple[torch.Tensor, int, int]:
    """Parallel PLD speculative decoding shared by both engine and demos.

    Returns the updated prefix together with draft/target forward counts that were
    consumed in this call. ``num_acc_tokens`` is mutated in-place when provided.
    """

    if num_acc_tokens is None:
        num_acc_tokens = []

    def log(message: str) -> None:
        if debug:
            print(f"[rank {accelerator.process_index}] {message}", flush=True)

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

    log(
        f"Starting parallel PLD decode: prefix_len={prefix_out.shape[1]}, "
        f"target_len={max_tokens}, mode={'pre-verify' if cur_mode else 'post-verify'}"
    )

    step = 0

    while prefix_out.shape[1] < max_tokens:
        step += 1
        prefix_len = prefix_out.shape[1]
        input_ids = prefix_out.to(device)

        log(
            f"Step {step}: prefix_len={prefix_len}, prev_tokens={previous_tokens}, "
            f"cur_mode={'pre' if cur_mode else 'post'}"
        )

        if accelerator.is_main_process:
            x = model.generate_with_pld(
                input_ids,
                args.gamma,
                max_ngram_size,
                num_pred_tokens,
            )
            current_tokens = model.get_actual_tokens_generated()
            log(
                f"Step {step}: draft produced current_tokens={current_tokens} (prev={previous_tokens})"
            )
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
            log(
                f"Step {step}: target updated history slice with shape={tuple(prob.shape)}"
            )

        accelerator.wait_for_everyone()

        all_prob = accelerator.gather(prob).to(device)
        log(
            f"Step {step}: gathered prob shape={tuple(all_prob.shape)}, "
            f"draft_fw={draft_forward_times}, target_fw={target_forward_times}"
        )
        if all_prob.shape[0] < 2:
            raise RuntimeError(
                "Parallel PLD decoding requires at least two accelerator processes; "
                "got gather output with fewer than two entries."
            )

        draft_ids = all_prob[0, [0], 1 : previous_tokens + current_tokens].int()
        draft_prob = all_prob[[0], 1:, :]
        target_prob = all_prob[[1], 1:, :]

        log(
            f"Step {step}: draft_ids shape={tuple(draft_ids.shape)}, draft_prob shape={tuple(draft_prob.shape)}, "
            f"target_prob shape={tuple(target_prob.shape)}"
        )

        if cur_mode:
            if previous_tokens + current_tokens == 0:
                log(f"Step {step}: no tokens to verify yet, continuing")
                continue

            log(f"Step {step}: entering find_candidate_pred_tokens")
            target_pld_candidates = find_candidate_pred_tokens(
                input_ids,
                max_ngram_size,
                num_pred_tokens,
                debug=debug,
                log=log,
            )
            log(
                f"Step {step}: find_candidate_pred_tokens returned length={len(target_pld_candidates)}"
            )
            target_pld_length = len(target_pld_candidates)

            total_tokens_to_verify = min(len(draft_ids[0]), target_pld_length + 1)
            verified_tokens = 0

            log(
                f"Step {step}: pre-verify total_tokens={total_tokens_to_verify}, "
                f"target_pld_length={target_pld_length}, draft_ids_len={len(draft_ids[0])}"
            )

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
                            log(
                                f"Step {step}: pre-verify candidate idx={i} accepted (target_prob={float(target_prob[0, target_pos, token]):.3f})"
                            )
                        else:
                            log(
                                f"Step {step}: pre-verify candidate idx={i} rejected (target_prob={float(target_prob[0, target_pos, token]):.3f})"
                            )
                            break
                    else:
                        draft_pos = i - target_pld_length
                        if 0 <= draft_pos < draft_prob.shape[1]:
                            if r <= target_prob[0, target_pos, token] / draft_prob[0, draft_pos, token]:
                                verified_tokens += 1
                                log(
                                    f"Step {step}: pre-verify idx={i} accepted ratio={float(target_prob[0, target_pos, token] / draft_prob[0, draft_pos, token]):.3f}"
                                )
                            else:
                                log(
                                    f"Step {step}: pre-verify idx={i} rejected ratio={float(target_prob[0, target_pos, token] / draft_prob[0, draft_pos, token]):.3f}"
                                )
                                break
                        else:
                            log(
                                f"Step {step}: pre-verify draft_pos out of range (draft_pos={draft_pos})"
                            )
                            break
                else:
                    log(
                        f"Step {step}: pre-verify target_pos out of range (target_pos={target_pos})"
                    )
                    break

            if verified_tokens > 0:
                if verified_tokens >= total_tokens_to_verify:
                    cur_mode = False
                    accepted_tokens = draft_ids[:, :verified_tokens]
                    prefix_out = torch.cat((input_ids, accepted_tokens), dim=1)
                    num_acc_token += verified_tokens
                    log(
                        f"Step {step}: accepted {verified_tokens} tokens (full), "
                        f"new_len={prefix_out.shape[1]}"
                    )
                else:
                    accepted_tokens = draft_ids[:, :verified_tokens]
                    prefix_out = torch.cat((input_ids, accepted_tokens), dim=1)
                    num_acc_token += verified_tokens
                    num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0
                    if accelerator.is_main_process:
                        model.rollback(prefix_out.shape[1])
                    log(
                        f"Step {step}: accepted {verified_tokens} tokens (partial), "
                        f"rolling back to len={prefix_out.shape[1]}"
                    )
            else:
                t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, 0, :]))
                prefix_out = torch.cat((input_ids, t), dim=1)
                num_acc_tokens.append(num_acc_token)
                num_acc_token = 0
                if accelerator.is_main_process:
                    model.rollback(prefix_len)
                log(
                    f"Step {step}: no acceptance, sampled target token, "
                    f"new_len={prefix_out.shape[1]}"
                )
                log(
                    f"Step {step}: sampled token id={int(t[0,0])}"
                )

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
                log(
                    f"Step {step}: post-verify accepted {total_tokens_to_verify} tokens, "
                    f"new_len={prefix_out.shape[1]}"
                )
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
                log(
                    f"Step {step}: post-verify rejection at offset {n}, "
                    f"rolled back to len={prefix_out.shape[1]}"
                )

        previous_tokens = current_tokens
        log(
            f"Step {step}: loop end -> prefix_len={prefix_out.shape[1]}, "
            f"prev_tokens set to {previous_tokens}, cur_mode={'pre' if cur_mode else 'post'}"
        )

    log(
        f"Finished parallel PLD decode: final_len={prefix_out.shape[1]}, "
        f"draft_fw={draft_forward_times}, target_fw={target_forward_times}"
    )
    return prefix_out, draft_forward_times, target_forward_times
