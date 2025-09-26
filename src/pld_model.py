import torch

from .kvcache import KVCacheModel
from .util import sample


@torch.no_grad()
def find_candidate_pred_tokens(
    input_ids: torch.Tensor,
    max_ngram_size: int = 3,
    num_pred_tokens: int = 10,
) -> torch.Tensor:
    """Return PLD candidate tokens by matching trailing n-grams in the prompt."""
    input_length = input_ids.size(1)

    if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
        return torch.tensor([], dtype=torch.long, device=input_ids.device)

    for ngram_size in range(max_ngram_size, 0, -1):
        ngram = input_ids[0, -ngram_size:].tolist()
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)
        matches = (windows == ngram_tensor).all(dim=2)
        match_indices = matches.nonzero(as_tuple=True)[1]

        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]

    return torch.tensor([], dtype=torch.long, device=input_ids.device)


class PLDKVCacheModel(KVCacheModel):
    """KV cache wrapper that augments generation with Prompt Lookup Decoding (PLD)."""

    def __init__(
        self,
        model: torch.nn.Module,
        temperature: float = 1,
        top_k: int = 0,
        top_p: float = 0,
    ) -> None:
        super().__init__(model, temperature, top_k, top_p)
        self._reset_pld_state()

    def _reset_pld_state(self):
        self._pld_tokens = None
        self._pld_probs = None
        self._pld_length = 0
        self._actual_tokens_generated = 0

    def _apply_one_hot_probs(self, start_pos: int, accepted_tokens: torch.Tensor):
        if self._prob_history is None:
            return
        for offset, token in enumerate(accepted_tokens):
            pos = start_pos + offset
            if pos < self._prob_history.shape[1]:
                self._prob_history[:, pos, :] = 0.0
                self._prob_history[:, pos, token] = 1.0

    def _forward_with_candidate_sequence(
        self,
        candidate_input: torch.Tensor,
    ) -> torch.Tensor:
        if self._past_key_values is None:
            outputs = self._model(candidate_input)
            normalized = self._normalize_logits(outputs.logits)
            self._prob_history = normalized
            self._past_key_values = outputs.past_key_values
            return normalized

        cached_len = self._get_cached_length()
        new_tokens = candidate_input[:, cached_len:]
        if new_tokens.numel() == 0:
            return self._prob_history

        outputs = self._model(new_tokens, past_key_values=self._past_key_values, use_cache=True)
        new_logits = self._normalize_logits(outputs.logits)
        if new_logits.dim() == 2:
            new_logits = new_logits.unsqueeze(1)
        self._prob_history = torch.cat([self._prob_history, new_logits], dim=1)
        self._past_key_values = outputs.past_key_values
        return new_logits

    def _generate_with_pld(
        self,
        prefix: torch.Tensor,
        gamma: int,
        max_ngram_size: int = 3,
        num_pred_tokens: int = 10,
    ) -> torch.Tensor:
        x = prefix
        self._reset_pld_state()

        for _ in range(gamma):
            current_len = x.shape[1]
            pld_candidates = find_candidate_pred_tokens(x, max_ngram_size, num_pred_tokens)

            if len(pld_candidates) > 0:
                candidate_input = torch.cat([x, pld_candidates.unsqueeze(0)], dim=1)
                candidate_length = len(pld_candidates)

                logits = self._forward_with_candidate_sequence(candidate_input)
                relevant_logits = logits[:, -candidate_length - 1 :, :]
                selected_tokens = relevant_logits.argmax(dim=-1)

                n_matches = 0
                for i in range(candidate_length):
                    if pld_candidates[i] == selected_tokens[0, i]:
                        n_matches += 1
                    else:
                        break

                if n_matches > 0:
                    accepted_pld = pld_candidates[:n_matches]
                    next_draft_token = selected_tokens[:, n_matches : n_matches + 1]
                    new_tokens = torch.cat([accepted_pld.unsqueeze(0), next_draft_token], dim=1)
                    x = torch.cat([x, new_tokens], dim=1)
                    self._apply_one_hot_probs(current_len, accepted_pld)
                    self._actual_tokens_generated += new_tokens.shape[1]
                    continue

                super().rollback(current_len)

            probs = super()._forward_with_kvcache(x)
            next_tok = sample(probs)
            x = torch.cat((x, next_tok), dim=1)
            self._actual_tokens_generated += 1

        return x

    @torch.no_grad()
    def generate_with_pld(
        self,
        input_ids: torch.Tensor,
        gamma: int,
        max_ngram_size: int = 3,
        num_pred_tokens: int = 10,
    ) -> torch.Tensor:
        return self._generate_with_pld(input_ids, gamma, max_ngram_size, num_pred_tokens)

    def get_actual_tokens_generated(self) -> int:
        return self._actual_tokens_generated

    @torch.no_grad()
    def rollback(self, end_pos: int):
        super().rollback(end_pos)
        self._reset_pld_state()
