import torch

from .util import norm_logits, sample


class KVCacheModel:
    """Lightweight wrapper around an autoregressive model with KV-cache support."""

    def __init__(
        self,
        model: torch.nn.Module,
        temperature: float = 1,
        top_k: int = 0,
        top_p: float = 0,
    ) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

        # populated externally (see engine.load_model)
        self.vocab_size = getattr(model.config, "vocab_size", None)

    def _normalize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Normalize logits across the time dimension using sampling config."""
        vocab_limit = self.vocab_size or logits.shape[-1]
        logits = logits[:, :, :vocab_limit]
        steps = logits.shape[1]
        normalized = []
        for i in range(steps):
            step_logits = logits[:, i, :]
            normalized.append(norm_logits(step_logits, self._temperature, self._top_k, self._top_p))
        return torch.stack(normalized, dim=1)

    def _get_cached_length(self) -> int:
        if self._past_key_values is None:
            return 0
        if hasattr(self._past_key_values, "get_seq_length"):
            return self._past_key_values.get_seq_length()
        if isinstance(self._past_key_values, (list, tuple)) and self._past_key_values:
            return self._past_key_values[0][0].shape[2]
        raise AttributeError("Unsupported past_key_values format")

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            normalized = self._normalize_logits(outputs.logits)
            self._prob_history = normalized
            self._past_key_values = outputs.past_key_values
            return normalized[:, -1, :]

        cached_len = self._get_cached_length()
        new_tokens = input_ids[:, cached_len:]
        if new_tokens.numel() == 0:
            # Already have logits for the current position
            return self._prob_history[:, -1, :]

        outputs = self._model(new_tokens, past_key_values=self._past_key_values, use_cache=True)
        new_logits = self._normalize_logits(outputs.logits)
        if new_logits.dim() == 2:
            new_logits = new_logits.unsqueeze(1)
        self._prob_history = torch.cat([self._prob_history, new_logits], dim=1)
        self._past_key_values = outputs.past_key_values
        return new_logits[:, -1, :]

    def _generate_with_kvcache(self, prefix: torch.Tensor, gamma: int) -> torch.Tensor:
        x = prefix
        for _ in range(gamma):
            probs = self._forward_with_kvcache(x)
            next_tok = sample(probs)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, gamma: int) -> torch.Tensor:
        return self._generate_with_kvcache(input_ids, gamma)

    @torch.no_grad()
    def rollback(self, end_pos: int):
        if self._past_key_values is None:
            return

        if hasattr(self._past_key_values, "crop"):
            self._past_key_values.crop(end_pos)
        else:
            trimmed = []
            for kv in self._past_key_values:
                k, v = kv
                trimmed.append((k[:, :, :end_pos, :], v[:, :, :end_pos, :]))
            self._past_key_values = trimmed

        if self._prob_history is not None:
            self._prob_history = self._prob_history[:, :end_pos, :]
