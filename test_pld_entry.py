"""Simple entry point to exercise PLD utilities without loading full models."""

import torch

from src.pld_engine import find_candidate_pred_tokens


def _run_positive_case():
    tokens = torch.tensor([[7, 8, 9, 7, 8, 9, 10, 11, 7, 8]], dtype=torch.long)
    candidates = find_candidate_pred_tokens(tokens, max_ngram_size=2, num_pred_tokens=2)
    expected = [9, 7]
    if candidates.tolist() != expected:
        raise AssertionError(f"Expected {expected}, got {candidates.tolist()}")
    print("Positive match case passed", candidates.tolist())


def _run_no_match_case():
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    candidates = find_candidate_pred_tokens(tokens, max_ngram_size=3, num_pred_tokens=2)
    if candidates.numel() != 0:
        raise AssertionError("Expected no candidates for unmatched prompt")
    print("No match case passed")


def main():
    _run_positive_case()
    _run_no_match_case()
    print("PLD candidate token tests completed successfully.")


if __name__ == "__main__":
    main()
