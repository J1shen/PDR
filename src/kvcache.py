import torch
from .util import norm_logits, sample


@torch.no_grad()
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
    """
    PLD function to find candidate tokens by matching n-grams in the prompt.
    Returns candidate tokens for speculative verification.
    """
    input_length = input_ids.size(1)

    # Ensure max_ngram_size and num_pred_tokens are valid
    if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
        return torch.tensor([], dtype=torch.long, device=input_ids.device)

    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:].tolist()

        # Create sliding windows of size ngram_size
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

        # Convert ngram to a tensor for comparison
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[1]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                candidate_tokens = input_ids[0, start_idx:end_idx]
                return candidate_tokens

    # If no match is found, return empty tensor
    return torch.tensor([], dtype=torch.long, device=input_ids.device)


class KVCacheModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        
        # PLD-specific state
        self._pld_tokens = None  # Store PLD retrieved tokens
        self._pld_probs = None  # Store one-hot probabilities for PLD tokens
        self._pld_length = 0    # Number of PLD tokens in current sequence

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids : torch.Tensor, max_ngram_size : int = 3, num_pred_tokens : int = 10) -> torch.Tensor:
        # Always try PLD retrieval first to enhance the input
        pld_candidates = find_candidate_pred_tokens(input_ids, max_ngram_size, num_pred_tokens)
        pld_length = len(pld_candidates)
        
        if pld_length > 0:
            # Enhance input with PLD candidates
            enhanced_input_ids = torch.cat([input_ids, pld_candidates.unsqueeze(0)], dim=1)
        else:
            # No PLD candidates, use original input
            enhanced_input_ids = input_ids
        
        if self._past_key_values is None:
            outputs = self._model(enhanced_input_ids)
            self._prob_history = outputs.logits[:, :, :self.vocab_size]
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            
            # Create one-hot probabilities for PLD tokens
            if pld_length > 0:
                pld_start_pos = input_ids.shape[1]
                for i, pld_token in enumerate(pld_candidates):
                    if pld_start_pos + i < self._prob_history.shape[1]:
                        self._prob_history[:, pld_start_pos + i, :] = 0.0
                        self._prob_history[:, pld_start_pos + i, pld_token] = 1.0
            
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = self._past_key_values.get_seq_length()
                
            last_input_id = enhanced_input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits[:, :, :self.vocab_size]
            
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
            
            # Create one-hot probabilities for PLD tokens in new logits
            if pld_length > 0 and last_input_id.shape[1] >= pld_length:
                for i in range(pld_length):
                    if i < not_cached_q.shape[1]:
                        pld_token = last_input_id[0, i]
                        not_cached_q[:, i, :] = 0.0
                        not_cached_q[:, i, pld_token] = 1.0
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x, 3, 10)  # Use default PLD parameters  
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output
    
    def _generate_with_pld(self, prefix : torch.Tensor, gamma : int, max_ngram_size : int = 3, num_pred_tokens : int = 10) -> torch.Tensor:
        """
        Generate tokens using PLD to accelerate draft model.
        gamma = number of PLD steps (not token count)
        Each step: try PLD retrieval → draft model verifies → accept/reject
        Returns: prefix + all generated tokens from gamma steps
        """
        x = prefix
        step_count = 0
        
        # Reset PLD tracking
        self._pld_tokens = None
        self._pld_probs = None
        self._pld_length = 0
        
        # Track actual tokens generated for verification
        self._actual_tokens_generated = 0
        
        # Execute gamma PLD steps
        for step in range(gamma):
            step_count += 1
            current_len = x.shape[1]
            
            # Step 1: Try PLD retrieval on current sequence
            pld_candidates = find_candidate_pred_tokens(x, max_ngram_size, num_pred_tokens)
            
            if len(pld_candidates) > 0:
                # PLD found candidates - create candidate sequence
                candidate_input = torch.cat([x, pld_candidates.unsqueeze(0)], dim=1)
                candidate_length = len(pld_candidates)
                
                # Step 2: Draft model forward pass to verify PLD candidates
                if self._past_key_values is None:
                    # First forward pass
                    outputs = self._model(candidate_input)
                    self._prob_history = outputs.logits[:, :, :self.vocab_size]
                    for i in range(self._prob_history.shape[-2]):   
                        self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
                    self._past_key_values = outputs.past_key_values
                else:
                    # Incremental forward pass
                    cached_len = self._past_key_values.get_seq_length()
                    new_tokens = candidate_input[:, cached_len:]
                    outputs = self._model(new_tokens, past_key_values=self._past_key_values, use_cache=True)
                    new_logits = outputs.logits[:, :, :self.vocab_size]
                    for i in range(new_logits.shape[-2]):   
                        new_logits[:, i, :] = norm_logits(new_logits[:, i, :], self._temperature, self._top_k, self._top_p)
                    self._prob_history = torch.cat([self._prob_history, new_logits], dim=1)
                    self._past_key_values = outputs.past_key_values
                
                # Step 3: Verify PLD candidates against draft model predictions
                new_logits = outputs.logits[:, -candidate_length-1:, :self.vocab_size]  # Include the +1 token
                selected_tokens = new_logits.argmax(dim=-1)
                
                # Compare PLD candidates with draft model predictions
                n_matches = 0
                for i in range(candidate_length):
                    if pld_candidates[i] == selected_tokens[0, i]:
                        n_matches += 1
                    else:
                        break
                
                # Step 4: Accept matching tokens + one additional from draft model
                if n_matches > 0:
                    # Accept n_matches PLD tokens + 1 draft model token
                    accepted_pld = pld_candidates[:n_matches]
                    next_draft_token = selected_tokens[:, n_matches:n_matches+1]
                    
                    # Update sequence
                    new_tokens = torch.cat([accepted_pld.unsqueeze(0), next_draft_token], dim=1)
                    x = torch.cat([x, new_tokens], dim=1)
                    
                    # Create one-hot probabilities for PLD tokens in prob_history
                    # PLD tokens should have probability = 1.0 at their token positions
                    pld_start_pos = current_len
                    for i, pld_token in enumerate(accepted_pld):
                        if pld_start_pos + i < self._prob_history.shape[1]:
                            # Set one-hot probability for PLD token
                            self._prob_history[:, pld_start_pos + i, :] = 0.0
                            self._prob_history[:, pld_start_pos + i, pld_token] = 1.0
                    
                    # Track actual tokens generated
                    self._actual_tokens_generated += len(new_tokens[0])
                else:
                    # No PLD matches, rollback and use draft model
                    self._past_key_values.crop(current_len)
                    self._prob_history = self._prob_history[:, :current_len, :]
                    
                    # Generate one token with draft model
                    q = self._forward_with_kvcache(x, max_ngram_size, num_pred_tokens)
                    next_tok = sample(q)
                    x = torch.cat((x, next_tok), dim=1)
                    self._actual_tokens_generated += 1
            else:
                # No PLD candidates - use draft model for one token
                q = self._forward_with_kvcache(x, max_ngram_size, num_pred_tokens)
                next_tok = sample(q)
                x = torch.cat((x, next_tok), dim=1)
                self._actual_tokens_generated += 1
        
        return x
    
    def get_pld_info(self):
        """Return PLD tokens info - simplified since PLD is now internal to draft model"""
        return self._pld_tokens, self._pld_probs, self._pld_length
    
    def get_actual_tokens_generated(self):
        """Return the actual number of tokens generated in the last PLD generation"""
        return getattr(self, '_actual_tokens_generated', 0)

    @torch.no_grad()
    def generate_with_pld(self, input : torch.Tensor, gamma : int, max_ngram_size : int = 3, num_pred_tokens : int = 10) -> torch.Tensor:
        """Generate tokens using PLD + draft model"""
        output = self._generate_with_pld(input, gamma, max_ngram_size, num_pred_tokens)
        return output

    @torch.no_grad()
    def rollback(self, end_pos : int):
        self._past_key_values.crop(end_pos)
        self._prob_history = self._prob_history[:, :end_pos, :]
        
        # Reset PLD state on rollback
        self._pld_tokens = None
        self._pld_probs = None
        self._pld_length = 0