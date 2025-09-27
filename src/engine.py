import torch
import transformers
import warnings
transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from accelerate import Accelerator
from .kvcache import KVCacheModel, find_candidate_pred_tokens
from .kvcache4RC import KVCacheModel as KVCache2Model
from .util import seed_everything, norm_logits, sample, max_fn
import time


class Decoding(ABC):
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()
        
        seed_everything(self.args.seed)
        self.seed = self.args.seed
        self.seed_set = set()
        
        # ! only parallel speculative decoding can use 2 processes
        assert (self.accelerator.num_processes == 1 and args.eval_mode in ["small", "large", "sd"]) or (self.accelerator.num_processes == 2 and args.eval_mode in ["para_sd", "para_sd_wo_1", "para_sd_wo_1", "rc_para_sd", "pld_para_sd"])

        # record metrics for report
        self.draft_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []
    
    def load_model(self):
        # * load models according to different evaluation methods.
        self.color_print(f"Loading models:\n{self.args.draft_model}\n{self.args.target_model}", 3)
        if self.args.eval_mode == "small":
            self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        elif self.args.eval_mode == "large":
            self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        elif self.args.eval_mode == "sd":
            self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
            self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="balanced_low_0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        
        elif self.args.eval_mode in ["para_sd", "para_sd_wo_1", "para_sd_wo_1", "pld_para_sd"]:
            if self.accelerator.is_main_process:
                self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="balanced_low_0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        
        elif self.args.eval_mode == "rc_para_sd":
            if self.accelerator.is_main_process:
                self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
                self.draft_model_2 = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map=f"cuda:{torch.cuda.device_count()-1}", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        
        self.vocab_size = self.args.vocab_size

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.draft_model}...", 3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.draft_model, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        
        # for llama models
        self.tokenizer.pad_token_id = 2

    @abstractmethod
    def load_data(self):
        pass
    
    @abstractmethod
    def preprocess(self, input_text):
        pass
    
    @abstractmethod
    def postprocess(self, input_text, output_text):
        pass
    
    @torch.no_grad()
    def autoregressive_sampling(self, prefix):
        if self.args.eval_mode == "small":
            model = self.draft_model
        elif self.args.eval_mode == "large":
            model = self.target_model
        else:
            raise RuntimeError("Auto-Regressive Decoding can be used only in small / large eval mode!")
        
        prefix = prefix.to(model.device)

        prefix_len = prefix.shape[1]
        max_tokens = prefix_len + self.args.max_tokens
        
        x = prefix
        past_key_values = None
        while x.shape[1] < max_tokens:
            if past_key_values:
                last_ids = x[:, -1]
                if last_ids.dim() == 1:
                    last_ids = last_ids.unsqueeze(0)
                outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
            else:
                outputs = model(x)

            if self.accelerator.is_main_process:
                if self.args.eval_mode == "small":
                    self.draft_forward_times += 1
                elif self.args.eval_mode == "large":
                    self.target_forward_times += 1

            last_p = norm_logits(outputs.logits[::, -1, :], self.args.temp, self.args.top_k, self.args.top_p)
            past_key_values = outputs.past_key_values
            idx_next = sample(last_p)
            x = torch.cat((x, idx_next), dim=1)
        return x

    @torch.no_grad()
    def speculative_decoding(self, prefix):
        max_tokens = prefix.shape[1] + self.args.max_tokens
        
        draft_device = self.draft_model.device
        target_device = self.target_model.device
        
        approx_model_cache = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
        target_model_cache.vocab_size = self.vocab_size

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            x = approx_model_cache.generate(prefix.to(draft_device), self.args.gamma)
            _ = target_model_cache.generate(x.to(target_device), 1)
            if self.accelerator.is_main_process:
                self.draft_forward_times += self.args.gamma
                self.target_forward_times += 1
            
            n = prefix_len + self.args.gamma - 1
            for i in range(self.args.gamma):
                r = torch.rand(1, device=draft_device)
                j = x[:, prefix_len + i]
                
                if r > (target_model_cache._prob_history.to(draft_device)[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                    n = prefix_len + i - 1
                    break

            self.num_acc_tokens.append(n - prefix_len + 1)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, :n + 1]
            
            approx_model_cache.rollback(n+1)

            if n < prefix_len + self.args.gamma - 1:
                # reject someone, sample from the pos n
                t = sample(max_fn(target_model_cache._prob_history[:, n, :self.vocab_size].to(draft_device) - approx_model_cache._prob_history[:, n, :self.vocab_size]))
                target_model_cache.rollback(n+1)
            else:
                # all approx model decoding accepted
                t = sample(target_model_cache._prob_history[:, -1, :self.vocab_size]).to(draft_device)
                target_model_cache.rollback(n+2)
            prefix = torch.cat((prefix, t), dim=1)
        return prefix
    
    @torch.no_grad()
    def parallel_speculative_decoding(self, prefix):
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens
        
        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            
            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                x = model.generate(input_ids, self.args.gamma)
                prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size].to(torch.float32)
                prob[:, 0, 0] = -1
                prob[:, 0, 1:self.args.gamma*2] = x[:, prefix_len-self.args.gamma+1:prefix_len+self.args.gamma]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size].to(torch.float32)
                prob = prob.to("cuda:1")
                self.target_forward_times += 1
            
            self.accelerator.wait_for_everyone()

            # verification
            all_prob = self.accelerator.gather(prob).to(device)
            draft_ids = all_prob[0, [0], 1:self.args.gamma*2].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]
            if cur_mode:
                first_token = draft_ids[:, -self.args.gamma]
                torch.manual_seed(self.seed + prefix_len)

                r = torch.rand(1, device=device)
                if  r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                    # reject the first token
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                    prefix = torch.cat((input_ids, t), dim=1)
                    
                    # record the number of accepted tokens
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0
                    
                    if self.accelerator.is_main_process:
                        # rollback the small model kv cache
                        model.rollback(prefix_len)
                else:
                    # accept the first token, change the mode
                    cur_mode = False
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                    num_acc_token += 1

            else:
                n = self.args.gamma
                for i in range(self.args.gamma):
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - self.args.gamma + i)
                    r = torch.rand(1, device=device)
                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break
                if n == self.args.gamma:
                    # accept all guess tokens
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                    num_acc_token += self.args.gamma
                else:
                    # reject someone, change the mode
                    assert n < self.args.gamma
                    cur_mode = True
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))
                    
                    prefix = torch.cat((input_ids[:, :prefix_len-self.args.gamma + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - self.args.gamma +n+1)
            
        return prefix

    
     
    @torch.no_grad()
    def parallel_speculative_decoding_with_pld(self, prefix):
        """
        PLD + PEARL combined parallel speculative decoding.
        Maintains PEARL structure but uses PLD-enhanced generation.
        """
        # parallel speculative decoding with PLD
        if self.accelerator.is_main_process:
            model = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens
        
        # PLD parameters
        max_ngram_size = getattr(self.args, 'pld_max_ngram_size', 3)
        num_pred_tokens = getattr(self.args, 'pld_num_pred_tokens', 10)
        
        # this flag is used to determine the current verify mode.
        cur_mode = True  # True: Pre-verify, False: Post-verify
        num_acc_token = 0
        
        # Track tokens for two-round mechanism
        previous_tokens = self.args.gamma  # Initialize with gamma for first round
        current_tokens = 0

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            input_ids = prefix.to(device)
            
            if self.accelerator.is_main_process:
                # Draft model side: Use PLD-enhanced generation (gamma = PLD steps, not token count)
                x = model.generate_with_pld(input_ids, self.args.gamma, max_ngram_size, num_pred_tokens)
                
                # Get current round's actual number of tokens generated
                current_tokens = model.get_actual_tokens_generated()
                
                # Store probabilities like standard PEARL using previous_tokens for window size
                prob = model._prob_history[:, prefix_len-previous_tokens-1:prefix_len, :self.vocab_size].to(torch.float32)
                prob[:, 0, 0] = -1
                prob[:, 0, 1:previous_tokens+current_tokens] = x[:, prefix_len-previous_tokens+1:prefix_len+current_tokens]
                
                # Store current_tokens count for next round verification
                if current_tokens*2 < prob.shape[2]:
                    prob[:, 0, current_tokens*2] = current_tokens
                
                # Count PLD steps as forward passes
                self.draft_forward_times += self.args.gamma
                
            else:
                # Target model side: use PLD-enhanced forward (sees same PLD tokens as draft model)
                x = model.generate(input_ids, 1)
                
                # Store probabilities like standard PEARL using previous_tokens for window size
                prob = model._prob_history[:, prefix_len-previous_tokens-1:prefix_len, :self.vocab_size].to(torch.float32)
                prob = prob.to("cuda:1")
                self.target_forward_times += 1
            
            self.accelerator.wait_for_everyone()

            # Verification phase - parse tokens using previous_tokens and get current_tokens
            all_prob = self.accelerator.gather(prob).to(device)
            
            # Extract tokens and probabilities like standard PEARL using previous_tokens
            draft_ids = all_prob[0, [0], 1:previous_tokens+current_tokens].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]
            
            # Get current_tokens for next round from communication data
            if previous_tokens*2 < all_prob.shape[2]:
                current_tokens = int(all_prob[0, 0, previous_tokens*2].item())
            else:
                current_tokens = self.args.gamma  # Fallback
            
            if cur_mode:
                # Pre-verify mode: enhanced to verify PLD tokens + first generated token
                if previous_tokens + current_tokens == 0:
                    continue
                
                # Target model also performed PLD retrieval, find PLD length for alignment
                target_pld_candidates = find_candidate_pred_tokens(input_ids, max_ngram_size, num_pred_tokens)
                target_pld_length = len(target_pld_candidates)
                
                # Verify tokens starting from PLD tokens to first generated token
                total_tokens_to_verify = min(len(draft_ids[0]), target_pld_length + 1)
                verified_tokens = 0
                
                for i in range(total_tokens_to_verify):
                    if i >= len(draft_ids[0]):
                        break
                        
                    token = draft_ids[0, i]
                    torch.manual_seed(self.seed + prefix_len + i)
                    r = torch.rand(1, device=device)
                    
                    # Find corresponding position in target_prob
                    target_pos = target_prob.shape[1] - total_tokens_to_verify + i
                    if target_pos >= 0 and target_pos < target_prob.shape[1]:
                        if i < target_pld_length:
                            # This is a PLD token - should have high confidence (one-hot)
                            if target_prob[0, target_pos, token] > 0.9:  # PLD tokens should be near 1.0
                                verified_tokens += 1
                            else:
                                break
                        else:
                            # This is the first generated token - normal verification
                            draft_pos = i - target_pld_length
                            if draft_pos >= 0 and draft_pos < draft_prob.shape[1]:
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
                        # All tokens verified (PLD + first generated), switch to post-verify
                        cur_mode = False
                        accepted_tokens = draft_ids[:, :verified_tokens]
                        prefix = torch.cat((input_ids, accepted_tokens), dim=1)
                        num_acc_token += verified_tokens
                    else:
                        # Partial verification
                        accepted_tokens = draft_ids[:, :verified_tokens]  
                        prefix = torch.cat((input_ids, accepted_tokens), dim=1)
                        num_acc_token += verified_tokens
                        self.num_acc_tokens.append(num_acc_token)
                        num_acc_token = 0
                        if self.accelerator.is_main_process:
                            model.rollback(prefix.shape[1])
                else:
                    # reject all tokens
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, 0, :]))
                    prefix = torch.cat((input_ids, t), dim=1)
                    
                    # record the number of accepted tokens
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0
                    
                    if self.accelerator.is_main_process:
                        # rollback the small model kv cache
                        model.rollback(prefix_len)

            else:
                # Post-verify mode: enhanced to verify all tokens including PLD tokens
                if previous_tokens + current_tokens == 0:
                    continue
                
                # Target model also performed PLD retrieval  
                target_pld_candidates = find_candidate_pred_tokens(input_ids, max_ngram_size, num_pred_tokens)
                target_pld_length = len(target_pld_candidates)
                
                # Verify all tokens from this round
                total_tokens_to_verify = len(draft_ids[0])
                n = total_tokens_to_verify
                
                for i in range(total_tokens_to_verify):
                    token = draft_ids[0, i]
                    torch.manual_seed(self.seed + prefix_len - total_tokens_to_verify + i)
                    r = torch.rand(1, device=device)
                    
                    # Find corresponding positions in target and draft probabilities
                    target_pos = target_prob.shape[1] - total_tokens_to_verify + i
                    
                    if target_pos >= 0 and target_pos < target_prob.shape[1]:
                        if i < target_pld_length:
                            # This is a PLD token - should have high confidence (one-hot)
                            if target_prob[0, target_pos, token] <= 0.9:  # Reject if not high confidence
                                n = i
                                break
                        else:
                            # This is a generated token - normal verification
                            draft_pos = i - target_pld_length
                            if draft_pos >= 0 and draft_pos < draft_prob.shape[1]:
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
                    # accept all tokens
                    prefix = torch.cat((input_ids, draft_ids), dim=1)
                    num_acc_token += total_tokens_to_verify
                else:
                    # reject someone, change the mode
                    assert n < total_tokens_to_verify
                    cur_mode = True
                    
                    # Find correct positions for rejection sampling
                    target_reject_pos = target_prob.shape[1] - total_tokens_to_verify + n
                    draft_reject_pos = max(0, n - target_pld_length)
                    
                    if target_reject_pos >= 0 and target_reject_pos < target_prob.shape[1] and draft_reject_pos < draft_prob.shape[1]:
                        t = sample(max_fn(target_prob[:, target_reject_pos, :] - draft_prob[:, draft_reject_pos, :]))
                    else:
                        t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                    
                    prefix = torch.cat((input_ids[:, :prefix_len-total_tokens_to_verify + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - total_tokens_to_verify + n + 1)
            
            # Update previous_tokens for next round
            previous_tokens = current_tokens
            
        return prefix
    
    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int=4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")
