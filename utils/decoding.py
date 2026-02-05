"""
Advanced CTC Decoding with KenLM Language Model Support.

Features:
- Greedy decoding
- Beam search with prefix merging
- KenLM language model integration (optional)
- Word-level and subword-level LM fusion
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
import math
from typing import Optional, List, Tuple, Dict, Any
import os


# =====================================================
# KenLM Wrapper
# =====================================================

class KenLMModel:
    """
    Wrapper for KenLM language model.
    
    Installation:
        pip install https://github.com/kpu/kenlm/archive/master.zip
        
    Training a model:
        1. Prepare text file with one sentence per line
        2. Run: lmplz -o 4 < corpus.txt > lm.arpa
        3. Convert: build_binary lm.arpa lm.bin
    """
    
    def __init__(self, model_path: str):
        """
        Load KenLM model.
        
        Args:
            model_path: Path to .arpa or .bin file
        """
        try:
            import kenlm
            self.model = kenlm.Model(model_path)
            self.order = self.model.order
            print(f"KenLM loaded: {model_path} (order={self.order})")
        except ImportError:
            raise ImportError(
                "KenLM not installed. Install with:\n"
                "pip install https://github.com/kpu/kenlm/archive/master.zip"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load KenLM model: {e}")
    
    def score(self, text: str, bos: bool = True, eos: bool = True) -> float:
        """
        Get log10 probability of text.
        
        Args:
            text: Text to score
            bos: Include beginning of sentence token
            eos: Include end of sentence token
            
        Returns:
            Log10 probability
        """
        return self.model.score(text, bos=bos, eos=eos)
    
    def score_word(self, word: str, context: str = "") -> float:
        """
        Get log10 probability of a word given context.
        
        Args:
            word: Word to score
            context: Previous context
            
        Returns:
            Log10 probability
        """
        full_text = f"{context} {word}".strip()
        # Score the full text minus context score
        if context:
            return self.score(full_text, bos=True, eos=False) - self.score(context, bos=True, eos=False)
        return self.score(word, bos=True, eos=False)


class NGramLanguageModel:
    """
    Simple N-gram language model (fallback when KenLM not available).
    """
    
    def __init__(self, order: int = 3):
        self.order = order
        self.counts: Dict[Tuple, int] = defaultdict(int)
        self.total_counts: Dict[Tuple, int] = defaultdict(int)
        
    def train(self, texts: List[str], tokenizer=None) -> None:
        """Train from list of texts."""
        for text in texts:
            if tokenizer:
                tokens = tokenizer.encode(text)
            else:
                tokens = text.lower().split()
                
            tokens = ["<s>"] * (self.order - 1) + list(tokens) + ["</s>"]
            
            for i in range(len(tokens) - self.order + 1):
                ngram = tuple(tokens[i:i + self.order])
                history = ngram[:-1]
                self.counts[ngram] += 1
                self.total_counts[history] += 1
                
    def score(self, history: Tuple, next_token: Any) -> float:
        """Get log probability of next token given history."""
        hist = tuple(history[-(self.order - 1):] if self.order > 1 else [])
        ngram = hist + (next_token,)
        
        count = self.counts.get(ngram, 0)
        total = self.total_counts.get(hist, 0)
        
        if total == 0:
            return -10.0  # Penalty for unknown
            
        return math.log(count / total + 1e-10)


# =====================================================
# CTC Decoders
# =====================================================

class GreedyDecoder:
    """Fast greedy CTC decoding."""
    
    def __init__(self, tokenizer, blank_id: int = 0):
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        
    def decode(self, logits: torch.Tensor) -> str:
        """
        Greedy decode logits.
        
        Args:
            logits: (Time, Vocab) tensor
            
        Returns:
            Decoded text
        """
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        return self.tokenizer.ctc_decode(pred_ids)
    
    def decode_batch(self, logits: torch.Tensor) -> List[str]:
        """
        Batch greedy decode.
        
        Args:
            logits: (Batch, Time, Vocab) tensor
            
        Returns:
            List of decoded texts
        """
        results = []
        pred_ids = torch.argmax(logits, dim=-1)
        
        for i in range(pred_ids.size(0)):
            text = self.tokenizer.ctc_decode(pred_ids[i].tolist())
            results.append(text)
            
        return results


class CTCBeamDecoder:
    """
    CTC Beam Search with optional Language Model fusion.
    
    Implements prefix beam search with proper CTC collapse handling.
    """
    
    def __init__(
        self,
        tokenizer,
        beam_width: int = 10,
        lm: Optional[KenLMModel] = None,
        lm_weight: float = 0.3,
        word_bonus: float = 0.5,
        blank_id: int = 0
    ):
        """
        Args:
            tokenizer: Tokenizer for decoding
            beam_width: Number of beams to keep
            lm: Language model (KenLM or NGramLanguageModel)
            lm_weight: Weight for LM score (alpha)
            word_bonus: Bonus per word to encourage longer outputs (beta)
            blank_id: CTC blank token ID
        """
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.lm = lm
        self.lm_weight = lm_weight
        self.word_bonus = word_bonus
        self.blank_id = blank_id
        
    def decode(self, logits: torch.Tensor) -> str:
        """
        Beam search decode single sequence.
        
        Args:
            logits: (Time, Vocab) tensor
            
        Returns:
            Decoded text
        """
        probs = F.softmax(logits, dim=-1)
        T, V = probs.shape
        
        # Beam state: {prefix_tuple: (prob_blank, prob_non_blank)}
        beam = {(): (0.0, float('-inf'))}
        
        for t in range(T):
            next_beam = defaultdict(lambda: (float('-inf'), float('-inf')))
            
            # Get top-k tokens for efficiency
            step_probs = probs[t]
            k = min(V, self.beam_width * 2)
            top_probs, top_indices = torch.topk(step_probs, k)
            
            for prefix, (p_b, p_nb) in beam.items():
                curr_p = _log_sum_exp(p_b, p_nb)
                
                for i in range(k):
                    token_id = top_indices[i].item()
                    p_token = torch.log(top_probs[i] + 1e-10).item()
                    
                    if token_id == self.blank_id:
                        # Blank: prefix stays same
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_b = _log_sum_exp(n_p_b, curr_p + p_token)
                        next_beam[prefix] = (n_p_b, n_p_nb)
                    else:
                        # Non-blank token
                        if len(prefix) > 0 and prefix[-1] == token_id:
                            # Repeat token: needs blank for new token
                            # Case 1: blank + token -> new token added
                            new_prefix = prefix + (token_id,)
                            n_p_b, n_p_nb = next_beam[new_prefix]
                            n_p_nb = _log_sum_exp(n_p_nb, p_b + p_token)
                            next_beam[new_prefix] = (n_p_b, n_p_nb)
                            
                            # Case 2: non_blank + same token -> merge
                            n_p_b, n_p_nb = next_beam[prefix]
                            n_p_nb = _log_sum_exp(n_p_nb, p_nb + p_token)
                            next_beam[prefix] = (n_p_b, n_p_nb)
                        else:
                            # Different token: extend prefix
                            new_prefix = prefix + (token_id,)
                            n_p_b, n_p_nb = next_beam[new_prefix]
                            new_score = curr_p + p_token
                            
                            # Add LM score
                            if self.lm is not None:
                                lm_score = self._get_lm_score(prefix, token_id)
                                new_score += self.lm_weight * lm_score
                                
                            n_p_nb = _log_sum_exp(n_p_nb, new_score)
                            next_beam[new_prefix] = (n_p_b, n_p_nb)
            
            # Prune to beam width
            scored_beams = [
                (prefix, scores, _log_sum_exp(*scores))
                for prefix, scores in next_beam.items()
            ]
            scored_beams.sort(key=lambda x: x[2], reverse=True)
            beam = {p: s for p, s, _ in scored_beams[:self.beam_width]}
            
        # Get best beam with word bonus
        best_prefix = None
        best_score = float('-inf')
        
        for prefix, (p_b, p_nb) in beam.items():
            score = _log_sum_exp(p_b, p_nb)
            
            # Add word count bonus
            if self.word_bonus > 0:
                text = self.tokenizer.decode(list(prefix))
                word_count = len(text.split())
                score += self.word_bonus * word_count
                
            if score > best_score:
                best_score = score
                best_prefix = prefix
                
        if best_prefix is None:
            return ""
            
        return self.tokenizer.decode(list(best_prefix))
    
    def _get_lm_score(self, prefix: Tuple, next_token: int) -> float:
        """Get LM score for next token."""
        if isinstance(self.lm, KenLMModel):
            # Decode to text for KenLM
            context = self.tokenizer.decode(list(prefix)) if prefix else ""
            next_text = self.tokenizer.decode([next_token])
            return self.lm.score_word(next_text, context)
        elif isinstance(self.lm, NGramLanguageModel):
            return self.lm.score(prefix, next_token)
        return 0.0


# =====================================================
# Flashlight Decoder (Optional - Best Performance)
# =====================================================

class FlashlightDecoder:
    """
    High-performance CTC decoder using Flashlight (if available).
    
    Installation:
        pip install flashlight-text
    """
    
    def __init__(
        self,
        tokenizer,
        lexicon_path: Optional[str] = None,
        lm_path: Optional[str] = None,
        beam_size: int = 100,
        lm_weight: float = 2.0,
        word_score: float = -1.0,
        sil_score: float = 0.0,
        beam_threshold: float = 25.0
    ):
        try:
            from flashlight.lib.text.decoder import (
                CriterionType,
                KenLM,
                LexiconDecoder,
                LexiconDecoderOptions,
                LexiconFreeDecoder,
                LexiconFreeDecoderOptions,
                SmearingMode,
                Trie,
            )
            self.fl_available = True
        except ImportError:
            print("Flashlight not available. Using fallback decoder.")
            self.fl_available = False
            self.fallback = CTCBeamDecoder(tokenizer, beam_width=beam_size)
            return
            
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        
        # Build vocabulary
        vocab = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
        
        # Configure decoder
        if lm_path and os.path.exists(lm_path):
            self.lm = KenLM(lm_path, vocab)
            self.use_lm = True
        else:
            self.lm = None
            self.use_lm = False
            
        # Decoder options
        self.options = LexiconFreeDecoderOptions(
            beam_size=beam_size,
            beam_size_token=beam_size,
            beam_threshold=beam_threshold,
            lm_weight=lm_weight if self.use_lm else 0.0,
            sil_score=sil_score
        )
        
        self.decoder = LexiconFreeDecoder(
            self.options,
            self.lm,
            0,  # sil_token
            0,  # blank
            vocab
        )
        
    def decode(self, logits: torch.Tensor) -> str:
        """Decode using Flashlight."""
        if not self.fl_available:
            return self.fallback.decode(logits)
            
        emissions = F.log_softmax(logits, dim=-1).cpu().numpy()
        results = self.decoder.decode([emissions], [emissions.shape[0]])
        
        if results and results[0]:
            best = results[0][0]
            return self.tokenizer.decode(best.tokens)
        return ""


# =====================================================
# Utility Functions
# =====================================================

def _log_sum_exp(a: float, b: float) -> float:
    """Numerically stable log-sum-exp."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def create_decoder(
    tokenizer,
    method: str = "greedy",
    lm_path: Optional[str] = None,
    beam_width: int = 10,
    lm_weight: float = 0.3
):
    """
    Factory function to create appropriate decoder.
    
    Args:
        tokenizer: Tokenizer instance
        method: "greedy", "beam", or "flashlight"
        lm_path: Path to KenLM model (optional)
        beam_width: Beam size for beam search
        lm_weight: LM weight for decoding
        
    Returns:
        Decoder instance
    """
    # Load LM if provided
    lm = None
    if lm_path and os.path.exists(lm_path):
        try:
            lm = KenLMModel(lm_path)
        except Exception as e:
            print(f"Warning: Could not load LM: {e}")
            
    if method == "greedy":
        return GreedyDecoder(tokenizer)
    elif method == "beam":
        return CTCBeamDecoder(tokenizer, beam_width=beam_width, lm=lm, lm_weight=lm_weight)
    elif method == "flashlight":
        return FlashlightDecoder(tokenizer, lm_path=lm_path, beam_size=beam_width)
    else:
        raise ValueError(f"Unknown decoder method: {method}")


# Legacy compatibility
CTCDecoder = CTCBeamDecoder
log_sum_exp = _log_sum_exp
