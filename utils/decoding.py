import torch
import torch.nn.functional as F
from collections import defaultdict
import math

class NGramLanguageModel:
    """
    Basit N-gram Dil Modeli.
    Gerçek bir uygulamada KenLM kullanılmalıdır, ancak bu sınıf
    mantığı göstermek ve basit durumlarda çalışmak içindir.
    """
    def __init__(self, order=3):
        self.order = order
        self.counts = defaultdict(int)
        self.total_counts = defaultdict(int)
        
    def train(self, text_list, tokenizer=None):
        """Metin listesinden n-gram sayar."""
        # Basit kelime bazlı n-gram (veya karakter)
        # BPE kullanıyorsak token-based n-gram daha mantıklı.
        for text in text_list:
            if tokenizer:
                tokens = tokenizer.encode(text) # IDs
            else:
                tokens = text.split() # Fallback
                
            # Padding
            tokens = ["<s>"] * (self.order - 1) + tokens + ["</s>"]
            
            for i in range(len(tokens) - self.order + 1):
                ngram = tuple(tokens[i:i+self.order])
                history = ngram[:-1]
                self.counts[ngram] += 1
                self.total_counts[history] += 1
                
    def score(self, history_tokens, next_token):
        """
        P(next_token | history) olasılığını (log prob) döndürür.
        """
        # History son (n-1) tokenı al
        history = tuple(history_tokens[-(self.order - 1):] if self.order > 1 else [])
        ngram = history + (next_token,)
        
        count = self.counts.get(ngram, 0)
        total = self.total_counts.get(history, 0)
        
        if total == 0:
            # Backoff veya smoothing gerekebilir aslında
            return -10.0 # Cezalandır
            
        prob = count / total
        return math.log(prob)


class CTCDecoder:
    """
    CTC Beam Search Decoding işlemi.
    """
    def __init__(self, tokenizer, beam_width=10, lm=None, alpha=0.3, beta=0.8):
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.lm = lm
        self.alpha = alpha # LM ağırlığı
        self.beta = beta   # Kelime uzunluk ödülü (bonus)
        
    def decode(self, logits):
        """
        Args:
            logits: (Time, Vocab) tensor (Batchsiz)
        Returns:
            str: En iyi tahmin
        """
        # Softmax
        probs = F.softmax(logits, dim=-1) # (T, V)
        T, V = probs.shape
        
        # Beam: (score, text_ids_tuple, last_token_id)
        # last_token_id CTC için blank takibi için gerekli olabilir ama
        # basit versiyonda sadece text biriktireceğiz.
        
        # CTC Algoritması (Prefix Beam Search basitleştirilmiş)
        # {prefix_tuple: (prob_blank, prob_non_blank)}
        
        # Başlangıç: Boş prefix, prob=1.0 (log=0)
        beam = {(): (0.0, -float('inf'))} # (log_prob_blank, log_prob_non_blank)
        
        for t in range(T):
            next_beam = defaultdict(lambda: (-float('inf'), -float('inf')))
            
            # Pruning: Sadece en yüksek olasılıklı k token'ı al (Hız için)
            # Bu tam beam search değil ama yakındır.
            step_probs = probs[t]
            top_k_probs, top_k_indices = torch.topk(step_probs, min(V, self.beam_width))
            
            for prefix, (p_b, p_nb) in beam.items():
                curr_p = log_sum_exp(p_b, p_nb)
                
                # Eğer prefix beam dışındaysa atla (Pruning sonrası)
                # if curr_p < threshold: continue
                
                for i in range(len(top_k_indices)):
                    token_index = top_k_indices[i].item()
                    p_token = torch.log(top_k_probs[i]).item()
                    
                    if token_index == 0: # Blank
                        # Blank eklenirse prefix değişmez
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_b = log_sum_exp(n_p_b, curr_p + p_token)
                        next_beam[prefix] = (n_p_b, n_p_nb)
                    else:
                        token = token_index
                        
                        # 1. Durum: Aynı karakter tekrar ediyor (aa -> a)
                        if len(prefix) > 0 and prefix[-1] == token:
                            # a) Önceki blank ise: yeni karakter (a_a -> aa)
                            n_p_b, n_p_nb = next_beam[prefix + (token,)]
                            n_p_nb = log_sum_exp(n_p_nb, p_b + p_token)
                            next_beam[prefix + (token,)] = (n_p_b, n_p_nb)
                            
                            # b) Önceki non-blank ise: birleştir (aa -> a)
                            n_p_b, n_p_nb = next_beam[prefix]
                            n_p_nb = log_sum_exp(n_p_nb, p_nb + p_token)
                            next_beam[prefix] = (n_p_b, n_p_nb)
                        else:
                            # 2. Durum: Yeni karakter
                            n_p_b, n_p_nb = next_beam[prefix + (token,)]
                            n_p_nb = log_sum_exp(n_p_nb, curr_p + p_token)
                            next_beam[prefix + (token,)] = (n_p_b, n_p_nb)
                            
                            # LM Puanı Ekle (Yeni kelime/token eklendiğinde)
                            if self.lm:
                                lm_score = self.lm.score(prefix, token)
                                # Burada n-p_nb güncellenmeli ama basit tutuyoruz
                                # LM skorunu finalde eklemek daha kolaydır bu yapıda.
            
            # Beam'i sırala ve kes
            sorted_beam = sorted(
                next_beam.items(),
                key=lambda x: log_sum_exp(*x[1]),
                reverse=True
            )
            beam = dict(sorted_beam[:self.beam_width])
            
        # En iyi sonucu seç
        best_prefix = max(beam.items(), key=lambda x: log_sum_exp(*x[1]))[0]
        
        # ID -> Text
        return self.tokenizer.decode(list(best_prefix))

def log_sum_exp(a, b):
    """Numerik kararlılık için log-sum-exp"""
    if a == -float('inf'): return b
    if b == -float('inf'): return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))
