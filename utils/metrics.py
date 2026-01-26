import jiwer
import torch

class ASRMetrics:
    """
    ASR modelleri için başarı metriklerini (WER, CER) hesaplar.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute(self, predictions, targets):
        """
        Batch halindeki tahminler ve hedefler için ortalama WER ve CER hesaplar.

        Args:
            predictions (Tensor): Model çıktıları (Logits) [Batch, Time, Classes]
            targets (Tensor): Hedef ID'ler [Batch, Len] (Paddingli)

        Returns:
            dict: {"wer": float, "cer": float}
        """
        # 1. Tahminleri Decode Et (Argmax + CTC Decode)
        # Logits -> ID
        pred_ids = torch.argmax(predictions, dim=2)
        
        preds_str = []
        targets_str = []
        
        for i in range(pred_ids.size(0)):
            # Tahmin
            p_ids = pred_ids[i].tolist()
            p_text = self.tokenizer.ctc_decode(p_ids)
            preds_str.append(p_text)
            
            # Hedef (Paddingleri temizle)
            # Not: Tokenizer'ın decode metodunda padding/blank ayrımı olmayabilir,
            # bu yüzden Dataset tarafında padding değeri (0) kullanıldıysa onu temizlemek gerek.
            # Bizim tokenizer'da 0 = Blank ve Dataset'te padding=0. 
            # Normal decode paddingleri de karaktere çevirebilir (Blank ise sorun yok).
            # Ancak target için "raw decode" yapsak daha iyi, ama 0 ları süzmeliyiz.
            t_ids = targets[i].tolist()
            # 0 (Blank/Pad) olanları çıkar
            t_ids_clean = [idx for idx in t_ids if idx != 0]
            t_text = self.tokenizer.decode(t_ids_clean)
            targets_str.append(t_text)

        # 2. Metrik Hesapla
        try:
            wer = jiwer.wer(targets_str, preds_str)
            cer = jiwer.cer(targets_str, preds_str)
        except Exception:
            # Boş string hatası vb. durumlar için
            wer = 1.0
            cer = 1.0
            
        return {"wer": wer, "cer": cer}, preds_str, targets_str
