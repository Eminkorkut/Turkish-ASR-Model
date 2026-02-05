from transformers import AutoTokenizer
import torch

class TurkishTokenizer:
    """
    HuggingFace AutoTokenizer tabanlı Tokenizer.
    Model: alibayram/turkish-mft-tokenizer
    """

    def __init__(self, model_name="alibayram/turkish-mft-tokenizer"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Pad token yoksa ekleyelim (pad_token_id genellikle EOS veya 0 olabilir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def vocab_size(self):
        return len(self.tokenizer)
        
    @property
    def chars(self):
        # Model n_classes için range döndürüyoruz.
        return range(self.vocab_size)

    def encode(self, text):
        """Metni ID dizisine çevirir."""
        return self.tokenizer.encode(text)

    def decode(self, ids):
        """ID dizisini metne çevirir."""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def ctc_decode(self, ids):
        """
        CTC Greedy Decode:
        1. Ardışık tekrarları sil.
        2. Blank/Pad tokenları sil.
        3. Kalanları decode et.
        """
        # HuggingFace tokenizer'da blank token ID'si modele göre değişir.
        # Genellikle pad_token_id kullanılır.
        blank_id = self.tokenizer.pad_token_id
        
        filtered_ids = []
        last_id = None
        
        for curr_id in ids:
            if isinstance(curr_id, torch.Tensor):
                curr_id = curr_id.item()
                
            if curr_id != last_id:
                if curr_id != blank_id: 
                    filtered_ids.append(curr_id)
            last_id = curr_id
            
        return self.tokenizer.decode(filtered_ids, skip_special_tokens=True)
