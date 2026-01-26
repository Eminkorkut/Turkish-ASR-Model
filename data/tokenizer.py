import sentencepiece as spm
import os

class TurkishTokenizer:
    """
    SentencePiece tabanlı BPE (Byte Pair Encoding) Tokenizer.
    
    CTC uyumluluğu için:
    - ID 0: Blank (Boşluk/Pad) token olarak kabul edilir.
    """

    def __init__(self, model_path="tokenizer_bpe.model"):
        self.sp = spm.SentencePieceProcessor()
        if os.path.exists(model_path):
            self.sp.load(model_path)
        else:
            # Model henüz eğitilmediyse dummy fallback veya hata
            # İlk kurulumda train_tokenizer.py çalıştırılmalı.
            pass
            
    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
        
    @property
    def chars(self):
        # Eski kodla uyumluluk için (len(tokenizer.chars) kullanılıyor)
        # Aslında bu "vocab" olmalı ama main.py'de n_classes için kullanılıyor.
        return range(self.vocab_size)

    def encode(self, text):
        """Metni ID dizisine çevirir."""
        return self.sp.encode_as_ids(text.lower())

    def decode(self, ids):
        """ID dizisini metne çevirir."""
        return self.sp.decode_ids(ids)

    def ctc_decode(self, ids):
        """
        CTC Greedy Decode:
        1. Ardışık tekrarları sil.
        2. Blank (0) tokenları sil.
        3. Kalanları decode et.
        """
        filtered_ids = []
        last_id = None
        
        for curr_id in ids:
            if curr_id != last_id:
                if curr_id != 0: # 0 = Blank (Pad)
                    filtered_ids.append(curr_id)
            last_id = curr_id
            
        return self.decode(filtered_ids)
