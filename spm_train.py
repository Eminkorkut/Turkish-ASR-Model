import sentencepiece as spm
import os
import glob
from utils.config import get_config

def train_tokenizer():
    config = get_config()
    data_path = config.data_path
    vocab_size = config.vocab_size
    model_prefix = "tokenizer_bpe"
    
    print(f"Veri yolu: {data_path}")
    print(f"Hedeflenen Sözlük Boyutu: {vocab_size}")
    
    # 1. Tüm metin dosyalarını bul
    txt_files = glob.glob(os.path.join(data_path, "**", "*.txt"), recursive=True)
    
    if not txt_files:
        print("HATA: Hiçbir .txt dosyası bulunamadı!")
        return

    print(f"Toplam {len(txt_files)} metin dosyası bulundu. Birleştiriliyor...")
    
    # 2. Hepsini geçici bir dosyaya yaz
    temp_corpus = "temp_corpus.txt"
    with open(temp_corpus, "w", encoding="utf-8") as outfile:
        for fpath in txt_files:
            with open(fpath, "r", encoding="utf-8") as infile:
                outfile.write(infile.read().strip() + "\n")
                
    # 3. SentencePiece Eğitimi
    # pad_id=0 -> CTC Blank olarak kullanılacak
    # bos_id=-1, eos_id=-1 -> CTC'de cümle başı/sonu sembolü genelde şeffaftır, kapatalım.
    # unk_id=1 -> Bilinmeyen karakterler
    print("SentencePiece eğitimi başlıyor...")
    spm.SentencePieceTrainer.train(
        input=temp_corpus,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0, # Tüm karakterleri kapsasın (Türkçe karakterler için önemli)
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1
    )
    
    print(f"Eğitim tamamlandı! Oluşturulan dosyalar: {model_prefix}.model, {model_prefix}.vocab")
    
    # Temizlik
    os.remove(temp_corpus)

if __name__ == "__main__":
    train_tokenizer()
