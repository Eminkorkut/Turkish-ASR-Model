import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import os
import glob
import scipy.io.wavfile as wav
import numpy as np

# Proje içi modüller
from data.preprocessing import pre_emphasis, framing, windowing, power_spectrum, get_filter_banks, normalize_features

class ASRDataset(Dataset):
    """
    ASR Modeli için PyTorch Dataset Sınıfı.
    Doğrudan WAV dosyalarını okur, preprocessing uygular ve modele besler.
    """
    def __init__(self, file_pairs, tokenizer, n_mel_channels=40):
        """
        Args:
            file_pairs (list): (wav_path, txt_path) tuple'larından oluşan liste.
            tokenizer (TurkishTokenizer): Metin tokenizer.
            n_mel_channels (int): Mel özellik sayısı.
        """
        self.file_pairs = file_pairs
        self.tokenizer = tokenizer
        self.n_mel_channels = n_mel_channels
        
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        wav_path, txt_path = self.file_pairs[idx]
        
        # 1. Sesi Yükle ve İşle (On-the-fly Feature Extraction)
        try:
            sample_rate, signal = wav.read(wav_path)
            # Sinyal float tipine çevir
            signal = signal.astype(float)
            
            # --- Preprocessing Adımları ---
            # 1. Pre-emphasis
            signal = pre_emphasis(signal)
            # 2. Framing
            frames = framing(signal, sample_rate)
            # 3. Windowing
            frames = windowing(frames)
            # 4. Power Spectrum
            pow_spec = power_spectrum(frames)
            # 5. Mel Filterbank Features
            features = get_filter_banks(pow_spec, sample_rate, nfilt=self.n_mel_channels)
            # 6. Normalization
            features = normalize_features(features)
            
            # Tensor'a çevir
            features = torch.FloatTensor(features) # (Time, Mel_Channels)
            
        except Exception as e:
            print(f"HATA: Ses dosyası işlenirken hata oluştu ({wav_path}): {e}")
            # Hata durumunda boş döndür veya dummy data (Handle edilmesi gerekir)
            return self.__getitem__((idx + 1) % len(self))

        # 2. Metni Okur
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
        except FileNotFoundError:
            transcript = ""
            
        # 3. Tokenize et
        target = torch.LongTensor(self.tokenizer.encode(transcript))
        
        return features, target

def find_files(root_dir):
    """
    Klasör içindeki (ve alt klasörlerdeki) tüm .wav dosyalarını bulur
    ve karşılık gelen .txt dosyalarıyla eşleştirir.
    
    Varsayım: "dosya_adı.wav" için "dosya_adı.txt" aranır.
    """
    wav_files = glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
    pairs = []
    
    for wav_path in wav_files:
        # İlgili .txt dosyasının yolunu tahmin et
        txt_path = wav_path.replace(".wav", ".txt")
        
        # Eğer .txt varsa listeye ekle
        if os.path.exists(txt_path):
            pairs.append((wav_path, txt_path))
            
    return pairs

def create_datasets(config, tokenizer):
    """
    Konfigürasyona göre Train, Valid ve Test datasetlerini oluşturur.
    Otomatik veya manuel bölme yapar.
    """
    train_pairs = []
    valid_pairs = []
    test_pairs = []
    
    # 1. Varsa Özel Yolları Kontrol Et
    if config.train_path:
        print(f"Train verisi taranıyor: {config.train_path}")
        train_pairs = find_files(config.train_path)
    
    if config.valid_path:
        print(f"Validasyon verisi taranıyor: {config.valid_path}")
        valid_pairs = find_files(config.valid_path)
        
    if config.test_path:
        print(f"Test verisi taranıyor: {config.test_path}")
        test_pairs = find_files(config.test_path)
        
    # 2. Eğer özel yollar yoksa, Ana Data Path'ten orantısal bölme yap
    if not train_pairs and config.data_path:
        print(f"Ana veri dizini taranıyor ve otomatik bölünüyor: {config.data_path}")
        all_pairs = find_files(config.data_path)
        total_count = len(all_pairs)
        
        if total_count == 0:
            raise ValueError(f"HATA: Belirtilen dizinde veri bulunamadı: {config.data_path}")
            
        # Oranları hesapla
        test_size = int(total_count * config.test_split)
        valid_size = int(total_count * config.val_split)
        train_size = total_count - test_size - valid_size
        
        # Rastgele karıştır ve böl
        # Sabit seed ile karıştırarak her çalıştırmada aynı seti elde edelim
        import random
        random.seed(config.seed)
        random.shuffle(all_pairs)
        
        train_pairs = all_pairs[:train_size]
        valid_pairs = all_pairs[train_size : train_size + valid_size]
        test_pairs = all_pairs[train_size + valid_size:]
        
        print(f"Otomatik Bölümleme Sonucu: Train: {len(train_pairs)}, Valid: {len(valid_pairs)}, Test: {len(test_pairs)}")

    # Dataset Nesnelerini Oluştur
    train_dataset = ASRDataset(train_pairs, tokenizer, config.n_mel_channels) if train_pairs else None
    valid_dataset = ASRDataset(valid_pairs, tokenizer, config.n_mel_channels) if valid_pairs else None
    test_dataset = ASRDataset(test_pairs, tokenizer, config.n_mel_channels) if test_pairs else None
    
    return train_dataset, valid_dataset, test_dataset

def collate_fn(batch):
    """
    DataLoader birleştirme fonksiyonu.
    """
    # Hatalı/Boş verileri filtrele
    batch = [item for item in batch if item is not None and item[0] is not None]
    if len(batch) == 0:
        return None, None, None, None
        
    features, targets = zip(*batch)
    
    input_lengths = torch.LongTensor([f.size(0) for f in features])
    target_lengths = torch.LongTensor([len(t) for t in targets])
    
    features_padded = nn.utils.rnn.pad_sequence(features, batch_first=True)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    
    return features_padded, targets_padded, input_lengths, target_lengths
