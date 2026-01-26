# 🎙️ Gelişmiş Türkçe ASR Projesi (Modern Conformer & BPE)

Bu proje, Türkçe konuşma tanıma (Automatic Speech Recognition) için uçtan uca, modern ve yüksek performanslı bir çözüm sunar. Google'ın **Conformer** mimarisini temel alır ve OpenAI Whisper gibi SOTA modellerde görülen gelişmiş tekniklerle (GELU, Relative Attention, BPE) güçlendirilmiştir.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-Proprietary-red)

## 🌟 Özellikler

- **Modern Mimari:** Conformer (Convolution-augmented Transformer) blokları.
  - **Relative Positional Encoding:** Uzun seslerde daha iyi zaman algısı.
  - **GELU Aktivasyonu:** Daha hızlı ve kararlı eğitim (Whisper tarzı).
  - **Relative Multi-Head Attention:** Bağımsız pozisyon kodlaması.
- **Gelişmiş Tokenizasyon:**
  - **SentencePiece (BPE):** Karakter yerine alt-kelime (Subword) parçalama. Bu sayede kelime dağarcığı (`vocab_size`) genişler ve model dilbilgisi kurallarını daha iyi öğrenir.
- **Güçlü Veri Hattı:**
  - **Otomatik Bölümleme:** Tek bir klasörü Train/Valid/Test olarak otomatik böler.
  - **Raw Wav Desteği:** Ön işleme gerekmeden `.wav` ve `.txt` dosyalarıyla çalışır.
  - **Data Augmentation:** SpecAugment (Time & Freq Masking) ile gürültüye direnç.
- **Profesyonel Eğitim Döngüsü:**
  - **Canlı Metrikler:** Loss değerinin yanında **WER (Word Error Rate)** ve **CER (Character Error Rate)** takibi.
  - **Mixed Precision:** FP16 eğitimi ile daha hızlı ve az bellek kullanımı.
  - **OneCycleLR:** Gelişmiş learning rate planlaması.
- **Gelişmiş Çıkarım (Inference):**
  - **Beam Search Decoding:** Greedy aramaya göre çok daha başarılı sonuçlar.
  - **N-gram Language Model:** Basit dil modeli entegrasyonu (Decoding aşamasında).

## 📂 Dizin Yapısı

```
ASR_Project/
├── data/                # Veri ve Tokenizasyon Modülleri
│   ├── dataset.py       # Wav okuma ve oto-split mantığı
│   ├── tokenizer.py     # SentencePiece wrapper
│   └── preprocessing.py # Mel-Spectrogram dönüşümleri
├── model/               # Derin Öğrenme Mimarisi
│   ├── conformer.py     # Conformer blokları ve ana model
│   └── attention.py     # Relative Multi-Head Attention
├── trainer/             # Eğitim Motoru
│   └── trainer.py       # Eğitim, Validasyon, Checkpoint, Metrikler
├── utils/               # Araçlar
│   ├── config.py        # Argüman yönetimi (argparse)
│   ├── decoding.py      # Beam Search ve LM
│   ├── logger.py        # Loglama
│   └── metrics.py       # WER/CER hesabı (jiwer)
├── main.py              # Eğitim Başlatıcı
├── inference.py         # Test/Tahmin Scripti
├── spm_train.py         # Tokenizer Eğitim Scripti
└── README.md            # Dokümantasyon
```

## 🚀 Kurulum

Gerekli kütüphaneleri yükleyin:

```bash
pip install torch torchaudio numpy scipy sentencepiece jiwer
```

## 🛠️ Kullanım

### 1. Veri Hazırlığı ve Tokenizer Eğitimi (Zorunlu)

Eğitime başlamadan önce, veri setinizdeki metinleri tarayarak bir BPE (Byte Pair Encoding) modeli eğitmelisiniz. Bu adım `tokenizer_bpe.model` dosyasını oluşturur.

```bash
# Veri yolunu kendi klasörünüze göre düzenleyin
python spm_train.py --data_path "C:/Veri/Klasorum" --vocab_size 1000
```

*Not: `vocab_size` değeri veri büyüklüğüne göre 1000, 2000, 5000 seçilebilir.*

### 2. Model Eğitimi (Training)

Eğitimi başlatmak için sadece veri klasörünü göstermeniz yeterlidir. Sistem otomatik olarak train/valid/test ayrımı yapar.

```bash
python main.py --data_path "C:/Veri/Klasorum" --epochs 50 --batch_size 16 --vocab_size 1000
```

**Opsiyonel Parametreler:**

- `--val_split 0.2`: Verinin %20'sini validasyon için ayırır.
- `--checkpoint_dir "./kayitlar"`: Modellerin kaydedileceği yer.
- `--n_blocks 8` `--d_model 256`: Modelin derinliğini ve genişliğini ayarlar.

### 3. Test ve Tahmin (Inference)

Eğitilmiş bir modeli kullanarak ses dosyalarını metne çevirmek için:

```bash
python inference.py --wav_path "ornek_ses.wav" --model_path "checkpoints/best_model.pt"
```

**Beam Search Kullanımı:**
Daha iyi sonuçlar için beam genişliğini artırabilirsiniz:

```bash
python inference.py --wav_path "test.wav" --model_path "model.pt" --beam_width 10
```

## 📊 Performans Takibi (Metrikler)

Eğitim sırasında konsolda her epoch sonunda şunları göreceksiniz:

- **Loss:** Modelin matematiksel hatası.
- **WER (Word Error Rate):** Kelime bazlı hata oranı (Düşük olması iyidir).
- **CER (Character Error Rate):** Harf bazlı hata oranı.

Örnek Çıktı:

```
Epoch 10 | Validation Loss: 0.4523 | WER: 0.1250 | CER: 0.0410
```

## 🧠 Model Mimarisi Detayları

Proje, **Conformer** makalesindeki (Gulati et al., 2020) mimariyi takip eder:

1. **SpecAugment:** Giriş spektrogramında rastgele maskeleme.
2. **Convolution Subsampling:** Zaman boyutunu 4 kat küçültür (Hız kazandırır).
3. **Relative Positional Encoding:** Sesin akış yönünü modele öğretir.
4. **Macaron Style FFN:** Blok başında ve sonunda yarımşar Feed-Forward katmanı.
5. **Multi-Head Self Attention:** Global bağlamı yakalar.
6. **Convolution Module:** Lokal özellikleri (fonem geçişleri) yakalar.

## 📄 License

This project is licensed under a modified MIT-style **Proprietary License**.

> **Permission is hereby granted, free of charge, to handle the Software, subject to the following restrictions:**
>
> 1. **Commercial Use:** Prohibited without written permission.
> 2. **Modification:** Prohibited without written permission.
> 3. **Distribution:** Prohibited without written permission.

See the `LICENSE` file for the full legal text.

---
*Developed by Muhammed Emin Korkut - Deep Zeka A.Ş*
