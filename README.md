# 🎙️ Türkçe ASR - Üretime Hazır Conformer Modeli

Modern tekniklerle geliştirilmiş **Conformer** mimarisine dayalı, yüksek performanslı Türkçe Otomatik Konuşma Tanıma (ASR) sistemi.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Özellikler

### Model Mimarisi

- **Conformer Kodlayıcı (Encoder)**: Modern iyileştirmelerle
- **Flash Attention**: PyTorch 2.0+ SDPA desteği ile hızlı dikkat mekanizması
- **Rotary Position Embeddings (RoPE)**: Göreceli pozisyon kodlama
- **Multi-Query Attention (MQA)**: Bellek verimliliği sağlayan dikkat yapısı
- **SwiGLU Aktivasyonu**: Modern İleri Beslemeli Ağ (FFN) yapısı
- **GroupNorm**: Batch boyutundan bağımsız normalizasyon

### Veri İşleme Hattı

- **torchaudio**: GPU hızlandırmalı Mel spektrogram çıkarımı
- **SpeedPerturbation**: 0.9x/1.0x/1.1x hız değişimleri ile veri çoğaltma
- **NoisePerturbation**: SNR tabanlı gürültü ekleme
- **SpecAugment**: Frekans ve zaman maskeleme
- **BucketingSampler**: Benzer uzunluktaki verileri gruplayarak verimli batch işleme

### Kod Çözme (Decoding)

- **Greedy Decoding**: Hızlı çıkarım
- **Beam Search**: Daha yüksek doğruluk
- **KenLM Entegrasyonu**: N-gram dil modeli desteği
- **Flashlight Decoder**: Yüksek performanslı kod çözücü seçeneği

### Üretim (Production)

- **ONNX Dışa Aktarma**: Platform bağımsız dağıtım
- **FastAPI Sunucusu**: REST API desteği
- **Docker**: Konteynerizasyon

## 📂 Proje Yapısı

```
Turkish-ASR-Model/
├── data/
│   ├── dataset.py        # BucketingSampler içeren Veri Seti sınıfı
│   ├── preprocessing.py  # torchaudio özellik çıkarımı
│   └── tokenizer.py      # HuggingFace tokenizer
├── model/
│   ├── conformer.py      # Conformer + SwiGLU + GroupNorm mimarisi
│   └── attention.py      # RoPE + MQA + Flash Attention
├── trainer/
│   └── trainer.py        # Gradyan kırpma/biriktirme özellikli eğitimci
├── utils/
│   ├── config.py         # Komut satırı argümanları
│   ├── decoding.py       # KenLM + Beam Search
│   ├── logger.py         # Loglama araçları
│   └── metrics.py        # WER/CER hesaplamaları
├── serve/
│   └── api.py            # FastAPI sunucusu
├── main.py               # Eğitim betiği
├── inference.py          # Tahmin/Çıkarım betiği
├── export_onnx.py        # ONNX dışa aktarma
├── Dockerfile            # Docker yapılandırması
└── requirements.txt      # Bağımlılıklar
```

## 🚀 Hızlı Başlangıç

### Kurulum

```bash
pip install -r requirements.txt
```

### Eğitim (Training)

```bash
# Temel eğitim
python main.py --data_path /veri/yolu --n_mel_channels 80

# Veri çoğaltma (augmentation) ile eğitim
python main.py --data_path /veri/yolu --augment --speed_perturb

# Gradyan biriktirme ile (efektif batch boyutu = 32 * 4 = 128)
python main.py --data_path /veri/yolu \
  --batch_size 32 \
  --accumulation_steps 4 \
  --gradient_clip 1.0

# Eğitime kaldığı yerden devam etme (Resume)
python main.py --resume
```

### Tahmin (Inference)

```bash
# Tek dosya için tahmin
python inference.py --audio ses.wav --model runs/best_model.pt

# Beam Search kullanarak tahmin
python inference.py --audio ses.wav --model runs/best_model.pt --beam_search
```

### ONNX Dışa Aktarma (Export)

```bash
python export_onnx.py --checkpoint runs/best_model.pt --output model.onnx
```

### API Sunucusu

```bash
# Yerel çalıştıma
python serve/api.py

# Docker ile çalıştırma
docker build -t turkish-asr .
docker run -p 8000:8000 -v ./runs:/app/models turkish-asr

# Test etme
curl -X POST http://localhost:8000/transcribe -F "file=@ses.wav"
```

## ⚙️ Yapılandırma

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--n_mel_channels` | 80 | Mel filtre sayısı |
| `--d_model` | 256 | Model boyutu |
| `--n_heads` | 4 | Dikkat başlığı sayısı |
| `--n_blocks` | 8 | Conformer blok sayısı |
| `--gradient_clip` | 1.0 | Maksimum gradyan normu |
| `--accumulation_steps` | 1 | Gradyan biriktirme adımları |
| `--augment` | False | SpecAugment aktif et |
| `--speed_perturb` | False | Hız değişimini aktif et |

## 📊 Metrikler

Eğitim çıktıları:

- **Loss**: CTC kaybı
- **WER**: Kelime Hata Oranı (Word Error Rate)
- **CER**: Karakter Hata Oranı (Character Error Rate)

## 🔧 İleri Düzey Konular

### KenLM Dil Modeli

```bash
# KenLM kurulumu
pip install https://github.com/kpu/kenlm/archive/master.zip

# Dil modeli eğitimi (corpus.txt üzerinden)
lmplz -o 4 < corpus.txt > lm.arpa
build_binary lm.arpa lm.bin

# Tahmin sırasında kullanma
python inference.py --audio ses.wav --model model.pt --lm lm.bin
```

### Docker Dağıtımı

```bash
# İnşa etme
docker build -t turkish-asr .

# GPU ile çalıştırma
docker run --gpus all -p 8000:8000 \
  -v ./runs:/app/models \
  -e ASR_MODEL_PATH=/app/models/best_model.pt \
  turkish-asr
```

## 📄 Lisans

MIT Lisansı - Detaylar için LICENSE dosyasına bakınız.

---
*Geliştirici: Muhammed Emin Korkut - Deep Zeka A.Ş*
