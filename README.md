# Türkçe ASR Projesi (Conformer Mimarisi)

Bu proje, Türkçe ses tanıma (Automatic Speech Recognition - ASR) için **Conformer** (Convolution-augmented Transformer) mimarisini temel alan, derin öğrenme tabanlı bir sistemdir.

Tüm kod yapısı modüler hale getirilmiş, açıklamalar Türkçeleştirilmiş ve parametre yönetimi kolaylaştırılmıştır.

## Dizin Yapısı

Proje aşağıdaki modüler yapıya sahiptir:

```
ASR_Project/
├── main.py              # Projenin ana çalıştırma dosyası.
├── run_train.bat        # Tek tıkla eğitim başlatmak için script.
├── data/                # Veri işleme modülleri
│   ├── dataset.py       # Veri seti (Dataset) sınıfı.
│   ├── tokenizer.py     # Metin <-> Sayı dönüşümü yapan Tokenizer.
│   └── preprocessing.py # Ses sinyal işleme (Spectrogram vb.) fonksiyonları.
├── model/               # Derin öğrenme modelleri
│   └── conformer.py     # Conformer model mimarisi.
├── trainer/             # Eğitim mantığı
│   └── trainer.py       # Eğitim döngüsünü yöneten Trainer sınıfı.
└── utils/               # Yardımcı araçlar
    ├── config.py        # Hiperparametre ve argüman yönetimi.
    └── logger.py        # Loglama işlemleri.
```

## Kurulum ve Gereksinimler

Projenin çalışması için PyTorch ve ilgili kütüphanelerin yüklü olması gerekir.
(Eğer `requirements.txt` varsa onu kullanabilirsiniz, yoksa temel olarak `torch`, `numpy` gereklidir).

## Kullanım

Eğitimi başlatmak için terminali açın ve şu komutu girin:

```bash
python main.py
```

Veya varsayılan parametreleri değiştirmek isterseniz:

```bash
python main.py --epochs 100 --batch_size 16 --d_model 512
```

### Parametreler

`python main.py --help` komutu ile tüm parametreleri görebilirsiniz. Öne çıkanlar:

- `--feature_path`: İşlenmiş özelliklerin (.pt) olduğu klasör.
- `--text_path`: Metin dosyalarının (.txt) olduğu klasör.
- `--epochs`: Eğitim tur sayısı.
- `--batch_size`: Batch boyutu.
- `--n_blocks`: Conformer blok sayısı (Derinlik).

## Model Hakkında

Model, Google'ın Conformer makalesinden esinlenmiştir. Hem **Self-Attention** (Global bağlam) hem de **Convolution** (Lokal detaylar) katmanlarını birleştirerek ses tanımada yüksek başarım hedefler.

- **Giriş:** Log-Mel Spektrogram
- **Kodlayıcı (Encoder):** Derin Conformer blokları
- **Çıkış:** CTC (Connectionist Temporal Classification) karakter olasılıkları.
