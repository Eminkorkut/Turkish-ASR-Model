import argparse

def get_config():
    """
    Komut satırı argümanlarını ve hiperparametreleri ayrıştırır.
    
    Returns:
        argparse.Namespace: Tüm konfigürasyon parametrelerini içeren nesne.
    """
    parser = argparse.ArgumentParser(description="Türkçe ASR Modeli Eğitim ve Test Altyapısı")

    # --- Veri Yolları (Revize Edildi) ---
    parser.add_argument("--data_path", type=str, default=r"C:\Users\memin\OneDrive\Desktop\ASR\ISSAI_TSC_218\Train", help="Verilerin bulunduğu ana klasör. (Hem wav hem txt içerebilir)")
    
    # Opsiyonel: Eğer kullanıcı train/test/val klasörlerini ayrı ayrı belirtirse bunlar kullanılır
    parser.add_argument("--train_path", type=str, default=None, help="Eğitim verisi klasörü (Opsiyonel). Belirtilmezse data_path'ten ayrılır.")
    parser.add_argument("--valid_path", type=str, default=None, help="Validasyon verisi klasörü (Opsiyonel). Belirtilmezse data_path'ten ayrılır.")
    parser.add_argument("--test_path", type=str, default=None, help="Test verisi klasörü (Opsiyonel). Belirtilmezse data_path'ten ayrılır.")
    
    # Bölümleme Oranları
    parser.add_argument("--val_split", type=float, default=0.1, help="Validasyon ayırma oranı (örn. 0.1 = %10).")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test ayırma oranı (örn. 0.1 = %10).")
    
    parser.add_argument("--vocab_size", type=int, default=1000, help="SentencePiece sözlük boyutu.")

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Model kayıtlarının (checkpoint) saklanacağı klasör.")
    parser.add_argument("--output_model_path", type=str, default="turkish_conformer_final.pt", help="Eğitim sonundaki final modelin kaydedileceği yol.")

    # --- Model Hiperparametreleri ---
    parser.add_argument("--n_mel_channels", type=int, default=40, help="Mel spektrogram kanal sayısı (giriş boyutu).")
    parser.add_argument("--d_model", type=int, default=256, help="Modelin iç katman boyutu (embedding dimension).")
    parser.add_argument("--n_heads", type=int, default=4, help="Multi-head attention kafa sayısı.")
    parser.add_argument("--n_blocks", type=int, default=8, help="Conformer blok sayısı (Derinlik).")
    parser.add_argument("--encoder_dropout", type=float, default=0.1, help="Encoder katmanlarındaki dropout oranı.")

    # --- Eğitim Hiperparametreleri ---
    parser.add_argument("--batch_size", type=int, default=32, help="Eğitim batch boyutu.")
    parser.add_argument("--epochs", type=int, default=70, help="Toplam epoch sayısı.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Maksimum öğrenme oranı (OneCycleLR için).")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Ağırlık azaltma (Regularization) katsayısı.")
    parser.add_argument("--num_workers", type=int, default=4, help="Veri yükleyici (DataLoader) çalışan sayısı.")
    parser.add_argument("--spec_augment_freq", type=int, default=15, help="SpecAugment frekans maskeleme genişliği.")
    parser.add_argument("--spec_augment_time", type=int, default=35, help="SpecAugment zaman maskeleme genişliği.")

    # --- Diğer ---
    parser.add_argument("--seed", type=int, default=42, help="Tekrarlanabilirlik için rastgelelik tohumu (seed).")
    parser.add_argument("--log_interval", type=int, default=10, help="Loglama sıklığı (batch bazında).")
    parser.add_argument("--save_interval", type=int, default=10, help="Modelin kaç epochta bir kaydedileceği.")

    config = parser.parse_args()
    return config
