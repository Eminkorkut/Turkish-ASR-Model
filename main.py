import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os

# Proje modülleri
from utils.config import get_config
from utils.logger import get_logger
from data.tokenizer import TurkishTokenizer
from data.dataset import create_datasets, collate_fn 
from model.conformer import TurkishASRModel
from trainer.trainer import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # 1. Konfigürasyon
    config = get_config()
    logger = get_logger(__name__)
    
    logger.info("Konfigürasyon yüklendi. İşlem Başlıyor...")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Kullanılan Cihaz: {device}")
    
    # 2. Tokenizer
    tokenizer = TurkishTokenizer()
    logger.info(f"Tokenizer Hazır. Alfabe boyutu: {len(tokenizer.chars)}")
    
    # 3. Veri Setleri (Otomatik veya Manuel)
    logger.info("Veri setleri hazırlanıyor...")
    train_dataset, valid_dataset, test_dataset = create_datasets(config, tokenizer)
    
    if train_dataset is None or len(train_dataset) == 0:
        logger.error("HATA: Eğitim veri seti boş! Lütfen --data_path veya --train_path'i kontrol edin.")
        return

    logger.info(f"Dataset İstatistikleri: Train={len(train_dataset)}, Valid={len(valid_dataset) if valid_dataset else 0}, Test={len(test_dataset) if test_dataset else 0}")

    # 4. DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    valid_loader = None
    if valid_dataset:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.num_workers
        )
        
    # 5. Model
    logger.info("Model oluşturuluyor...")
    model = TurkishASRModel(
        n_mel_channels=config.n_mel_channels,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_blocks=config.n_blocks,
        n_classes=len(tokenizer.chars),
        dropout=config.encoder_dropout
    ).to(device)
    
    # 6. Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.learning_rate, 
        steps_per_epoch=len(train_loader), 
        epochs=config.epochs, 
        pct_start=0.1
    )
    
    # 7. Trainer (Valid loader eklendi, Tokenizer eklendi)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger,
        tokenizer=tokenizer # Metrikler için
    )
    
    # 8. Başlat
    try:
        trainer.fit()
    except KeyboardInterrupt:
        logger.info("Eğitim kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.exception(f"Beklenmedik bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
