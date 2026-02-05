"""
Turkish ASR Model - Main Training Script
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os

from utils.config import get_config
from utils.logger import get_logger
from data.tokenizer import TurkishTokenizer
from data.dataset import create_datasets, collate_fn, BucketingSampler
from model.conformer import TurkishASRModel
from trainer.trainer import Trainer


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Configuration
    config = get_config()
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("Turkish ASR Model Training")
    logger.info("=" * 60)
    
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # 2. Tokenizer
    tokenizer = TurkishTokenizer()
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    
    # 3. Datasets
    logger.info("Preparing datasets...")
    train_dataset, valid_dataset, test_dataset = create_datasets(
        config, 
        tokenizer,
        augment_train=config.augment
    )
    
    if train_dataset is None or len(train_dataset) == 0:
        logger.error("Training dataset is empty! Check --data_path or --train_path.")
        return

    logger.info(f"Datasets: Train={len(train_dataset)}, Valid={len(valid_dataset) if valid_dataset else 0}, Test={len(test_dataset) if test_dataset else 0}")

    # 4. DataLoaders with BucketingSampler
    train_sampler = BucketingSampler(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        sampler=train_sampler,
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
    logger.info("Building model...")
    model = TurkishASRModel(
        n_mel_channels=config.n_mel_channels,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_blocks=config.n_blocks,
        n_classes=tokenizer.vocab_size,
        dropout=config.encoder_dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # 6. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Calculate total steps for scheduler
    steps_per_epoch = len(train_loader) // config.accumulation_steps
    total_steps = steps_per_epoch * config.epochs
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.learning_rate, 
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 7. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger,
        tokenizer=tokenizer,
        gradient_clip=config.gradient_clip,
        accumulation_steps=config.accumulation_steps
    )
    
    # 8. Train
    try:
        trainer.fit()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        trainer.save_checkpoint(trainer.start_epoch, name="interrupted_checkpoint.pt")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
