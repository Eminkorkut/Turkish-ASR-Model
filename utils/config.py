import argparse

def get_config():
    """
    Parse command line arguments and hyperparameters.
    
    Returns:
        argparse.Namespace: Configuration object with all parameters.
    """
    parser = argparse.ArgumentParser(description="Turkish ASR Model Training")

    # --- Data Paths ---
    parser.add_argument("--data_path", type=str, default=None, help="Main data directory (wav + txt files)")
    parser.add_argument("--train_path", type=str, default=None, help="Training data directory (optional)")
    parser.add_argument("--valid_path", type=str, default=None, help="Validation data directory (optional)")
    parser.add_argument("--test_path", type=str, default=None, help="Test data directory (optional)")
    parser.add_argument("--noise_dir", type=str, default=None, help="Directory with noise files for augmentation")
    
    # Split ratios
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    
    # Tokenizer
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size")

    # Checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="./runs", help="Checkpoint save directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--output_model_path", type=str, default="turkish_conformer_final.pt", help="Final model filename")

    # --- Model Architecture ---
    parser.add_argument("--n_mel_channels", type=int, default=80, help="Number of mel filterbanks")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_blocks", type=int, default=8, help="Number of Conformer blocks")
    parser.add_argument("--encoder_dropout", type=float, default=0.1, help="Dropout rate")

    # --- Training Hyperparameters ---
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=70, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Max learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Gradient Management
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Augmentation
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--speed_perturb", action="store_true", help="Enable speed perturbation")
    parser.add_argument("--spec_augment_freq", type=int, default=27, help="SpecAugment frequency mask param")
    parser.add_argument("--spec_augment_time", type=int, default=100, help="SpecAugment time mask param")

    # --- Other ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging frequency (batches)")
    parser.add_argument("--save_interval", type=int, default=5, help="Checkpoint save frequency (epochs)")

    config = parser.parse_args()
    return config
