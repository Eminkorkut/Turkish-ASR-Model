"""
Professional ASR Trainer with gradient clipping, accumulation, and resumable checkpointing.
"""

import torch
import torch.nn as nn
import os
import time
import glob
from typing import Optional, Dict, Any
from utils.metrics import ASRMetrics


class Trainer:
    """
    Trainer class for Turkish ASR Model.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient clipping for stability
    - Gradient accumulation for larger effective batch sizes
    - Resumable checkpointing
    - Comprehensive logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        optimizer,
        scheduler,
        device: torch.device,
        config,
        logger,
        valid_loader=None,
        tokenizer=None,
        gradient_clip: float = 1.0,
        accumulation_steps: int = 1
    ):
        """
        Args:
            model: ASR model
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Training device
            config: Configuration object
            logger: Logger instance
            valid_loader: Optional validation data loader
            tokenizer: Tokenizer for metrics
            gradient_clip: Max gradient norm for clipping
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger
        self.tokenizer = tokenizer
        
        # Training hyperparameters
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        
        # Metrics
        if tokenizer:
            self.metrics = ASRMetrics(tokenizer)
        else:
            self.metrics = None
            self.logger.warning("Tokenizer not provided! WER/CER calculation disabled.")
        
        # Loss and AMP
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Training state
        self.start_epoch = 1
        self.best_val_loss = float('inf')
        self.global_step = 0

    def save_checkpoint(self, epoch: int, name: Optional[str] = None, is_best: bool = False) -> None:
        """Save complete training state to checkpoint."""
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
            
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config)
        }
        
        if name is None:
            name = f"checkpoint_epoch_{epoch}.pt"
            
        path = os.path.join(self.config.checkpoint_dir, name)
        torch.save(state, path)
        self.logger.info(f"Checkpoint saved: {path}")
        
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(state, best_path)
            self.logger.info(f"Best model updated: {best_path}")

    def load_checkpoint(self) -> None:
        """Load latest checkpoint if resuming."""
        if not self.config.resume:
            return

        checkpoints = sorted(
            glob.glob(os.path.join(self.config.checkpoint_dir, "checkpoint_epoch_*.pt")),
            key=os.path.getmtime
        )
        
        if not checkpoints:
            self.logger.warning("No checkpoint found! Starting from scratch.")
            return

        latest_ckpt = checkpoints[-1]
        self.logger.info(f"Resuming from: {latest_ckpt}")
        
        try:
            checkpoint = torch.load(latest_ckpt, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            self.logger.info(f"Loaded checkpoint. Resuming from Epoch {self.start_epoch}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with gradient accumulation and clipping."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            if batch[0] is None:
                continue
                
            features, targets, input_lengths, target_lengths = batch
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda'):
                output = self.model(features, input_lengths=input_lengths)
                output = output.permute(1, 0, 2)  # (T, B, C) for CTC
                log_probs = torch.nn.functional.log_softmax(output, dim=2)
                
                # Adjust input lengths for subsampling
                current_input_lengths = input_lengths // 4
                
                loss = self.criterion(log_probs, targets, current_input_lengths, target_lengths)
                
                # Scale loss for accumulation
                loss = loss / self.accumulation_steps
            
            # Skip NaN losses
            if torch.isnan(loss):
                self.logger.warning(f"Epoch {epoch}, Batch {batch_idx}: NaN loss, skipping...")
                continue
                
            # Backward pass with scaling
            self.scaler.scale(loss).backward()
            
            # Optimizer step every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            epoch_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch [{epoch}/{self.config.epochs}] "
                    f"Batch [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item() * self.accumulation_steps:.4f} "
                    f"LR: {current_lr:.2e}"
                )
        
        # Handle remaining gradients
        if num_batches % self.accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        duration = time.time() - start_time
        
        self.logger.info(f"Epoch {epoch} Complete | Loss: {avg_loss:.4f} | Time: {duration:.1f}s")
        return avg_loss

    def validate(self, epoch: int) -> Optional[float]:
        """Validate model and compute metrics."""
        if not self.valid_loader:
            return None
            
        self.model.eval()
        val_loss = 0.0
        total_wer = 0.0
        total_cer = 0.0
        num_batches = 0
        
        example_preds = []
        example_targets = []
        
        with torch.no_grad():
            for batch in self.valid_loader:
                if batch[0] is None:
                    continue
                    
                features, targets, input_lengths, target_lengths = batch
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                output = self.model(features, input_lengths=input_lengths)
                output_permuted = output.permute(1, 0, 2)
                log_probs = torch.nn.functional.log_softmax(output_permuted, dim=2)
                
                current_input_lengths = input_lengths // 4
                loss = self.criterion(log_probs, targets, current_input_lengths, target_lengths)
                val_loss += loss.item()
                
                if self.metrics:
                    result, preds, targs = self.metrics.compute(output, targets)
                    total_wer += result["wer"]
                    total_cer += result["cer"]
                    
                    if num_batches == 0:
                        example_preds = preds[:2]
                        example_targets = targs[:2]
                        
                num_batches += 1
                
        avg_val_loss = val_loss / max(num_batches, 1)
        avg_wer = total_wer / max(num_batches, 1)
        avg_cer = total_cer / max(num_batches, 1)
        
        self.logger.info(
            f"Epoch {epoch} Validation | Loss: {avg_val_loss:.4f} | "
            f"WER: {avg_wer:.2%} | CER: {avg_cer:.2%}"
        )
        
        if example_preds:
            self.logger.info(f"  Pred: {example_preds[0]}")
            self.logger.info(f"  True: {example_targets[0]}")
            
        return avg_val_loss

    def fit(self) -> None:
        """Main training loop."""
        self.logger.info("=" * 60)
        self.logger.info("Starting Training")
        self.logger.info("=" * 60)
        
        # Load checkpoint if resuming
        self.load_checkpoint()
        
        if self.start_epoch > self.config.epochs:
            self.logger.info("Training already completed.")
            return

        self.logger.info(f"Epochs: {self.start_epoch} -> {self.config.epochs}")
        self.logger.info(f"Gradient Clipping: {self.gradient_clip}")
        self.logger.info(f"Accumulation Steps: {self.accumulation_steps}")
        self.logger.info("=" * 60)
        
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Periodic checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
                
            # Best model checkpoint
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, name="best_model.pt", is_best=True)

        # Final checkpoint
        self.save_checkpoint(self.config.epochs, name=self.config.output_model_path)
        self.logger.info("=" * 60)
        self.logger.info("Training Complete!")
        self.logger.info("=" * 60)
