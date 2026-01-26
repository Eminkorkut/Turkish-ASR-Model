import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import time

class SpecAugment(nn.Module):
    """
    Modelin gürültüye karşı direncini artıran spektrogram maskeleme (Data Augmentation).
    Eğitim sırasında rastgele frekans ve zaman bloklarını maskeler.
    """
    def __init__(self, freq_mask=15, time_mask=35, device='cpu'):
        super(SpecAugment, self).__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.device = device

    def forward(self, x):
        # x: (Batch, Time, Mel_Features)
        if not self.training: # Sadece eğitim sırasında uygula
            return x
            
        # Frekans maskeleme
        mask_value = x.mean()
        num_mel = x.shape[2]
        f = torch.randint(0, self.freq_mask, (1,), device=self.device).item()
        f0 = torch.randint(0, max(1, num_mel - f), (1,), device=self.device).item()
        x[:, :, f0:f0 + f] = mask_value
        
        # Zaman maskeleme
        num_time = x.shape[1]
        t = torch.randint(0, self.time_mask, (1,), device=self.device).item()
        t0 = torch.randint(0, max(1, num_time - t), (1,), device=self.device).item()
        x[:, t0:t0 + t, :] = mask_value
        
        return x

from utils.metrics import ASRMetrics

class Trainer:
    """
    Eğitim sürecini yöneten sınıf.
    Modeli başlatır, eğitir, checkpoint alır ve loglar.
    """
    def __init__(self, model, train_loader, optimizer, scheduler, device, config, logger, valid_loader=None, tokenizer=None): # Tokenizer eklendi
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger
        self.tokenizer = tokenizer # Metrikler için gerekli
        
        # Metrik Hesaplayıcı
        if tokenizer:
            self.metrics = ASRMetrics(tokenizer)
        else:
            self.metrics = None
            self.logger.warning("Tokenizer verilmediği için WER/CER hesaplanamayacak!")
        
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.scaler = GradScaler()
        self.spec_aug = SpecAugment(
            freq_mask=config.spec_augment_freq,
            time_mask=config.spec_augment_time,
            device=device
        )
        
    def train_epoch(self, epoch):
        """Bir epoch boyunca eğitim yapar."""
        self.model.train() 
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, (features, targets, input_lengths, target_lengths) in enumerate(self.train_loader):
            features, targets = features.to(self.device), targets.to(self.device)
            
            # Augmentation
            features = self.spec_aug(features)
            
            self.optimizer.zero_grad() 

            with autocast():
                output = self.model(features)
                output = output.permute(1, 0, 2)
                log_probs = torch.nn.functional.log_softmax(output, dim=2)

                current_input_lengths = input_lengths // 4 

                loss = self.criterion(log_probs, targets, current_input_lengths, target_lengths)
            
            if not torch.isnan(loss):
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                epoch_loss += loss.item()
            else:
                self.logger.warning(f"Epoch {epoch}, Batch {batch_idx}: Loss is NaN, skipping...")

            if (batch_idx + 1) % self.config.log_interval == 0:
                self.logger.info(f"Epoch [{epoch}/{self.config.epochs}] Batch [{batch_idx+1}/{len(self.train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(self.train_loader)
        duration = time.time() - start_time
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.logger.info(f"End of Epoch {epoch} | Train Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {duration:.2f}s")
        
        return avg_loss

    def validate(self, epoch):
        """Validasyon seti üzerinde performans ölçer (Loss + WER + CER)."""
        if not self.valid_loader:
            return None
            
        self.model.eval()
        val_loss = 0
        total_wer = 0
        total_cer = 0
        num_batches = 0
        
        # Örnek tahminleri loglamak için
        example_preds = []
        example_targets = []
        
        with torch.no_grad():
            for features, targets, input_lengths, target_lengths in self.valid_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                
                output = self.model(features) # (Batch, Time/4, Class)
                # Loss İçin Permute gerekli
                output_permuted = output.permute(1, 0, 2)
                log_probs = torch.nn.functional.log_softmax(output_permuted, dim=2)
                
                current_input_lengths = input_lengths // 4
                
                loss = self.criterion(log_probs, targets, current_input_lengths, target_lengths)
                val_loss += loss.item()
                
                # Metrik Hesabı (Metric class'ı (Batch, Time, Class) bekliyor, yani 'output' doğrudan kullanılabilir)
                if self.metrics:
                    metric_res, preds, targs = self.metrics.compute(output, targets)
                    total_wer += metric_res["wer"]
                    total_cer += metric_res["cer"]
                    
                    if num_batches == 0: # İlk batch'ten örnek al
                        example_preds = preds[:2]
                        example_targets = targs[:2]
                        
                num_batches += 1
                
        avg_val_loss = val_loss / num_batches
        avg_wer = total_wer / num_batches if num_batches > 0 else 0
        avg_cer = total_cer / num_batches if num_batches > 0 else 0
        
        self.logger.info(f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f} | WER: {avg_wer:.4f} | CER: {avg_cer:.4f}")
        
        # Örnekleri yazdır
        if example_preds:
            self.logger.info(f"Örnek Tahmin:  {example_preds[0]}")
            self.logger.info(f"Örnek Hedef:   {example_targets[0]}")
            
        return avg_val_loss

    def save_checkpoint(self, epoch, name=None):
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
            
        if name is None:
            name = f"checkpoint_epoch_{epoch}.pt"
            
        path = os.path.join(self.config.checkpoint_dir, name)
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model kaydedildi: {path}")

    def fit(self):
        self.logger.info(f"Eğitim başlıyor... Toplam Epoch: {self.config.epochs}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
                
            # En iyi modeli kaydet (Validasyon varsa)
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, name="best_model.pt")
        
        self.save_checkpoint(self.config.epochs, name=self.config.output_model_path)
        self.logger.info("Eğitim tamamlandı.")
