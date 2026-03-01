"""
CRNN Recognition Model Egitim Scripti
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Proje kokunu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_engine.recognition.model import CRNN, CRNNLoss, build_crnn
from ocr_engine.recognition.vocab import Vocabulary
from ocr_engine.recognition.decoder import CTCDecoder
from training.recognition_dataset import RecognitionDataset, collate_recognition, create_recognition_dataloader
from training.augmentation_recognition import RecognitionAugmentor


class RecognitionTrainer:
    """Recognition model egitici"""
    
    def __init__(
        self,
        config: dict,
        vocab: Vocabulary,
        device: str = 'cuda'
    ):
        self.config = config
        self.vocab = vocab
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Model
        self.model = self._build_model()
        
        # Loss
        self.criterion = CRNNLoss(blank_idx=vocab.blank_idx)
        
        # Decoder (validation icin)
        self.decoder = CTCDecoder(vocab)
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        
        # Scheduler
        self.scheduler = None  # train() icinde olusturulacak
        self.plateau_scheduler = None  # val mevcut oldugunda ReduceLROnPlateau
        
        # Metrikler
        self.best_accuracy = 0.0
        self.epoch = 0
    
    def _build_model(self) -> CRNN:
        """Model olustur"""
        model_cfg = self.config.get('recognition', {}).get('model', {})
        
        model = CRNN(
            num_classes=self.vocab.num_classes,
            input_channels=1,
            hidden_size=model_cfg.get('hidden_size', 256),
            num_layers=model_cfg.get('num_layers', 2),
            dropout=model_cfg.get('dropout', 0.1),
            encoder_type=model_cfg.get('encoder_type', 'vgg')
        )
        
        return model.to(self.device)
    
    def _build_optimizer(self):
        """Optimizer olustur"""
        train_cfg = self.config.get('training', {}).get('recognition', {})
        lr = train_cfg.get('learning_rate', 0.001)
        
        return AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
    
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Bir epoch egit"""
        self.model.train()
        
        total_loss = 0
        num_correct = 0
        num_total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            # Veriyi GPU'ya tasi
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            label_lengths = batch['label_lengths']
            texts = batch['texts']
            
            # Input uzunluklari (CRNN cikis sequence uzunlugu)
            batch_size = images.size(0)
            input_width = images.size(3)
            input_length = self.model.get_sequence_length(input_width)
            input_lengths = torch.full((batch_size,), input_length, dtype=torch.long)
            
            # Forward
            self.optimizer.zero_grad()
            log_probs = self.model(images)  # [T, B, C]
            
            # CTC loss icin labels'i flatten
            targets = []
            for i in range(batch_size):
                targets.extend(labels[i, :label_lengths[i]].tolist())
            targets = torch.tensor(targets, dtype=torch.long)
            
            # Loss
            loss = self.criterion(
                log_probs,
                targets,
                input_lengths,
                label_lengths
            )
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Metrikler
            total_loss += loss.item()
            
            # Accuracy (greedy decode)
            with torch.no_grad():
                preds = self.decoder.decode_greedy(log_probs)
                for pred, target in zip(preds, texts):
                    if pred == target:
                        num_correct += 1
                    num_total += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{num_correct / max(num_total, 1) * 100:.1f}%"
            })
        
        num_batches = len(dataloader)
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': num_correct / max(num_total, 1)
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Validation"""
        self.model.eval()
        
        total_loss = 0
        num_correct = 0
        num_char_correct = 0
        num_total = 0
        num_chars = 0
        
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            label_lengths = batch['label_lengths']
            texts = batch['texts']
            
            batch_size = images.size(0)
            input_width = images.size(3)
            input_length = self.model.get_sequence_length(input_width)
            input_lengths = torch.full((batch_size,), input_length, dtype=torch.long)
            
            log_probs = self.model(images)
            
            # Loss
            targets = []
            for i in range(batch_size):
                targets.extend(labels[i, :label_lengths[i]].tolist())
            targets = torch.tensor(targets, dtype=torch.long)
            
            loss = self.criterion(log_probs, targets, input_lengths, label_lengths)
            total_loss += loss.item()
            
            # Accuracy
            preds = self.decoder.decode_greedy(log_probs)
            for pred, target in zip(preds, texts):
                if pred == target:
                    num_correct += 1
                num_total += 1
                
                # Karakter bazli accuracy
                for c1, c2 in zip(pred, target):
                    if c1 == c2:
                        num_char_correct += 1
                num_chars += max(len(pred), len(target))
        
        return {
            'val_loss': total_loss / len(dataloader),
            'accuracy': num_correct / max(num_total, 1),
            'char_accuracy': num_char_correct / max(num_chars, 1)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 100,
        save_dir: str = 'checkpoints',
        freeze_encoder_epochs: int = 0,
    ):
        """Egitim dongusu"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        train_cfg = self.config.get('training', {}).get('recognition', {})
        save_interval = train_cfg.get('save_interval', 10)
        val_interval = train_cfg.get('val_interval', 5)
        
        # Scheduler secimi
        if val_loader:
            # Validation varsa: ReduceLROnPlateau (val_loss izler, per-step yok)
            self.scheduler = None
            self.plateau_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=True,
            )
            print("[LR] ReduceLROnPlateau aktif (val_loss izleniyor, patience=5)")
        else:
            # Validation yoksa: OneCycleLR (per-step)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=train_cfg.get('learning_rate', 0.001),
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1
            )
            self.plateau_scheduler = None
            print("[LR] OneCycleLR aktif")
        
        for epoch in range(epochs):
            self.epoch = epoch + 1

            # --- Encoder dondurma / cozme (2-asamali fine-tune icin) ---
            if freeze_encoder_epochs > 0:
                if epoch == 0:
                    for p in self.model.encoder.parameters():
                        p.requires_grad = False
                    print(f"[FREEZE] CNN encoder donduruldu ({freeze_encoder_epochs} epoch)")
                elif epoch == freeze_encoder_epochs:
                    for p in self.model.encoder.parameters():
                        p.requires_grad = True
                    print(f"[UNFREEZE] Epoch {self.epoch}: CNN encoder aktiflestirildi")
            
            # Egit
            train_metrics = self.train_epoch(train_loader)
            print(f"\nEpoch {self.epoch} - Loss: {train_metrics['loss']:.4f}, "
                  f"Accuracy: {train_metrics['accuracy']*100:.1f}%")
            
            # Validation
            if val_loader and (self.epoch % val_interval == 0):
                val_metrics = self.validate(val_loader)
                print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"Accuracy: {val_metrics['accuracy']*100:.1f}%, "
                      f"Char Acc: {val_metrics['char_accuracy']*100:.1f}%")

                # ReduceLROnPlateau adimi
                if self.plateau_scheduler is not None:
                    self.plateau_scheduler.step(val_metrics['val_loss'])
                
                # En iyi modeli kaydet
                if val_metrics['accuracy'] > self.best_accuracy:
                    self.best_accuracy = val_metrics['accuracy']
                    self.save_checkpoint(save_dir / 'best_recognition.pth')
                    print("Best model saved!")
            
            # Periyodik kayit
            if self.epoch % save_interval == 0:
                self.save_checkpoint(save_dir / f'recognition_epoch_{self.epoch}.pth')
        
        # Son modeli kaydet
        self.save_checkpoint(save_dir / 'recognition_final.pth')
        print("\nEgitim tamamlandi!")
    
    def save_checkpoint(self, path: str):
        """Checkpoint kaydet"""
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'plateau_scheduler_state_dict': self.plateau_scheduler.state_dict() if self.plateau_scheduler else None,
            'best_accuracy': self.best_accuracy,
            'vocab': {
                'chars': self.vocab.chars,
                'include_blank': self.vocab.include_blank,
                'include_unk': self.vocab.include_unk
            }
        }, path)
    
    def load_pretrained(self, path: str) -> None:
        """
        Stage-1 pretrained agirliklarini yukle.

        Sadece model agirliklarini yukler; optimizer, scheduler ve epoch
        sifirlanir. 2-asamali egitim icin:
          Stage 1: train_recognition_mjsynth.py  -> best_model.pth
          Stage 2: train_recognition.py --pretrain best_model.pth
                                         --freeze_encoder_epochs 5
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"Pretrained agirliklar yuklendi: {path}")
        if missing:
            print(f"  Eksik katmanlar: {len(missing)}")
        if unexpected:
            print(f"  Beklenmeyen katmanlar: {len(unexpected)}")
        # Epoch ve metrikler sifirlanir (yeni fine-tune baslangici)
        self.epoch = 0
        self.best_accuracy = 0.0

    def load_checkpoint(self, path: str):
        """Checkpoint yukle"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        if checkpoint.get('scheduler_state_dict') and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('plateau_scheduler_state_dict') and self.plateau_scheduler is not None:
            self.plateau_scheduler.load_state_dict(checkpoint['plateau_scheduler_state_dict'])
        print(f"Checkpoint yuklendi: epoch {self.epoch}, best_acc {self.best_accuracy*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='CRNN Recognition Egitimi')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config dosyasi')
    parser.add_argument('--data_dir', type=str, default=None, help='Veri klasoru')
    parser.add_argument('--train_ann', type=str, default=None, help='Train annotation dosyasi')
    parser.add_argument('--val_ann', type=str, default=None, help='Validation annotation dosyasi')
    parser.add_argument('--epochs', type=int, default=100, help='Epoch sayisi')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch boyutu')
    parser.add_argument('--synthetic_ratio', type=float, default=1.0, 
                        help='Sentetik veri orani (0-1, 1=tamamen sentetik)')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint yolu')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Stage-1 pretrained agirlik (sadece model yukle, optimizer sifirla)')
    parser.add_argument('--freeze_encoder_epochs', type=int, default=0,
                        help='Bu kadar epoch CNN encoder dondur (2-asamali fine-tune icin)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Kayit klasoru')
    parser.add_argument('--device', type=str, default='cuda', help='Cihaz')
    
    args = parser.parse_args()
    
    # Config yukle
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Vocabulary
    vocab = Vocabulary()
    print(f"Vocabulary size: {vocab.size}")
    
    # Dataset
    rec_cfg = config.get('recognition', {}).get('model', {})
    
    augmentor = RecognitionAugmentor() if config.get('training', {}).get('augmentation', {}).get('enabled', True) else None
    
    train_dataset = RecognitionDataset(
        data_dir=args.data_dir,
        annotation_file=args.train_ann,
        vocab=vocab,
        image_height=rec_cfg.get('input_height', 32),
        image_width=rec_cfg.get('input_width', 256),
        augmentor=augmentor,
        synthetic_ratio=args.synthetic_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.get('general', {}).get('num_workers', 4),
        collate_fn=collate_recognition,
        pin_memory=True
    )
    
    val_loader = None
    if args.val_ann and args.data_dir:
        val_dataset = RecognitionDataset(
            data_dir=args.data_dir,
            annotation_file=args.val_ann,
            vocab=vocab,
            image_height=rec_cfg.get('input_height', 32),
            image_width=rec_cfg.get('input_width', 256),
            augmentor=None,
            synthetic_ratio=0.0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_recognition
        )
    
    # Trainer
    trainer = RecognitionTrainer(config, vocab=vocab, device=args.device)
    
    # Resume veya pretrain
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif args.pretrain:
        trainer.load_pretrained(args.pretrain)
    
    # Egit
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
    )


if __name__ == '__main__':
    main()
