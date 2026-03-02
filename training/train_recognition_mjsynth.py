"""
Gelismis MJSynth Egitim Script'i
- Augmentation destegi
- Detayli loglar
- Ornek tahminler
- Karakter bazli accuracy
- Learning rate takibi
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import random
import time
from datetime import datetime, timedelta

from ocr_engine.recognition.vocab import Vocabulary
from ocr_engine.recognition.model import CRNN, CRNNLoss
from ocr_engine.recognition.decoder import CTCDecoder
from training.recognition_dataset import RecognitionDataset, collate_recognition
from training.augmentation_recognition import RecognitionAugmentor


def set_seed(seed=42):
    """Reproducibility icin seed ayarla"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(preds, targets):
    """Detayli metrikler hesapla"""
    word_correct = 0
    char_correct = 0
    char_total = 0
    
    for pred, target in zip(preds, targets):
        # Kelime accuracy
        if pred == target:
            word_correct += 1
        
        # Karakter accuracy (edit distance yerine basit karsilastirma)
        min_len = min(len(pred), len(target))
        for i in range(min_len):
            if pred[i] == target[i]:
                char_correct += 1
        char_total += max(len(pred), len(target))
    
    return {
        'word_acc': word_correct / len(preds) if preds else 0,
        'char_acc': char_correct / char_total if char_total > 0 else 0,
        'word_correct': word_correct,
        'total': len(preds)
    }


def format_time(seconds):
    """Saniyeyi okunabilir formata cevir"""
    return str(timedelta(seconds=int(seconds)))


class AdvancedTrainer:
    """Gelismis egitim sinifi"""
    
    def __init__(self, config, vocab, device='cuda', use_amp=True):
        self.config = config
        self.vocab = vocab
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Mixed Precision Training icin scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Model
        model_cfg = config.get('recognition', {}).get('model', {})
        self.model = CRNN(
            num_classes=vocab.size,
            input_channels=1,
            hidden_size=model_cfg.get('hidden_size', 256),
            num_layers=model_cfg.get('num_layers', 2),
            dropout=model_cfg.get('dropout', 0.1),
            encoder_type=model_cfg.get('encoder_type', 'vgg')
        ).to(self.device)
        
        # Loss & Decoder
        self.criterion = CRNNLoss(blank_idx=vocab.blank_idx)
        self.decoder = CTCDecoder(vocab)
        
        # Optimizer
        train_cfg = config.get('training', {}).get('recognition', {})
        self.lr = train_cfg.get('learning_rate', 0.001)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Metrikler
        self.best_val_acc = 0
        self.epoch = 0
        self.start_epoch = 0       # resume edilince load_checkpoint tarafindan guncellenir
        self.global_step = 0
        self.scheduler = None      # train() icinde olusturulacak
        self.plateau_scheduler = None  # val mevcut oldugunda ReduceLROnPlateau
        self._pending_plateau_state = None  # resume sonrasi scheduler yaratilinca uygulanir
        
        # Model parametreleri
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n[MODEL INFO]")
        print(f"  Toplam parametre: {total_params:,}")
        print(f"  Egitilir parametre: {trainable_params:,}")
        print(f"  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  Mixed Precision (AMP): {'Aktif' if self.use_amp else 'Kapali'}")
    
    def train_epoch(self, train_loader, epoch, total_epochs):
        """Bir epoch egit"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        sample_preds = []  # Ornek tahminler icin
        
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", 
                    ncols=120, leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            label_lengths = batch['label_lengths']
            texts = batch['texts']
            
            batch_size = images.size(0)
            input_length = self.model.get_sequence_length(images.size(3))
            input_lengths = torch.full((batch_size,), input_length, dtype=torch.long)
            
            # Forward with Mixed Precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    log_probs = self.model(images)
                    
                    # CTC Loss
                    targets = []
                    for i in range(batch_size):
                        targets.extend(labels[i, :label_lengths[i]].tolist())
                    targets = torch.tensor(targets, dtype=torch.long)
                    
                    loss = self.criterion(log_probs, targets, input_lengths, label_lengths)
                
                # Backward with scaler
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                log_probs = self.model(images)
                
                # CTC Loss
                targets = []
                for i in range(batch_size):
                    targets.extend(labels[i, :label_lengths[i]].tolist())
                targets = torch.tensor(targets, dtype=torch.long)
                
                loss = self.criterion(log_probs, targets, input_lengths, label_lengths)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            self.global_step += 1
            total_loss += loss.item()
            
            # Decode & metrikler (her 50 batch'te bir)
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    preds = self.decoder.decode_greedy(log_probs)
                    all_preds.extend(preds)
                    all_targets.extend(texts)
                    
                    # Ornek tahminler kaydet (ilk 3)
                    if batch_idx % 200 == 0 and len(sample_preds) < 10:
                        for p, t in zip(preds[:3], texts[:3]):
                            sample_preds.append((t, p))
            
            # Progress bar guncelle
            current_lr = self.optimizer.param_groups[0]['lr']
            metrics = calculate_metrics(all_preds, all_targets) if all_preds else {'word_acc': 0, 'char_acc': 0}
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'word': f"{metrics['word_acc']*100:.1f}%",
                'char': f"{metrics['char_acc']*100:.1f}%",
                'lr': f"{current_lr:.2e}"
            })
        
        elapsed = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        final_metrics = calculate_metrics(all_preds, all_targets) if all_preds else {'word_acc': 0, 'char_acc': 0}
        
        return {
            'loss': avg_loss,
            'word_acc': final_metrics['word_acc'],
            'char_acc': final_metrics['char_acc'],
            'elapsed': elapsed,
            'samples': sample_preds
        }
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validation"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        sample_preds = []
        
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", ncols=100)):
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            label_lengths = batch['label_lengths']
            texts = batch['texts']
            
            batch_size = images.size(0)
            input_length = self.model.get_sequence_length(images.size(3))
            input_lengths = torch.full((batch_size,), input_length, dtype=torch.long)
            
            log_probs = self.model(images)
            
            # Loss
            targets = []
            for i in range(batch_size):
                targets.extend(labels[i, :label_lengths[i]].tolist())
            targets = torch.tensor(targets, dtype=torch.long)
            
            loss = self.criterion(log_probs, targets, input_lengths, label_lengths)
            total_loss += loss.item()
            
            # Decode
            preds = self.decoder.decode_greedy(log_probs)
            all_preds.extend(preds)
            all_targets.extend(texts)
            
            # Ornek tahminler
            if batch_idx == 0:
                for p, t in zip(preds[:5], texts[:5]):
                    sample_preds.append((t, p))
        
        metrics = calculate_metrics(all_preds, all_targets)
        
        return {
            'loss': total_loss / len(val_loader),
            'word_acc': metrics['word_acc'],
            'char_acc': metrics['char_acc'],
            'samples': sample_preds
        }
    
    def train(self, train_loader, val_loader, epochs, save_dir):
        """Ana egitim dongusu"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Scheduler secimi
        if val_loader:
            # Validation varsa: ReduceLROnPlateau (val_loss izler, per-step yok)
            self.scheduler = None
            self.plateau_scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5,
                patience=3, min_lr=1e-7
            )
            print("[LR] ReduceLROnPlateau aktif (val_loss izleniyor, patience=3)")
            # Resume: plateau_scheduler state'i geri yukle
            if self._pending_plateau_state is not None:
                self.plateau_scheduler.load_state_dict(self._pending_plateau_state)
                self._pending_plateau_state = None
                print("[LR] ReduceLROnPlateau state checkpoint'ten geri yuklendi")
        else:
            # Validation yoksa: OneCycleLR
            # Resume durumunda kalan step sayisini hesapla
            remaining_epochs = epochs - self.start_epoch
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.lr,
                epochs=remaining_epochs if remaining_epochs > 0 else epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,
                anneal_strategy='cos'
            )
            self.plateau_scheduler = None
            print("[LR] OneCycleLR aktif")
        
        print("\n" + "="*70)
        print(f"EGITIM BASLIYOR")
        print("="*70)
        print(f"  Train ornekleri: {len(train_loader.dataset):,}")
        if val_loader:
            print(f"  Val ornekleri: {len(val_loader.dataset):,}")
        print(f"  Batch boyutu: {train_loader.batch_size}")
        print(f"  Epoch sayisi: {epochs}")
        print(f"  Batch/epoch: {len(train_loader)}")
        print(f"  Baslangic LR: {self.lr}")
        print("="*70 + "\n")
        
        training_start = time.time()

        for epoch in range(self.start_epoch + 1, epochs + 1):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch, epochs)
            
            # Epoch sonucu
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch}/{epochs} TAMAMLANDI")
            print(f"{'='*70}")
            print(f"  [TRAIN] Loss: {train_metrics['loss']:.4f} | "
                  f"Word Acc: {train_metrics['word_acc']*100:.2f}% | "
                  f"Char Acc: {train_metrics['char_acc']*100:.2f}%")
            print(f"  [SURE]  {format_time(train_metrics['elapsed'])}")
            
            # Ornek tahminler
            if train_metrics['samples']:
                print(f"\n  [ORNEKLER - Train]")
                for i, (target, pred) in enumerate(train_metrics['samples'][:5]):
                    match = "[OK]" if target == pred else "[X]"
                    print(f"    {match} '{target}' -> '{pred}'")
            
            # Validation
            if val_loader and epoch % 2 == 0:  # Her 2 epoch'ta bir
                val_metrics = self.validate(val_loader)
                print(f"\n  [VAL]   Loss: {val_metrics['loss']:.4f} | "
                      f"Word Acc: {val_metrics['word_acc']*100:.2f}% | "
                      f"Char Acc: {val_metrics['char_acc']*100:.2f}%")
                
                # Ornek tahminler
                if val_metrics['samples']:
                    print(f"\n  [ORNEKLER - Val]")
                    for target, pred in val_metrics['samples'][:5]:
                        match = "[OK]" if target == pred else "[X]"
                        print(f"    {match} '{target}' -> '{pred}'")
                
                # En iyi modeli kaydet
                if val_metrics['word_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['word_acc']
                    self.save_checkpoint(save_dir / 'best_model.pth')
                    print(f"\n  [KAYIT] Yeni en iyi model! Val Acc: {self.best_val_acc*100:.2f}%")

                # ReduceLROnPlateau adimi
                if self.plateau_scheduler is not None:
                    self.plateau_scheduler.step(val_metrics['loss'])
            
            # Her epoch checkpoint kaydet (uzun suruyor)
            self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"\n  [KAYIT] Checkpoint kaydedildi: epoch_{epoch}.pth")
            
            # Kalan sure tahmini
            elapsed_total = time.time() - training_start
            avg_epoch_time = elapsed_total / epoch
            remaining_epochs = epochs - epoch
            eta = avg_epoch_time * remaining_epochs
            print(f"\n  [ETA]   Tahmini kalan sure: {format_time(eta)}")
            print(f"{'='*70}\n")
        
        # Son modeli kaydet
        self.save_checkpoint(save_dir / 'final_model.pth')
        
        total_time = time.time() - training_start
        print(f"\n{'='*70}")
        print(f"EGITIM TAMAMLANDI!")
        print(f"{'='*70}")
        print(f"  Toplam sure: {format_time(total_time)}")
        print(f"  En iyi Val Acc: {self.best_val_acc*100:.2f}%")
        print(f"  Modeller: {save_dir}")
        print(f"{'='*70}\n")
    
    def save_checkpoint(self, path):
        """Checkpoint kaydet"""
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'plateau_scheduler_state_dict': self.plateau_scheduler.state_dict() if self.plateau_scheduler else None,
            'best_val_acc': self.best_val_acc,
            'global_step': self.global_step,
            'vocab_size': self.vocab.size
        }, path)
    
    def load_checkpoint(self, path):
        """Checkpoint yukle"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.start_epoch = checkpoint['epoch']   # egitim dongusu buradan devam eder
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.global_step = checkpoint.get('global_step', 0)
        # OneCycleLR state (varsa)
        if checkpoint.get('scheduler_state_dict') and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # ReduceLROnPlateau state: train() scheduler'i yaratinca uygulanacak
        self._pending_plateau_state = checkpoint.get('plateau_scheduler_state_dict')
        print(f"[CHECKPOINT] Yuklendi: epoch {self.epoch}, best_acc {self.best_val_acc*100:.2f}%")
        print(f"[CHECKPOINT] Egitim epoch {self.start_epoch + 1}'den devam edecek")


def main():
    parser = argparse.ArgumentParser(description='Gelismis MJSynth Egitimi')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--val_split', type=float, default=0.05,
                        help='Validation icin ayrilacak oran (0.05 = %%5)')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=96,
                        help='Batch boyutu (AMP ile arttirildi)')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Augmentation aktif et')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint yolu (ornek: checkpoints/1M_augmented/checkpoint_epoch_3.pth)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/advanced')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # -------------------------------------------------------------- #
    # CPU Affinity: P-cekirdekler (0-7) + 6 E-cekirdek (8-13)       #
    # Logical 14 ve 15 (2 E-cekirdek) kullaniciya birakildi          #
    # -------------------------------------------------------------- #
    try:
        import psutil
        _use_cpus = list(range(14))  # logical 0-13
        psutil.Process(os.getpid()).cpu_affinity(_use_cpus)
        print("[CPU] Affinity: 0-13 aktif (P-core 0-7 + E-core 8-13) | 14,15 serbest")
    except Exception as e:
        print(f"[CPU] Affinity ayarlanamadi: {e}")

    # Ana process torch thread sayisi (val/metrik icin)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    # CUDA optimizasyonlari
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # En hizli algoritmayi sec
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 hizlandir
        torch.backends.cudnn.allow_tf32 = True
        print("[CUDA] Optimizasyonlar aktif: cudnn.benchmark=True, tf32=True")
    
    # Seed
    set_seed(args.seed)
    
    # Banner
    print("\n" + "="*70)
    print("   OCR RECOGNITION MODEL - GELISMIS EGITIM")
    print("   " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    # Config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Vocabulary
    vocab = Vocabulary()
    print(f"\n[VOCABULARY] Boyut: {vocab.size} karakter")
    
    # Augmentor
    augmentor = None
    if args.augment:
        augmentor = RecognitionAugmentor()
        print("[AUGMENTATION] Aktif")
    else:
        print("[AUGMENTATION] Kapali")
    
    # Dataset
    rec_cfg = config.get('recognition', {}).get('model', {})
    
    print(f"\n[DATASET] Yukleniyor: {args.train_json}")
    full_dataset = RecognitionDataset(
        data_dir=args.data_root,
        annotation_file=args.train_json,
        vocab=vocab,
        image_height=rec_cfg.get('input_height', 32),
        image_width=rec_cfg.get('input_width', 256),
        augmentor=augmentor,
        synthetic_ratio=0.0
    )
    
    # Train/Val split
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"[SPLIT] Train: {train_size:,} | Val: {val_size:,}")
    
    # DataLoaders - Optimize edilmis
    # 6 worker x ~2 thread = 12 thread (14 mevcut threadden)
    # Ana process 2 thread kullanir, 14-15 kullaniciya kalir
    num_workers = 6 if torch.cuda.is_available() else 0
    pf = 4 if num_workers > 0 else None  # 4 batch onceden hazirla

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_recognition,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=pf,
        drop_last=True  # son kucuk batch'i at (BatchNorm + AMP icin stabil)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # val'da gradient yok, 2x batch guvende
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_recognition,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=pf
    )

    print(f"[DATALOADER] num_workers: {num_workers}, prefetch_factor: {pf}, drop_last: True")
    
    # Trainer with AMP
    trainer = AdvancedTrainer(config, vocab, device=args.device, use_amp=True)
    
    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Egit
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
