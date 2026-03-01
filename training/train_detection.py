"""
DBNet Detection Model Egitim Scripti

Ozellikler:
- Mixed Precision Training (AMP)
- Learning rate warmup + cosine decay
- Validation metrics (Precision/Recall/F1)
- OneCycleLR / LambdaLR scheduler destegi
- Per-epoch checkpoint kayit
- Detayli loglama
- SynthText ve ICDAR veri seti destegi
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, OneCycleLR
from torch.amp import GradScaler
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import numpy as np
import cv2
import time
from datetime import datetime, timedelta

# Proje kokunu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_engine.detection.model import DBNet, DBLoss, build_dbnet
from ocr_engine.detection.postprocess import DBPostProcessor
from training.detection_dataset import DetectionDataset, SynthTextDataset, ICDARDataset
from training.augmentation_detection import Augmentor, DetectionAugmentor


class DetectionMetrics:
    """Detection metrikleri hesaplayici"""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives

    def update(self, pred_boxes: list, gt_boxes: list):
        """Bir gorsel icin metrikleri guncelle"""
        if len(gt_boxes) == 0:
            self.fp += len(pred_boxes)
            return

        if len(pred_boxes) == 0:
            self.fn += len(gt_boxes)
            return

        # IoU matrix hesapla
        matched_gt = set()

        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                iou = self._compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold:
                self.tp += 1
                matched_gt.add(best_gt_idx)
            else:
                self.fp += 1

        self.fn += len(gt_boxes) - len(matched_gt)

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Iki polygon arasindaki IoU"""
        try:
            from shapely.geometry import Polygon

            poly1 = Polygon(box1)
            poly2 = Polygon(box2)

            if not poly1.is_valid:
                poly1 = poly1.buffer(0)
            if not poly2.is_valid:
                poly2 = poly2.buffer(0)

            intersection = poly1.intersection(poly2).area
            union = poly1.union(poly2).area

            if union == 0:
                return 0

            return intersection / union
        except Exception:
            return 0

    def compute(self) -> dict:
        """Final metrikleri hesapla"""
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn
        }


class DetectionTrainer:
    """Detection model egitici"""
    
    def __init__(
        self,
        config: dict,
        device: str = 'cuda',
        use_amp: bool = True
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and torch.cuda.is_available()

        print(f"\n{'='*60}")
        print(f"DETECTION TRAINER BASLATILIYOR")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"AMP (Mixed Precision): {self.use_amp}")

        # CUDA optimizasyonlari
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Model
        self.model = self._build_model()

        # Loss
        self.criterion = DBLoss()

        # Optimizer
        self.optimizer = self._build_optimizer()

        # AMP Scaler
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Postprocessor (validation icin)
        self.postprocessor = DBPostProcessor(
            threshold=0.3,
            box_threshold=0.5,
            max_candidates=1000,
            unclip_ratio=1.5
        )

        # Scheduler: T_max placeholder — rebuilt with actual epochs inside train().
        # load_checkpoint() may store pending state here before train() is called.
        train_cfg = self.config.get('training', {}).get('detection', {})
        self.scheduler = self._build_scheduler_cosine(train_cfg.get('epochs', 100))
        self._pending_scheduler_state: Optional[dict] = None

        # Egitim metrikleri
        self.best_f1 = 0
        self.best_loss = float('inf')
        self.epoch = 0
        self.global_step = 0

        print(f"{'='*60}\n")
    
    def _build_model(self) -> DBNet:
        """Model olustur"""
        model_cfg = self.config.get('detection', {}).get('model', {})

        model = DBNet(
            backbone=model_cfg.get('backbone', 'resnet18'),
            pretrained=model_cfg.get('pretrained', True)
        )

        # Model parametrelerini say
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parametreleri: {total_params:,} (trainable: {trainable_params:,})")

        return model.to(self.device)

    def _build_optimizer(self):
        """Optimizer olustur"""
        train_cfg = self.config.get('training', {}).get('detection', {})
        lr = train_cfg.get('learning_rate', 0.001)
        weight_decay = train_cfg.get('weight_decay', 1e-4)

        # AdamW daha iyi regularization
        return AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

    def _build_scheduler_cosine(self, epochs: int):
        """CosineAnnealingLR scheduler olustur (epoch-level, checkpoint resume icin)"""
        return CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6
        )

    def train_epoch(self, dataloader: DataLoader, scheduler=None) -> dict:
        """Bir epoch egit (AMP destekli)"""
        self.model.train()

        total_loss = 0
        prob_loss_sum = 0
        binary_loss_sum = 0
        thresh_loss_sum = 0
        num_batches = 0

        start_time = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Veriyi GPU'ya tasi
            images = batch['image'].to(self.device, non_blocking=True)
            gt_prob = batch['prob_map'].to(self.device, non_blocking=True)
            gt_thresh = batch['thresh_map'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)

            # Forward + Backward
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images, return_maps=True)
                    losses = self.criterion(outputs, gt_prob, gt_thresh, mask)
                    loss = losses['total_loss']

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, return_maps=True)
                losses = self.criterion(outputs, gt_prob, gt_thresh, mask)
                loss = losses['total_loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            # Per-step scheduler update (OneCycleLR / LambdaLR)
            if scheduler is not None:
                scheduler.step()

            # Metrikleri guncelle
            total_loss += loss.item()
            prob_loss_sum += losses['prob_loss'].item()
            binary_loss_sum += losses['binary_loss'].item()
            thresh_loss_sum += losses['thresh_loss'].item()
            num_batches += 1
            self.global_step += 1

            # Progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'prob': f"{losses['prob_loss'].item():.4f}",
                'lr': f"{current_lr:.2e}"
            })

        elapsed = time.time() - start_time

        return {
            'total_loss': total_loss / num_batches,
            'prob_loss': prob_loss_sum / num_batches,
            'binary_loss': binary_loss_sum / num_batches,
            'thresh_loss': thresh_loss_sum / num_batches,
            'elapsed_time': elapsed,
            'samples_per_sec': len(dataloader.dataset) / elapsed
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader, compute_metrics: bool = True) -> dict:
        """Validation with P/R/F1 metrics"""
        self.model.eval()

        total_loss = 0
        num_batches = 0
        metrics = DetectionMetrics(iou_threshold=0.5)

        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(self.device)
            gt_prob = batch['prob_map'].to(self.device)
            gt_thresh = batch['thresh_map'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images, return_maps=True)
                    losses = self.criterion(outputs, gt_prob, gt_thresh, mask)
            else:
                outputs = self.model(images, return_maps=True)
                losses = self.criterion(outputs, gt_prob, gt_thresh, mask)

            total_loss += losses['total_loss'].item()
            num_batches += 1

            # Detection metrics
            if compute_metrics:
                prob_maps = outputs['prob_map'].cpu().numpy()

                for i in range(prob_maps.shape[0]):
                    # Prediction boxes
                    prob_map = prob_maps[i, 0]
                    pred_boxes, _ = self.postprocessor(
                        prob_map,
                        (images.shape[2], images.shape[3])
                    )

                    # GT boxes (prob_map'ten cikar)
                    gt_prob_np = gt_prob[i, 0].cpu().numpy()
                    gt_boxes = self._extract_boxes_from_map(gt_prob_np)

                    metrics.update(pred_boxes, gt_boxes)

        result = {
            'val_loss': total_loss / num_batches
        }

        if compute_metrics:
            metric_values = metrics.compute()
            result.update(metric_values)

        return result

    def _extract_boxes_from_map(self, prob_map: np.ndarray, thresh: float = 0.5) -> list:
        """Probability map'ten boxes cikar"""
        binary = (prob_map > thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            boxes.append(box.astype(np.float32))

        return boxes
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 100,
        save_dir: str = 'checkpoints',
        warmup_epochs: float = 1.0
    ):
        """Egitim dongusu"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Rebuild epoch-level cosine scheduler with the actual number of training epochs.
        # If a checkpoint was loaded, restore its LR state so resume continues correctly.
        self.scheduler = self._build_scheduler_cosine(epochs)
        if self._pending_scheduler_state is not None:
            self.scheduler.load_state_dict(self._pending_scheduler_state)
            self._pending_scheduler_state = None

        train_cfg = self.config.get('training', {}).get('detection', {})
        save_interval = train_cfg.get('save_interval', 10)
        val_interval = train_cfg.get('val_interval', 5)

        print(f"\n{'='*60}")
        print(f"EGITIM BASLIYOR")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset) if val_loader else 0}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Save directory: {save_dir}")
        print(f"{'='*60}\n")

        training_start = time.time()

        for epoch in range(epochs):
            self.epoch = epoch + 1
            epoch_start = time.time()

            print(f"\n{'='*60}")
            print(f"EPOCH {self.epoch}/{epochs}")
            print(f"{'='*60}")

            # Egit
            train_metrics = self.train_epoch(train_loader)

            print(f"\n[TRAIN] Loss: {train_metrics['total_loss']:.4f}")
            print(f"  - Prob Loss: {train_metrics['prob_loss']:.4f}")
            print(f"  - Binary Loss: {train_metrics['binary_loss']:.4f}")
            print(f"  - Thresh Loss: {train_metrics['thresh_loss']:.4f}")
            print(f"  - Speed: {train_metrics['samples_per_sec']:.1f} samples/sec")

            # Validation
            if val_loader and (self.epoch % val_interval == 0):
                val_metrics = self.validate(val_loader)

                print(f"\n[VAL] Loss: {val_metrics['val_loss']:.4f}")
                print(f"  - Precision: {val_metrics['precision']:.4f}")
                print(f"  - Recall: {val_metrics['recall']:.4f}")
                print(f"  - F1: {val_metrics['f1']:.4f}")

                # En iyi modeli F1'e gore kaydet
                if val_metrics['f1'] > self.best_f1:
                    self.best_f1 = val_metrics['f1']
                    self.save_checkpoint(save_dir / 'best_detection.pth')
                    print(f"  [BEST MODEL] F1: {self.best_f1:.4f}")

                # Loss'a gore de kaydet
                if val_metrics['val_loss'] < self.best_loss:
                    self.best_loss = val_metrics['val_loss']
                    self.save_checkpoint(save_dir / 'best_loss_detection.pth')

            # Periyodik kayit
            if self.epoch % save_interval == 0:
                self.save_checkpoint(save_dir / f'detection_epoch_{self.epoch}.pth')

            # Epoch-level cosine scheduler step
            self.scheduler.step()

            # ETA hesapla
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - training_start
            remaining_epochs = epochs - self.epoch
            eta = timedelta(seconds=int(remaining_epochs * epoch_time))

            print(f"\n[ZAMAN] Epoch: {epoch_time:.0f}s | Toplam: {timedelta(seconds=int(elapsed))} | ETA: {eta}")

        # Son modeli kaydet
        self.save_checkpoint(save_dir / 'detection_final.pth')

        total_time = time.time() - training_start
        print(f"\n{'='*60}")
        print(f"EGITIM TAMAMLANDI!")
        print(f"{'='*60}")
        print(f"Toplam sure: {timedelta(seconds=int(total_time))}")
        print(f"En iyi F1: {self.best_f1:.4f}")
        print(f"En iyi Loss: {self.best_loss:.4f}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, path: str):
        """Checkpoint kaydet"""
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'best_loss': self.best_loss,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Checkpoint yukle"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Scheduler train() icinde dogru T_max ile yeniden olusturulur;
        # state'i pending olarak sakla, train() onu uygular.
        self._pending_scheduler_state = checkpoint.get('scheduler_state_dict')

        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_f1 = checkpoint.get('best_f1', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        print(f"Checkpoint yuklendi: epoch {self.epoch}, best_f1 {self.best_f1:.4f}, best_loss {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='DBNet Detection Egitimi')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config dosyasi')
    parser.add_argument('--data_dir', type=str, required=True, help='Veri klasoru')
    parser.add_argument('--train_ann', type=str, required=True, help='Train annotation dosyasi')
    parser.add_argument('--val_ann', type=str, default=None, help='Validation annotation dosyasi')
    parser.add_argument('--epochs', type=int, default=100, help='Epoch sayisi')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch boyutu')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint yolu')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Kayit klasoru')
    parser.add_argument('--device', type=str, default='cuda', help='Cihaz')
    parser.add_argument('--no_amp', action='store_true', help='AMP devre disi')
    parser.add_argument('--warmup', type=float, default=1.0, help='Warmup epoch sayisi')

    args = parser.parse_args()

    # Config yukle
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Dataset
    det_cfg = config.get('detection', {})
    input_size = tuple(det_cfg.get('input_size', [640, 640]))

    augmentor = Augmentor() if config.get('training', {}).get('augmentation', {}).get('enabled', True) else None

    train_dataset = DetectionDataset(
        data_dir=args.data_dir,
        annotation_file=args.train_ann,
        input_size=input_size,
        augmentor=augmentor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.get('general', {}).get('num_workers', 4),
        pin_memory=True
    )

    val_loader = None
    if args.val_ann:
        val_dataset = DetectionDataset(
            data_dir=args.data_dir,
            annotation_file=args.val_ann,
            input_size=input_size,
            augmentor=None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )

    # Trainer
    trainer = DetectionTrainer(config, device=args.device, use_amp=not args.no_amp)

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Egit
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir,
        warmup_epochs=args.warmup
    )


if __name__ == '__main__':
    main()
