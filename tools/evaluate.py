"""
Degerlendirme Scripti

Detection icin F1/Precision/Recall, Recognition icin Word Accuracy ve CER (Character Error Rate)
hesaplar.

Kullanim:
    # Recognition degerlendirme
    python tools/evaluate.py recognition \
        --data_dir DataSets/mnt/ramdisk/max/90kDICT32px \
        --ann_file  data/mjsynth_1M.json \
        --checkpoint checkpoints/recognition_best.pth \
        --device cuda

    # Detection degerlendirme (ICDAR formatinda GT gerektirir)
    python tools/evaluate.py detection \
        --images_dir DataSets/icdar2015/test_images \
        --gt_dir     DataSets/icdar2015/test_gts \
        --checkpoint checkpoints/detection_best.pth \
        --device cuda
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Recognition metrikleri
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Edit mesafesi (Levenshtein distance)"""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
    return dp[n]


def evaluate_recognition(
    data_dir: str,
    ann_file: str,
    checkpoint: str,
    device: str = 'cpu',
    batch_size: int = 128,
    max_samples: int = None,
    encoder_type: str = 'vgg',
) -> dict:
    """
    Recognition modelini degerlendir.

    Returns:
        {
            'word_accuracy': float,       # Tam kelime eslestirme orani (case-insensitive)
            'word_accuracy_exact': float, # Tam kelime eslestirme (case-sensitive)
            'cer': float,                 # Character Error Rate
            'num_samples': int,
            'num_correct': int,
        }
    """
    from ocr_engine.recognition.vocab import Vocabulary
    from ocr_engine.recognition.model import CRNN
    from ocr_engine.recognition.decoder import CTCPrefixDecoder
    from training.dataset import RecognitionDataset, collate_recognition
    from torch.utils.data import DataLoader

    print(f"\n=== Recognition Degerlendirme ===")
    print(f"Checkpoint : {checkpoint}")
    print(f"Device     : {device}")

    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')

    # Vocabulary
    vocab = Vocabulary()

    # Model
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    model = CRNN(
        num_classes=vocab.size,
        input_channels=1,
        hidden_size=256,
        num_layers=2,
        encoder_type=encoder_type,
    ).to(dev)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Model yuklendi (epoch {ckpt.get('epoch', '?')})")

    # Decoder
    decoder = CTCPrefixDecoder(vocab, beam_width=5)

    # Dataset
    dataset = RecognitionDataset(
        data_dir=data_dir,
        annotation_file=ann_file,
        vocab=vocab,
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_recognition,
    )
    print(f"Dataset: {len(dataset)} ornek")

    n_correct_exact = 0
    n_correct_ci    = 0
    total_edit      = 0
    total_chars     = 0
    n_total         = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Degerlendiriliyor"):
            images  = batch['images'].to(dev)
            texts   = batch['texts']

            log_probs = model(images)  # [T, B, C]
            results   = decoder.decode_batch(log_probs)

            for (pred, _), text in zip(results, texts):
                n_total += 1

                # Exact match
                if pred == text:
                    n_correct_exact += 1

                # Case-insensitive match
                if pred.lower() == text.lower():
                    n_correct_ci += 1

                # CER
                edit = levenshtein(pred, text)
                total_edit  += edit
                total_chars += max(len(text), 1)

    word_acc_exact = n_correct_exact / max(n_total, 1)
    word_acc_ci    = n_correct_ci    / max(n_total, 1)
    cer            = total_edit      / max(total_chars, 1)

    metrics = {
        'word_accuracy_exact': round(word_acc_exact, 4),
        'word_accuracy_ci'   : round(word_acc_ci, 4),
        'cer'                : round(cer, 4),
        'num_samples'        : n_total,
        'num_correct_exact'  : n_correct_exact,
        'num_correct_ci'     : n_correct_ci,
    }

    print(f"\nSonuclar:")
    print(f"  Word Accuracy (exact)      : {word_acc_exact*100:.2f}%")
    print(f"  Word Accuracy (case-insens): {word_acc_ci*100:.2f}%")
    print(f"  CER                        : {cer*100:.2f}%")
    print(f"  Toplam Ornek               : {n_total}")
    return metrics


# ---------------------------------------------------------------------------
# Detection metrikleri
# ---------------------------------------------------------------------------

def iou_boxes(a: np.ndarray, b: np.ndarray) -> float:
    """2D polygon IoU: Shapely kullaniyor"""
    try:
        from shapely.geometry import Polygon as SPoly
        pa, pb = SPoly(a), SPoly(b)
        if not pa.is_valid or not pb.is_valid:
            return 0.0
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return inter / (union + 1e-9)
    except Exception:
        return 0.0


def evaluate_detection(
    images_dir: str,
    gt_dir: str,
    checkpoint: str,
    device: str = 'cpu',
    iou_threshold: float = 0.5,
    det_threshold: float = 0.3,
    box_threshold: float = 0.5,
) -> dict:
    """
    Detection modelini degerlendir (ICDAR-style GT).

    GT dosya formati: <image_name>.txt, her satir:
        x1,y1,x2,y2,x3,y3,x4,y4[,IGNORE]

    Returns:
        {'precision': float, 'recall': float, 'f1': float, 'num_images': int}
    """
    from ocr_engine.detection.model import DBNet
    from ocr_engine.detection.postprocess import DBPostProcessor

    print(f"\n=== Detection Degerlendirme ===")
    print(f"Checkpoint    : {checkpoint}")
    print(f"Images dir    : {images_dir}")
    print(f"GT dir        : {gt_dir}")
    print(f"IoU threshold : {iou_threshold}")

    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')

    # Model
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    model = DBNet(backbone='resnet18', pretrained=False).to(dev)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Model yuklendi (epoch {ckpt.get('epoch', '?')})")

    postproc = DBPostProcessor(
        threshold=det_threshold,
        box_threshold=box_threshold,
        max_candidates=1000,
        unclip_ratio=1.5,
    )

    images_path = Path(images_dir)
    gt_path     = Path(gt_dir)
    image_files = sorted(images_path.glob('*.jpg')) + sorted(images_path.glob('*.png'))

    tp = fp = fn = 0

    for img_file in tqdm(image_files, desc="Degerlendiriliyor"):
        # GT yukle
        gt_file = gt_path / (img_file.stem + '.txt')
        if not gt_file.exists():
            continue

        gt_boxes = []
        with open(gt_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                try:
                    coords = list(map(float, parts[:8]))
                    box = np.array(coords).reshape(4, 2)
                    gt_boxes.append(box)
                except ValueError:
                    continue

        if not gt_boxes:
            continue

        # Gorsel isle
        image = cv2.imread(str(img_file))
        if image is None:
            continue

        h, w = image.shape[:2]
        # Resize (uzun kenar 640)
        scale = 640.0 / max(h, w)
        rsz = cv2.resize(image, (int(w * scale), int(h * scale)))
        rsz_h, rsz_w = rsz.shape[:2]

        # Tensor
        tensor = torch.from_numpy(rsz).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(dev)

        with torch.no_grad():
            out = model(tensor, return_maps=True)
        prob_map = out['prob_map'][0, 0].cpu().numpy()
        pred_boxes = postproc(prob_map, (rsz_h, rsz_w))

        # Scale pred boxes back to original size
        sx, sy = w / rsz_w, h / rsz_h
        scaled_preds = []
        for box in pred_boxes:
            sb = box.astype(float).copy()
            sb[:, 0] *= sx
            sb[:, 1] *= sy
            scaled_preds.append(sb)

        # Match
        matched_gt = set()
        matched_pred = set()
        for pi, pb in enumerate(scaled_preds):
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                if iou_boxes(pb, gb) >= iou_threshold:
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    break

        tp += len(matched_pred)
        fp += len(scaled_preds) - len(matched_pred)
        fn += len(gt_boxes)      - len(matched_gt)

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    metrics = {
        'precision'  : round(precision, 4),
        'recall'     : round(recall, 4),
        'f1'         : round(f1, 4),
        'tp': tp, 'fp': fp, 'fn': fn,
        'num_images' : len(image_files),
    }

    print(f"\nSonuclar:")
    print(f"  Precision : {precision*100:.2f}%")
    print(f"  Recall    : {recall*100:.2f}%")
    print(f"  F1        : {f1*100:.2f}%")
    print(f"  Gorsel    : {len(image_files)}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='OCR Model Degerlendirme')
    sub = parser.add_subparsers(dest='task', required=True)

    # -- recognition --
    rec = sub.add_parser('recognition', help='Recognition degerlendirme')
    rec.add_argument('--data_dir',    required=True)
    rec.add_argument('--ann_file',    required=True)
    rec.add_argument('--checkpoint',  required=True)
    rec.add_argument('--device',      default='cpu')
    rec.add_argument('--batch_size',  type=int, default=128)
    rec.add_argument('--max_samples', type=int, default=None)
    rec.add_argument('--encoder_type', default='vgg', choices=['vgg', 'resnet'])
    rec.add_argument('--output_json', default=None, help='Sonuclari JSON olarak kaydet')

    # -- detection --
    det = sub.add_parser('detection', help='Detection degerlendirme')
    det.add_argument('--images_dir',   required=True)
    det.add_argument('--gt_dir',       required=True)
    det.add_argument('--checkpoint',   required=True)
    det.add_argument('--device',       default='cpu')
    det.add_argument('--iou_threshold', type=float, default=0.5)
    det.add_argument('--det_threshold', type=float, default=0.3)
    det.add_argument('--box_threshold', type=float, default=0.5)
    det.add_argument('--output_json',  default=None)

    args = parser.parse_args()

    if args.task == 'recognition':
        metrics = evaluate_recognition(
            data_dir     = args.data_dir,
            ann_file     = args.ann_file,
            checkpoint   = args.checkpoint,
            device       = args.device,
            batch_size   = args.batch_size,
            max_samples  = args.max_samples,
            encoder_type = args.encoder_type,
        )
    else:
        metrics = evaluate_detection(
            images_dir    = args.images_dir,
            gt_dir        = args.gt_dir,
            checkpoint    = args.checkpoint,
            device        = args.device,
            iou_threshold = args.iou_threshold,
            det_threshold = args.det_threshold,
            box_threshold = args.box_threshold,
        )

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nSonuclar kaydedildi: {args.output_json}")


if __name__ == '__main__':
    main()
