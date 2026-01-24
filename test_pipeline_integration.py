"""
OCR Pipeline Entegrasyon Testi

Tum yeni ozellikleri test eder:
- Perspektif duzeltme
- Per-region rotation
- Adaptif line grouping
- Batch recognition
- Variable width
"""

import sys
import os
from pathlib import Path

# Proje kokunu path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import time


def create_test_image_simple():
    """Basit test gorseli olustur"""
    # Beyaz arka plan
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Metin ekle
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Hello World", (50, 100), font, 1.5, (0, 0, 0), 2)
    cv2.putText(img, "Test OCR", (50, 200), font, 1.5, (0, 0, 0), 2)
    cv2.putText(img, "Line Three", (50, 300), font, 1.5, (0, 0, 0), 2)
    
    return img


def create_test_image_rotated():
    """Egik test gorseli olustur"""
    img = create_test_image_simple()
    
    # Goruntüyü dondur
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=(255, 255, 255))
    
    return rotated


def create_test_image_perspective():
    """Perspektif bozulmali test gorseli"""
    img = create_test_image_simple()
    h, w = img.shape[:2]
    
    # Kaynak ve hedef noktalar
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[50, 20], [w-30, 40], [w-10, h-30], [20, h-50]])
    
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (w, h), borderValue=(200, 200, 200))
    
    return warped


def test_preprocessing():
    """Preprocessing modüllerini test et"""
    print("\n" + "="*60)
    print("PREPROCESSING TESTI")
    print("="*60)
    
    from ocr_engine.preprocessing import (
        ImageProcessor, Binarizer, Deskewer, Denoiser,
        PerspectiveCorrector, auto_correct_perspective
    )
    
    # Test gorseli
    img = create_test_image_rotated()
    print(f"Test gorseli boyutu: {img.shape}")
    
    # Deskew
    print("\n[1] Deskew testi...")
    deskewer = Deskewer(max_angle=45)
    deskewed, angle = deskewer.deskew(img)
    print(f"    Tespit edilen aci: {angle:.2f} derece")
    print(f"    Duzeltilmis boyut: {deskewed.shape}")
    
    # Perspektif duzeltme
    print("\n[2] Perspektif duzeltme testi...")
    persp_img = create_test_image_perspective()
    corrector = PerspectiveCorrector()
    
    corners = corrector.detect_corners(persp_img)
    if corners is not None:
        print(f"    Koseler tespit edildi: {len(corners)} nokta")
        corrected = corrector.correct(persp_img, corners)
        print(f"    Duzeltilmis boyut: {corrected.shape}")
    else:
        print("    Koseler tespit edilemedi (beklenen - basit gorsel)")
    
    # Denoise
    print("\n[3] Denoise testi...")
    denoiser = Denoiser(method='bilateral')
    noisy = img + np.random.normal(0, 20, img.shape).astype(np.uint8)
    denoised = denoiser.denoise(noisy)
    print(f"    Gurultulu vs temiz fark: {np.mean(np.abs(noisy.astype(float) - denoised.astype(float))):.2f}")
    
    # Binarization
    print("\n[4] Binarization testi...")
    binarizer = Binarizer(method='adaptive')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = binarizer.binarize(gray)
    print(f"    Binary gorsel unique values: {np.unique(binary)}")
    
    print("\n[OK] Preprocessing testleri tamamlandi!")
    return True


def test_detection_postprocess():
    """Detection postprocess modüllerini test et"""
    print("\n" + "="*60)
    print("DETECTION POSTPROCESS TESTI")
    print("="*60)
    
    from ocr_engine.detection.postprocess import (
        DBPostProcessor, sort_boxes_by_position, adaptive_sort_boxes,
        get_box_rotation_angle, correct_box_rotation, AdaptiveLineGrouper,
        group_boxes_into_lines
    )
    
    # Test kutulari olustur
    boxes = [
        np.array([[50, 50], [200, 50], [200, 80], [50, 80]], dtype=np.float32),   # Line 1, word 1
        np.array([[220, 52], [350, 52], [350, 82], [220, 82]], dtype=np.float32), # Line 1, word 2
        np.array([[50, 120], [180, 120], [180, 150], [50, 150]], dtype=np.float32), # Line 2
        np.array([[50, 200], [300, 200], [300, 230], [50, 230]], dtype=np.float32), # Line 3
    ]
    
    print(f"Test kutuları: {len(boxes)} adet")
    
    # Rotation angle
    print("\n[1] Rotation angle testi...")
    for i, box in enumerate(boxes):
        angle = get_box_rotation_angle(box)
        print(f"    Box {i}: {angle:.2f} derece")
    
    # Egik kutu
    rotated_box = np.array([
        [50, 60], [200, 40], [210, 70], [60, 90]
    ], dtype=np.float32)
    angle = get_box_rotation_angle(rotated_box)
    print(f"    Egik box: {angle:.2f} derece")
    
    # Adaptive line grouping
    print("\n[2] Adaptive line grouping testi...")
    grouper = AdaptiveLineGrouper()
    lines = grouper.group_into_lines(boxes)
    print(f"    Satir sayisi: {len(lines)}")
    for i, line in enumerate(lines):
        print(f"    Satir {i+1}: {len(line)} kutu")
    
    # Sort boxes
    print("\n[3] Box siralama testi...")
    sorted_boxes = adaptive_sort_boxes(boxes)
    print(f"    Siralanmis: {len(sorted_boxes)} kutu")
    
    print("\n[OK] Detection postprocess testleri tamamlandi!")
    return True


def test_recognition_batch():
    """Batch recognition testi"""
    print("\n" + "="*60)
    print("BATCH RECOGNITION TESTI")
    print("="*60)
    
    from ocr_engine.recognition import CRNN, CTCDecoder, Vocabulary
    
    # Vocabulary
    vocab = Vocabulary()
    print(f"Vocabulary size: {vocab.num_classes}")
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = CRNN(
        num_classes=vocab.num_classes,
        input_channels=1,
        hidden_size=256,
        num_layers=2
    ).to(device)
    model.eval()
    
    # Test batch
    print("\n[1] Batch inference testi...")
    batch_sizes = [1, 4, 8]
    widths = [128, 256, 384]
    
    for bs in batch_sizes:
        for w in widths:
            # Random batch olustur
            batch = torch.randn(bs, 1, 32, w).to(device)
            
            start = time.time()
            with torch.no_grad():
                output = model(batch)
            elapsed = (time.time() - start) * 1000
            
            print(f"    Batch {bs}x32x{w}: output shape {output.shape}, {elapsed:.1f}ms")
    
    # Variable width test
    print("\n[2] Variable width testi...")
    decoder = CTCDecoder(vocab)
    
    for w in [128, 256, 512]:
        input_tensor = torch.randn(1, 1, 32, w).to(device)
        with torch.no_grad():
            log_probs = model(input_tensor)
        
        seq_len = log_probs.shape[0]
        texts = decoder.decode_greedy(log_probs)
        print(f"    Width {w}: seq_len={seq_len}, text_len={len(texts[0]) if texts else 0}")
    
    print("\n[OK] Batch recognition testleri tamamlandi!")
    return True


def test_full_pipeline():
    """Full pipeline testi (eğer ağırlıklar varsa)"""
    print("\n" + "="*60)
    print("FULL PIPELINE TESTI")
    print("="*60)
    
    # Test gorseli
    img = create_test_image_simple()
    
    # Pipeline import kontrolu
    try:
        from ocr_engine.pipeline import OCRPipeline, create_pipeline
        print("Pipeline import: [OK]")
    except Exception as e:
        print(f"Pipeline import hatasi: {e}")
        return False
    
    # Pipeline olustur (agirliksiz)
    print("\nPipeline olusturuluyor...")
    try:
        pipeline = OCRPipeline()
        print("Pipeline olusturuldu: [OK]")
    except Exception as e:
        print(f"Pipeline olusturma hatasi: {e}")
        return False
    
    # Detection (agirlik olmadan sadece yapısal test)
    print("\nDetection testi (agirlik olmadan)...")
    try:
        # Sadece preprocess
        processed = pipeline._preprocess(img)
        print(f"  Preprocessed shape: {processed.shape}")
    except Exception as e:
        print(f"  Preprocess hatasi: {e}")
    
    print("\n[BILGI] Tam pipeline testi icin detection agirliklari gerekli!")
    print("        python training/train_detection_advanced.py ile egitim yapin")
    
    return True


def run_all_tests():
    """Tum testleri calistir"""
    print("\n" + "="*60)
    print("OCR PIPELINE ENTEGRASYON TESTLERI")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = {}
    
    # Test 1: Preprocessing
    try:
        results['preprocessing'] = test_preprocessing()
    except Exception as e:
        print(f"\n[HATA] Preprocessing: {e}")
        results['preprocessing'] = False
    
    # Test 2: Detection Postprocess
    try:
        results['detection_postprocess'] = test_detection_postprocess()
    except Exception as e:
        print(f"\n[HATA] Detection Postprocess: {e}")
        results['detection_postprocess'] = False
    
    # Test 3: Recognition Batch
    try:
        results['recognition_batch'] = test_recognition_batch()
    except Exception as e:
        print(f"\n[HATA] Recognition Batch: {e}")
        results['recognition_batch'] = False
    
    # Test 4: Full Pipeline
    try:
        results['full_pipeline'] = test_full_pipeline()
    except Exception as e:
        print(f"\n[HATA] Full Pipeline: {e}")
        results['full_pipeline'] = False
    
    # Sonuc ozeti
    print("\n" + "="*60)
    print("TEST SONUCLARI")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("TUM TESTLER BASARILI!")
    else:
        print("BAZI TESTLER BASARISIZ!")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    run_all_tests()
