"""
Detection Veri Seti Hazirlama Araci

Desteklenen veri setleri:
1. SynthText - Sentetik, ~800K gorsel
2. ICDAR 2015 - Gercek dunya, 1000 train + 500 test
3. ICDAR 2017 MLT - Cok dilli (Turkce dahil)

Kullanim:
    python tools/prepare_detection_data.py --help
"""

import os
import sys
import argparse
import json
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil

# Proje kokunu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))


DATASET_INFO = {
    'synthtext': {
        'name': 'SynthText',
        'description': 'Sentetik metin gorselleri (~800K)',
        'size': '~41GB',
        'url': 'https://www.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip',
        'format': 'mat'
    },
    'icdar2015': {
        'name': 'ICDAR 2015',
        'description': 'Scene text detection benchmark',
        'size': '~130MB',
        'train_url': 'https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',
        'train_gt_url': 'https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip',
        'test_url': 'https://rrc.cvc.uab.es/downloads/ch4_test_images.zip',
        'format': 'icdar'
    },
    'icdar2017': {
        'name': 'ICDAR 2017 MLT',
        'description': 'Cok dilli metin tespiti (9 dil)',
        'size': '~7GB',
        'note': 'Manuel indirme gerekli: https://rrc.cvc.uab.es/?ch=8'
    }
}


def download_file(url: str, dest_path: str, desc: str = None):
    """URL'den dosya indir"""
    if desc is None:
        desc = os.path.basename(dest_path)
    
    print(f"Indiriliyor: {desc}")
    
    # Progress bar ile indir
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
    
    print(f"Tamamlandi: {dest_path}")


def extract_archive(archive_path: str, dest_dir: str):
    """Arsivi cikar"""
    print(f"Cikartiliyor: {archive_path}")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(dest_dir)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tf:
            tf.extractall(dest_dir)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tf:
            tf.extractall(dest_dir)
    
    print(f"Cikartildi: {dest_dir}")


def prepare_icdar2015(data_dir: str, download: bool = False):
    """ICDAR 2015 veri setini hazirla"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    
    if download:
        print("\n" + "="*60)
        print("ICDAR 2015 INDIRME")
        print("="*60)
        print("NOT: ICDAR indirmek icin RRC hesabi gerekli!")
        print("Manuel indirme: https://rrc.cvc.uab.es/?ch=4&com=downloads")
        print("="*60 + "\n")
        return
    
    # Mevcut dosyalari kontrol et
    if train_dir.exists():
        images = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))
        gts = list(train_dir.glob('gt_*.txt'))
        print(f"Train: {len(images)} gorsel, {len(gts)} annotation")
    
    if test_dir.exists():
        images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        print(f"Test: {len(images)} gorsel")


def convert_icdar_to_json(icdar_dir: str, output_file: str):
    """ICDAR formatini JSON'a cevir"""
    icdar_dir = Path(icdar_dir)
    samples = []
    
    # Tum gorselleri bul
    image_files = list(icdar_dir.glob('*.jpg')) + list(icdar_dir.glob('*.png'))
    
    for img_path in tqdm(image_files, desc="ICDAR donusturuluyor"):
        # GT dosyasi
        gt_name = f"gt_{img_path.stem}.txt"
        gt_path = icdar_dir / gt_name
        
        if not gt_path.exists():
            continue
        
        boxes = []
        texts = []
        
        with open(gt_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 8:
                    continue
                
                try:
                    coords = [float(parts[i]) for i in range(8)]
                    box = [
                        [coords[0], coords[1]],
                        [coords[2], coords[3]],
                        [coords[4], coords[5]],
                        [coords[6], coords[7]]
                    ]
                    boxes.append(box)
                    
                    if len(parts) > 8:
                        text = ','.join(parts[8:])
                        texts.append(text)
                    else:
                        texts.append('')
                except:
                    continue
        
        if boxes:
            samples.append({
                'image_path': img_path.name,
                'boxes': boxes,
                'texts': texts
            })
    
    # JSON kaydet
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"Kaydedildi: {output_file} ({len(samples)} ornek)")


def show_dataset_info():
    """Veri seti bilgilerini goster"""
    print("\n" + "="*60)
    print("DESTEKLENEN VERI SETLERI")
    print("="*60)
    
    for key, info in DATASET_INFO.items():
        print(f"\n[{key.upper()}] {info['name']}")
        print(f"  Aciklama: {info['description']}")
        print(f"  Boyut: {info['size']}")
        if 'url' in info:
            print(f"  URL: {info['url']}")
        if 'note' in info:
            print(f"  Not: {info['note']}")
    
    print("\n" + "="*60)
    print("HIZLI BASLANGIC")
    print("="*60)
    print("""
1. SynthText (Onerilen - Buyuk veri seti):
   - https://www.robots.ox.ac.uk/~vgg/data/scenetext/ adresinden indir
   - DataSets/SynthText/ klasorune cikar
   - Egitim:
     python training/train_detection_advanced.py \\
       --dataset synthtext \\
       --data_dir DataSets/SynthText \\
       --epochs 50 \\
       --batch_size 8

2. ICDAR 2015 (Kucuk ama kaliteli):
   - https://rrc.cvc.uab.es/?ch=4 adresinden indir
   - DataSets/ICDAR2015/train/ klasorune cikar
   - Egitim:
     python training/train_detection_advanced.py \\
       --dataset icdar \\
       --data_dir DataSets/ICDAR2015/train \\
       --val_dir DataSets/ICDAR2015/test \\
       --epochs 200 \\
       --batch_size 8

3. Kendi Veriniz:
   - JSON format: [{"image_path": "img.jpg", "boxes": [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]]}]
   - python training/train_detection_advanced.py \\
       --dataset custom \\
       --data_dir your_data_folder
""")


def main():
    parser = argparse.ArgumentParser(
        description='Detection Veri Seti Hazirlama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  # Veri seti bilgilerini goster
  python tools/prepare_detection_data.py --info
  
  # ICDAR formatini JSON'a cevir
  python tools/prepare_detection_data.py --convert icdar --input DataSets/ICDAR2015/train --output data/icdar_train.json
        """
    )
    
    parser.add_argument('--info', action='store_true', help='Veri seti bilgilerini goster')
    parser.add_argument('--convert', type=str, choices=['icdar', 'synthtext'], help='Format donusumu')
    parser.add_argument('--input', type=str, help='Girdi klasoru')
    parser.add_argument('--output', type=str, help='Cikti dosyasi')
    parser.add_argument('--download', type=str, choices=['icdar2015'], help='Veri seti indir')
    parser.add_argument('--data_dir', type=str, default='DataSets', help='Veri klasoru')
    
    args = parser.parse_args()
    
    if args.info:
        show_dataset_info()
        return
    
    if args.convert:
        if not args.input or not args.output:
            print("HATA: --input ve --output gerekli")
            return
        
        if args.convert == 'icdar':
            convert_icdar_to_json(args.input, args.output)
    
    if args.download:
        if args.download == 'icdar2015':
            prepare_icdar2015(args.data_dir, download=True)
    
    if not any([args.info, args.convert, args.download]):
        parser.print_help()


if __name__ == '__main__':
    main()
