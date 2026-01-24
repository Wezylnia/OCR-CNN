"""
Veri seti ve veri yukleyici sinifları
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Callable
from pathlib import Path
import json
import random
import pyclipper
from shapely.geometry import Polygon
import scipy.io as sio
from tqdm import tqdm

from ocr_engine.recognition.vocab import Vocabulary


class SyntheticTextGenerator:
    """Sentetik metin gorseli uretici"""
    
    def __init__(
        self,
        vocab: Vocabulary,
        font_paths: Optional[List[str]] = None,
        font_sizes: Tuple[int, int] = (20, 48),
        image_height: int = 32,
        max_text_length: int = 25,
        bg_colors: List[Tuple[int, int, int]] = None,
        text_colors: List[Tuple[int, int, int]] = None
    ):
        """
        Args:
            vocab: Vocabulary nesnesi
            font_paths: Font dosya yollari
            font_sizes: Font boyut araligi
            image_height: Cikis gorsel yuksekligi
            max_text_length: Maksimum metin uzunlugu
            bg_colors: Arka plan renkleri
            text_colors: Metin renkleri
        """
        self.vocab = vocab
        self.font_sizes = font_sizes
        self.image_height = image_height
        self.max_text_length = max_text_length
        
        # Varsayilan renkler
        self.bg_colors = bg_colors or [
            (255, 255, 255),  # Beyaz
            (245, 245, 245),  # Acik gri
            (255, 255, 240),  # Krem
            (240, 248, 255),  # Acik mavi
        ]
        
        self.text_colors = text_colors or [
            (0, 0, 0),        # Siyah
            (50, 50, 50),     # Koyu gri
            (0, 0, 128),      # Koyu mavi
            (128, 0, 0),      # Koyu kirmizi
        ]
        
        # OpenCV font
        self.fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
        ]
    
    def generate(
        self,
        text: Optional[str] = None,
        min_length: int = 3,
        max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Sentetik metin gorseli uret
        
        Args:
            text: Kullanilacak metin (None ise rastgele)
            min_length: Minimum metin uzunlugu
            max_length: Maksimum metin uzunlugu
            
        Returns:
            (gorsel, metin) tuple
        """
        if max_length is None:
            max_length = self.max_text_length
        
        # Metin olustur
        if text is None:
            text = self._generate_random_text(min_length, max_length)
        
        # Font sec
        font = random.choice(self.fonts)
        font_scale = random.uniform(0.5, 1.5)
        thickness = random.randint(1, 2)
        
        # Renkler sec
        bg_color = random.choice(self.bg_colors)
        text_color = random.choice(self.text_colors)
        
        # Metin boyutunu hesapla
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Padding
        padding = random.randint(5, 15)
        
        # Gorsel boyutu
        img_h = text_h + baseline + 2 * padding
        img_w = text_w + 2 * padding
        
        # Gorsel olustur
        image = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)
        
        # Metin yaz
        x = padding
        y = padding + text_h
        cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)
        
        # Hedef yukseklige boyutlandir
        scale = self.image_height / img_h
        new_w = int(img_w * scale)
        image = cv2.resize(image, (new_w, self.image_height))
        
        return image, text
    
    def _generate_random_text(
        self,
        min_length: int,
        max_length: int
    ) -> str:
        """Rastgele metin olustur"""
        chars = self.vocab.get_all_chars()
        
        # Bosluk haric karakterler
        non_space = [c for c in chars if c != ' ']
        
        length = random.randint(min_length, max_length)
        
        # Kelime bazli veya karakter bazli
        if random.random() < 0.7:
            # Kelime benzeri
            words = []
            current_len = 0
            while current_len < length:
                word_len = random.randint(2, min(8, length - current_len))
                word = ''.join(random.choices(non_space, k=word_len))
                words.append(word)
                current_len += word_len + 1
            return ' '.join(words)
        else:
            # Rastgele karakterler
            return ''.join(random.choices(chars, k=length))
    
    def generate_batch(
        self,
        batch_size: int,
        texts: Optional[List[str]] = None
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Batch gorsel uret"""
        images = []
        labels = []
        
        for i in range(batch_size):
            text = texts[i] if texts else None
            image, label = self.generate(text)
            images.append(image)
            labels.append(label)
        
        return images, labels


class RecognitionDataset(Dataset):
    """Metin tanima veri seti"""
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        annotation_file: Optional[str] = None,
        vocab: Optional[Vocabulary] = None,
        image_height: int = 32,
        image_width: int = 256,
        augmentor: Optional[Callable] = None,
        synthetic_ratio: float = 0.0,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: Gorsel klasoru
            annotation_file: Annotation dosyasi (JSON)
            vocab: Vocabulary nesnesi
            image_height: Gorsel yuksekligi
            image_width: Gorsel genisligi
            augmentor: Augmentation fonksiyonu
            synthetic_ratio: Sentetik veri orani (0-1)
            max_samples: Maksimum ornek sayisi
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.vocab = vocab or Vocabulary()
        self.image_height = image_height
        self.image_width = image_width
        self.augmentor = augmentor
        self.synthetic_ratio = synthetic_ratio
        
        # Veri yukle
        self.samples = []
        
        if annotation_file and Path(annotation_file).exists():
            self._load_annotations(annotation_file)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        # Sentetik uretici
        if synthetic_ratio > 0:
            self.synthetic_generator = SyntheticTextGenerator(
                vocab=self.vocab,
                image_height=image_height
            )
        else:
            self.synthetic_generator = None
    
    def _load_annotations(self, annotation_file: str):
        """Annotation dosyasini yukle"""
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            image_path = item.get('image_path', item.get('image'))
            text = item.get('text', item.get('label'))
            
            if image_path and text:
                if self.data_dir:
                    image_path = self.data_dir / image_path
                self.samples.append({
                    'image_path': str(image_path),
                    'text': text
                })
    
    def __len__(self) -> int:
        base_len = len(self.samples)
        if self.synthetic_ratio > 0:
            return int(base_len / (1 - self.synthetic_ratio))
        return max(base_len, 1000)  # Minimum 1000 ornek
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sentetik mi gercek mi
        use_synthetic = (
            self.synthetic_generator is not None and
            (len(self.samples) == 0 or random.random() < self.synthetic_ratio)
        )
        
        if use_synthetic:
            image, text = self.synthetic_generator.generate()
        else:
            sample = self.samples[idx % len(self.samples)]
            image = cv2.imread(sample['image_path'])
            text = sample['text']
            
            if image is None:
                # Fallback to synthetic
                image, text = self.synthetic_generator.generate()
        
        # Grayscale'e cevir
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Augmentation
        if self.augmentor:
            image = self.augmentor(image)
        
        # Boyutlandir
        image = self._resize_and_pad(image)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Tensor'e cevir
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
        
        # Label encode
        label = self.vocab.encode(text)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'label_length': len(label),
            'text': text
        }
    
    def _resize_and_pad(self, image: np.ndarray) -> np.ndarray:
        """Boyutlandir ve padding ekle"""
        h, w = image.shape[:2]
        
        # Yukseklige gore olcekle
        scale = self.image_height / h
        new_w = int(w * scale)
        
        image = cv2.resize(image, (new_w, self.image_height))
        
        # Genislik kontrolu
        if new_w > self.image_width:
            image = cv2.resize(image, (self.image_width, self.image_height))
        elif new_w < self.image_width:
            # Padding
            padded = np.zeros((self.image_height, self.image_width), dtype=image.dtype)
            padded[:, :new_w] = image
            image = padded
        
        return image


class DetectionDataset(Dataset):
    """Metin tespit veri seti"""
    
    def __init__(
        self,
        data_dir: str,
        annotation_file: str,
        input_size: Tuple[int, int] = (640, 640),
        augmentor: Optional[Callable] = None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: Gorsel klasoru
            annotation_file: Annotation dosyasi
            input_size: Model girdi boyutu (width, height)
            augmentor: Augmentation fonksiyonu
            max_samples: Maksimum ornek sayisi
        """
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.augmentor = augmentor
        
        # Veri yukle
        self.samples = self._load_annotations(annotation_file)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
    
    def _load_annotations(self, annotation_file: str) -> List[Dict]:
        """Annotation yukle"""
        samples = []
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            image_path = self.data_dir / item['image_path']
            boxes = [np.array(box, dtype=np.float32) for box in item['boxes']]
            
            samples.append({
                'image_path': str(image_path),
                'boxes': boxes
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Gorsel yukle
        image = cv2.imread(sample['image_path'])
        boxes = [box.copy() for box in sample['boxes']]
        
        # Augmentation
        if self.augmentor:
            image, boxes = self.augmentor(image, boxes)
        
        # Boyutlandir
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, self.input_size)
        
        # Kutulari olcekle
        scale_x = self.input_size[0] / orig_w
        scale_y = self.input_size[1] / orig_h
        
        scaled_boxes = []
        for box in boxes:
            scaled = box.copy()
            scaled[:, 0] *= scale_x
            scaled[:, 1] *= scale_y
            scaled_boxes.append(scaled)
        
        # Ground truth map'ler olustur
        prob_map, thresh_map, mask = self._generate_maps(scaled_boxes)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Tensor'e cevir
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        prob_tensor = torch.from_numpy(prob_map).unsqueeze(0).float()
        thresh_tensor = torch.from_numpy(thresh_map).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            'image': image_tensor,
            'prob_map': prob_tensor,
            'thresh_map': thresh_tensor,
            'mask': mask_tensor
        }
    
    def _generate_maps(
        self,
        boxes: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        DBNet icin ground truth map'leri olustur
        
        1. prob_map (shrink map): Polygon ice dogru kucultulerek olusturulur
        2. thresh_map: Distance transform ile kenar gecisleri
        3. mask: Egitimde kullanilacak bolgeleri isaretler
        """
        h, w = self.input_size[1], self.input_size[0]
        
        prob_map = np.zeros((h, w), dtype=np.float32)
        thresh_map = np.zeros((h, w), dtype=np.float32)
        thresh_mask = np.zeros((h, w), dtype=np.float32)  # thresh_map icin mask
        mask = np.ones((h, w), dtype=np.float32)
        
        for box in boxes:
            # Polygon'u validate et
            try:
                poly = Polygon(box)
                if not poly.is_valid or poly.area < 1:
                    continue
            except:
                continue
            
            # Cok kucuk kutulari atla (minimum 8 pixel alan)
            area = cv2.contourArea(box.astype(np.int32))
            if area < 8:
                # Kucuk kutuyu mask'tan cikar (ignore)
                pts = box.astype(np.int32)
                cv2.fillPoly(mask, [pts], 0.0)
                continue
            
            # === PROBABILITY MAP (Shrink Map) ===
            shrink_ratio = 0.4  # DBNet paper'da 0.4
            shrunk_box = self._shrink_polygon(box, shrink_ratio)
            
            if shrunk_box is not None and len(shrunk_box) > 0:
                pts = shrunk_box.astype(np.int32)
                cv2.fillPoly(prob_map, [pts], 1.0)
            else:
                # Shrink basarisiz olursa orijinal kutuyu kullan
                pts = box.astype(np.int32)
                cv2.fillPoly(prob_map, [pts], 1.0)
            
            # === THRESHOLD MAP (Distance Transform) ===
            self._generate_threshold_map(box, thresh_map, thresh_mask)
        
        # Threshold map'i normalize et (0.3 - 0.7 arasi)
        thresh_map = thresh_map * 0.4 + 0.3  # [0.3, 0.7] araligina olcekle
        thresh_map = np.where(thresh_mask > 0, thresh_map, 0)
        
        return prob_map, thresh_map, mask
    
    def _shrink_polygon(
        self,
        polygon: np.ndarray,
        shrink_ratio: float
    ) -> Optional[np.ndarray]:
        """
        Polygon'u Vatti clipping algoritmasi ile ice dogru kucult
        
        Args:
            polygon: [N, 2] nokta dizisi
            shrink_ratio: Kucultme orani (0-1)
        
        Returns:
            Kucultulmus polygon veya None
        """
        try:
            poly = Polygon(polygon)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix invalid polygon
            
            # Perimeter ve area
            perimeter = poly.length
            area = poly.area
            
            if perimeter == 0:
                return None
            
            # Shrink distance hesapla (DBNet formulu)
            # D = A * (1 - r^2) / L
            # A = area, L = perimeter, r = shrink_ratio
            distance = area * (1 - shrink_ratio ** 2) / perimeter
            
            # Pyclipper ile kucult
            subject = [tuple(p) for p in polygon]
            
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            
            shrunk = pco.Execute(-distance)
            
            if not shrunk:
                return None
            
            # En buyuk polygon'u al
            shrunk = max(shrunk, key=lambda x: cv2.contourArea(np.array(x)))
            shrunk = np.array(shrunk, dtype=np.float32)
            
            return shrunk
            
        except Exception as e:
            return None
    
    def _generate_threshold_map(
        self,
        polygon: np.ndarray,
        thresh_map: np.ndarray,
        thresh_mask: np.ndarray,
        shrink_ratio: float = 0.4,
        thresh_min: float = 0.3,
        thresh_max: float = 0.7
    ):
        """
        Threshold map olustur (distance transform tabanli)
        
        DBNet'te threshold map, text sinirlari etrafinda yumusak gecis saglar.
        """
        h, w = thresh_map.shape
        
        try:
            poly = Polygon(polygon)
            if not poly.is_valid:
                poly = poly.buffer(0)
            
            # Expand ve shrink mesafelerini hesapla
            perimeter = poly.length
            area = poly.area
            
            if perimeter == 0:
                return
            
            # Expand distance (text siniri disina dogru)
            expand_distance = area * (1 - shrink_ratio ** 2) / perimeter
            
            # Expanded polygon olustur
            subject = [tuple(p) for p in polygon]
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            
            expanded = pco.Execute(expand_distance)
            if not expanded:
                return
            
            expanded = max(expanded, key=lambda x: cv2.contourArea(np.array(x)))
            expanded = np.array(expanded, dtype=np.int32)
            
            # Bounding box
            x_min = max(0, int(np.min(expanded[:, 0])) - 1)
            x_max = min(w, int(np.max(expanded[:, 0])) + 1)
            y_min = max(0, int(np.min(expanded[:, 1])) - 1)
            y_max = min(h, int(np.max(expanded[:, 1])) + 1)
            
            # Threshold map bolgesini hesapla
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    # Noktanin polygon'a mesafesi
                    point = (x, y)
                    
                    # Polygon sinirlarinda mi?
                    distance = self._point_to_polygon_distance(point, polygon)
                    
                    if distance <= expand_distance:
                        # Normalize distance to [0, 1]
                        normalized = distance / expand_distance
                        # Threshold degerini guncelle (max al)
                        thresh_map[y, x] = max(thresh_map[y, x], 1.0 - normalized)
                        thresh_mask[y, x] = 1.0
                        
        except Exception as e:
            pass
    
    def _point_to_polygon_distance(
        self,
        point: Tuple[int, int],
        polygon: np.ndarray
    ) -> float:
        """Noktanin polygon sinirina olan minimum mesafesi"""
        x, y = point
        n = len(polygon)
        min_dist = float('inf')
        
        for i in range(n):
            # Kenar: polygon[i] -> polygon[(i+1) % n]
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            
            # Noktanin dogru parcasina mesafesi
            dist = self._point_to_line_segment_distance(x, y, x1, y1, x2, y2)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _point_to_line_segment_distance(
        self,
        px: float, py: float,
        x1: float, y1: float,
        x2: float, y2: float
    ) -> float:
        """Noktanin dogru parcasina mesafesi"""
        # Kenar vektoru
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # Nokta
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        
        # Parametrik t degeri
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        # En yakin nokta
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


class SynthTextDataset(Dataset):
    """
    SynthText veri seti loader
    
    SynthText formati:
    - gt.mat dosyasi MATLAB formatinda annotations icerir
    - wordBB: word-level bounding boxes [2, 4, num_words] veya [2, 4]
    - charBB: character-level bounding boxes
    - imnames: gorsel dosya adlari
    - txt: metin icerikleri
    """
    
    def __init__(
        self,
        data_dir: str,
        mat_file: str = 'gt.mat',
        input_size: Tuple[int, int] = (640, 640),
        augmentor: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        preload: bool = False
    ):
        """
        Args:
            data_dir: SynthText klasoru
            mat_file: Ground truth .mat dosyasi
            input_size: Model girdi boyutu (width, height)
            augmentor: Augmentation fonksiyonu
            max_samples: Maksimum ornek sayisi
            preload: Tum annotations'i onceden yukle
        """
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.augmentor = augmentor
        self.max_samples = max_samples
        
        # .mat dosyasini yukle
        print(f"SynthText gt.mat yukleniyor: {self.data_dir / mat_file}")
        self.gt_data = sio.loadmat(str(self.data_dir / mat_file))
        
        # Gorsel listesi
        self.imnames = self.gt_data['imnames'][0]
        self.wordBB = self.gt_data['wordBB'][0]
        self.txt = self.gt_data['txt'][0]
        
        self.num_samples = len(self.imnames)
        if max_samples:
            self.num_samples = min(self.num_samples, max_samples)
        
        print(f"SynthText: {self.num_samples} gorsel yuklendi")
        
        # Preload
        self.preloaded = {}
        if preload:
            self._preload_annotations()
    
    def _preload_annotations(self):
        """Tum annotations'i onceden isle"""
        print("Annotations on yukleniyor...")
        for idx in tqdm(range(self.num_samples)):
            self.preloaded[idx] = self._parse_annotation(idx)
    
    def _parse_annotation(self, idx: int) -> Dict:
        """Tek bir annotation'i parse et"""
        # Gorsel yolu
        imname = self.imnames[idx][0]
        image_path = self.data_dir / imname
        
        # Word bounding boxes
        wordBB = self.wordBB[idx]
        
        # wordBB formati: [2, 4, num_words] veya [2, 4] (tek kelime)
        if len(wordBB.shape) == 2:
            # Tek kelime: [2, 4] -> [2, 4, 1]
            wordBB = wordBB[:, :, np.newaxis]
        
        # [2, 4, N] -> N x [4, 2] polygon listesi
        boxes = []
        num_words = wordBB.shape[2]
        
        for i in range(num_words):
            # [2, 4] -> [4, 2] (x,y koordinatlari)
            box = wordBB[:, :, i].T  # [4, 2]
            boxes.append(box.astype(np.float32))
        
        # Text
        txt = self.txt[idx]
        if isinstance(txt, np.ndarray):
            texts = []
            for t in txt:
                if isinstance(t, str):
                    texts.extend(t.split())
                else:
                    texts.extend(str(t).split())
        else:
            texts = str(txt).split()
        
        return {
            'image_path': str(image_path),
            'boxes': boxes,
            'texts': texts
        }
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Annotation al
        if idx in self.preloaded:
            ann = self.preloaded[idx]
        else:
            ann = self._parse_annotation(idx)
        
        # Gorsel yukle
        image = cv2.imread(ann['image_path'])
        if image is None:
            # Fallback: bos gorsel
            image = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
            boxes = []
        else:
            boxes = [box.copy() for box in ann['boxes']]
        
        # Augmentation
        if self.augmentor and len(boxes) > 0:
            try:
                image, boxes = self.augmentor(image, boxes)
            except:
                pass
        
        # Boyutlandir
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, self.input_size)
        
        # Kutulari olcekle
        scale_x = self.input_size[0] / orig_w
        scale_y = self.input_size[1] / orig_h
        
        scaled_boxes = []
        for box in boxes:
            if len(box) >= 4:
                scaled = box.copy()
                scaled[:, 0] *= scale_x
                scaled[:, 1] *= scale_y
                scaled_boxes.append(scaled)
        
        # Ground truth map'ler olustur (DetectionDataset ile ayni)
        prob_map, thresh_map, mask = self._generate_maps(scaled_boxes)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Tensor'e cevir
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        prob_tensor = torch.from_numpy(prob_map).unsqueeze(0).float()
        thresh_tensor = torch.from_numpy(thresh_map).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            'image': image_tensor,
            'prob_map': prob_tensor,
            'thresh_map': thresh_tensor,
            'mask': mask_tensor
        }
    
    def _generate_maps(
        self,
        boxes: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ground truth map'leri olustur (DetectionDataset ile ayni mantik)"""
        h, w = self.input_size[1], self.input_size[0]
        
        prob_map = np.zeros((h, w), dtype=np.float32)
        thresh_map = np.zeros((h, w), dtype=np.float32)
        thresh_mask = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        
        for box in boxes:
            try:
                poly = Polygon(box)
                if not poly.is_valid or poly.area < 1:
                    continue
            except:
                continue
            
            area = cv2.contourArea(box.astype(np.int32))
            if area < 8:
                pts = box.astype(np.int32)
                cv2.fillPoly(mask, [pts], 0.0)
                continue
            
            # Shrink map
            shrink_ratio = 0.4
            shrunk_box = self._shrink_polygon(box, shrink_ratio)
            
            if shrunk_box is not None and len(shrunk_box) > 0:
                pts = shrunk_box.astype(np.int32)
                cv2.fillPoly(prob_map, [pts], 1.0)
            else:
                pts = box.astype(np.int32)
                cv2.fillPoly(prob_map, [pts], 1.0)
            
            # Threshold map (simplified - fast version)
            self._generate_threshold_map_fast(box, thresh_map, thresh_mask)
        
        thresh_map = thresh_map * 0.4 + 0.3
        thresh_map = np.where(thresh_mask > 0, thresh_map, 0)
        
        return prob_map, thresh_map, mask
    
    def _shrink_polygon(
        self,
        polygon: np.ndarray,
        shrink_ratio: float
    ) -> Optional[np.ndarray]:
        """Polygon shrink"""
        try:
            poly = Polygon(polygon)
            if not poly.is_valid:
                poly = poly.buffer(0)
            
            perimeter = poly.length
            area = poly.area
            
            if perimeter == 0:
                return None
            
            distance = area * (1 - shrink_ratio ** 2) / perimeter
            
            subject = [tuple(p) for p in polygon]
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            
            shrunk = pco.Execute(-distance)
            
            if not shrunk:
                return None
            
            shrunk = max(shrunk, key=lambda x: cv2.contourArea(np.array(x)))
            return np.array(shrunk, dtype=np.float32)
            
        except:
            return None
    
    def _generate_threshold_map_fast(
        self,
        polygon: np.ndarray,
        thresh_map: np.ndarray,
        thresh_mask: np.ndarray
    ):
        """Hizli threshold map generation (dilate-based)"""
        h, w = thresh_map.shape
        
        try:
            # Polygon mask olustur
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            pts = polygon.astype(np.int32)
            cv2.fillPoly(poly_mask, [pts], 255)
            
            # Dilate ile expand
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            dilated = cv2.dilate(poly_mask, kernel, iterations=1)
            
            # Border region = dilated - original
            border = dilated - poly_mask
            
            # Distance transform
            dist = cv2.distanceTransform(border, cv2.DIST_L2, 3)
            dist = dist / (dist.max() + 1e-6)
            
            # Update maps
            thresh_map[border > 0] = np.maximum(thresh_map[border > 0], 1.0 - dist[border > 0])
            thresh_mask[dilated > 0] = 1.0
            
        except:
            pass


class ICDARDataset(Dataset):
    """
    ICDAR 2015/2017 veri seti loader
    
    ICDAR formati:
    - Gorsel: img_*.jpg
    - Annotation: gt_img_*.txt (her satir: x1,y1,x2,y2,x3,y3,x4,y4,text)
    """
    
    def __init__(
        self,
        data_dir: str,
        input_size: Tuple[int, int] = (640, 640),
        augmentor: Optional[Callable] = None,
        max_samples: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.augmentor = augmentor
        
        # Gorselleri bul
        self.samples = self._load_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"ICDAR: {len(self.samples)} gorsel yuklendi")
    
    def _load_samples(self) -> List[Dict]:
        """ICDAR formatindaki dosyalari yukle"""
        samples = []
        
        # Gorsel dosyalarini bul
        image_files = list(self.data_dir.glob('*.jpg')) + list(self.data_dir.glob('*.png'))
        
        for img_path in image_files:
            # Annotation dosyasi
            gt_name = f"gt_{img_path.stem}.txt"
            gt_path = self.data_dir / gt_name
            
            if not gt_path.exists():
                # Alternatif format
                gt_path = img_path.with_suffix('.txt')
            
            if gt_path.exists():
                boxes, texts = self._parse_annotation(gt_path)
                samples.append({
                    'image_path': str(img_path),
                    'boxes': boxes,
                    'texts': texts
                })
        
        return samples
    
    def _parse_annotation(self, gt_path: Path) -> Tuple[List[np.ndarray], List[str]]:
        """ICDAR annotation dosyasini parse et"""
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
                    # x1,y1,x2,y2,x3,y3,x4,y4,text
                    coords = [float(parts[i]) for i in range(8)]
                    box = np.array([
                        [coords[0], coords[1]],
                        [coords[2], coords[3]],
                        [coords[4], coords[5]],
                        [coords[6], coords[7]]
                    ], dtype=np.float32)
                    boxes.append(box)
                    
                    # Text (opsiyonel)
                    if len(parts) > 8:
                        text = ','.join(parts[8:])
                        # ### veya * ile baslayan ignore
                        if text.startswith('###') or text.startswith('*'):
                            text = ''
                        texts.append(text)
                    else:
                        texts.append('')
                except:
                    continue
        
        return boxes, texts
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Gorsel yukle
        image = cv2.imread(sample['image_path'])
        if image is None:
            image = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
            boxes = []
        else:
            boxes = [box.copy() for box in sample['boxes']]
        
        # Augmentation
        if self.augmentor and len(boxes) > 0:
            try:
                image, boxes = self.augmentor(image, boxes)
            except:
                pass
        
        # Boyutlandir
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, self.input_size)
        
        # Kutulari olcekle
        scale_x = self.input_size[0] / orig_w
        scale_y = self.input_size[1] / orig_h
        
        scaled_boxes = []
        for box in boxes:
            scaled = box.copy()
            scaled[:, 0] *= scale_x
            scaled[:, 1] *= scale_y
            scaled_boxes.append(scaled)
        
        # Ground truth map'ler
        prob_map, thresh_map, mask = self._generate_maps(scaled_boxes)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        prob_tensor = torch.from_numpy(prob_map).unsqueeze(0).float()
        thresh_tensor = torch.from_numpy(thresh_map).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            'image': image_tensor,
            'prob_map': prob_tensor,
            'thresh_map': thresh_tensor,
            'mask': mask_tensor
        }
    
    def _generate_maps(self, boxes: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """DetectionDataset ile ayni"""
        h, w = self.input_size[1], self.input_size[0]
        
        prob_map = np.zeros((h, w), dtype=np.float32)
        thresh_map = np.zeros((h, w), dtype=np.float32)
        thresh_mask = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        
        for box in boxes:
            try:
                poly = Polygon(box)
                if not poly.is_valid or poly.area < 1:
                    continue
            except:
                continue
            
            area = cv2.contourArea(box.astype(np.int32))
            if area < 8:
                pts = box.astype(np.int32)
                cv2.fillPoly(mask, [pts], 0.0)
                continue
            
            shrunk_box = self._shrink_polygon(box, 0.4)
            
            if shrunk_box is not None:
                pts = shrunk_box.astype(np.int32)
                cv2.fillPoly(prob_map, [pts], 1.0)
            else:
                pts = box.astype(np.int32)
                cv2.fillPoly(prob_map, [pts], 1.0)
            
            self._generate_threshold_map_fast(box, thresh_map, thresh_mask)
        
        thresh_map = thresh_map * 0.4 + 0.3
        thresh_map = np.where(thresh_mask > 0, thresh_map, 0)
        
        return prob_map, thresh_map, mask
    
    def _shrink_polygon(self, polygon: np.ndarray, shrink_ratio: float) -> Optional[np.ndarray]:
        try:
            poly = Polygon(polygon)
            if not poly.is_valid:
                poly = poly.buffer(0)
            
            perimeter = poly.length
            area = poly.area
            
            if perimeter == 0:
                return None
            
            distance = area * (1 - shrink_ratio ** 2) / perimeter
            
            subject = [tuple(p) for p in polygon]
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            
            shrunk = pco.Execute(-distance)
            
            if not shrunk:
                return None
            
            shrunk = max(shrunk, key=lambda x: cv2.contourArea(np.array(x)))
            return np.array(shrunk, dtype=np.float32)
        except:
            return None
    
    def _generate_threshold_map_fast(self, polygon: np.ndarray, thresh_map: np.ndarray, thresh_mask: np.ndarray):
        h, w = thresh_map.shape
        
        try:
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            pts = polygon.astype(np.int32)
            cv2.fillPoly(poly_mask, [pts], 255)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            dilated = cv2.dilate(poly_mask, kernel, iterations=1)
            
            border = dilated - poly_mask
            dist = cv2.distanceTransform(border, cv2.DIST_L2, 3)
            dist = dist / (dist.max() + 1e-6)
            
            thresh_map[border > 0] = np.maximum(thresh_map[border > 0], 1.0 - dist[border > 0])
            thresh_mask[dilated > 0] = 1.0
        except:
            pass


def collate_recognition(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Recognition batch collate fonksiyonu"""
    images = torch.stack([item['image'] for item in batch])
    
    # Label'lari padding ile birlestir
    labels = [item['label'] for item in batch]
    label_lengths = torch.tensor([item['label_length'] for item in batch])
    
    # Max uzunluga padding
    max_len = max(len(l) for l in labels)
    padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
    
    return {
        'images': images,
        'labels': padded_labels,
        'label_lengths': label_lengths,
        'texts': [item['text'] for item in batch]
    }


def create_recognition_dataloader(
    data_dir: Optional[str] = None,
    annotation_file: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    synthetic_ratio: float = 0.5
) -> DataLoader:
    """Recognition DataLoader olustur"""
    from .augmentation import RecognitionAugmentor
    
    augmentor = RecognitionAugmentor() if augment else None
    
    dataset = RecognitionDataset(
        data_dir=data_dir,
        annotation_file=annotation_file,
        augmentor=augmentor,
        synthetic_ratio=synthetic_ratio
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_recognition,
        pin_memory=True
    )
