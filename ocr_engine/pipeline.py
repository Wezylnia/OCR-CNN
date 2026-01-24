"""
OCR Pipeline - Detection ve Recognition entegrasyonu
"""

import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict
from pathlib import Path
import yaml

from .preprocessing import ImageProcessor, Binarizer, Deskewer, Denoiser
from .detection import DBNet, DBPostProcessor
from .detection.postprocess import sort_boxes_by_position, adaptive_sort_boxes, AdaptiveLineGrouper
from .recognition import CRNN, CTCDecoder, Vocabulary


@dataclass
class TextBox:
    """Tespit edilen metin kutusu"""
    box: np.ndarray  # [4, 2] polygon koordinatlari
    text: str = ""
    confidence: float = 0.0
    
    @property
    def x1(self) -> int:
        return int(np.min(self.box[:, 0]))
    
    @property
    def y1(self) -> int:
        return int(np.min(self.box[:, 1]))
    
    @property
    def x2(self) -> int:
        return int(np.max(self.box[:, 0]))
    
    @property
    def y2(self) -> int:
        return int(np.max(self.box[:, 1]))
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    def to_dict(self) -> dict:
        return {
            'box': self.box.tolist(),
            'text': self.text,
            'confidence': self.confidence,
            'bbox': [self.x1, self.y1, self.x2, self.y2]
        }


@dataclass
class OCRResult:
    """OCR sonucu"""
    text_boxes: List[TextBox] = field(default_factory=list)
    full_text: str = ""
    processing_time: float = 0.0
    
    @property
    def text(self) -> str:
        """Tum metni birlestir"""
        if self.full_text:
            return self.full_text
        return '\n'.join([tb.text for tb in self.text_boxes if tb.text])
    
    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'boxes': [tb.to_dict() for tb in self.text_boxes],
            'processing_time': self.processing_time
        }


class OCRPipeline:
    """
    Ana OCR pipeline
    
    Gorsellerden metin cikarmak icin detection ve recognition
    modellerini birlestiren end-to-end pipeline.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        detection_weights: Optional[str] = None,
        recognition_weights: Optional[str] = None
    ):
        """
        Args:
            config_path: Yapilandirma dosyasi yolu
            device: Cihaz (cuda veya cpu)
            detection_weights: Detection model agirliklari
            recognition_weights: Recognition model agirliklari
        """
        # Yapilandirma yukle
        self.config = self._load_config(config_path)
        
        # Cihaz ayarla
        if device is None:
            device = self.config.get('general', {}).get('device', 'cuda')
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA mevcut degil, CPU kullaniliyor.")
            device = 'cpu'
        
        self.device = torch.device(device)
        
        # Preprocessing bilesenleri
        self._init_preprocessing()
        
        # Detection modeli
        self._init_detection(detection_weights)
        
        # Recognition modeli
        self._init_recognition(recognition_weights)
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Yapilandirma dosyasini yukle"""
        if config_path is None:
            # Varsayilan config yolu
            config_path = Path(__file__).parent.parent / 'config.yaml'
        
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Varsayilan yapilandirma
        return {
            'general': {'device': 'cuda'},
            'preprocessing': {
                'target_size': [1280, 960],
                'denoise': {'enabled': True, 'method': 'bilateral', 'strength': 10},
                'deskew': {'enabled': True, 'max_angle': 45}
            },
            'detection': {
                'model': {'backbone': 'resnet18'},
                'inference': {
                    'threshold': 0.3,
                    'box_threshold': 0.5,
                    'max_candidates': 1000,
                    'unclip_ratio': 1.5
                }
            },
            'recognition': {
                'model': {
                    'input_height': 32,
                    'input_width': 256,
                    'hidden_size': 256,
                    'num_layers': 2
                },
                'inference': {'beam_width': 5}
            }
        }
    
    def _init_preprocessing(self):
        """Preprocessing bilesenlerini baslat"""
        preproc_cfg = self.config.get('preprocessing', {})
        
        # Image processor
        self.image_processor = ImageProcessor(
            target_size=tuple(preproc_cfg.get('target_size', [1280, 960]))
        )
        
        # Denoiser
        denoise_cfg = preproc_cfg.get('denoise', {})
        if denoise_cfg.get('enabled', True):
            self.denoiser = Denoiser(
                method=denoise_cfg.get('method', 'bilateral'),
                strength=denoise_cfg.get('strength', 10)
            )
        else:
            self.denoiser = None
        
        # Deskewer
        deskew_cfg = preproc_cfg.get('deskew', {})
        if deskew_cfg.get('enabled', True):
            self.deskewer = Deskewer(
                max_angle=deskew_cfg.get('max_angle', 45)
            )
        else:
            self.deskewer = None
        
        # Binarizer (opsiyonel, recognition icin kullanilabilir)
        self.binarizer = Binarizer(method='adaptive')
    
    def _init_detection(self, weights_path: Optional[str]):
        """Detection modelini baslat"""
        det_cfg = self.config.get('detection', {})
        model_cfg = det_cfg.get('model', {})
        
        # Model
        self.detection_model = DBNet(
            backbone=model_cfg.get('backbone', 'resnet18'),
            pretrained=model_cfg.get('pretrained', True)
        ).to(self.device)
        
        # Agirliklar
        if weights_path is None:
            weights_path = det_cfg.get('weights_path')
        
        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            self.detection_model.load_state_dict(state_dict)
            print(f"Detection agirliklari yuklendi: {weights_path}")
        
        self.detection_model.eval()
        
        # Post-processor
        inf_cfg = det_cfg.get('inference', {})
        self.detection_postprocessor = DBPostProcessor(
            threshold=inf_cfg.get('threshold', 0.3),
            box_threshold=inf_cfg.get('box_threshold', 0.5),
            max_candidates=inf_cfg.get('max_candidates', 1000),
            unclip_ratio=inf_cfg.get('unclip_ratio', 1.5)
        )
    
    def _init_recognition(self, weights_path: Optional[str]):
        """Recognition modelini baslat"""
        rec_cfg = self.config.get('recognition', {})
        model_cfg = rec_cfg.get('model', {})
        
        # Vocabulary
        self.vocab = Vocabulary()
        
        # Model
        self.recognition_model = CRNN(
            num_classes=self.vocab.num_classes,
            input_channels=1,  # Grayscale
            hidden_size=model_cfg.get('hidden_size', 256),
            num_layers=model_cfg.get('num_layers', 2),
            dropout=model_cfg.get('dropout', 0.1)
        ).to(self.device)
        
        # Agirliklar
        if weights_path is None:
            weights_path = rec_cfg.get('weights_path')
        
        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            self.recognition_model.load_state_dict(state_dict)
            print(f"Recognition agirliklari yuklendi: {weights_path}")
        
        self.recognition_model.eval()
        
        # Decoder
        inf_cfg = rec_cfg.get('inference', {})
        self.decoder = CTCDecoder(self.vocab)
        self.beam_width = inf_cfg.get('beam_width', 5)
        
        # Recognition girdi boyutlari
        self.rec_input_height = model_cfg.get('input_height', 32)
        self.rec_input_width = model_cfg.get('input_width', 256)
        self.rec_max_width = model_cfg.get('max_width', 512)
        self.variable_width = inf_cfg.get('variable_width', True)
    
    def recognize(
        self,
        image: Union[str, Path, np.ndarray],
        detect_only: bool = False,
        recognize_only: bool = False,
        boxes: Optional[List[np.ndarray]] = None
    ) -> OCRResult:
        """
        Gorseldeki metni tanir
        
        Args:
            image: Gorsel yolu veya numpy array
            detect_only: Sadece tespit yap, tanima yapma
            recognize_only: Sadece tanima yap (boxes gerekli)
            boxes: Onceden tespit edilmis kutular (recognize_only icin)
            
        Returns:
            OCRResult nesnesi
        """
        import time
        start_time = time.time()
        
        # Gorseli yukle
        if isinstance(image, (str, Path)):
            image = self.image_processor.load_image(str(image))
        
        original_image = image.copy()
        original_size = image.shape[:2]  # (height, width)
        
        # Preprocessing
        image = self._preprocess(image)
        
        # Detection
        if not recognize_only:
            boxes = self._detect(image)
            # Adaptif siralama (satir gruplama ile)
            boxes = adaptive_sort_boxes(boxes)
        
        if detect_only or boxes is None or len(boxes) == 0:
            processing_time = time.time() - start_time
            text_boxes = [TextBox(box=box) for box in (boxes or [])]
            return OCRResult(
                text_boxes=text_boxes,
                processing_time=processing_time
            )
        
        # Recognition
        text_boxes = self._recognize(original_image, boxes)
        
        processing_time = time.time() - start_time
        
        return OCRResult(
            text_boxes=text_boxes,
            processing_time=processing_time
        )
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Gorseli on islemden gecir"""
        # Boyutlandir
        image, _ = self.image_processor.resize_with_aspect_ratio(image)
        
        # Gurultu gider
        if self.denoiser is not None:
            image = self.denoiser.denoise(image)
        
        # Aci duzelt
        if self.deskewer is not None:
            image, _ = self.deskewer.deskew(image)
        
        return image
    
    @torch.no_grad()
    def _detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Metin bolgelerini tespit et"""
        original_size = image.shape[:2]
        
        # Detection icin hazirla
        # Resize to detection input size
        det_cfg = self.config.get('detection', {})
        det_size = det_cfg.get('input_size', [640, 640])
        
        resized = cv2.resize(image, (det_size[0], det_size[1]))
        
        # RGB'ye cevir ve normalize et
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = self.image_processor.normalize(rgb)
        
        # Tensor'e cevir: [B, C, H, W]
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        # Detection
        outputs = self.detection_model(tensor)
        prob_map = outputs['prob_map'][0, 0].cpu().numpy()
        
        # Post-process
        boxes = self.detection_postprocessor(prob_map, original_size)
        
        return boxes
    
    @torch.no_grad()
    def _recognize(
        self,
        image: np.ndarray,
        boxes: List[np.ndarray],
        batch_size: int = 32
    ) -> List[TextBox]:
        """
        Tespit edilen bolgelerdeki metni tanir (Batch processing)
        
        Args:
            image: Kaynak gorsel
            boxes: Tespit edilen kutular
            batch_size: Batch boyutu
            
        Returns:
            TextBox listesi
        """
        from .detection.postprocess import correct_box_rotation
        
        if len(boxes) == 0:
            return []
        
        # 1. Tum crop'lari hazirla
        crops = []
        valid_indices = []
        
        for i, box in enumerate(boxes):
            try:
                # Per-region rotation correction
                crop, _ = correct_box_rotation(image, box, angle_threshold=5.0)
                
                if crop.size == 0:
                    crops.append(None)
                    continue
                
                # Grayscale'e cevir
                if len(crop.shape) == 3:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                else:
                    gray = crop
                
                # Yuksekligi ayarla
                h, w = gray.shape[:2]
                scale = self.rec_input_height / h
                new_w = int(w * scale)
                gray = cv2.resize(gray, (new_w, self.rec_input_height))
                
                crops.append(gray)
                valid_indices.append(i)
                
            except Exception:
                crops.append(None)
        
        # 2. Batch halinde isle
        text_boxes = [TextBox(box=box, text="", confidence=0.0) for box in boxes]
        
        # Valid crop'lari batch'le
        valid_crops = [(valid_indices[j], crops[valid_indices[j]]) 
                       for j in range(len(valid_indices)) if crops[valid_indices[j]] is not None]
        
        if not valid_crops:
            return text_boxes
        
        # Batch'lere bol
        for batch_start in range(0, len(valid_crops), batch_size):
            batch_items = valid_crops[batch_start:batch_start + batch_size]
            batch_indices = [item[0] for item in batch_items]
            batch_crops = [item[1] for item in batch_items]
            
            # Batch icindeki tum crop'lari ayni genislige getir
            # Variable width: max_width'e kadar izin ver
            target_width = self.rec_max_width if self.variable_width else self.rec_input_width
            max_width = min(
                max(crop.shape[1] for crop in batch_crops),
                target_width
            )
            
            # Batch tensor olustur
            batch_tensors = []
            for crop in batch_crops:
                # Truncate veya pad
                h, w = crop.shape[:2]
                if w > max_width:
                    crop = crop[:, :max_width]
                elif w < max_width:
                    padded = np.zeros((h, max_width), dtype=crop.dtype)
                    padded[:, :w] = crop
                    crop = padded
                
                # Normalize
                normalized = crop.astype(np.float32) / 255.0
                batch_tensors.append(normalized)
            
            # Stack: [B, H, W] -> [B, 1, H, W]
            batch_tensor = np.stack(batch_tensors, axis=0)
            batch_tensor = torch.from_numpy(batch_tensor).unsqueeze(1)
            batch_tensor = batch_tensor.to(self.device)
            
            # Recognition (single forward pass for batch)
            log_probs = self.recognition_model(batch_tensor)
            
            # Decode
            texts = self.decoder.decode_greedy(log_probs)
            
            # Confidence hesapla
            probs = torch.exp(log_probs)
            max_probs, _ = probs.max(dim=2)  # [B, T]
            confidences = max_probs.mean(dim=1).cpu().numpy()  # [B]
            
            # Sonuclari yerlestir
            for j, (idx, text) in enumerate(zip(batch_indices, texts)):
                text_boxes[idx] = TextBox(
                    box=boxes[idx],
                    text=text,
                    confidence=float(confidences[j])
                )
        
        return text_boxes
    
    @torch.no_grad()
    def _recognize_single(
        self,
        image: np.ndarray,
        box: np.ndarray
    ) -> Tuple[str, float]:
        """
        Tek bir kutu icin recognition (eski yontem, yedek olarak)
        
        Args:
            image: Kaynak gorsel
            box: Kutu koordinatlari
            
        Returns:
            (text, confidence) tuple
        """
        # Bolgeyi kes
        crop = self.image_processor.crop_polygon(
            image, box,
            target_height=self.rec_input_height
        )
        
        if crop.size == 0:
            return "", 0.0
        
        # Grayscale'e cevir
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        # Pad to width
        padded = self.image_processor.pad_to_width(
            gray,
            self.rec_input_width,
            pad_value=0
        )
        
        # Normalize
        normalized = padded.astype(np.float32) / 255.0
        
        # Tensor'e cevir: [B, C, H, W]
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        # Recognition
        log_probs = self.recognition_model(tensor)
        
        # Decode
        texts = self.decoder.decode_greedy(log_probs)
        text = texts[0] if texts else ""
        
        # Confidence
        probs = torch.exp(log_probs)
        max_probs, _ = probs.max(dim=2)
        confidence = float(max_probs.mean())
        
        return text, confidence
    
    def visualize(
        self,
        image: Union[str, Path, np.ndarray],
        result: OCRResult,
        show_text: bool = True,
        font_scale: float = 0.5,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Sonuclari gorsel uzerinde goster
        
        Args:
            image: Gorsel
            result: OCR sonucu
            show_text: Metni goster
            font_scale: Font boyutu
            thickness: Cizgi kalinligi
            
        Returns:
            Sonuclari gosterilen gorsel
        """
        if isinstance(image, (str, Path)):
            image = self.image_processor.load_image(str(image))
        
        vis = image.copy()
        
        for tb in result.text_boxes:
            # Kutuyu ciz
            pts = tb.box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), thickness)
            
            # Metni yaz
            if show_text and tb.text:
                x, y = tb.x1, tb.y1 - 5
                if y < 15:
                    y = tb.y2 + 15
                
                # Arka plan
                (w, h), _ = cv2.getTextSize(
                    tb.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                cv2.rectangle(vis, (x, y - h - 5), (x + w + 5, y + 5), (0, 255, 0), -1)
                
                # Metin
                cv2.putText(
                    vis, tb.text,
                    (x + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    1
                )
        
        return vis
    
    def __call__(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> OCRResult:
        """Kisayol: recognize() metodunu cagir"""
        return self.recognize(image)


def create_pipeline(
    config_path: Optional[str] = None,
    device: Optional[str] = None
) -> OCRPipeline:
    """
    OCR pipeline olustur
    
    Args:
        config_path: Yapilandirma dosyasi
        device: Cihaz (cuda/cpu)
        
    Returns:
        OCRPipeline nesnesi
    """
    return OCRPipeline(config_path=config_path, device=device)
