"""
Veri artirma (data augmentation) islemleri
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Callable
import random


class Augmentor:
    """Veri artirma sinifi"""
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-10, 10),
        scale_range: Tuple[float, float] = (0.8, 1.2),
        blur_prob: float = 0.3,
        noise_prob: float = 0.3,
        brightness_range: Tuple[float, float] = (-30, 30),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        perspective_prob: float = 0.2,
        erosion_dilation_prob: float = 0.2
    ):
        """
        Args:
            rotation_range: Donme acisi araligi (derece)
            scale_range: Olcekleme araligi
            blur_prob: Blur uygulama olasiligi
            noise_prob: Gurultu ekleme olasiligi
            brightness_range: Parlaklik araligi
            contrast_range: Kontrast araligi
            perspective_prob: Perspektif bozulma olasiligi
            erosion_dilation_prob: Erosion/dilation olasiligi
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.perspective_prob = perspective_prob
        self.erosion_dilation_prob = erosion_dilation_prob
    
    def __call__(
        self,
        image: np.ndarray,
        boxes: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Augmentation uygula
        
        Args:
            image: Giris gorseli
            boxes: Bounding box listesi (opsiyonel)
            
        Returns:
            Augmented gorsel ve (varsa) guncellanmis kutular
        """
        # Geometrik donusumler
        if random.random() < 0.5:
            image, boxes = self.rotate(image, boxes)
        
        if random.random() < 0.3:
            image, boxes = self.scale(image, boxes)
        
        if random.random() < self.perspective_prob:
            image, boxes = self.perspective(image, boxes)
        
        # Renk donusumleri
        if random.random() < 0.5:
            image = self.adjust_brightness_contrast(image)
        
        # Blur
        if random.random() < self.blur_prob:
            image = self.blur(image)
        
        # Gurultu
        if random.random() < self.noise_prob:
            image = self.add_noise(image)
        
        # Morfolojik islemler
        if random.random() < self.erosion_dilation_prob:
            image = self.morphological(image)
        
        return image, boxes
    
    def rotate(
        self,
        image: np.ndarray,
        boxes: Optional[List[np.ndarray]] = None,
        angle: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """Donme uygula"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        if angle is None:
            angle = random.uniform(*self.rotation_range)
        
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        if boxes is not None:
            new_boxes = []
            for box in boxes:
                # Her noktayi donustur
                ones = np.ones((box.shape[0], 1))
                points = np.hstack([box, ones])
                new_points = points @ matrix.T
                new_boxes.append(new_points.astype(np.float32))
            boxes = new_boxes
        
        return rotated, boxes
    
    def scale(
        self,
        image: np.ndarray,
        boxes: Optional[List[np.ndarray]] = None,
        scale: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """Olcekleme uygula"""
        if scale is None:
            scale = random.uniform(*self.scale_range)
        
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        scaled = cv2.resize(image, (new_w, new_h))
        
        if boxes is not None:
            boxes = [box * scale for box in boxes]
        
        return scaled, boxes
    
    def perspective(
        self,
        image: np.ndarray,
        boxes: Optional[List[np.ndarray]] = None,
        strength: float = 0.1
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """Perspektif bozulma uygula"""
        h, w = image.shape[:2]
        
        # Kaynak noktalar
        src_pts = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ])
        
        # Hedef noktalar (rastgele kaydirma)
        offset = int(min(w, h) * strength)
        dst_pts = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [w - 1 - random.randint(0, offset), random.randint(0, offset)],
            [w - 1 - random.randint(0, offset), h - 1 - random.randint(0, offset)],
            [random.randint(0, offset), h - 1 - random.randint(0, offset)]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (w, h))
        
        if boxes is not None:
            new_boxes = []
            for box in boxes:
                ones = np.ones((box.shape[0], 1))
                points = np.hstack([box, ones])
                new_points = (matrix @ points.T).T
                new_points = new_points[:, :2] / new_points[:, 2:3]
                new_boxes.append(new_points.astype(np.float32))
            boxes = new_boxes
        
        return warped, boxes
    
    def adjust_brightness_contrast(
        self,
        image: np.ndarray,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None
    ) -> np.ndarray:
        """Parlaklik ve kontrast ayarla"""
        if brightness is None:
            brightness = random.uniform(*self.brightness_range)
        if contrast is None:
            contrast = random.uniform(*self.contrast_range)
        
        adjusted = image.astype(np.float32)
        adjusted = contrast * adjusted + brightness
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def blur(
        self,
        image: np.ndarray,
        blur_type: Optional[str] = None
    ) -> np.ndarray:
        """Blur uygula"""
        if blur_type is None:
            blur_type = random.choice(['gaussian', 'motion', 'median'])
        
        if blur_type == 'gaussian':
            ksize = random.choice([3, 5, 7])
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        
        elif blur_type == 'motion':
            ksize = random.randint(5, 15)
            kernel = np.zeros((ksize, ksize))
            kernel[ksize // 2, :] = 1 / ksize
            return cv2.filter2D(image, -1, kernel)
        
        elif blur_type == 'median':
            ksize = random.choice([3, 5])
            return cv2.medianBlur(image, ksize)
        
        return image
    
    def add_noise(
        self,
        image: np.ndarray,
        noise_type: Optional[str] = None
    ) -> np.ndarray:
        """Gurultu ekle"""
        if noise_type is None:
            noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
        
        if noise_type == 'gaussian':
            std = random.uniform(5, 25)
            noise = np.random.normal(0, std, image.shape).astype(np.float32)
            noisy = image.astype(np.float32) + noise
            return np.clip(noisy, 0, 255).astype(np.uint8)
        
        elif noise_type == 'salt_pepper':
            prob = random.uniform(0.01, 0.05)
            noisy = image.copy()
            
            # Salt
            salt = np.random.random(image.shape[:2]) < prob / 2
            noisy[salt] = 255
            
            # Pepper
            pepper = np.random.random(image.shape[:2]) < prob / 2
            noisy[pepper] = 0
            
            return noisy
        
        elif noise_type == 'speckle':
            noise = np.random.randn(*image.shape) * 0.1
            noisy = image + image * noise
            return np.clip(noisy, 0, 255).astype(np.uint8)
        
        return image
    
    def morphological(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Morfolojik islem uygula"""
        operation = random.choice(['erosion', 'dilation'])
        ksize = random.choice([2, 3])
        kernel = np.ones((ksize, ksize), np.uint8)
        
        if operation == 'erosion':
            return cv2.erode(image, kernel, iterations=1)
        else:
            return cv2.dilate(image, kernel, iterations=1)


class RecognitionAugmentor:
    """Metin tanima icin ozellestirilmis augmentation"""
    
    def __init__(
        self,
        stretch_range: Tuple[float, float] = (0.8, 1.2),
        shear_range: Tuple[float, float] = (-0.1, 0.1),
        noise_prob: float = 0.3,
        blur_prob: float = 0.3,
        invert_prob: float = 0.1
    ):
        self.stretch_range = stretch_range
        self.shear_range = shear_range
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
        self.invert_prob = invert_prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Augmentation uygula - Windows-safe (sadece numpy islemleri)"""
        try:
            # Goruntunun contiguous oldugundan emin ol
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
            
            # Parlaklik ayari (numpy ile)
            if random.random() < 0.3:
                factor = random.uniform(0.7, 1.3)
                image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            
            # Kontrast ayari (numpy ile)
            if random.random() < 0.3:
                factor = random.uniform(0.8, 1.2)
                mean = np.mean(image)
                image = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
            
            # Gurultu (numpy ile - guvenli)
            if random.random() < self.noise_prob:
                std = random.uniform(5, 15)
                noise = np.random.normal(0, std, image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            # Renk ters cevirme (numpy ile - guvenli)
            if random.random() < self.invert_prob:
                image = 255 - image
            
            # Salt & pepper noise
            if random.random() < 0.1:
                noise_mask = np.random.random(image.shape)
                image = np.where(noise_mask < 0.02, 0, image)  # Salt
                image = np.where(noise_mask > 0.98, 255, image)  # Pepper
                image = image.astype(np.uint8)
                
        except Exception as e:
            # Herhangi bir hata durumunda orijinal gorseli dondur
            pass
        
        return image
    
    def horizontal_stretch(self, image: np.ndarray) -> np.ndarray:
        """Yatay germe"""
        factor = random.uniform(*self.stretch_range)
        h, w = image.shape[:2]
        new_w = int(w * factor)
        return cv2.resize(image, (new_w, h))
    
    def shear(self, image: np.ndarray) -> np.ndarray:
        """Shear donusumu"""
        shear_factor = random.uniform(*self.shear_range)
        h, w = image.shape[:2]
        
        matrix = np.array([
            [1, shear_factor, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        
        new_w = w + int(abs(shear_factor) * h)
        return cv2.warpAffine(image, matrix, (new_w, h))


class DetectionAugmentor:
    """
    Detection egitimi icin ozellestirilmis augmentation
    
    - Kutulari da donusturur
    - Geometrik + fotometrik augmentations
    - Windows uyumlu (OpenCV multiprocessing sorunu yok)
    """
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-15, 15),
        scale_range: Tuple[float, float] = (0.8, 1.2),
        flip_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        noise_prob: float = 0.2,
        blur_prob: float = 0.2
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
    
    def __call__(
        self,
        image: np.ndarray,
        boxes: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Augmentation uygula
        
        Args:
            image: BGR gorsel [H, W, 3]
            boxes: Polygon listesi, her biri [4, 2] veya [N, 2]
            
        Returns:
            Augmented gorsel ve kutular
        """
        h, w = image.shape[:2]
        
        # Horizontal flip
        if random.random() < self.flip_prob:
            image = cv2.flip(image, 1)
            boxes = [self._flip_box_horizontal(box, w) for box in boxes]
        
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(*self.rotation_range)
            image, boxes = self._rotate(image, boxes, angle)
        
        # Random scale
        if random.random() < 0.3:
            scale = random.uniform(*self.scale_range)
            image, boxes = self._scale(image, boxes, scale)
        
        # Brightness
        if random.random() < 0.5:
            factor = random.uniform(*self.brightness_range)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        
        # Contrast
        if random.random() < 0.3:
            factor = random.uniform(*self.contrast_range)
            mean = np.mean(image)
            image = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # Gaussian noise
        if random.random() < self.noise_prob:
            std = random.uniform(5, 20)
            noise = np.random.normal(0, std, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Blur (safe version)
        if random.random() < self.blur_prob:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
        # Color jitter
        if random.random() < 0.3 and len(image.shape) == 3:
            # HSV space manipulation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180  # Hue
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)  # Saturation
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return image, boxes
    
    def _flip_box_horizontal(self, box: np.ndarray, width: int) -> np.ndarray:
        """Kutuyu yatay flip et"""
        flipped = box.copy()
        flipped[:, 0] = width - box[:, 0]
        return flipped
    
    def _rotate(
        self,
        image: np.ndarray,
        boxes: List[np.ndarray],
        angle: float
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Gorsel ve kutulari dondur"""
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        # Rotation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Gorsel boyutunu ayarla (kesilmesin)
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Matrix'i guncelle
        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2
        
        # Gorseli dondur
        rotated = cv2.warpAffine(image, matrix, (new_w, new_h), borderValue=(128, 128, 128))
        
        # Kutulari dondur
        new_boxes = []
        for box in boxes:
            ones = np.ones((box.shape[0], 1))
            points = np.hstack([box, ones])
            new_points = (matrix @ points.T).T
            new_boxes.append(new_points.astype(np.float32))
        
        return rotated, new_boxes
    
    def _scale(
        self,
        image: np.ndarray,
        boxes: List[np.ndarray],
        scale: float
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Gorsel ve kutulari olcekle"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        scaled = cv2.resize(image, (new_w, new_h))
        scaled_boxes = [box * scale for box in boxes]
        
        return scaled, scaled_boxes


def create_augmentation_pipeline(
    for_detection: bool = True,
    strength: str = 'medium'
) -> Callable:
    """
    Augmentation pipeline olustur
    
    Args:
        for_detection: Detection icin mi
        strength: Augmentation gucu (light, medium, strong)
        
    Returns:
        Augmentation fonksiyonu
    """
    if strength == 'light':
        params = {
            'rotation_range': (-5, 5),
            'scale_range': (0.9, 1.1),
            'blur_prob': 0.1,
            'noise_prob': 0.1
        }
    elif strength == 'strong':
        params = {
            'rotation_range': (-15, 15),
            'scale_range': (0.7, 1.3),
            'blur_prob': 0.5,
            'noise_prob': 0.5,
            'perspective_prob': 0.3
        }
    else:  # medium
        params = {
            'rotation_range': (-10, 10),
            'scale_range': (0.8, 1.2),
            'blur_prob': 0.3,
            'noise_prob': 0.3
        }
    
    if for_detection:
        return Augmentor(**params)
    else:
        return RecognitionAugmentor()
