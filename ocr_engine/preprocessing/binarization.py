"""
Goruntu ikili (binary) hale getirme islemleri
"""

import cv2
import numpy as np
from typing import Optional


class Binarizer:
    """Goruntu binarization sinifi"""
    
    def __init__(
        self,
        method: str = "adaptive",
        block_size: int = 11,
        c: int = 2
    ):
        """
        Args:
            method: Binarization yontemi (adaptive, otsu, sauvola, niblack)
            block_size: Adaptive threshold icin blok boyutu (tek sayi olmali)
            c: Adaptive threshold icin sabit deger
        """
        self.method = method
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.c = c
    
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Gorseli ikili hale getirir
        
        Args:
            image: Giris gorseli (gri tonlama veya renkli)
            
        Returns:
            Binary gorsel (0 ve 255)
        """
        # Gri tonlamaya cevir
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        fn = getattr(self, f'_{self.method}_threshold', None)
        if fn is None:
            raise ValueError(f"Bilinmeyen binarization yontemi: {self.method}")
        return fn(gray)
    
    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """OpenCV adaptive threshold"""
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.c
        )
    
    def _otsu_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Otsu's threshold"""
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary
    
    def _sauvola_threshold(
        self,
        gray: np.ndarray,
        k: float = 0.5,
        r: float = 128
    ) -> np.ndarray:
        """
        Sauvola threshold - dokuman gorselleri icin daha iyi sonuc verir
        
        T(x,y) = mean(x,y) * (1 + k * ((std(x,y) / r) - 1))
        """
        # Lokal ortalama
        mean = cv2.blur(gray.astype(np.float64), (self.block_size, self.block_size))
        
        # Lokal standart sapma
        mean_sq = cv2.blur(gray.astype(np.float64) ** 2, (self.block_size, self.block_size))
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
        
        # Sauvola threshold
        threshold = mean * (1 + k * ((std / r) - 1))
        
        binary = np.zeros_like(gray)
        binary[gray > threshold] = 255
        
        return binary.astype(np.uint8)
    
    def _niblack_threshold(
        self,
        gray: np.ndarray,
        k: float = -0.2
    ) -> np.ndarray:
        """
        Niblack threshold
        
        T(x,y) = mean(x,y) + k * std(x,y)
        """
        # Lokal ortalama
        mean = cv2.blur(gray.astype(np.float64), (self.block_size, self.block_size))
        
        # Lokal standart sapma
        mean_sq = cv2.blur(gray.astype(np.float64) ** 2, (self.block_size, self.block_size))
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
        
        # Niblack threshold
        threshold = mean + k * std
        
        binary = np.zeros_like(gray)
        binary[gray > threshold] = 255
        
        return binary.astype(np.uint8)
    
    def morphological_clean(
        self,
        binary: np.ndarray,
        kernel_size: int = 3,
        iterations: int = 1
    ) -> np.ndarray:
        """
        Morfolojik islemlerle gurultu temizleme
        
        Args:
            binary: Binary gorsel
            kernel_size: Kernel boyutu
            iterations: Iterasyon sayisi
            
        Returns:
            Temizlenmis binary gorsel
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size)
        )
        
        # Opening (erosion -> dilation) - kucuk noktalar temizler
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        # Closing (dilation -> erosion) - kucuk bosluklari kapatir
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        return cleaned
    
    def remove_small_components(
        self,
        binary: np.ndarray,
        min_size: int = 50
    ) -> np.ndarray:
        """
        Kucuk baglantili bilesenleri kaldir
        
        Args:
            binary: Binary gorsel
            min_size: Minimum bilesen boyutu (piksel)
            
        Returns:
            Filtrelenmis binary gorsel
        """
        # Baglantili bilesenler
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
        
        # Yeni gorsel olustur
        cleaned = np.zeros_like(binary)
        
        for i in range(1, num_labels):  # 0 arka plan
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned[labels == i] = 255
        
        return cleaned
