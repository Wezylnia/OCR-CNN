"""
Goruntu gurultu giderme islemleri
"""

import cv2
import numpy as np
from typing import Tuple


class Denoiser:
    """Goruntu gurultu giderme sinifi"""
    
    def __init__(
        self,
        method: str = "bilateral",
        strength: int = 10
    ):
        """
        Args:
            method: Gurultu giderme yontemi (bilateral, gaussian, nlmeans, median)
            strength: Gurultu giderme gucu (1-20)
        """
        self.method = method
        self.strength = strength
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Goruntudeki gurultuyu giderir
        
        Args:
            image: Giris gorseli
            
        Returns:
            Gurultusu giderilmis gorsel
        """
        fn = getattr(self, f'_{self.method}_filter', None)
        if fn is None:
            raise ValueError(f"Bilinmeyen denoise yontemi: {self.method}")
        return fn(image)
    
    def _bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Bilateral filter - kenarlari koruyarak gurultu giderir
        """
        d = max(5, self.strength // 2)
        sigma_color = self.strength * 7.5
        sigma_space = self.strength * 7.5
        
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def _gaussian_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Gaussian blur - basit gurultu giderme
        """
        kernel_size = max(3, self.strength) | 1  # Tek sayi olmali
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _nlmeans_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Non-Local Means - yuksek kaliteli gurultu giderme (yavas)
        """
        h = self.strength
        template_window = 7
        search_window = 21
        
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h,
                h,
                template_window,
                search_window
            )
        else:
            return cv2.fastNlMeansDenoising(
                image,
                None,
                h,
                template_window,
                search_window
            )
    
    def _median_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Median filter - tuz-biber gurultusune karsi etkili
        """
        kernel_size = max(3, self.strength) | 1  # Tek sayi olmali
        return cv2.medianBlur(image, kernel_size)
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Golgeleri kaldirir (dokuman gorselleri icin)
        
        Args:
            image: Giris gorseli
            
        Returns:
            Golgeleri kaldirilmis gorsel
        """
        if len(image.shape) == 3:
            # RGB -> LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = image.copy()
        
        # Dilasyon ile arka plan tahmini
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        background = cv2.dilate(l, kernel)
        background = cv2.GaussianBlur(background, (31, 31), 0)
        
        # Golge kaldirma
        result = 255 - cv2.absdiff(l, background)
        
        # Kontrast artirma
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        
        if len(image.shape) == 3:
            lab = cv2.merge([result, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def auto_adjust(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Otomatik goruntu iyilestirme
        
        Args:
            image: Giris gorseli
            
        Returns:
            Iyilestirilmis gorsel ve uygulanan islemler
        """
        result = image.copy()
        operations = {}
        
        # Gurultu seviyesini tara
        noise_level = self._estimate_noise(result)
        operations['noise_level'] = noise_level
        
        # Gurultu varsa gider
        if noise_level > 10:
            result = self.denoise(result)
            operations['denoised'] = True
        
        # Kontrast dusukse artir
        contrast = self._estimate_contrast(result)
        operations['contrast'] = contrast
        
        if contrast < 50:
            result = self._improve_contrast(result)
            operations['contrast_improved'] = True
        
        return result, operations
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Gurultu seviyesini tahmin eder"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Laplacian varyans yontemi
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def _estimate_contrast(self, image: np.ndarray) -> float:
        """Kontrast seviyesini tahmin eder"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(gray.std())
    
    def _improve_contrast(self, image: np.ndarray) -> np.ndarray:
        """Kontrasti arttirir"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
