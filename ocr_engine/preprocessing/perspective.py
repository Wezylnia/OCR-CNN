"""
Perspektif Duzeltme Modulu

Gercek dunya fotograflarinda:
- Dokuman koselerini tespit eder
- Perspektif bozulmasini duzeltir
- Egik/yamuk gorselleri duzlestirir
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import math


class PerspectiveCorrector:
    """
    Gorsel perspektif duzeltici
    
    Kullanim:
        corrector = PerspectiveCorrector()
        corrected = corrector.correct(image)
        
        # Veya sadece koseler:
        corners = corrector.detect_corners(image)
    """
    
    def __init__(
        self,
        min_area_ratio: float = 0.1,
        max_angle_deviation: float = 30.0,
        edge_detection_method: str = 'canny'
    ):
        """
        Args:
            min_area_ratio: Minimum alan orani (gorsel alanina gore)
            max_angle_deviation: Maksimum aci sapmasi (derece)
            edge_detection_method: Kenar tespiti yontemi ('canny', 'adaptive')
        """
        self.min_area_ratio = min_area_ratio
        self.max_angle_deviation = max_angle_deviation
        self.edge_detection_method = edge_detection_method
    
    def correct(
        self,
        image: np.ndarray,
        corners: Optional[np.ndarray] = None,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Perspektif duzeltmesi uygula
        
        Args:
            image: Girdi gorseli (BGR)
            corners: Opsiyonel - bilinen koseler [4, 2]
            target_size: Opsiyonel - hedef boyut (width, height)
            
        Returns:
            Duzeltilmis gorsel
        """
        h, w = image.shape[:2]
        
        # Koseler verilmemisse tespit et
        if corners is None:
            corners = self.detect_corners(image)
        
        if corners is None:
            # Kose bulunamadi, orijinal gorseli dondur
            return image
        
        # Koseleri sirala (sol-ust, sag-ust, sag-alt, sol-alt)
        corners = self._order_corners(corners)
        
        # Hedef boyutu hesapla
        if target_size is None:
            target_size = self._calculate_target_size(corners)
        
        target_w, target_h = target_size
        
        # Hedef koseler
        dst_corners = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype=np.float32)
        
        # Perspektif matrisi
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
        
        # Donusumu uygula
        corrected = cv2.warpPerspective(
            image, matrix, (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return corrected
    
    def detect_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Dokuman/kart koselerini tespit et
        
        Args:
            image: Girdi gorseli
            
        Returns:
            4 kose noktasi [4, 2] veya None
        """
        h, w = image.shape[:2]
        min_area = h * w * self.min_area_ratio
        
        # Gri tonlama
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Kenar tespiti
        edges = self._detect_edges(gray)
        
        # Kontur bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # En buyuk konturu bul
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:5]:  # Ilk 5 kontura bak
            area = cv2.contourArea(contour)
            
            if area < min_area:
                continue
            
            # Polygon yaklasimi
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # 4 koseli mi?
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                
                # Gecerli dikdortgen mi kontrol et
                if self._is_valid_quadrilateral(corners, w, h):
                    return corners
        
        # Hough Lines ile alternatif yontem
        corners = self._detect_corners_hough(edges, w, h)
        if corners is not None:
            return corners
        
        return None
    
    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Kenar tespiti"""
        # Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if self.edge_detection_method == 'canny':
            # Otsu ile threshold belirle
            high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            low_thresh = high_thresh * 0.5
            
            edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        elif self.edge_detection_method == 'adaptive':
            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            edges = cv2.Canny(binary, 50, 150)
        
        else:
            edges = cv2.Canny(blurred, 50, 150)
        
        # Morfolojik islemler
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        return edges
    
    def _detect_corners_hough(
        self,
        edges: np.ndarray,
        width: int,
        height: int
    ) -> Optional[np.ndarray]:
        """Hough Lines ile kose tespiti"""
        # Cizgileri bul
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 100,
            minLineLength=min(width, height) * 0.3,
            maxLineGap=20
        )
        
        if lines is None or len(lines) < 4:
            return None
        
        # Cizgileri yatay ve dikey olarak grupla
        horizontal = []
        vertical = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            if abs(angle) < 30 or abs(angle) > 150:
                horizontal.append(line[0])
            elif 60 < abs(angle) < 120:
                vertical.append(line[0])
        
        if len(horizontal) < 2 or len(vertical) < 2:
            return None
        
        # Ust/alt yatay cizgiler
        horizontal = sorted(horizontal, key=lambda l: (l[1] + l[3]) / 2)
        top_line = horizontal[0]
        bottom_line = horizontal[-1]
        
        # Sol/sag dikey cizgiler
        vertical = sorted(vertical, key=lambda l: (l[0] + l[2]) / 2)
        left_line = vertical[0]
        right_line = vertical[-1]
        
        # Kesisim noktalarini bul
        corners = []
        for h_line in [top_line, bottom_line]:
            for v_line in [left_line, right_line]:
                intersection = self._line_intersection(h_line, v_line)
                if intersection is not None:
                    corners.append(intersection)
        
        if len(corners) != 4:
            return None
        
        return np.array(corners, dtype=np.float32)
    
    def _line_intersection(
        self,
        line1: np.ndarray,
        line2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Iki cizginin kesisim noktasi"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-6:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (x, y)
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Koseleri saat yonunde sirala:
        0: sol-ust, 1: sag-ust, 2: sag-alt, 3: sol-alt
        """
        # Merkez
        center = corners.mean(axis=0)
        
        # Her koseyi merkeze gore aciyla sirala
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        indices = np.argsort(angles)
        
        ordered = corners[indices]
        
        # Sol-ust'u bul (x+y en kucuk)
        sums = ordered[:, 0] + ordered[:, 1]
        start_idx = np.argmin(sums)
        
        # Koseleri dondur
        ordered = np.roll(ordered, -start_idx, axis=0)
        
        return ordered
    
    def _is_valid_quadrilateral(
        self,
        corners: np.ndarray,
        img_width: int,
        img_height: int
    ) -> bool:
        """Gecerli bir dortgen mi kontrol et"""
        # Tum koseler gorsel icinde mi
        for corner in corners:
            x, y = corner
            if x < 0 or x > img_width or y < 0 or y > img_height:
                return False
        
        # Alan kontrolu
        area = cv2.contourArea(corners)
        img_area = img_width * img_height
        
        if area < img_area * self.min_area_ratio:
            return False
        
        # Aci kontrolu (cok egik mi)
        angles = self._calculate_angles(corners)
        for angle in angles:
            if abs(angle - 90) > self.max_angle_deviation:
                return False
        
        return True
    
    def _calculate_angles(self, corners: np.ndarray) -> List[float]:
        """Dortgenin ic acilarini hesapla"""
        angles = []
        n = len(corners)
        
        for i in range(n):
            p1 = corners[(i - 1) % n]
            p2 = corners[i]
            p3 = corners[(i + 1) % n]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
            angles.append(angle)
        
        return angles
    
    def _calculate_target_size(self, corners: np.ndarray) -> Tuple[int, int]:
        """Hedef boyutu hesapla"""
        # Kenar uzunluklari
        width1 = np.linalg.norm(corners[1] - corners[0])
        width2 = np.linalg.norm(corners[2] - corners[3])
        height1 = np.linalg.norm(corners[3] - corners[0])
        height2 = np.linalg.norm(corners[2] - corners[1])
        
        width = int(max(width1, width2))
        height = int(max(height1, height2))
        
        return (width, height)


class DocumentScanner:
    """
    Dokuman tarayici
    
    Ozellikler:
    - Otomatik dokuman tespiti
    - Perspektif duzeltme
    - Kontrast iyilestirme
    """
    
    def __init__(self):
        self.corrector = PerspectiveCorrector()
    
    def scan(
        self,
        image: np.ndarray,
        enhance: bool = True
    ) -> np.ndarray:
        """
        Dokumani tara ve duzelt
        
        Args:
            image: Girdi gorseli
            enhance: Kontrast iyilestirmesi uygula
            
        Returns:
            Taranan dokuman gorseli
        """
        # Perspektif duzelt
        scanned = self.corrector.correct(image)
        
        # Iyilestirme
        if enhance:
            scanned = self._enhance(scanned)
        
        return scanned
    
    def _enhance(self, image: np.ndarray) -> np.ndarray:
        """Dokuman kalitesini iyilestir"""
        # Gri tonlama
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # BGR'ye geri don
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced


def auto_correct_perspective(image: np.ndarray) -> np.ndarray:
    """
    Otomatik perspektif duzeltme (convenience function)
    
    Args:
        image: Girdi gorseli
        
    Returns:
        Duzeltilmis gorsel
    """
    corrector = PerspectiveCorrector()
    return corrector.correct(image)


def detect_document_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Dokuman koselerini tespit et (convenience function)
    
    Args:
        image: Girdi gorseli
        
    Returns:
        4 kose noktasi [4, 2] veya None
    """
    corrector = PerspectiveCorrector()
    return corrector.detect_corners(image)
