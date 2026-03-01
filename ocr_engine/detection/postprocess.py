"""
DBNet post-processing - bounding box cikarimi ve NMS
"""

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from typing import List, Tuple, Optional, Union


class DBPostProcessor:
    """DBNet cikislarindan metin kutularini cikarir"""
    
    def __init__(
        self,
        threshold: float = 0.3,
        box_threshold: float = 0.5,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        min_size: int = 3,
        use_polygon: bool = False
    ):
        """
        Args:
            threshold: Probability map threshold
            box_threshold: Box score threshold
            max_candidates: Maksimum kutu adayi sayisi
            unclip_ratio: Kutu genisletme orani
            min_size: Minimum kutu boyutu
            use_polygon: Dortgen yerine poligon kullan
        """
        self.threshold = threshold
        self.box_threshold = box_threshold
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size
        self.use_polygon = use_polygon
    
    def __call__(
        self,
        prob_map: np.ndarray,
        original_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        """
        Post-processing uygula
        
        Args:
            prob_map: Probability map [H, W] veya [1, H, W]
            original_size: Orijinal gorsel boyutu (height, width)
            
        Returns:
            Bounding box listesi, her biri [4, 2] veya [N, 2] numpy array
        """
        # Boyut kontrolu
        if len(prob_map.shape) == 3:
            prob_map = prob_map[0]
        
        # Binary mask
        mask = (prob_map > self.threshold).astype(np.uint8)
        
        # Kontur bul
        boxes = self._extract_boxes(prob_map, mask)
        
        # Olcekle
        boxes = self._rescale_boxes(boxes, prob_map.shape, original_size)
        
        return boxes
    
    def _extract_boxes(
        self,
        prob_map: np.ndarray,
        mask: np.ndarray
    ) -> List[np.ndarray]:
        """Konturlerden kutulari cikar"""
        # Konturleri bul
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Alan buyuklugune gore sirala: en buyuk (olasi metin bolgeleri) once gelsin
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        boxes = []
        scores = []
        
        for contour in contours[:self.max_candidates]:
            # Cok kucuk konturlari atla
            if contour.shape[0] < 4:
                continue
            
            # Skor hesapla
            score = self._get_box_score(prob_map, contour)
            if score < self.box_threshold:
                continue
            
            if self.use_polygon:
                # Poligon olarak kullan
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if approx.shape[0] < 4:
                    continue
                
                box = approx.reshape(-1, 2)
            else:
                # Minimum alan dikdortgeni
                box = self._get_min_box(contour)
                
                if box is None:
                    continue
            
            # Unclip (genislet)
            box = self._unclip(box)
            
            if box is None:
                continue
            
            # Boyut kontrolu
            box = self._validate_box(box)
            
            if box is not None:
                boxes.append(box)
                scores.append(score)
        
        # NMS uygula
        if len(boxes) > 0:
            boxes = self._nms(boxes, scores)
        
        return boxes
    
    def _get_box_score(
        self,
        prob_map: np.ndarray,
        contour: np.ndarray
    ) -> float:
        """Kutu icindeki ortalama skoru hesapla"""
        h, w = prob_map.shape
        
        # Maske olustur
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 1)
        
        # Ortalama skor
        return cv2.mean(prob_map, mask)[0]
    
    def _get_min_box(
        self,
        contour: np.ndarray
    ) -> Optional[np.ndarray]:
        """Minimum alan dikdortgeni"""
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        
        # Boyut kontrolu
        w, h = rect[1]
        if min(w, h) < self.min_size:
            return None
        
        # Noktalari sirala (sol-ust'ten baslayarak saat yonunde)
        box = self._order_points(box)
        
        return box.astype(np.float32)
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Noktalari saat yonunde sirala"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sol-ust ve sag-alt
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Sag-ust ve sol-alt
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _unclip(self, box: np.ndarray) -> Optional[np.ndarray]:
        """Kutuyu genislet"""
        try:
            poly = Polygon(box)
            
            if not poly.is_valid:
                return None
            
            distance = poly.area * self.unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(
                box.astype(np.int64).tolist(),
                pyclipper.JT_ROUND,
                pyclipper.ET_CLOSEDPOLYGON
            )
            
            expanded = offset.Execute(distance)
            
            if not expanded:
                return None
            
            expanded = np.array(expanded[0])
            
            if self.use_polygon:
                return expanded.astype(np.float32)
            else:
                # Dikdortgene donustur
                rect = cv2.minAreaRect(expanded)
                box = cv2.boxPoints(rect)
                box = self._order_points(box)
                return box.astype(np.float32)
                
        except Exception:
            return None
    
    def _validate_box(self, box: np.ndarray) -> Optional[np.ndarray]:
        """Kutu boyutunu dogrula"""
        if self.use_polygon:
            # Poligon alan kontrolu
            if cv2.contourArea(box.astype(np.int32)) < self.min_size ** 2:
                return None
        else:
            # Dikdortgen boyut kontrolu
            w = np.linalg.norm(box[0] - box[1])
            h = np.linalg.norm(box[1] - box[2])
            if min(w, h) < self.min_size:
                return None
        
        return box
    
    def _rescale_boxes(
        self,
        boxes: List[np.ndarray],
        map_size: Tuple[int, int],
        original_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        """Kutulari orijinal boyuta olcekle"""
        if not boxes:
            return boxes
        
        map_h, map_w = map_size
        orig_h, orig_w = original_size
        
        scale_x = orig_w / map_w
        scale_y = orig_h / map_h
        
        scaled_boxes = []
        for box in boxes:
            scaled = box.copy()
            scaled[:, 0] *= scale_x
            scaled[:, 1] *= scale_y
            
            # Sinirlari kontrol et
            scaled[:, 0] = np.clip(scaled[:, 0], 0, orig_w - 1)
            scaled[:, 1] = np.clip(scaled[:, 1], 0, orig_h - 1)
            
            scaled_boxes.append(scaled)
        
        return scaled_boxes
    
    def _nms(
        self,
        boxes: List[np.ndarray],
        scores: List[float],
        threshold: float = 0.5
    ) -> List[np.ndarray]:
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return boxes
        
        # Skorlara gore sirala (buyukten kucuge)
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # IoU hesapla
            rest = indices[1:]
            ious = np.array([
                self._polygon_iou(boxes[current], boxes[i])
                for i in rest
            ])
            
            # Dusuk IoU olanları tut
            indices = rest[ious < threshold]
        
        return [boxes[i] for i in keep]
    
    def _polygon_iou(
        self,
        poly1: np.ndarray,
        poly2: np.ndarray
    ) -> float:
        """Iki poligon arasindaki IoU"""
        try:
            p1 = Polygon(poly1)
            p2 = Polygon(poly2)
            
            if not p1.is_valid or not p2.is_valid:
                return 0.0
            
            intersection = p1.intersection(p2).area
            union = p1.area + p2.area - intersection
            
            if union <= 0:
                return 0.0
            
            return intersection / union
            
        except Exception:
            return 0.0
    
    def visualize(
        self,
        image: np.ndarray,
        boxes: List[np.ndarray],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Kutulari gorsel uzerinde ciz
        
        Args:
            image: Gorsel
            boxes: Kutu listesi
            color: Cizgi rengi (BGR)
            thickness: Cizgi kalinligi
            
        Returns:
            Kutulari cizilmis gorsel
        """
        result = image.copy()
        
        for box in boxes:
            pts = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(result, [pts], True, color, thickness)
        
        return result


def boxes_to_rects(boxes: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
    """
    Poligon kutulari dikdortgene donustur
    
    Args:
        boxes: Poligon kutulari listesi [N, 4, 2]
        
    Returns:
        Dikdortgen listesi [(x, y, w, h), ...]
    """
    rects = []
    for box in boxes:
        x_min = int(np.min(box[:, 0]))
        y_min = int(np.min(box[:, 1]))
        x_max = int(np.max(box[:, 0]))
        y_max = int(np.max(box[:, 1]))
        
        rects.append((x_min, y_min, x_max - x_min, y_max - y_min))
    
    return rects


# ---------------------------------------------------------------------------
# Backward-compat shims – real implementations live in line_grouping.py
# ---------------------------------------------------------------------------
from .line_grouping import (  # noqa: F401
    sort_boxes_by_position,
    get_box_rotation_angle,
    order_points,
    crop_polygon,
    correct_box_rotation,
    AdaptiveLineGrouper,
    adaptive_sort_boxes,
    group_boxes_into_lines,
)
