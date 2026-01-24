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


def sort_boxes_by_position(
    boxes: List[np.ndarray],
    line_threshold: int = 10
) -> List[np.ndarray]:
    """
    Kutulari okuma sirasina gore sirala (sol-ustten sag-alta)
    
    Args:
        boxes: Kutu listesi
        line_threshold: Ayni satir icin Y toleransi
        
    Returns:
        Siralanmis kutu listesi
    """
    if not boxes:
        return boxes
    
    # Y koordinatina gore grupla (satirlar)
    boxes_with_pos = []
    for box in boxes:
        center_y = np.mean(box[:, 1])
        center_x = np.mean(box[:, 0])
        boxes_with_pos.append((box, center_x, center_y))
    
    # Y'ye gore sirala
    boxes_with_pos.sort(key=lambda x: x[2])
    
    # Satirlari grupla
    lines = []
    current_line = [boxes_with_pos[0]]
    current_y = boxes_with_pos[0][2]
    
    for item in boxes_with_pos[1:]:
        if abs(item[2] - current_y) < line_threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
            current_y = item[2]
    
    lines.append(current_line)
    
    # Her satiri X'e gore sirala
    sorted_boxes = []
    for line in lines:
        line.sort(key=lambda x: x[1])
        sorted_boxes.extend([item[0] for item in line])
    
    return sorted_boxes


def get_box_rotation_angle(box: np.ndarray) -> float:
    """
    Kutunun rotasyon acisini hesapla
    
    Args:
        box: 4 noktali polygon [4, 2]
        
    Returns:
        Rotasyon acisi (derece, -45 ile 45 arasi)
    """
    # minAreaRect kullan
    rect = cv2.minAreaRect(box.astype(np.float32))
    angle = rect[2]
    
    # OpenCV minAreaRect'in acisini normalize et
    width, height = rect[1]
    
    if width < height:
        angle = angle - 90
    
    # -45 ile 45 arasina getir
    while angle < -45:
        angle += 90
    while angle > 45:
        angle -= 90
    
    return angle


def correct_box_rotation(
    image: np.ndarray,
    box: np.ndarray,
    angle_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Text region icin rotasyon duzeltmesi
    
    Args:
        image: Kaynak gorsel
        box: Text bounding box [4, 2]
        angle_threshold: Duzeltme yapilacak minimum aci
        
    Returns:
        (duzeltilmis_crop, yeni_box)
    """
    import math
    
    # Rotasyon acisini hesapla
    angle = get_box_rotation_angle(box)
    
    # Kucuk acilari atla
    if abs(angle) < angle_threshold:
        # Direkt crop yap
        return crop_polygon(image, box), box
    
    # Bounding rect
    x_min = int(max(0, np.min(box[:, 0]) - 5))
    y_min = int(max(0, np.min(box[:, 1]) - 5))
    x_max = int(min(image.shape[1], np.max(box[:, 0]) + 5))
    y_max = int(min(image.shape[0], np.max(box[:, 1]) + 5))
    
    # Region crop
    region = image[y_min:y_max, x_min:x_max]
    if region.size == 0:
        return crop_polygon(image, box), box
    
    # Local box koordinatlari
    local_box = box.copy()
    local_box[:, 0] -= x_min
    local_box[:, 1] -= y_min
    
    # Rotation center
    h, w = region.shape[:2]
    center = (w / 2, h / 2)
    
    # Rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Yeni boyutu hesapla
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Matrix'i guncelle
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    
    # Gorseli dondur
    rotated = cv2.warpAffine(region, matrix, (new_w, new_h), borderValue=(255, 255, 255))
    
    # Box'i dondur
    ones = np.ones((4, 1))
    points = np.hstack([local_box, ones])
    new_box = (matrix @ points.T).T.astype(np.float32)
    
    # Yeni box'tan crop
    x1 = max(0, int(np.min(new_box[:, 0])))
    y1 = max(0, int(np.min(new_box[:, 1])))
    x2 = min(new_w, int(np.max(new_box[:, 0])))
    y2 = min(new_h, int(np.max(new_box[:, 1])))
    
    cropped = rotated[y1:y2, x1:x2]
    
    if cropped.size == 0:
        return crop_polygon(image, box), box
    
    return cropped, new_box


def crop_polygon(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Polygon seklindeki bolgeden crop al (perspektif duzeltmeli)
    
    Args:
        image: Kaynak gorsel
        box: 4 noktali polygon [4, 2]
        
    Returns:
        Crop edilmis gorsel
    """
    # Siralı koseler (sol-ust, sag-ust, sag-alt, sol-alt)
    pts = order_points(box)
    
    # Hedef boyut
    width = int(max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3])
    ))
    height = int(max(
        np.linalg.norm(pts[0] - pts[3]),
        np.linalg.norm(pts[1] - pts[2])
    ))
    
    if width <= 0 or height <= 0:
        return np.zeros((32, 100, 3), dtype=np.uint8)
    
    # Hedef noktalar
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # Perspektif transform
    matrix = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    
    return warped


def order_points(pts: np.ndarray) -> np.ndarray:
    """Noktalari saat yonunde sirala (sol-ust'ten baslayarak)"""
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # sol-ust
    rect[2] = pts[np.argmax(s)]  # sag-alt
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # sag-ust
    rect[3] = pts[np.argmax(diff)]  # sol-alt
    
    return rect


class AdaptiveLineGrouper:
    """
    Adaptif satir gruplama
    
    Sabit threshold yerine box yuksekliklerine gore dinamik gruplama yapar.
    """
    
    def __init__(
        self,
        overlap_threshold: float = 0.5,
        y_tolerance_ratio: float = 0.5
    ):
        """
        Args:
            overlap_threshold: Ayni satir icin minimum Y overlap orani
            y_tolerance_ratio: Box yuksekligine gore Y toleransi orani
        """
        self.overlap_threshold = overlap_threshold
        self.y_tolerance_ratio = y_tolerance_ratio
    
    def group_into_lines(
        self,
        boxes: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        """
        Kutulari satirlara grupla
        
        Args:
            boxes: Kutu listesi
            
        Returns:
            Satir listesi, her satir kutu listesi icerir
        """
        if not boxes:
            return []
        
        # Her box icin y-range hesapla
        box_info = []
        for box in boxes:
            y_min = np.min(box[:, 1])
            y_max = np.max(box[:, 1])
            x_center = np.mean(box[:, 0])
            height = y_max - y_min
            box_info.append({
                'box': box,
                'y_min': y_min,
                'y_max': y_max,
                'y_center': (y_min + y_max) / 2,
                'x_center': x_center,
                'height': height
            })
        
        # Y center'a gore sirala
        box_info.sort(key=lambda x: x['y_center'])
        
        # Satirlara grupla
        lines = []
        used = set()
        
        for i, info in enumerate(box_info):
            if i in used:
                continue
            
            # Yeni satir baslat
            current_line = [info]
            used.add(i)
            
            # Bu satira ait diger kutulari bul
            for j, other in enumerate(box_info):
                if j in used:
                    continue
                
                # Ayni satirda mi kontrol et
                if self._is_same_line(info, other):
                    current_line.append(other)
                    used.add(j)
            
            lines.append(current_line)
        
        # Her satiri X'e gore sirala
        sorted_lines = []
        for line in lines:
            line.sort(key=lambda x: x['x_center'])
            sorted_lines.append([item['box'] for item in line])
        
        # Satirlari Y'ye gore sirala
        sorted_lines.sort(key=lambda line: np.mean([np.mean(box[:, 1]) for box in line]))
        
        return sorted_lines
    
    def _is_same_line(self, box1: dict, box2: dict) -> bool:
        """Iki kutunun ayni satirda olup olmadigini kontrol et"""
        # Y-overlap hesapla
        overlap_start = max(box1['y_min'], box2['y_min'])
        overlap_end = min(box1['y_max'], box2['y_max'])
        overlap = max(0, overlap_end - overlap_start)
        
        # Her iki box'un yuksekligine gore normalize et
        height1 = box1['height']
        height2 = box2['height']
        min_height = min(height1, height2)
        
        if min_height <= 0:
            return False
        
        overlap_ratio = overlap / min_height
        
        # Y-center farki kontrolu
        avg_height = (height1 + height2) / 2
        y_diff = abs(box1['y_center'] - box2['y_center'])
        y_tolerance = avg_height * self.y_tolerance_ratio
        
        return overlap_ratio >= self.overlap_threshold or y_diff <= y_tolerance
    
    def group_and_sort(
        self,
        boxes: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Kutulari satirlara grupla ve duz liste olarak dondur
        
        Args:
            boxes: Kutu listesi
            
        Returns:
            Okuma sirasina gore siralanmis kutu listesi
        """
        lines = self.group_into_lines(boxes)
        
        result = []
        for line in lines:
            result.extend(line)
        
        return result


def adaptive_sort_boxes(boxes: List[np.ndarray]) -> List[np.ndarray]:
    """
    Kutulari adaptif olarak okuma sirasina gore sirala
    
    Args:
        boxes: Kutu listesi
        
    Returns:
        Siralanmis kutu listesi
    """
    grouper = AdaptiveLineGrouper()
    return grouper.group_and_sort(boxes)


def group_boxes_into_lines(boxes: List[np.ndarray]) -> List[List[np.ndarray]]:
    """
    Kutulari satirlara grupla
    
    Args:
        boxes: Kutu listesi
        
    Returns:
        Satir listesi
    """
    grouper = AdaptiveLineGrouper()
    return grouper.group_into_lines(boxes)
