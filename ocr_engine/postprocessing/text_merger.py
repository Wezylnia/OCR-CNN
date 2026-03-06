"""
Metin birlestirme modulu - satirlari ve paragraflari birlestirme
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TextLine:
    """Metin satiri"""
    text: str
    box: np.ndarray  # [4, 2]
    confidence: float = 0.0
    
    @property
    def center_y(self) -> float:
        return np.mean(self.box[:, 1])
    
    @property
    def center_x(self) -> float:
        return np.mean(self.box[:, 0])
    
    @property
    def height(self) -> float:
        return np.max(self.box[:, 1]) - np.min(self.box[:, 1])
    
    @property
    def width(self) -> float:
        return np.max(self.box[:, 0]) - np.min(self.box[:, 0])
    
    @property
    def left(self) -> float:
        return np.min(self.box[:, 0])
    
    @property
    def right(self) -> float:
        return np.max(self.box[:, 0])
    
    @property
    def top(self) -> float:
        return np.min(self.box[:, 1])
    
    @property
    def bottom(self) -> float:
        return np.max(self.box[:, 1])


class TextMerger:
    """Metin kutularini birlestirme sinifi"""
    
    def __init__(
        self,
        line_threshold: float = 10.0,
        paragraph_threshold: float = 20.0,
        word_spacing_ratio: float = 1.5
    ):
        """
        Args:
            line_threshold: Ayni satir icin Y toleransi (piksel)
            paragraph_threshold: Ayni paragraf icin Y boslugu (piksel)
            word_spacing_ratio: Kelime araligi orani (karakter genisligine gore)
        """
        self.line_threshold = line_threshold
        self.paragraph_threshold = paragraph_threshold
        self.word_spacing_ratio = word_spacing_ratio
    
    def merge_to_lines(
        self,
        text_boxes: List[TextLine]
    ) -> List[TextLine]:
        """
        Metin kutularini satirlara birlestir
        
        Args:
            text_boxes: Metin kutusu listesi
            
        Returns:
            Birlestirilmis satir listesi
        """
        if not text_boxes:
            return []
        
        # Y koordinatina gore grupla
        lines = self._group_by_line(text_boxes)
        
        # Her satiri birlestir
        merged_lines = []
        for line_boxes in lines:
            merged = self._merge_line(line_boxes)
            merged_lines.append(merged)
        
        return merged_lines
    
    def merge_to_paragraphs(
        self,
        text_boxes: List[TextLine]
    ) -> List[str]:
        """
        Metin kutularini paragraflara birlestir
        
        Args:
            text_boxes: Metin kutusu listesi
            
        Returns:
            Paragraf listesi
        """
        # Oncelikle satirlara birlestir
        lines = self.merge_to_lines(text_boxes)
        
        if not lines:
            return []
        
        # Paragraflara grupla
        paragraphs = []
        current_paragraph = [lines[0].text]
        last_bottom = lines[0].bottom
        
        for line in lines[1:]:
            gap = line.top - last_bottom
            
            if gap > self.paragraph_threshold:
                # Yeni paragraf
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [line.text]
            else:
                # Ayni paragraf
                current_paragraph.append(line.text)
            
            last_bottom = line.bottom
        
        # Son paragraf
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return paragraphs
    
    def merge_to_text(
        self,
        text_boxes: List[TextLine],
        preserve_lines: bool = True
    ) -> str:
        """
        Tum metin kutularini tek metne birlestir
        
        Args:
            text_boxes: Metin kutusu listesi
            preserve_lines: Satir sonlarini koru
            
        Returns:
            Birlestirilmis metin
        """
        if preserve_lines:
            lines = self.merge_to_lines(text_boxes)
            return '\n'.join([line.text for line in lines])
        else:
            paragraphs = self.merge_to_paragraphs(text_boxes)
            return '\n\n'.join(paragraphs)
    
    def _group_by_line(
        self,
        text_boxes: List[TextLine]
    ) -> List[List[TextLine]]:
        """Kutuları satırlara grupla"""
        if not text_boxes:
            return []
        
        # Y koordinatina gore sirala
        sorted_boxes = sorted(text_boxes, key=lambda x: x.center_y)
        
        lines = []
        current_line = [sorted_boxes[0]]
        current_y = sorted_boxes[0].center_y
        
        for box in sorted_boxes[1:]:
            # Ayni satirda mi?
            if abs(box.center_y - current_y) <= self.line_threshold:
                current_line.append(box)
            else:
                # Yeni satir
                lines.append(current_line)
                current_line = [box]
                current_y = box.center_y
        
        # Son satir
        if current_line:
            lines.append(current_line)
        
        # Her satiri X koordinatina gore sirala
        for line in lines:
            line.sort(key=lambda x: x.left)
        
        return lines
    
    def _merge_line(self, boxes: List[TextLine]) -> TextLine:
        """Bir satirdaki kutuları birlestir"""
        if len(boxes) == 1:
            return boxes[0]
        
        # X koordinatina gore sirali olmali
        boxes = sorted(boxes, key=lambda x: x.left)
        
        # Metinleri birlestir
        texts = []
        last_right = boxes[0].left
        avg_char_width = sum(b.width / max(len(b.text), 1) for b in boxes) / len(boxes)
        
        for box in boxes:
            gap = box.left - last_right
            
            # Bosluk ekle
            if gap > avg_char_width * self.word_spacing_ratio:
                texts.append(' ')
            
            texts.append(box.text)
            last_right = box.right
        
        merged_text = ''.join(texts)
        
        # Birlesik kutu
        all_points = np.vstack([b.box for b in boxes])
        (x1, y1), (x2, y2) = all_points.min(axis=0), all_points.max(axis=0)
        merged_box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        
        # Ortalama guven
        avg_confidence = sum(b.confidence for b in boxes) / len(boxes)
        
        return TextLine(
            text=merged_text,
            box=merged_box,
            confidence=avg_confidence
        )

