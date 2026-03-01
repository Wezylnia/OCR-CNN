"""
Sentetik metin gorseli uretici
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import random
from PIL import Image, ImageDraw, ImageFont

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
        
        # Font dosya yollari - her generate() cagrisinda rastgele sec
        self._font_paths: List[str] = self._collect_font_paths(font_paths)

    def _collect_font_paths(
        self,
        extra_paths: Optional[List[str]] = None
    ) -> List[str]:
        """
        Kullanilabilir tum TTF/OTF dosyalarini topla.

        Oncelik sirasi:
          1. Kullanici tarafindan verilen yollar
          2. data/fonts/ (alt klasorler dahil)
          3. C:/Windows/Fonts  (Windows sisteminde 400+ font, hepsi Turkce destekli)

        Fontlar bellegde TUTULMAZ — her uretimde dosya yolundan on-the-fly acilir,
        bu sayede bellek kullanimi minimal kalir ve maksimum cesitlilik saglanir.
        """
        from pathlib import Path as _Path
        paths: List[str] = []

        seen: set = set()

        def add(p: str) -> None:
            if p not in seen:
                seen.add(p)
                paths.append(p)

        # 1. Kullanici tarafindan verilen yollar
        if extra_paths:
            for p in extra_paths:
                if _Path(p).exists():
                    add(p)

        # 2. data/fonts/ (NotoSans ve diger proje fontlari)
        fonts_dir = _Path(__file__).parent.parent / 'data' / 'fonts'
        if fonts_dir.exists():
            for ext in ('*.ttf', '*.otf', '*.TTF', '*.OTF'):
                for f in sorted(fonts_dir.rglob(ext)):
                    add(str(f))

        # 3. Windows sistem fontlari (Arial, Times, Calibri, Georgia, Courier…)
        # Sembol / dekoratif fontlar atlandi (belge OCR'da kullanilmaz)
        _SKIP_KEYWORDS = (
            'wing', 'ding', 'symbol', 'webding', 'marlett',
            'mtextra', 'bssym', 'symbol', 'emoji',
        )
        win_fonts = _Path('C:/Windows/Fonts')
        if win_fonts.exists():
            for ext in ('*.ttf', '*.otf'):
                for f in sorted(win_fonts.glob(ext)):
                    name_lower = f.name.lower()
                    if any(kw in name_lower for kw in _SKIP_KEYWORDS):
                        continue
                    add(str(f))

        if not paths:
            # Son care: PIL dahili bitmap fontu
            return []  # generate() None kontrolu yapar

        return paths

    
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
        
        # Font sec — dosya yolundan on-the-fly olustur (dusuk bellek, yuksek cesitlilik)
        size = random.randint(self.font_sizes[0], self.font_sizes[1])
        if self._font_paths:
            font_path = random.choice(self._font_paths)
            try:
                pil_font = ImageFont.truetype(font_path, size)
            except (IOError, OSError):
                pil_font = ImageFont.load_default()
        else:
            try:
                pil_font = ImageFont.load_default(size=size)
            except TypeError:
                pil_font = ImageFont.load_default()

        # Renkler sec
        bg_color = random.choice(self.bg_colors)
        text_color = random.choice(self.text_colors)

        # PIL ile metin olcut hesapla
        dummy = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        bbox = dummy.textbbox((0, 0), text, font=pil_font)
        text_w = max(bbox[2] - bbox[0], 1)
        text_h = max(bbox[3] - bbox[1], 1)

        # Padding
        padding = random.randint(5, 15)

        # Gorsel boyutu
        img_h = text_h + 2 * padding
        img_w = text_w + 2 * padding

        # PIL gorsel olustur ve metni yaz
        pil_img = Image.new('RGB', (img_w, img_h), bg_color)
        draw = ImageDraw.Draw(pil_img)
        draw.text((padding - bbox[0], padding - bbox[1]), text, font=pil_font, fill=text_color)

        # numpy'a cevir
        image = np.array(pil_img)

        # Hedef yukseklige boyutlandir
        scale = self.image_height / img_h
        new_w = max(int(img_w * scale), 1)
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
            # Kelime benzeli
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
