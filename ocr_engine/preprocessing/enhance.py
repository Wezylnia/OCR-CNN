"""
Gelismis goruntu iyilestirme modulu

Dusuk kaliteli, gurultulu, az kontrast veya golge iceren
gorseller icin otomatik iyilestirme teknikleri.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImageEnhancer:
    """
    Goruntu kalitesini artiran islemler.

    Uygulama sirasi:
        1. Goruntu kalitesi olcumu
        2. Golgeden arindirma
        3. CLAHE kontrast iyilestirme
        4. Keskinlestirme
        5. Morfolojik temizleme
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        sharpen_strength: float = 0.5,
        shadow_removal: bool = True,
        auto_mode: bool = True
    ):
        """
        Args:
            clahe_clip_limit: CLAHE clip limit (yukseltce daha fazla kontrast)
            clahe_tile_size: CLAHE pencere boyutu
            sharpen_strength: Keskinlestirme siddeti [0..1]
            shadow_removal: Golge gidermeyi etkinlestir
            auto_mode: Goruntu kalitesine gore otomatik islem sec
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.sharpen_strength = sharpen_strength
        self.shadow_removal = shadow_removal
        self.auto_mode = auto_mode

        # CLAHE nesnesi
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size
        )

    # ------------------------------------------------------------------
    # Ana metot
    # ------------------------------------------------------------------

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Goruntu iyilestirme pipeline'ini uygula.

        Args:
            image: BGR veya gri gorsel

        Returns:
            Iyilestirilmis BGR gorsel (giris ile ayni format)
        """
        is_gray = len(image.shape) == 2

        # BGR'ye cevir
        if is_gray:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image.copy()

        # Kalite olcumu
        quality = self.measure_quality(bgr) if self.auto_mode else None

        # Golge giderme
        if self.shadow_removal:
            bgr = self._remove_shadows(bgr)

        # CLAHE kontrast iyilestirme (LAB uzayinda L kanalina)
        bgr = self._apply_clahe(bgr, quality)

        # Keskinlestirme
        if self.sharpen_strength > 0:
            bgr = self._sharpen(bgr)

        # Morfolojik kapatma (kucuk delikleri doldur)
        bgr = self._morphological_close(bgr)

        if is_gray:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        return bgr

    # ------------------------------------------------------------------
    # Kalite olcumu
    # ------------------------------------------------------------------

    def measure_quality(self, image: np.ndarray) -> dict:
        """
        Goruntu kalite metriklerini hesapla.

        Returns:
            {'sharpness': float, 'contrast': float,
             'brightness': float, 'noise': float, 'score': float}
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Keskinlik — Laplacian varyans
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Kontrast — standart sapma
        contrast = float(gray.std())

        # Parlaklik — ortalama
        brightness = float(gray.mean())

        # Gurultu — yerel varyans farki
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = float(np.abs(gray.astype(np.float32) - blurred.astype(np.float32)).mean())

        # Toplam skor (0..1, yuksek = iyi)
        sharpness_norm = min(sharpness / 500.0, 1.0)
        contrast_norm = min(contrast / 80.0, 1.0)
        brightness_ok = 1.0 - abs(brightness - 128) / 128.0
        noise_ok = max(0.0, 1.0 - noise / 20.0)

        score = 0.35 * sharpness_norm + 0.30 * contrast_norm + \
                0.20 * brightness_ok + 0.15 * noise_ok

        return {
            'sharpness': sharpness,
            'contrast': contrast,
            'brightness': brightness,
            'noise': noise,
            'score': score
        }

    # ------------------------------------------------------------------
    # Golge giderme
    # ------------------------------------------------------------------

    def _remove_shadows(self, bgr: np.ndarray) -> np.ndarray:
        """Duz aydinlatma golgelerini buyuk-pencereli blur ile normalize ederek gider."""
        k = max(bgr.shape[0] // 8, bgr.shape[1] // 8, 31) | 1
        bgr_f = bgr.astype(np.float32)
        bg = cv2.GaussianBlur(bgr_f, (k, k), 0)
        return np.clip(bgr_f / (bg + 1e-6) * 128.0, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # CLAHE
    # ------------------------------------------------------------------

    def _apply_clahe(
        self,
        bgr: np.ndarray,
        quality: Optional[dict] = None
    ) -> np.ndarray:
        """LAB uzayinda L kanalina CLAHE uygula."""
        # Dusuk kontrastsa clip_limit'i artir
        if quality is not None and quality['contrast'] < 40:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit * 1.5,
                tileGridSize=self.clahe_tile_size
            )
        else:
            clahe = self._clahe

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Keskinlestirme
    # ------------------------------------------------------------------

    def _sharpen(self, bgr: np.ndarray) -> np.ndarray:
        """Unsharp masking ile keskinlestirme."""
        blur = cv2.GaussianBlur(bgr, (0, 0), 3)
        sharp = cv2.addWeighted(
            bgr, 1.0 + self.sharpen_strength,
            blur, -self.sharpen_strength,
            0
        )
        return sharp

    # ------------------------------------------------------------------
    # Morfolojik islemler
    # ------------------------------------------------------------------

    def _morphological_close(self, bgr: np.ndarray) -> np.ndarray:
        """Kucuk delikleri kapatmak icin morfolojik kapatma."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(bgr, cv2.MORPH_CLOSE, kernel)

    # ------------------------------------------------------------------
    # Yardimci: el yazisi on isleme
    # ------------------------------------------------------------------

    def prepare_for_handwriting(self, image: np.ndarray) -> np.ndarray:
        """
        El yazisi tanima icin ozel on isleme.

        - Kuvvetli golge giderme
        - Kontrast artirma
        - Ince/kalin cizgi normalizasyonu
        - Gurultu bastirma

        Args:
            image: Giris gorseli (BGR veya gri)

        Returns:
            On islenmis gorsel (giris ile ayni format)
        """
        is_gray = len(image.shape) == 2
        if is_gray:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image.copy()

        # 1. Daha kuvvetli golge giderme (kucuk pencere)
        bgr = self._remove_shadows(bgr)

        # 2. Non-local means ile gurultu giderme
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)

        # 3. Gri tonlamaya gec
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 4. Kuvvetli CLAHE
        clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray = clahe_strong.apply(gray)

        # 5. Otsu binarizasyonu + morfolojik ince hat normalizasyonu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Kalin hatlar => iyilestir (genisletme + asindirma)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)

        # 6. Geri BGR'ye cevir
        result_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        if is_gray:
            return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
        return result_bgr

    # ------------------------------------------------------------------
    # Yardimci: belge tarama on isleme
    # ------------------------------------------------------------------

    def prepare_for_scan(self, image: np.ndarray) -> np.ndarray:
        """
        Taranmis belge / fotograf edilmis dokumanlar icin on isleme.

        - Perspektif duzeltme ipucu hazirligi
        - Agresif golge giderme
        - CLAHE + keskinlestirme
        """
        is_gray = len(image.shape) == 2
        if is_gray:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image.copy()

        # Agresif golge giderme
        bgr = self._remove_shadows(bgr)

        # CLAHE (daha buyuk pencere = daha tutarli)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # Unsharp mask
        bgr = self._sharpen(bgr)

        if is_gray:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return bgr
