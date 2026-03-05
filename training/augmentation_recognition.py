"""
Metin tanima icin ozellestirilmis augmentation: RecognitionAugmentor
"""

import cv2
import numpy as np
from typing import Tuple
import random


class RecognitionAugmentor:
    """Metin tanima icin ozellestirilmis augmentation"""

    def __init__(
        self,
        stretch_range: Tuple[float, float] = (0.8, 1.2),
        shear_range: Tuple[float, float] = (-0.1, 0.1),
        noise_prob: float = 0.15,
        blur_prob: float = 0.15,
        invert_prob: float = 0.05,
        elastic_prob: float = 0.15,
        grid_distort_prob: float = 0.1,
        handwriting_mode: bool = False
    ):
        self.stretch_range = stretch_range
        self.shear_range = shear_range
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
        self.invert_prob = invert_prob
        self.elastic_prob = elastic_prob
        self.grid_distort_prob = grid_distort_prob
        self.handwriting_mode = handwriting_mode
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Augmentation uygula - Windows-safe (sadece numpy islemleri)"""
        try:
            # Goruntunun contiguous oldugundan emin ol
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)

            # El yazisi modunda oncelikli distorsiyon
            if self.handwriting_mode:
                if random.random() < self.elastic_prob:
                    image = self.elastic_transform(image)
                if random.random() < self.grid_distort_prob:
                    image = self.grid_distortion(image)

            # Parlaklik ayari (numpy ile)
            if random.random() < 0.15:
                factor = random.uniform(0.7, 1.3)
                image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

            # Kontrast ayari (numpy ile)
            if random.random() < 0.15:
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
            if random.random() < 0.05:
                noise_mask = np.random.random(image.shape)
                image = np.where(noise_mask < 0.02, 0, image)   # Salt
                image = np.where(noise_mask > 0.98, 255, image)  # Pepper
                image = image.astype(np.uint8)

            # Normal mod distorsiyon
            if not self.handwriting_mode:
                if random.random() < self.elastic_prob * 0.5:
                    image = self.elastic_transform(image, alpha=12, sigma=3)

            # Yatay germe
            if random.random() < 0.1:
                image = self.horizontal_stretch(image)

            # Shear donusumu
            if random.random() < 0.08:
                image = self.shear(image)

            # CutOut: rastgele dikdortgen maskeleme
            if random.random() < 0.12:
                h, w = image.shape[:2]
                rh = random.randint(h // 4, h // 2)
                rw = random.randint(w // 8, max(w // 4, 1))
                x0 = random.randint(0, max(1, w - rw))
                y0 = random.randint(0, max(1, h - rh))
                fill_val = random.randint(64, 192)
                image[y0:y0 + rh, x0:x0 + rw] = fill_val

            # JPEG sikistirma artifaktlari
            if random.random() < 0.1:
                quality = random.randint(30, 80)
                _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                read_flag = cv2.IMREAD_GRAYSCALE if image.ndim == 2 else cv2.IMREAD_COLOR
                decoded = cv2.imdecode(buf, read_flag)
                if decoded is not None and decoded.shape == image.shape:
                    image = decoded

        except Exception as e:
            # Herhangi bir hata durumunda orijinal gorseli dondur
            import logging
            logging.getLogger(__name__).debug("Augmentation hatasi (orijinal gorsel kullanilacak): %s", e)

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

    # ------------------------------------------------------------------
    # Elastik distorsiyon (el yazisi icin kritik)
    # ------------------------------------------------------------------

    def elastic_transform(
        self,
        image: np.ndarray,
        alpha: float = 20.0,
        sigma: float = 4.0
    ) -> np.ndarray:
        """
        Elastik distorsiyon.

        Simmons & Chollet (2016) tabanli: piksel düzeyinde rastgele
        yerinden etme alanı üretilir, Gaussian blur ile pürüzsüzleştirilir.
        El yazısının kaligrafik varyasyonlarını simüle eder.

        Args:
            image: Gri veya BGR gorsel
            alpha: Distorsiyon siddeti (büyük = daha fazla bozulma)
            sigma: Gaussian pürüzsüzleştirme std (büyük = daha yumuşak)
        """
        h, w = image.shape[:2]

        # Rastgele yerinden etme alanı
        dx = np.random.randn(h, w).astype(np.float32) * alpha
        dy = np.random.randn(h, w).astype(np.float32) * alpha

        # Pürüzsüzleştir
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        dx = cv2.GaussianBlur(dx, (ksize, ksize), sigma)
        dy = cv2.GaussianBlur(dy, (ksize, ksize), sigma)

        # Koordinat grid'i
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def grid_distortion(
        self,
        image: np.ndarray,
        num_steps: int = 5,
        distort_limit: float = 0.25
    ) -> np.ndarray:
        """
        Grid (kafes) distorsiyon.

        Görüntüyü eşit parçalara böler, her parçanın köşelerini rastgele
        kaydırır. Yazı eğimleri ve perspektif varyasyonlarını taklit eder.

        Args:
            image:         Girdi
            num_steps:     Kafes bölme sayısı (her eksende)
            distort_limit: Maksimum kaydırma oranı [0..0.5]
        """
        h, w = image.shape[:2]
        x_step = w // num_steps
        y_step = h // num_steps

        # Kontrol noktaları
        x_pts = np.linspace(0, w, num_steps + 1, dtype=np.float32)
        y_pts = np.linspace(0, h, num_steps + 1, dtype=np.float32)

        # Rastgele pertürbasyon
        x_pts[1:-1] += np.random.uniform(
            -x_step * distort_limit, x_step * distort_limit, len(x_pts) - 2
        ).astype(np.float32)
        y_pts[1:-1] += np.random.uniform(
            -y_step * distort_limit, y_step * distort_limit, len(y_pts) - 2
        ).astype(np.float32)

        x_pts = np.clip(x_pts, 0, w).astype(np.float32)
        y_pts = np.clip(y_pts, 0, h).astype(np.float32)

        # Dense map üret
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for i in range(num_steps):
            for j in range(num_steps):
                # Kaynak ve hedef dikdörtgenler
                src_x1, src_x2 = int(j * w / num_steps), int((j + 1) * w / num_steps)
                src_y1, src_y2 = int(i * h / num_steps), int((i + 1) * h / num_steps)

                dst_x1, dst_x2 = int(x_pts[j]), int(x_pts[j + 1])
                dst_y1, dst_y2 = int(y_pts[i]), int(y_pts[i + 1])

                if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
                    continue

                cell_w = dst_x2 - dst_x1
                cell_h = dst_y2 - dst_y1

                # Lineer interpolasyon ile map doldur
                xx = np.linspace(src_x1, src_x2 - 1, cell_w, dtype=np.float32)
                yy = np.linspace(src_y1, src_y2 - 1, cell_h, dtype=np.float32)
                gx, gy = np.meshgrid(xx, yy)

                map_x[dst_y1:dst_y2, dst_x1:dst_x2] = gx
                map_y[dst_y1:dst_y2, dst_x1:dst_x2] = gy

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
