"""
Karakter sozlugu (vocabulary) yonetimi
"""

from typing import List, Dict, Optional
import json
from pathlib import Path


class Vocabulary:
    """Karakter sozlugu sinifi"""
    
    # Varsayilan Turkce + Ingilizce karakter seti
    DEFAULT_CHARS = (
        # Rakamlar
        "0123456789"
        # Kucuk harfler (Ingilizce)
        "abcdefghijklmnopqrstuvwxyz"
        # Buyuk harfler (Ingilizce)
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # Turkce ozel karakterler
        "çÇğĞıİöÖşŞüÜ"
        # Noktalama isaretleri
        ".,;:!?-'\"()[]{}/"
        # Ozel karakterler
        "@#$%&*+=<>~`\\|^_"
        # Para birimi ve tipografi karakterleri
        "\u20ba\u20ac\u00a3\u2014\u2013\u2026\u201c\u201d\u2018\u2019\u00b0\u00d7\u00f7\u00b1\u00b2\u00b3\u00bd\u00bc"
        # Bosluk
        " "
    )
    
    # CTC blank token
    BLANK_TOKEN = "<BLANK>"

    # Bilinmeyen karakter token
    UNK_TOKEN = "<UNK>"

    # Attention decoder icin baslangic/bitis tokenlari
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    
    def __init__(
        self,
        chars: Optional[str] = None,
        include_blank: bool = True,
        include_unk: bool = True,
        include_sos_eos: bool = False
    ):
        """
        Args:
            chars: Karakter dizisi (None ise varsayilan)
            include_blank: CTC blank token ekle (idx 0)
            include_unk: Bilinmeyen token ekle (idx 1)
            include_sos_eos: Attention decoder icin SOS/EOS token ekle
        """
        if chars is None:
            chars = self.DEFAULT_CHARS

        self.chars = chars
        self.include_blank = include_blank
        self.include_unk = include_unk
        self.include_sos_eos = include_sos_eos

        # Karakter -> index ve index -> karakter mapping'leri olustur
        self._build_vocab()
    
    def _build_vocab(self):
        """Vocabulary mapping'lerini olustur"""
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        
        idx = 0
        
        # Blank token (CTC icin index 0)
        if self.include_blank:
            self.char_to_idx[self.BLANK_TOKEN] = idx
            self.idx_to_char[idx] = self.BLANK_TOKEN
            self.blank_idx = idx
            idx += 1
        else:
            self.blank_idx = -1
        
        # Bilinmeyen token
        if self.include_unk:
            self.char_to_idx[self.UNK_TOKEN] = idx
            self.idx_to_char[idx] = self.UNK_TOKEN
            self.unk_idx = idx
            idx += 1
        else:
            self.unk_idx = -1

        # Attention icin SOS / EOS tokenlari
        if self.include_sos_eos:
            self.char_to_idx[self.SOS_TOKEN] = idx
            self.idx_to_char[idx] = self.SOS_TOKEN
            self.sos_idx = idx
            idx += 1

            self.char_to_idx[self.EOS_TOKEN] = idx
            self.idx_to_char[idx] = self.EOS_TOKEN
            self.eos_idx = idx
            idx += 1
        else:
            self.sos_idx = -1
            self.eos_idx = -1

        # Normal karakterler
        for char in self.chars:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                idx += 1
        
        self._size = idx
    
    @property
    def size(self) -> int:
        """Vocabulary boyutu (blank ve unk dahil)"""
        return self._size
    
    @property
    def num_classes(self) -> int:
        """Model icin sinif sayisi (size ile ayni)"""
        return self._size
    
    def encode(self, text: str) -> List[int]:
        """Metni index listesine donustur"""
        default = self.unk_idx if self.include_unk else None
        return [i for c in text if (i := self.char_to_idx.get(c, default)) is not None]
    
    def decode(
        self,
        indices: List[int],
        remove_blank: bool = True,
        remove_unk: bool = False,
        remove_sos_eos: bool = True
    ) -> str:
        """Index listesini metne donustur."""
        skip: set = set()
        if remove_blank and self.blank_idx >= 0:
            skip.add(self.blank_idx)
        if remove_unk and self.unk_idx >= 0:
            skip.add(self.unk_idx)
        if remove_sos_eos:
            if self.sos_idx >= 0: skip.add(self.sos_idx)
            if self.eos_idx >= 0: skip.add(self.eos_idx)
        return ''.join(
            self.idx_to_char[i] for i in indices
            if i in self.idx_to_char and i not in skip
        )
    
    def __len__(self) -> int:
        return self._size
    
    def __contains__(self, char: str) -> bool:
        return char in self.char_to_idx
    
    def get_char(self, idx: int) -> str:
        """Index'e karsilik gelen karakteri getir"""
        return self.idx_to_char.get(idx, self.UNK_TOKEN)
    
    def get_idx(self, char: str) -> int:
        """Karaktere karsilik gelen index'i getir"""
        return self.char_to_idx.get(char, self.unk_idx)
    
    def save(self, path: str):
        """Vocabulary'yi dosyaya kaydet"""
        data = {
            'chars': self.chars,
            'include_blank': self.include_blank,
            'include_unk': self.include_unk,
            'include_sos_eos': self.include_sos_eos,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()}
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Vocabulary'yi dosyadan yukle"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(
            chars=data['chars'],
            include_blank=data['include_blank'],
            include_unk=data['include_unk'],
            include_sos_eos=data.get('include_sos_eos', False)
        )
        
        return vocab
    
    def get_all_chars(self, include_special: bool = False) -> List[str]:
        """
        Tum karakterleri listele

        Args:
            include_special: Ozel token'lari dahil et (BLANK, UNK, SOS, EOS)

        Returns:
            Karakter listesi
        """
        special = {self.BLANK_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN}
        if include_special:
            return list(self.char_to_idx.keys())
        else:
            return [ch for ch in self.char_to_idx if ch not in special]
    
    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size}, chars='{self.chars[:20]}...')"


# Hazir vocabulary'ler
def get_turkish_vocab() -> Vocabulary:
    """Turkce icin optimize edilmis vocabulary"""
    chars = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "çÇğĞıİöÖşŞüÜ"
        ".,;:!?-'\"()[]{}/ "
    )
    return Vocabulary(chars)


def get_english_vocab() -> Vocabulary:
    """Ingilizce icin vocabulary"""
    chars = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ".,;:!?-'\"()[]{}/ "
    )
    return Vocabulary(chars)


def get_alphanumeric_vocab() -> Vocabulary:
    """Sadece alfanumerik karakterler"""
    chars = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        " "
    )
    return Vocabulary(chars)


def get_full_vocab() -> Vocabulary:
    """Tam karakter seti"""
    return Vocabulary()  # Varsayilan kullan
