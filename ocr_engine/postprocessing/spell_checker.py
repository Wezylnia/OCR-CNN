"""
Yazim duzeltme modulu
"""

from typing import List, Dict, Optional, Set, Tuple
import re
from collections import Counter


class SpellChecker:
    """Basit yazim duzeltme sinifi"""
    
    # Turkce karakter eslestirmeleri (OCR hatalari icin)
    TURKISH_CHAR_MAP = {
        'i': ['ı', 'l', '1', '|'],
        'ı': ['i', 'l', '1', '|'],
        'o': ['ö', '0'],
        'ö': ['o', '0'],
        'u': ['ü', 'v'],
        'ü': ['u', 'v'],
        's': ['ş', '5'],
        'ş': ['s', '5'],
        'c': ['ç'],
        'ç': ['c'],
        'g': ['ğ', '9'],
        'ğ': ['g', '9'],
    }
    
    def __init__(
        self,
        language: str = "tr",
        dictionary: Optional[Set[str]] = None,
        max_edit_distance: int = 2
    ):
        """
        Args:
            language: Dil (tr, en veya both)
            dictionary: Ozel sozluk (set)
            max_edit_distance: Maksimum edit mesafesi
        """
        self.language = language
        self.max_edit_distance = max_edit_distance
        
        # Sozluk
        if dictionary:
            self.dictionary = dictionary
        else:
            self.dictionary = self._load_default_dictionary()
        
        # Kelime frekanslari (varsa)
        self.word_frequencies: Dict[str, int] = {}

        # SymSpell entegrasyonu (varsa)
        self._symspell = None
        self._try_load_symspell()

    def _try_load_symspell(self) -> None:
        """symspellpy ile hizli sozluk yukle (varsa)"""
        try:
            from symspellpy import SymSpell
            from pathlib import Path
            sym = SymSpell(max_dictionary_edit_distance=self.max_edit_distance)
            loaded = False
            langs = ['tr', 'en'] if self.language == 'both' else [self.language]
            for lang in langs:
                dict_path = (
                    Path(__file__).parent.parent.parent / 'data' / f'{lang}_frequency_dict.txt'
                )
                if dict_path.exists():
                    sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
                    loaded = True
            if loaded:
                self._symspell = sym
        except ImportError:
            pass
        except Exception:
            pass

    def _load_default_dictionary(self) -> Set[str]:
        """Varsayilan sozluk yukle"""
        # Basit Turkce kelimeler
        turkish_words = {
            # Yaygın kelimeler
            've', 'bir', 'bu', 'da', 'de', 'ile', 'için', 'var', 'olan',
            'ben', 'sen', 'o', 'biz', 'siz', 'onlar',
            'ne', 'nasıl', 'nerede', 'neden', 'kim', 'hangi',
            'evet', 'hayır', 'tamam', 'lütfen', 'teşekkür',
            'gün', 'ay', 'yıl', 'saat', 'dakika',
            'sayfa', 'tarih', 'numara', 'adres', 'telefon',
            # Sayilar (yazi ile)
            'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz', 'on',
            'yüz', 'bin', 'milyon',
        }
        
        english_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might',
            'and', 'or', 'but', 'if', 'then', 'else',
            'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'where', 'when', 'why', 'how', 'who',
            'yes', 'no', 'not', 'all', 'any', 'some',
            'page', 'date', 'number', 'address', 'phone', 'email',
        }
        
        if self.language == "tr":
            return turkish_words
        elif self.language == "en":
            return english_words
        else:
            return turkish_words | english_words
    
    def add_words(self, words: List[str]):
        """Sozluge kelime ekle"""
        for word in words:
            self.dictionary.add(word.lower())
    
    def check(self, text: str) -> List[Tuple[str, bool]]:
        """
        Metindeki kelimeleri kontrol et
        
        Args:
            text: Kontrol edilecek metin
            
        Returns:
            [(kelime, dogru_mu), ...] listesi
        """
        words = self._tokenize(text)
        results = []
        
        for word in words:
            is_correct = self._is_correct(word)
            results.append((word, is_correct))
        
        return results
    
    def correct(self, text: str) -> str:
        """
        Metindeki yazim hatalarini duzelt
        
        Args:
            text: Duzeltilecek metin
            
        Returns:
            Duzeltilmis metin
        """
        words = self._tokenize(text)
        corrected_words = []
        
        for word in words:
            if self._is_correct(word):
                corrected_words.append(word)
            else:
                suggestion = self.suggest(word)
                corrected_words.append(suggestion if suggestion else word)
        
        return ' '.join(corrected_words)
    
    def suggest(self, word: str) -> Optional[str]:
        """
        Kelime icin oneri getir
        
        Args:
            word: Yanlis kelime
            
        Returns:
            Onerilen duzeltme veya None
        """
        word_lower = word.lower()

        # symspellpy ile hizli arama
        if self._symspell is not None:
            try:
                suggestions = self._symspell.lookup(
                    word_lower,
                    verbosity=0,
                    max_edit_distance=self.max_edit_distance
                )
                if suggestions:
                    return self._preserve_case(word, suggestions[0].term)
            except Exception:
                pass

        # Zaten dogru mu?
        if word_lower in self.dictionary:
            return word
        
        # Edit mesafesi 1 olanlar
        candidates = self._edits1(word_lower)
        valid = [w for w in candidates if w in self.dictionary]
        
        if valid:
            # En yuksek frekansli veya ilk eslesme
            if self.word_frequencies:
                valid.sort(key=lambda w: self.word_frequencies.get(w, 0), reverse=True)
            return self._preserve_case(word, valid[0])
        
        # Edit mesafesi 2 olanlar
        if self.max_edit_distance >= 2:
            candidates2 = set()
            for c in candidates:
                candidates2.update(self._edits1(c))
            valid = [w for w in candidates2 if w in self.dictionary]
            
            if valid:
                if self.word_frequencies:
                    valid.sort(key=lambda w: self.word_frequencies.get(w, 0), reverse=True)
                return self._preserve_case(word, valid[0])
        
        return None
    
    def _tokenize(self, text: str) -> List[str]:
        """Metni kelimelere ayir"""
        # Basit tokenization
        words = re.findall(r'\b[\wçÇğĞıİöÖşŞüÜ]+\b', text)
        return words
    
    def _is_correct(self, word: str) -> bool:
        """Kelime dogru mu?"""
        word_lower = word.lower()
        
        # Sozlukte var mi?
        if word_lower in self.dictionary:
            return True
        
        # Sayi mi?
        if word.isdigit():
            return True
        
        # Tek karakter mi?
        if len(word) <= 1:
            return True
        
        return False
    
    def _edits1(self, word: str) -> Set[str]:
        """Edit mesafesi 1 olan tum kelimeler"""
        letters = 'abcçdefgğhıijklmnoöpqrsştuüvwxyz'
        
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        # Silme
        deletes = [L + R[1:] for L, R in splits if R]
        
        # Degistirme
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        
        # Yerine koyma
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        
        # Ekleme
        inserts = [L + c + R for L, R in splits for c in letters]
        
        return set(deletes + transposes + replaces + inserts)
    
    def _preserve_case(self, original: str, corrected: str) -> str:
        """Orijinal kelimenin buyuk/kucuk harf durumunu koru"""
        if original.isupper():
            return corrected.upper()
        elif original[0].isupper():
            return corrected.capitalize()
        return corrected
    
    def correct_ocr_errors(self, text: str) -> str:
        """
        OCR'a ozgu hatalari duzelt
        
        Args:
            text: OCR cikisi
            
        Returns:
            Duzeltilmis metin
        """
        # Yaygın OCR hatalari
        replacements = [
            (r'\bl\b', 'I'),        # Tek 'l' -> 'I'
            (r'0(?=[a-zA-Z])', 'O'),  # Harf oncesi 0 -> O
            (r'(?<=[a-zA-Z])0', 'O'),  # Harf sonrasi 0 -> O
            (r'1(?=[a-zA-Z])', 'l'),  # Harf oncesi 1 -> l
            (r'\brn\b', 'm'),        # rn -> m
            (r'vv', 'w'),            # vv -> w
            (r'cl', 'd'),            # cl -> d
        ]
        
        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)
        
        return result


class ConfidenceBasedCorrector:
    """Guven skoruna dayali duzeltme"""
    
    def __init__(
        self,
        spell_checker: SpellChecker,
        confidence_threshold: float = 0.8
    ):
        """
        Args:
            spell_checker: SpellChecker nesnesi
            confidence_threshold: Duzeltme esik degeri
        """
        self.spell_checker = spell_checker
        self.confidence_threshold = confidence_threshold
    
    def correct_with_confidence(
        self,
        text: str,
        char_confidences: Optional[List[float]] = None
    ) -> str:
        """
        Guven skorlarina gore duzelt
        
        Args:
            text: Metin
            char_confidences: Her karakter icin guven skoru
            
        Returns:
            Duzeltilmis metin
        """
        if char_confidences is None:
            return self.spell_checker.correct(text)
        
        # Dusuk guvenli karakterleri isaretlebirlikte
        low_confidence_indices = [
            i for i, conf in enumerate(char_confidences)
            if conf < self.confidence_threshold
        ]
        
        if not low_confidence_indices:
            return text
        
        # Bu karakterleri iceren kelimeleri duzelt
        words = text.split()
        char_idx = 0
        corrected_words = []
        
        for word in words:
            word_end = char_idx + len(word)
            
            # Bu kelimede dusuk guvenli karakter var mi?
            has_low_conf = any(
                char_idx <= i < word_end
                for i in low_confidence_indices
            )
            
            if has_low_conf:
                suggestion = self.spell_checker.suggest(word)
                corrected_words.append(suggestion if suggestion else word)
            else:
                corrected_words.append(word)
            
            char_idx = word_end + 1  # +1 for space
        
        return ' '.join(corrected_words)
