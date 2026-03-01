"""
CTC Decoder - Model cikislarindan metin uretme
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from .vocab import Vocabulary


class UnigramLM:
    """
    Kelime duzeyinde tek-gram dil modeli.

    CTC prefix beam search sonuclarini kelime frekansina gore
    yeniden puanlar: final = ctc_score + lm_weight * sum(log P(word)).

    Kullanim::
        lm = UnigramLM.from_file('data/tr_frequency_dict.txt')
        decoder = CTCPrefixDecoder(vocab, beam_width=10)
        decoder.set_lm(lm, lm_weight=0.3)
    """

    def __init__(
        self,
        word_log_probs: Optional[Dict[str, float]] = None,
        unk_log_prob: float = -10.0,
    ):
        self.word_log_probs: Dict[str, float] = word_log_probs or {}
        self.unk_log_prob = unk_log_prob

    @classmethod
    def from_file(cls, path: str, max_words: int = 200_000) -> 'UnigramLM':
        """Frekans sozlugundan yukle (format: '<kelime>\\t<frekans>' veya boslukla ayrilmis)"""
        word_freqs: Dict[str, float] = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        word_freqs[parts[0].lower()] = float(parts[1])
                    except ValueError:
                        continue
                if len(word_freqs) >= max_words:
                    break
        total = sum(word_freqs.values()) or 1.0
        log_probs = {w: float(np.log(c / total)) for w, c in word_freqs.items()}
        return cls(log_probs)

    def score(self, word: str) -> float:
        """Kelimenin log-olasiligini dondur"""
        return self.word_log_probs.get(word.lower(), self.unk_log_prob)

    def score_text(self, text: str) -> float:
        """Metindeki tum kelimelerin log-olasiliklari toplami"""
        words = text.strip().split()
        return sum(self.score(w) for w in words) if words else 0.0


class CTCDecoder:
    """CTC cikislarini metne donusturur"""
    
    def __init__(self, vocab: Vocabulary):
        """
        Args:
            vocab: Vocabulary nesnesi
        """
        self.vocab = vocab
        self.blank_idx = vocab.blank_idx
    
    def decode_greedy(
        self,
        log_probs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> List[str]:
        """
        Greedy decoding - en yuksek olasilikli yolu sec
        
        Args:
            log_probs: [seq_len, batch, num_classes]
            lengths: [batch] - her ornek icin sequence uzunlugu
            
        Returns:
            Decode edilmis metin listesi
        """
        # Argmax
        _, max_indices = log_probs.max(dim=2)  # [seq_len, batch]
        max_indices = max_indices.permute(1, 0)  # [batch, seq_len]
        
        if lengths is None:
            lengths = [max_indices.shape[1]] * max_indices.shape[0]
        else:
            lengths = lengths.cpu().tolist()
        
        decoded = []
        for indices, length in zip(max_indices, lengths):
            indices = indices[:length].cpu().tolist()
            text = self._collapse_repeated(indices)
            decoded.append(text)
        
        return decoded

    def _collapse_repeated(self, indices: List[int]) -> str:
        """
        Tekrarli karakterleri birlestir ve blank'lari kaldir
        
        Args:
            indices: Index listesi
            
        Returns:
            Decode edilmis metin
        """
        collapsed = []
        prev_idx = None
        
        for idx in indices:
            # Blank atla
            if idx == self.blank_idx:
                prev_idx = None
                continue
            
            # Tekrar atla
            if idx == prev_idx:
                continue
            
            collapsed.append(idx)
            prev_idx = idx
        
        return self.vocab.decode(collapsed, remove_blank=True)
    
    def decode_batch(
        self,
        log_probs: torch.Tensor,
        method: str = "greedy",
        beam_width: int = 5,
        lengths: Optional[torch.Tensor] = None
    ) -> List[str]:
        """
        Batch decoding

        Args:
            log_probs: [seq_len, batch, num_classes]
            method: Decoding yontemi ('greedy' desteklenir)
            beam_width: Kullanilmiyor, geri uyumluluk icin tutuldu
            lengths: [batch] - her ornek icin sequence uzunlugu

        Returns:
            Decode edilmis metin listesi
        """
        if method == "greedy":
            return self.decode_greedy(log_probs, lengths)
        else:
            raise ValueError(
                f"Bilinmeyen decoding yontemi: '{method}'. "
                "Beam search icin CTCPrefixDecoder kullanin."
            )


class CTCPrefixDecoder:
    """
    CTC Prefix Beam Search Decoder
    
    Daha dogru beam search implementasyonu
    """
    
    def __init__(
        self,
        vocab: Vocabulary,
        beam_width: int = 10,
        blank_idx: Optional[int] = None
    ):
        self.vocab = vocab
        self.beam_width = beam_width
        self.blank_idx = blank_idx if blank_idx is not None else vocab.blank_idx
        self.lm: Optional[UnigramLM] = None
        self.lm_weight: float = 0.3

    def set_lm(self, lm: UnigramLM, lm_weight: float = 0.3) -> None:
        """Unigram dil modelini etkinlestir."""
        self.lm = lm
        self.lm_weight = lm_weight
    
    def decode(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None
    ) -> Tuple[str, float]:
        """
        Tek bir sequence icin decode
        
        Args:
            log_probs: [seq_len, num_classes]
            length: Sequence uzunlugu
            
        Returns:
            (metin, log_prob)
        """
        T, V = log_probs.shape
        
        if length is not None:
            T = min(T, length)
            log_probs = log_probs[:T]
        
        # Beams: prefix -> (prob_blank, prob_non_blank)
        # prob_blank: prefix'in blank ile bittigi olasilik
        # prob_non_blank: prefix'in karakter ile bittigi olasilik
        beams = {(): (0.0, float('-inf'))}  # Bos prefix ile basla
        
        for t in range(T):
            new_beams = {}
            
            for prefix, (pb, pnb) in beams.items():
                # Mevcut prefix'in toplam log olasiligi
                p = np.logaddexp(pb, pnb)
                
                # Her karakter icin
                for c in range(V):
                    lp = log_probs[t, c]
                    
                    if c == self.blank_idx:
                        # Blank - prefix ayni kalir
                        key = prefix
                        new_pb = np.logaddexp(pb + lp, pnb + lp)
                        
                        if key in new_beams:
                            new_beams[key] = (
                                np.logaddexp(new_beams[key][0], new_pb),
                                new_beams[key][1]
                            )
                        else:
                            new_beams[key] = (new_pb, float('-inf'))
                    
                    else:
                        # Karakter
                        # Eger son karakter ayni ise, sadece blank'tan gelebilir
                        # Degilse hem blank hem non-blank'tan gelebilir
                        
                        if len(prefix) > 0 and prefix[-1] == c:
                            # Tekrar - sadece blank'tan
                            new_pnb = pb + lp
                        else:
                            # Yeni karakter
                            new_pnb = p + lp
                        
                        key = prefix + (c,)
                        
                        if key in new_beams:
                            new_beams[key] = (
                                new_beams[key][0],
                                np.logaddexp(new_beams[key][1], new_pnb)
                            )
                        else:
                            new_beams[key] = (float('-inf'), new_pnb)
            
            # En iyi beam_width kadarini tut
            scored = [
                (prefix, np.logaddexp(pb, pnb))
                for prefix, (pb, pnb) in new_beams.items()
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            beams = {}
            for prefix, _ in scored[:self.beam_width]:
                beams[prefix] = new_beams[prefix]
        
        # En iyi sonucu al (opsiyonel LM ile yeniden puanlama)
        if not beams:
            return "", float('-inf')

        if self.lm is not None:
            # Tum beam adaylari uzerinde CTC + LM birlesik puani hesapla
            best_prefix: tuple = ()
            best_combined = float('-inf')
            best_ctc_score = float('-inf')
            for prefix, (pb, pnb) in beams.items():
                ctc_score = np.logaddexp(pb, pnb)
                candidate = self.vocab.decode(list(prefix), remove_blank=True)
                lm_score = self.lm.score_text(candidate)
                combined = ctc_score + self.lm_weight * lm_score
                if combined > best_combined:
                    best_combined = combined
                    best_prefix = prefix
                    best_ctc_score = ctc_score
            best_score = best_ctc_score
        else:
            best_prefix = max(
                beams.keys(),
                key=lambda p: np.logaddexp(beams[p][0], beams[p][1])
            )
            best_score = np.logaddexp(beams[best_prefix][0], beams[best_prefix][1])

        text = self.vocab.decode(list(best_prefix), remove_blank=True)

        return text, best_score
    
    def decode_batch(
        self,
        log_probs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> List[Tuple[str, float]]:
        """
        Batch decoding
        
        Args:
            log_probs: [seq_len, batch, num_classes]
            lengths: [batch]
            
        Returns:
            List of (text, score)
        """
        log_probs = log_probs.cpu().numpy()
        batch_size = log_probs.shape[1]
        
        if lengths is None:
            lengths = [log_probs.shape[0]] * batch_size
        else:
            lengths = lengths.cpu().tolist()
        
        results = []
        for b in range(batch_size):
            probs = log_probs[:, b, :]
            text, score = self.decode(probs, lengths[b])
            results.append((text, score))
        
        return results
