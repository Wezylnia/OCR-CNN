"""
Attention tabanli metin tanima modulu

El yazisi ve egri/bozuk metinler icin CTC yerine Attention Decoder kullanir.
CRNN encoder cikisina Bahdanau (additive) attention mekanizmasi uygular.

Avantajlari (CTC'ye gore):
  - El yazisinda daha yuksek dogruluk
  - Karakter hizalamasi gorunulebilir
  - Uzun ve karmasik metinlerde daha kararsiz degil
  - Specials (noktalama, rakam) tespitinde daha guvenilir

Kullanim:
    from ocr_engine.recognition.attention import AttentionCRNN, build_attention_crnn
    model = build_attention_crnn(num_classes=vocab.size)
    logits, attn_weights = model(images, targets, target_lengths)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .model import VGGEncoder, BidirectionalLSTM


# ---------------------------------------------------------------------------
# Attention mekanizmasi
# ---------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention.

    Kaynak: "Neural Machine Translation by Jointly Learning to Align and Translate"
    (Bahdanau et al., 2015)

    h_enc : encoder gizli durumu  [batch, T, enc_dim]
    h_dec : decoder gizli durumu  [batch, dec_dim]
    cikis : context vektor         [batch, enc_dim]
            attention agirliklari  [batch, T]
    """

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 256):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        enc_out: torch.Tensor,
        dec_hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            enc_out:    [batch, T, enc_dim]
            dec_hidden: [batch, dec_dim]
            mask:       [batch, T] — maskeli zaman adimlarini sifirla (bool)

        Returns:
            context:   [batch, enc_dim]
            attn_w:    [batch, T]
        """
        # enc_out  : [batch, T, attn_dim]
        proj_enc = self.W_enc(enc_out)

        # dec_hidden: [batch, 1, attn_dim]
        proj_dec = self.W_dec(dec_hidden).unsqueeze(1)

        # energy: [batch, T, 1] -> [batch, T]
        energy = self.v(torch.tanh(proj_enc + proj_dec)).squeeze(-1)

        if mask is not None:
            energy = energy.masked_fill(mask, float('-inf'))

        attn_w = F.softmax(energy, dim=-1)           # [batch, T]

        # NaN'leri sifirla (tum pozisyonlar maskeli ise)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)

        context = torch.bmm(attn_w.unsqueeze(1), enc_out).squeeze(1)  # [batch, enc_dim]
        return context, attn_w


# ---------------------------------------------------------------------------
# Attention Decoder
# ---------------------------------------------------------------------------

class AttentionDecoder(nn.Module):
    """
    Karakter seviyesinde attention decoder.

    Her adimda:
        1. Onceki karakter gomme (embedding)
        2. Attention hesaplama
        3. GRU guncelleme
        4. Cikis projeksiyonu
    """

    def __init__(
        self,
        num_classes: int,
        enc_dim: int = 512,
        dec_hidden: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dec_hidden = dec_hidden

        # Karakter embedding
        self.embedding = nn.Embedding(num_classes, 64)

        # Attention
        self.attention = BahdanauAttention(enc_dim, dec_hidden, attn_dim)

        # GRU decoder
        self.gru = nn.GRU(
            input_size=64 + enc_dim,   # embedding + context
            hidden_size=dec_hidden,
            num_layers=1,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(dec_hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        prev_char: torch.Tensor,
        hidden: torch.Tensor,
        enc_out: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tek adim.

        Args:
            prev_char: [batch]            — onceki karakter indisi
            hidden:    [1, batch, dec_dim] — GRU gizli durum
            enc_out:   [batch, T, enc_dim]
            mask:      [batch, T]

        Returns:
            logit:    [batch, num_classes]
            hidden:   [1, batch, dec_dim]
            attn_w:   [batch, T]
        """
        # [batch] -> [batch, 64]
        emb = self.embedding(prev_char)

        # Decoder gizli durumu [1, batch, dim] -> [batch, dim]
        h = hidden.squeeze(0)

        context, attn_w = self.attention(enc_out, h, mask)

        # GRU giris: [batch, 1, 64 + enc_dim]
        gru_input = torch.cat([emb, context], dim=-1).unsqueeze(1)
        gru_out, hidden = self.gru(gru_input, hidden)   # [batch, 1, dec_dim]
        gru_out = self.dropout(gru_out.squeeze(1))      # [batch, dec_dim]

        logit = self.output_proj(gru_out)               # [batch, num_classes]
        return logit, hidden, attn_w

    def forward(
        self,
        enc_out: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        max_len: int = 100,
        teacher_forcing_ratio: float = 0.5,
        sos_idx: int = 1,
        eos_idx: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            enc_out:     [batch, T, enc_dim]
            targets:     [batch, max_target_len] (egitim, opsiyonel)
            target_lengths: [batch]
            max_len:     maksimum cikti uzunlugu (inference)
            teacher_forcing_ratio: ogretmen zorlama orani
            sos_idx:     baslangic token indisi
            eos_idx:     bitis token indisi

        Returns:
            all_logits:  [batch, seq_len, num_classes]
            all_attns:   [batch, seq_len, T]
        """
        batch = enc_out.size(0)
        device = enc_out.device

        # Baslangic gizli durumu sifir
        hidden = torch.zeros(1, batch, self.dec_hidden, device=device)

        # Baslangic token
        prev_char = torch.full((batch,), sos_idx, dtype=torch.long, device=device)

        is_train = (targets is not None)
        seq_len = targets.size(1) if is_train else max_len

        all_logits = []
        all_attns = []

        for t in range(seq_len):
            logit, hidden, attn_w = self.forward_step(prev_char, hidden, enc_out)
            all_logits.append(logit.unsqueeze(1))   # [batch, 1, C]
            all_attns.append(attn_w.unsqueeze(1))   # [batch, 1, T]

            if is_train:
                use_teacher = (torch.rand(1).item() < teacher_forcing_ratio)
                prev_char = targets[:, t] if use_teacher else logit.argmax(-1)
            else:
                prev_char = logit.argmax(-1)
                # Tum batch EOS'e ulastiysa dur
                if (prev_char == eos_idx).all():
                    break

        all_logits = torch.cat(all_logits, dim=1)  # [batch, seq, C]
        all_attns = torch.cat(all_attns, dim=1)    # [batch, seq, T]
        return all_logits, all_attns


# ---------------------------------------------------------------------------
# AttentionCRNN (end-to-end)
# ---------------------------------------------------------------------------

class AttentionCRNN(nn.Module):
    """
    Attention mekanizmali CRNN modeli.

    Mimari:
        CNN (VGG/ResNet) -> BiLSTM -> Attention Decoder

    Egitimde: cross-entropy loss + teacher forcing
    Inference: greedy veya beam search
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        attn_dim: int = 256,
        dropout: float = 0.1,
        encoder_type: str = "vgg",
        sos_idx: int = 1,
        eos_idx: int = 2
    ):
        """
        Args:
            num_classes:    Vocab boyutu (blank + unk + karakterler)
            input_channels: 1=gri, 3=BGR
            hidden_size:    BiLSTM gizli durum boyutu
            num_layers:     BiLSTM katman sayisi
            attn_dim:       Attention projeksiyon boyutu
            dropout:        Dropout orani
            encoder_type:   "vgg" veya "resnet"
            sos_idx:        Baslangic token indisi
            eos_idx:        Bitis token indisi
        """
        super().__init__()

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.num_classes = num_classes

        # 1. CNN Encoder
        if encoder_type == "vgg":
            self.encoder = VGGEncoder(input_channels)
        else:
            raise ValueError(f"Bilinmeyen encoder tipi: {encoder_type}")

        enc_channels = self.encoder.output_channels  # 512

        # 2. BiLSTM sequence modelleme
        self.bilstm_layers = nn.ModuleList()
        for i in range(num_layers):
            inp = enc_channels if i == 0 else hidden_size
            self.bilstm_layers.append(
                BidirectionalLSTM(inp, hidden_size, hidden_size, dropout if i < num_layers - 1 else 0)
            )

        # 3. Attention Decoder
        self.decoder = AttentionDecoder(
            num_classes=num_classes,
            enc_dim=hidden_size,
            dec_hidden=hidden_size * 2,
            attn_dim=attn_dim,
            dropout=dropout
        )

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """CNN + BiLSTM ile goruntu encode et."""
        # CNN: [B, C, H, W] -> [B, 512, 1, W']
        feat = self.encoder(x)

        # [B, 512, 1, W'] -> [B, W', 512]
        B, C, H, W = feat.size()
        feat = feat.squeeze(2).permute(0, 2, 1)

        # BiLSTM katmanlari: [B, W', 512] -> [B, W', hidden]
        for layer in self.bilstm_layers:
            feat = layer(feat)

        return feat  # [B, W', hidden]

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:               [B, C, H, W]
            targets:         [B, T] — egitim icin
            target_lengths:  [B]    — egitim icin
            teacher_forcing_ratio: ogretmen zorlama orani

        Returns:
            logits:    [B, seq_len, num_classes]
            attn_maps: [B, seq_len, T_enc]
        """
        enc_out = self._encode(x)   # [B, T_enc, hidden]

        logits, attns = self.decoder(
            enc_out,
            targets=targets,
            target_lengths=target_lengths,
            teacher_forcing_ratio=teacher_forcing_ratio,
            sos_idx=self.sos_idx,
            eos_idx=self.eos_idx
        )

        return logits, attns

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        max_len: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy inference.

        Returns:
            char_indices: [B, seq_len]
            attn_maps:    [B, seq_len, T_enc]
        """
        self.eval()
        enc_out = self._encode(x)
        logits, attns = self.decoder(
            enc_out,
            targets=None,
            max_len=max_len,
            teacher_forcing_ratio=0.0,
            sos_idx=self.sos_idx,
            eos_idx=self.eos_idx
        )
        char_indices = logits.argmax(-1)        # [B, seq_len]
        return char_indices, attns

    def get_sequence_length(self, input_width: int) -> int:
        """VGG encoder icin yaklasik cikis uzunlugu."""
        return (input_width // 4) - 1


# ---------------------------------------------------------------------------
# Cross-Entropy Loss (Attention icin)
# ---------------------------------------------------------------------------

class AttentionLoss(nn.Module):
    """Attention decoder icin masked cross-entropy loss."""

    def __init__(self, pad_idx: int = 0, label_smoothing: float = 0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits:  [B, seq_len, num_classes]
            targets: [B, max_target_len]
            target_lengths: [B]

        Returns:
            Skalar loss
        """
        B, seq_len, C = logits.size()
        tgt_len = targets.size(1)
        min_len = min(seq_len, tgt_len)

        loss = self.criterion(
            logits[:, :min_len, :].reshape(-1, C),
            targets[:, :min_len].reshape(-1)
        )
        return loss


# ---------------------------------------------------------------------------
# Decoder: indislerden metin olusturma
# ---------------------------------------------------------------------------

class AttentionDecodeHelper:
    """AttentionCRNN cikislarini metne donusturur."""

    def __init__(self, vocab, sos_idx: int = 1, eos_idx: int = 2):
        self.vocab = vocab
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def indices_to_text(self, indices: torch.Tensor) -> str:
        """
        [seq_len] tensorunu metne cevir.

        Args:
            indices: Tek ornek icin karakter indis tensoru

        Returns:
            Metin string
        """
        chars = []
        for idx in indices.tolist():
            if idx == self.eos_idx:
                break
            if idx in (self.sos_idx, self.vocab.blank_idx, self.vocab.unk_idx):
                continue
            ch = self.vocab.idx_to_char.get(idx, '')
            if ch:
                chars.append(ch)
        return ''.join(chars)

    def batch_indices_to_texts(self, batch_indices: torch.Tensor) -> list:
        """
        [B, seq_len] tensorunu metin listesine cevir.

        Args:
            batch_indices: [B, seq_len]

        Returns:
            Liste[str]
        """
        return [self.indices_to_text(row) for row in batch_indices]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_attention_crnn(
    num_classes: int,
    input_channels: int = 1,
    hidden_size: int = 256,
    num_layers: int = 2,
    attn_dim: int = 256,
    dropout: float = 0.1,
    encoder_type: str = "vgg",
    sos_idx: int = 1,
    eos_idx: int = 2,
    weights_path: Optional[str] = None
) -> AttentionCRNN:
    """
    AttentionCRNN modeli olustur.

    Args:
        num_classes:    Vocab boyutu
        input_channels: 1=gri, 3=renk
        hidden_size:    BiLSTM gizli boyut
        num_layers:     BiLSTM katman sayisi
        attn_dim:       Attention projeksiyon boyutu
        dropout:        Dropout orani
        encoder_type:   "vgg" veya "resnet"
        sos_idx:        Baslangic token
        eos_idx:        Bitis token
        weights_path:   Onceden egitilmis agirlik dosyasi

    Returns:
        AttentionCRNN modeli
    """
    model = AttentionCRNN(
        num_classes=num_classes,
        input_channels=input_channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        attn_dim=attn_dim,
        dropout=dropout,
        encoder_type=encoder_type,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    )

    if weights_path:
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        print(f"Attention CRNN agirliklari yuklendi: {weights_path}")

    return model
