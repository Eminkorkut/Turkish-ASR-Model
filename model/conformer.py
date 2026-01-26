import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Transformer tabanlı modeller için Sinüzoidal Pozisyon Kodlaması (Sinusoidal Positional Encoding).

    Amaç:
    - Self-attention mekanizması, RNN'lerin aksine girdinin sırasını (zaman algısını) doğrudan anlayamaz.
    - Bu modül, modele her anın (saniyenin/token'ın) sırasını bildiren benzersiz bir sinyal ekler.
    - Öğrenilebilir parametre içermez (deterministik formül kullanılır).

    Formül:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # (max_len, d_model) boyutunda sıfır matrisi
        pe = torch.zeros(max_len, d_model)

        # Pozisyon indeksleri (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Frekans terimleri (log uzayında hesaplanır, sayısal kararlılık için)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # Çift indekslere sinüs, tek indekslere kosinüs uygula
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Modül parametresi değil, sabit bir buffer olarak kaydet (state_dict içinde saklanır ama güncellenmez)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x (Tensor): Giriş vektörü (Batch, Time, D_Model)

        Returns:
            Tensor: Pozisyon bilgisi eklenmiş vektör (Batch, Time, D_Model)
        """
        # Giriş uzunluğuna kadar olan kısmı al ve ekle
        return x + self.pe[:, :x.size(1), :]


class ConformerConvModule(nn.Module):
    """
    Conformer mimarisinin kalbi: Konvolüsyon Modülü.

    Neden Conformer?:
    - Transformer'lar global bağlamı (cümlenin başı ile sonu arasındaki ilişkiyi) çok iyi yakalar.
    - CNN'ler ise lokal detayları (fonemler, heceler arasındaki geçişleri) çok iyi yakalar.
    - Conformer, bu iki dünyayı birleştirir.

    Yapı:
    LayerNorm -> Pointwise Conv -> GLU -> Depthwise Conv -> BatchNorm -> Swish -> Pointwise Conv
    """

    def __init__(self, d_model, kernel_size=31):
        super(ConformerConvModule, self).__init__()

        # 1. Giriş Normalizasyonu
        self.layer_norm = nn.LayerNorm(d_model)

        # 2. Pointwise Conv (Kanal Karıştırma) + GLU (Gated Linear Unit)
        # GLU çıkış boyutunu yarıya düşürdüğü için kanalları 2 katına çıkarıyoruz.
        self.pointwise_conv1 = nn.Conv1d(
            d_model, 2 * d_model, kernel_size=1
        )
        self.glu = nn.GLU(dim=1)

        # 3. Depthwise Conv (Uzamsal Filtreleme)
        # Her kanal kendi içinde filtrelenir, parametre sayısı azdır.
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, # Boyutu korumak için padding
            groups=d_model # Groups=Input_Canal olması Depthwise yapar
        )

        # 4. Batch Norm + Aktivasyon
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU() # Swish aktivasyonu (x * sigmoid(x))

        # 5. Çıkış Pointwise Conv
        self.pointwise_conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=1
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): (Batch, Time, D_Model)
        """
        # Conv1d, (Batch, Kanal, Zaman) bekler, bu yüzden transpose alıyoruz.
        x = self.layer_norm(x).transpose(1, 2) 

        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)

        # Orijinal boyuta geri dön
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """
    Tek bir Conformer Bloğu. Bu bloklar art arda eklenerek derin bir model oluşturulur.

    Mimari (Sırasıyla):
    1. Feed Forward (Macaron Style - 1. Yarı)
    2. Multi-Head Self Attention (Global Bağlam)
    3. Convolution Module (Lokal Detaylar)
    4. Feed Forward (Macaron Style - 2. Yarı)
    5. Layer Norm
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(ConformerBlock, self).__init__()

        # --- 1. Feed Forward (Yarım Adım) ---
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        # --- 2. Multi-Head Self Attention ---
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_attn = nn.LayerNorm(d_model)

        # --- 3. Convolution Module ---
        self.conv = ConformerConvModule(d_model)
        self.norm_conv = nn.LayerNorm(d_model)

        # --- 4. Feed Forward (Yarım Adım) ---
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        # --- 5. Final Normalizasyon ---
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Macaron Net stili: FF katmanları ikiye bölünür ve 0.5 ile toplanır.
        
        # 1. FF1 + Residual
        x = x + 0.5 * self.ff1(x)
        
        # 2. Attention + Residual
        # Norm içeride, residual dışarıda (Pre-norm yapısı)
        attn_out, _ = self.attn(
            self.norm_attn(x),
            self.norm_attn(x),
            self.norm_attn(x)
        )
        x = x + attn_out
        
        # 3. Conv + Residual
        x = x + self.conv(self.norm_conv(x))
        
        # 4. FF2 + Residual
        x = x + 0.5 * self.ff2(x)
        
        return self.final_norm(x)


class TurkishASRModel(nn.Module):
    """
    Türkçe ASR (Speech-to-Text) Modeli.
    
    Bu model, Conformer mimarisini temel alır. Ses spektrogramını alır ve
    harf olasılıklarını (CTC için) üretir.

    Bileşenler:
    1. Subsampling: Giriş spektrogramını zaman ekseninde küçültür (4x).
        - Bu, hesaplama maliyetini çok düşürür.
        - 1000 frame'lik bir sesi 250 frame'e indirir.
    2. Linear Projection: Özellikleri model boyutuna (d_model) taşır.
    3. Positional Encoding: Sıra bilgisi ekler.
    4. Conformer Encoder: Asıl işi yapan derin bloklar.
    5. Classifier: Çıkış karakterleri için olasılık üretir.
    """

    def __init__(
        self,
        n_mel_channels, # Giriş özellik sayısı (örn: 40 veya 80)
        d_model=256,    # Model boyutu
        n_heads=4,      # Attention kafa sayısı
        n_blocks=6,     # Derinlik (Blok sayısı)
        n_classes=31,   # Çıktı sınıf sayısı (Alfabe + Blank)
        dropout=0.1
    ):
        super(TurkishASRModel, self).__init__()
        
        # --- 1. Subsampling (CNN) ---
        # 2 adet Conv2d katmanı ile zamanı 4 kat, frekansı 4 kat küçültüyoruz (stride=2,2)
        self.subsample = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.SiLU()
        )
        
        # Subsampling sonrası boyut hesabı:
        # Zaman: T -> T/2 -> T/4
        # Frekans: F -> F/2 -> F/4
        # Çıkış boyutu = d_model * (F/4)
        flattened_dim = d_model * (n_mel_channels // 4)
        
        self.input_proj = nn.Linear(flattened_dim, d_model)

        # --- 2. Positional Encoding ---
        self.pos_encoding = PositionalEncoding(d_model)
        
        # --- 3. Conformer Encoder ---
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # --- 4. Sınıflandırıcı (Classifier) ---
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        """
        Args:
            x (Tensor): Log-Mel Spectrogram (Batch, Time, Mel_Channels)

        Returns:
            Tensor: Logits (Batch, Subsampled_Time, Num_Classes)
        """

        # CNN 4 boyutlu giriş ister: (Batch, Channel, Time, Freq)
        # Girişimiz (B, T, F) olduğu için kanal boyutunu (1) ekliyoruz.
        x = x.unsqueeze(1)  # (B, 1, T, F)
        
        # 1. Subsampling
        x = self.subsample(x) # (B, d_model, T/4, F/4)
        
        # 2. Boyutları düzenle
        b, c, t, f = x.size()
        # (B, T, C, F) -> (B, T, C*F)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, t, -1) 
        
        # 3. Projeksiyon + Pozisyon
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # 4. Conformer Blokları
        for block in self.blocks:
            x = block(x)
            
        # 5. Sınıflandırma
        x = self.fc(x)
        
        return x
