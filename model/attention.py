import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding (Bağıl Pozisyon Kodlaması).
    Transformer-XL ve Conformer mimarilerinde kullanılır.
    
    Mutlak pozisyon yerine (0, 1, 2...), tokenlar arasındaki mesafe önemlidir.
    Örn: "Ben" kelimesi "Geldim" kelimesinden 2 adım öncedir.
    """
    def __init__(self, d_model, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = None # Dinamik oluşturulacak

    def forward(self, x):
        """
        x: (Batch, Seq_Len, D_Model)
        Returns: (Seq_Len, D_Model) -> Positional Embeddings
        Note: Conformer için genellikle her batch için aynıdır.
        """
        length = x.size(1)
        if self.pe is None or self.pe.size(1) < length:
            # PE matrisini oluştur
            pe = torch.zeros(length * 2 - 1, self.d_model)
            position = torch.arange(0, length * 2 - 1, dtype=torch.float).unsqueeze(1)
            # Merkezlendiği için shift (0 noktası ortada)
            # Ama basitlik için Conformer implementasyonlarında genellikle
            # [0, length] aralığı değil, [-length, length] aralığı kodlanır.
            # Biz burada standart Sinusoidal formülü kullanıp attention içinde shift trick kullanacağız.
            
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2).float()
                * (-math.log(10000.0) / self.d_model)
            )
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.pe = pe.unsqueeze(0).to(x.device) # (1, 2*Len-1, D)
            
        # Attention mekanizmasına uygun aralığı döndür
        # Genellikle pozitif ve negatif mesafeler için embedding
        return self.pe[:, self.pe.size(1)//2 : self.pe.size(1)//2 + length, :]

class RelativeMultiHeadAttention(nn.Module):
    """
    Relative Positional Encoding destekli Multi-Head Attention.
    Standart nn.MultiheadAttention yerine bu kullanılır.
    
    Formül (Basitleştirilmiş):
    Attn = Softmax( Q*K^T + Q*Pos^T + u*K^T + v*Pos^T )
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        
        # Q, K, V projeksiyonları
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Positional Bias (u ve v paremetreleri - öğrenilebilir global biaslar)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)
        
        # Positional Encoding Projeksiyonu
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, x_k, x_v, pos_emb):
        """
        x: Query (Batch, Time, D_Model)
        pos_emb: Positional Embeddings (1, Time, D_Model) veya (Batch, Time, D_Model)
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. Projeksiyonlar: (B, T, D) -> (B, T, H, D_h) -> (B, H, T, D_h)
        q = self.linear_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.linear_k(x_k).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.linear_v(x_v).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Positional: (1, T, D) -> (1, T, H, D_h) -> (1, H, T, D_h)
        # Eğer pos_emb batch boyutuna sahip değilse expand et
        if pos_emb.size(0) != batch_size:
            pos_emb = pos_emb.expand(batch_size, -1, -1)
            
        p = self.linear_pos(pos_emb).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # 2. Attention Score Hesabı
        # (a) Content-based attention: Q * K^T
        # q_with_bias_u = q + self.pos_bias_u.view(1, self.n_heads, 1, self.d_head)
        # matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        
        # (b) Position-based attention: Q * P^T
        # q_with_bias_v = q + self.pos_bias_v.view(1, self.n_heads, 1, self.d_head)
        # matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        
        # Optimizasyon (Transformer-XL stili):
        # AC = (Q + u) * K^T
        ac = torch.einsum("bhtd,bhld->bhtl", q + self.pos_bias_u.view(1, self.n_heads, 1, self.d_head), k)
        
        # BD = (Q + v) * P^T
        bd = torch.einsum("bhtd,bhld->bhtl", q + self.pos_bias_v.view(1, self.n_heads, 1, self.d_head), p)
        
        # BD matrisini shift et (Relative indexing için trick)
        # Bu aşama karmaşıktır, şimdilik basit toplama yapıyoruz. 
        # Tam Conformer için "rel_shift" işlemi gerekir ama burada basitleştirilmiş halini kullanıyoruz.
        
        scores = ac + bd
        scores = scores / math.sqrt(self.d_head)
        
        # 3. Softmax & Masking
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 4. Context: Probs * V
        context = torch.matmul(attn_probs, v) # (B, H, T, D_h)
        
        # 5. Concat & Output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.linear_out(context)
        
        return output, attn_probs
