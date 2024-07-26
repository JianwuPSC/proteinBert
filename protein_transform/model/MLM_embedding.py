import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=128):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1) # [0, max_len, 1]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(nn.Embedding):
    def __init__(self, seg_size, embed_size=512):
        super().__init__(seg_size, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding : TokenEmbedding; PositionalEmbedding; SegmentEmbedding
    """

    def __init__(self, vocab_size, max_len, seg_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size default:24
        :param max_len: max sequence length default: max([len(str(text)) for text in lines[:]]) 5446
        :param seg_size: embedding segment class default:4
        :param embed_size: embedding size default feature size: 512
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size, embed_size)
        self.position = PositionalEmbedding(max_len, embed_size)
        self.segment = SegmentEmbedding(embed_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
    

# 1d absolute sin_cos encode
class Absolute_sin_cos_embedding(nn.Module):
    
    def __init__(self, pos_len, dim):
        super(Absolute_sin_cos_embedding, self).__init__()
        
        self.pos_len = pos_len
        self.dim = dim
        
        assert self.dim % 2 == 0, "wrong dimension!"
        position_emb = torch.zeros(self.pos_len, self.dim, dtype=torch.float)
        position_emb.require_grad = False
        # i矩阵
        i_matrix = torch.arange(self.dim//2, dtype=torch.float)
        i_matrix /= self.dim / 2.
        i_matrix = torch.pow(10000, i_matrix)
        i_matrix = 1. / i_matrix
        i_matrix = i_matrix.to(torch.long)
        # pos矩阵
        pos_vec = torch.arange(self.pos_len).to(torch.long)
        # 矩阵相乘，pos变成列向量，i_matrix变成行向量
        out = pos_vec[:, None] @ i_matrix[None, :]
        # 奇/偶数列
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        # 赋值
        position_emb[:, 0::2] = emb_sin
        position_emb[:, 1::2] = emb_cos
        #position_emb.unsqueeze(0)
        self.register_buffer('position_emb', position_emb)
        
    def forward(self, x):
        return self.position_emb[:, :x.size(1)].unsqueeze(0)
