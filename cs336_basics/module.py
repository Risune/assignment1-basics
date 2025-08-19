import torch
import torch.nn as nn
import einops
import math
from . import utils


class Linear(nn.Module):
  def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
    super().__init__()
    weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
    std = math.sqrt(2 / (in_features + out_features))
    nn.init.trunc_normal_(weight, 0, std, -3*std, 3*std)
    self.weight = nn.Parameter(weight, requires_grad=True)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
  def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
    super().__init__()
    embedding = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
    nn.init.trunc_normal_(embedding, 0, 1, -3, 3)
    self.embedding = nn.Parameter(embedding, requires_grad=True)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T = x.shape
    flat_x = einops.rearrange(x, "b t -> (b t)")
    flat_y = torch.index_select(self.embedding, 0, flat_x)
    return einops.rearrange(flat_y, "(b t) ... -> b t ...", b=B, t=T)


class RoPE(nn.Module):
  def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
    super().__init__()
    angle = 1.0 / torch.pow(theta, torch.arange(0, d_k, 2, device=device).float() / d_k)
    pos_angle = einops.einsum(angle, torch.arange(0, max_seq_len, device=device), "d, t -> t d")
    cos = einops.repeat(torch.cos(pos_angle), "... d -> ...(d 2)")
    sin = einops.repeat(torch.sin(pos_angle), "... d -> ...(d 2)")
    neg_trans = torch.ones(d_k, device=device)
    neg_trans[1::2] = -neg_trans[1::2]
    self.d_k = d_k
    self.register_buffer("cos", cos, persistent=False)
    self.register_buffer("sin", sin, persistent=False)
    self.register_buffer("neg_trans", neg_trans, persistent=False)
  
  def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
    length = token_positions.shape[-1]
    truc_cos = self.cos[..., :length, :]
    truc_sin = self.sin[..., :length, :]

    neg_trans = einops.rearrange(x * self.neg_trans, "... (d d1) -> ... d d1", d1=2)
    neg_trans = einops.rearrange(neg_trans.flip(-1), "... d d1 -> ... (d d1)", d1=2)

    return x * truc_cos + neg_trans * truc_sin
    

class RMSNorm(nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    super().__init__()
    self.d_model = d_model
    self.eps = eps
    self.g = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype), requires_grad=True)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    rms = torch.sqrt((einops.reduce(torch.pow(x, 2), "... d2 -> ... 1", "sum")) / self.d_model + self.eps)
    result = x / rms * self.g
    return result.to(dtype)
  

class SwiGlu(nn.Module):
  def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
    super().__init__()
    self.w1 = Linear(d_model, d_ff, device, dtype)
    self.w2 = Linear(d_ff, d_model, device, dtype)
    self.w3 = Linear(d_model, d_ff, device, dtype)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    xw1 = self.w1(x)
    silu = xw1 * torch.sigmoid(xw1)
    xw3 = self.w3(x)
    xw1_dot_xw3 = silu * xw3
    return self.w2(xw1_dot_xw3)


class MultiHeadSelfAttention(nn.Module):
  def __init__(self, d_model: int, num_heads: int,  max_seq_len: int=0, theta: float=0, device=None, dtype=None):
    super().__init__()
    self.num_heads = num_heads
    self.wq = Linear(d_model, d_model, device, dtype)
    self.wk = Linear(d_model, d_model, device, dtype)
    self.wv = Linear(d_model, d_model, device, dtype)
    self.wo = Linear(d_model, d_model, device, dtype)
    if max_seq_len > 0:
      self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device)
    else:
      self.rope = None

  def forward(self, x: torch.Tensor, token_positions: torch.Tensor|None=None) -> torch.Tensor:
    seq_length = x.shape[-2]
    Q, K, V = self.wq(x), self.wk(x), self.wv(x)
    mh_Q, mh_K, mh_V = (einops.rearrange(y, "... seq (heads dh) -> ... heads seq dh", heads=self.num_heads) 
                        for y in [Q, K, V])
    mask = torch.tril(torch.ones(seq_length, seq_length))
    if self.rope is not None:
      mh_Q, mh_K = self.rope(mh_Q, token_positions), self.rope(mh_K, token_positions)
    attn_v = utils.scaled_dot_product_attention(mh_Q, mh_K, mh_V, mask)
    concat_heads = einops.rearrange(attn_v, "... heads seq dh -> ... seq (heads dh)")
    return self.wo(concat_heads)
  

class TransformerBlock(nn.Module):
  def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
    super().__init__()
    self.rms_norm = RMSNorm(d_model)
    self.attn = MultiHeadSelfAttention(d_model, n_heads, max_seq_len, theta, device=device, dtype=dtype)
    self.ffn = nn.Sequential(
      RMSNorm(d_model),
      SwiGlu(d_model, d_ff, device=device, dtype=dtype)
    )
  
  def forward(self, x: torch.Tensor):
    seq_len = x.shape[-2]
    token_positions = torch.arange(seq_len)
    attn = self.attn(self.rms_norm(x), token_positions)
    r1 = attn + x
    return self.ffn(r1) + r1
    

class Transformer(nn.Module):
  def __init__(self, vocab_size: int, context_length: int, d_model: int, 
               num_layers: int, num_heads: int, d_ff: int, rope_theta: float,
               device=None, dtype=None):
    super().__init__()
    self.embedding = Embedding(vocab_size, d_model, device, dtype)
    self.blocks = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
                                  for _ in range(num_layers)])
    self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
    self.final = Linear(d_model, vocab_size, device=device, dtype=dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    emb = self.embedding(x)
    attn = self.blocks(emb)
    return self.final(self.final_norm(attn))


if __name__ == "__main__":
  t = Transformer(50257, 1024, 1600, 48, 25, 6400, 10000)
  print(sum(x.numel() for x in t.parameters() if x.requires_grad))