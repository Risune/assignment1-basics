import torch
import einops
import math
from collections.abc import Iterable

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor|None) -> torch.Tensor:
  d_sqrt = math.sqrt(Q.shape[-1])
  qk = einops.einsum(Q, K, "... queries d, ... keys d -> ... queries keys") / d_sqrt
  if mask is not None:
    qk = qk.masked_fill(mask == 0, float("-inf"))
  return einops.einsum(softmax(qk, -1), V, "... queries keys, ... keys dv -> ... queries dv")


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
  max_num = torch.max(x, dim=dim, keepdim=True)
  scaled = x - max_num.values
  exp = torch.exp(scaled)
  sums = torch.sum(exp, dim=dim, keepdim=True)
  return exp / sums


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
  max_num = torch.max(inputs, dim=-1, keepdim=True)
  scaled = inputs - max_num.values
  exp_sum = einops.reduce(torch.exp(scaled), "... d -> ... 1", "sum")
  log_softmax = scaled - torch.log(exp_sum)
  selection = log_softmax.gather(-1, einops.rearrange(targets, "... idx -> ... idx 1"))
  return -selection.mean()


def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
  l2_norm = torch.Tensor([x.grad.norm() for x in parameters if x.grad is not None]).norm()
  if l2_norm <= max_l2_norm:
    return
  for params in parameters:
    grad = params.grad
    if grad is None:
      continue
    params.grad = grad * max_l2_norm / (l2_norm + eps)
