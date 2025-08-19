import torch
import math
from typing import Optional
from collections.abc import Callable

class AdamW(torch.optim.Optimizer):
  def __init__(self, params, **kwargs):
    super().__init__(params, kwargs)
  
  def step(self, closure: Optional[Callable] = None):
    loss = None if closure is None else closure()
    for group in self.param_groups:
      base_lr, weight_decay, betas, eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
      for p in group["params"]:
        if p.grad is None:
          continue

        state = self.state[p]
        m, v, t = state.get("m", 0), state.get("v", 0), state.get("t", 1)
        grad = p.grad.data
        m = betas[0] * m + (1 - betas[0]) * grad
        v = betas[1] * v + (1 - betas[1]) * (grad ** 2)
        lr = base_lr * math.sqrt(1 - math.pow(betas[1], t)) / (1 - math.pow(betas[0], t))
        p.data -= lr * m / (torch.sqrt(v) + eps)
        p.data -= base_lr * weight_decay * p.data
        state["m"], state["v"], state["t"] = m, v, t+1
    return loss


class CosinLRScheduler():
  def __init__(self, min_lr: float, max_lr: float, warmup_iters: int, cosine_cycle_iters: int):
    self.min_lr = min_lr
    self.max_lr = max_lr
    self.warmup_iters = warmup_iters
    self.cosin_cycle_iters = cosine_cycle_iters
  
  def get_lr(self, iters: int):
    if iters < self.warmup_iters:
      return self.max_lr * iters / self.warmup_iters
    elif iters <= self.cosin_cycle_iters:
      return self.min_lr + 1/2 * (self.max_lr - self.min_lr) * \
        (1 + math.cos(math.pi * (iters - self.warmup_iters) / (self.cosin_cycle_iters - self.warmup_iters)))
    else:
      return self.min_lr