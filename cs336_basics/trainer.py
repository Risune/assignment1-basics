import torch
import random
import os
import typing


def get_batch(dataset: torch.Tensor, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
  length = dataset.shape[-1]
  inputs = torch.empty((batch_size, context_length))
  targets = torch.empty((batch_size, context_length))
  for i in range(batch_size):
    start = random.randint(0, length - context_length - 1)
    inputs[i, :] = dataset[start:start+context_length]
    targets[i, :] = dataset[start+1:start+context_length+1]
  return (inputs.to(device), targets.to(device))


def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
  state = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iters": iteration
  }
  torch.save(state, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer):
  state = torch.load(src)
  model.load_state_dict(state["model"])
  optimizer.load_state_dict(state["optimizer"])
  return state["iters"]


if __name__ == "__main__":
  for _ in range(10):
    print(random.randint(1, 2))