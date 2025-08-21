import torch
import random
import os
import typing
import numpy as np
from cs336_basics.tokenizer import Tokenizer, SimpleChineseTokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.module import Transformer
from cs336_basics.opt import AdamW
from cs336_basics.utils import cross_entropy_loss, clip_gradient


def get_batch(dataset, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
  length = len(dataset)
  inputs = torch.empty((batch_size, context_length))
  targets = torch.empty((batch_size, context_length))
  for i in range(batch_size):
    start = random.randint(0, length - context_length - 1)
    inputs[i, :] = torch.from_numpy(dataset[start:start+context_length].copy())
    targets[i, :] = torch.from_numpy(dataset[start+1:start+context_length+1].copy())
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
  if model is not None:
    model.load_state_dict(state["model"])
  if optimizer is not None:
    optimizer.load_state_dict(state["optimizer"])
  return state["iters"]


def load_tokened_text_to_file(
    tokenizer: Tokenizer,
    text_file: str,
    mmap_file: str,
    num_processes: int = 16,
    special_token: bytes = b"<|endoftext|>",
    dtype="uint16"):
  token_length = 0
  np_fp = np.memmap(mmap_file, dtype=dtype, mode="w+", shape=(token_length))
  with open(text_file, "rb") as fp:
    boundaries = find_chunk_boundaries(fp, num_processes, special_token)
    print(boundaries)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
      fp.seek(start)
      chunk = fp.read(end - start).decode("utf-8", errors="ignore")
      result = tokenizer.encode(chunk)
      token_length += len(result)
      np_fp = np.memmap(mmap_file, dtype=dtype, mode="r+", shape=(token_length))
      np_fp[-len(result):] = result
      np_fp.flush()


if __name__ == "__main__":
  device = "mps"

  memmap_path = "data/memmap.dat"
  vocab_path = "vocab/tiny-story.vocab"
  merges_path = "vocab/tiny-story.merges"

  train_file = "data/TinyStoriesV2-GPT4-train.txt"
  model_file = "model/LLM-TinyStories.model"

  hlm_file = "data/hlm.txt"
  hlm_data_file = "data/hlm.dat"
  hlm_model_file = "model/hlm.model"

  iters = 5000
  batch_size = 32
  context_length = 256
  d_model = 512
  num_layers = 4
  num_heads = 16
  d_ff = 1344
  rope_theta = 10000

  max_l2_norm = 1e-2

  tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])
  # load_tokened_text_to_file(tokenizer, train_file, memmap_path)
  tokenized_train_data = np.memmap(memmap_path, dtype="uint16", mode="r")

  # tokenizer = SimpleChineseTokenizer(hlm_file)
  # tokenized_train_data = np.memmap(hlm_data_file, dtype="uint16", mode="r")

  model = Transformer(tokenizer.vocab_size(), context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=device)
  optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2, eps=1e-8, betas=(0.9, 0.999))

  for iter in range(iters):
    # sample train data
    inputs, targets = get_batch(tokenized_train_data, batch_size, context_length, device=device)

    # forward and loss
    outputs = model(inputs)
    loss = cross_entropy_loss(outputs, targets)

    if iter % 100 == 0:
      print("iter", iter, "loss", loss)

    # backward and update
    optimizer.zero_grad()
    loss.backward()
    clip_gradient(model.parameters(), max_l2_norm)
    optimizer.step()

  save_checkpoint(model, optimizer, iters, model_file)
  # save_checkpoint(model, optimizer, iters, hlm_model_file)
