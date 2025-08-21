from cs336_basics.module import Transformer
from cs336_basics.tokenizer import Tokenizer, SimpleChineseTokenizer
from cs336_basics.utils import softmax
from collections.abc import Iterable
import einops
import torch

class Generator():
  def __init__(self, 
               model: Transformer, 
               tokenizer: Tokenizer, 
               max_generate_length: int, 
               stop_word: str="<|endoftext|>", 
               temperature: float=1.0, 
               top_p: int=-1, 
               device=None):
    self.model = model
    self.tokenizer = tokenizer
    self.max_generate_length = max_generate_length
    self.temperature = temperature
    self.top_p = top_p
    self.stop_token_id = tokenizer.encode(stop_word)[0] if stop_word is not None else -1
    self.device = device

  def generate(self, text: str) -> Iterable[str]:
    generated = 0
    inputs = einops.rearrange(torch.Tensor(self.tokenizer.encode(text)), "... -> 1 ...").to(self.device)
    
    while generated < self.max_generate_length:
      # print("inputs", inputs[:, -self.model.get_max_seq_length():])
      outputs = self.model(inputs[:, -self.model.get_max_seq_length():])
      logits = outputs[:, -1, :]
      if self.temperature > 0:
        logits = logits / self.temperature
      probs = softmax(logits, dim=-1)
      if self.top_p > 0:
        V = probs.shape[-1]
        _, index = probs.topk(V - self.top_p, largest=False)
        probs.scatter_(-1, index, 0)
        probs = probs / torch.sum(probs, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)
      yield self.tokenizer.decode(next_token[0].tolist())
      if next_token == self.stop_token_id:
        break
      inputs = torch.concat((inputs, next_token), dim=1)
      generated += 1


if __name__ == "__main__":
  device = "mps"

  vocab_path = "vocab/tiny-story.vocab"
  merges_path = "vocab/tiny-story.merges"

  model_file = "model/LLM-TinyStories.model"

  hlm_file = "data/hlm.txt"
  hlm_data_file = "data/hlm.dat"
  hlm_model_file = "model/hlm.model"

  context_length = 256
  d_model = 512
  num_layers = 4
  num_heads = 16
  d_ff = 1344
  rope_theta = 10000

  stop_word = "<|endoftext|>"
  # stop_word = None

  tokenizer = Tokenizer.from_files(vocab_path, merges_path, [stop_word])
  # tokenizer = SimpleChineseTokenizer(hlm_file)
  model = Transformer(tokenizer.vocab_size(), context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=device)
  from cs336_basics.trainer import load_checkpoint
  # load_checkpoint(model_file, model, None)
  load_checkpoint(model_file, model, None)

  generator = Generator(model, tokenizer, 200, stop_word, device=device)
  
  print("> ", end="")
  text = input().strip()
  while len(text) > 0 and text != "exit":
    print("> ", end="")
    for c in generator.generate(text):
      if c != stop_word:
        print(c, end="")
    print("\n> ", end="")
    text = input().strip()
  
