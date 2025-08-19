import regex
from collections import Counter, defaultdict
import time
from collections.abc import Iterable

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Pair():
  def __init__(self, bs1:bytes, bs2:bytes, bs1_id:int, bs2_id:int):
    self.bs1 = bs1
    self.bs2 = bs2
    self.bs1_id = bs1_id
    self.bs2_id = bs2_id

  def to_tuple(self):
    return (self.bs1, self.bs2)
  
  def __hash__(self):
    return ("%s|%s" % (self.bs1_id, self.bs2_id)).__hash__()
  
  def __eq__(self, value):
    return self.bs1 == value.bs1 and self.bs2 == value.bs2
  
  def __repr__(self):
    return (self.bs1, self.bs2).__repr__()


def construct_word_counter(text: list[str]) -> Counter[str]:
  counter = Counter()
  for t in text:
    for m in regex.finditer(PAT, t):
      counter[m[0]] += 1
  return counter


class Word():
  def __init__(self, word, count):
    self.word = word
    self.count = count
    self.encode = list(word.encode("utf-8"))

  def get_bs(self, i):
    return self.encode[i]

  def update_encode(self, pair:Pair, merged_id:int):
    new_encode = list()
    it = 0
    match_count = 0
    lost_pairs = list()
    new_pairs = list()
    while it < len(self.encode):
      if it < len(self.encode) - 1 and self.encode[it] == pair.bs1_id and self.encode[it+1] == pair.bs2_id:
        new_encode.append(merged_id)
        if it > 0:
          lost_pairs.append((self.encode[it-1], self.encode[it]))
          new_pairs.append((self.encode[it-1], merged_id))
        if it + 1 < len(self.encode) - 1:
          lost_pairs.append((self.encode[it+1], self.encode[it+2]))
          new_pairs.append((merged_id, self.encode[it+2]))
        it += 2
        match_count += 1
      else:
        new_encode.append(self.encode[it])
        it += 1
    self.encode = new_encode
    return match_count, lost_pairs, new_pairs

  def __repr__(self):
    return self.word
  
  def __hash__(self):
    return self.word.__hash__()
  
  def __eq__(self, value):
    return self.word == value.word


def get_pairs(vocab:dict[int, bytes], word:Word) -> list[Pair]:
  encoded = word.encode
  pairs = list()
  for i in range(len(encoded)-1):
    bs1_id = word.get_bs(i)
    bs2_id = word.get_bs(i+1)
    pairs.append(Pair(vocab[bs1_id], vocab[bs2_id], bs1_id, bs2_id))
  return pairs


def build_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  
  start = time.time()
  
  with open(input_path) as fp:
    text = fp.read()
    print("file loaded, cost: %ss" % (time.time() - start))
  word_counter = construct_word_counter(regex.split("|".join([regex.escape(x) for x in special_tokens]), text))
  words = {word: Word(word, count) for word, count in word_counter.items()}
  print("construct word counter, cost: %ss" % (time.time() - start))
  
  # init vocab
  vocab = dict[int, bytes]()
  merges = list[tuple[bytes, bytes]]()
  idx = 0
  # special_token in vocab
  for i in range(0, 256):
    vocab[idx] = bytes([i])
    idx += 1
  for special_token in special_tokens:
    vocab[idx] = special_token.encode("utf-8")
    idx += 1
  
  pair_counter = Counter[Pair]()
  pair_to_word_cache = defaultdict[Pair, set[Word]](set)
  for word in words.values():
      pairs = get_pairs(vocab, word)
      for pair in pairs:
        pair_counter[pair] += word.count
        pair_to_word_cache[pair].add(word)
  print("init pair counter, cost: %ss" % (time.time() - start))

  while len(vocab) < vocab_size:
    if len(vocab) % 1000 == 0:
      print("processing %s, cost: %s", (len(vocab), time.time() - start))
    items = list(pair_counter.items())
    pair, _ = max(items, key=lambda x:(x[1], x[0].to_tuple()))
    merges.append(pair.to_tuple())
    vocab[idx] = b"".join(pair.to_tuple())
    del pair_counter[pair]

    word_set = pair_to_word_cache[pair]
    for word in word_set:
      _, lost_pairs, new_pairs = word.update_encode(pair, idx)
      for id1, id2 in lost_pairs:
        lp = Pair(vocab[id1], vocab[id2], id1, id2)
        pair_counter[lp] -= word.count
      for id1, id2 in new_pairs:
        np = Pair(vocab[id1], vocab[id2], id1, id2)
        pair_counter[np] += word.count
    
    for word in word_set:
      for p in get_pairs(vocab, word):
        pair_to_word_cache[p].add(word)

    del pair_to_word_cache[pair]
    idx += 1

  print("done, cost: %ss" % (time.time() - start))
  return vocab, merges


class Tokenizer():
  def __init__(self, vocab:dict[int, bytes], merges:list[tuple[bytes, bytes]], special_tokens:list[str] | None=None):
    self.vocab = vocab
    self.vocab_revert = {y: x for x, y in vocab.items()}
    self.merges = {y: x for x, y in enumerate(merges)}
    if special_tokens is not None:
      special_tokens.sort(key=lambda x: -len(x))
      self.special_tokens_pattern = regex.compile("|".join([regex.escape(x) for x in special_tokens]))
    else:
      self.special_tokens_pattern = None

  @staticmethod
  def from_files(vocab_filepath:str, merge_filepath:str, special_tokens:list[str] | None=None):
    import pickle
    with open(vocab_filepath, "rb") as vp:
      with open(merge_filepath, "rb") as mp:
        vocab = pickle.load(vp)
        merges = pickle.load(mp)
        return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
      
  def _encode(self, text:str) -> list[int]:
    result = list[int]()
    words= regex.findall(PAT, text)
    for word in words:
      encode = [self.vocab_revert[bytes([x])] for x in word.encode("utf-8")]
      merged = True
      while merged:
        idx, merged_id, priority = -1, -1, len(self.merges)
        for i in range(len(encode) - 1):
          target = (self.vocab[encode[i]], self.vocab[encode[i+1]])
          if target in self.merges:
            if priority > self.merges[target]:
              priority = self.merges[target]
              idx = i
              merged_id = self.vocab_revert[b"".join(target)]
        merged = idx >= 0
        if merged:
            encode[idx] = merged_id
            del encode[idx+1]
      result.extend(encode)
    return result

  def encode(self, text:str) -> list[int]:
    if self.special_tokens_pattern is None:
      return self._encode(text)
    
    result = list()
    pos = 0
    while True:
      m = self.special_tokens_pattern.search(text, pos)
      if m is None:
        chuck = text[pos:]
        result.extend(self._encode(chuck))
        break
      matched_token, start, end = m[0], m.start(), m.end()
      special_token_id = self.vocab_revert[matched_token.encode("utf-8")]
      chuck = text[pos:start]
      result.extend(self._encode(chuck))
      result.append(special_token_id)
      pos = end
    return result

  def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
    for text in iterable:
      for x in self.encode(text):
        yield x

  def decode(self, ids: list[int]) -> str:
    bs = b"".join([self.vocab[id] for id in ids if id in self.vocab])
    return bs.decode("utf-8", errors="replace")


if __name__ == "__main__":
  pass