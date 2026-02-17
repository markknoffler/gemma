"""Demo script that tests Gemma2 tool-token bug fix.

This script uses the actual Sampler.sample() code path. When the fix is NOT
applied, it crashes with AttributeError when accessing BEGIN_OF_TOOL_RESPONSE
at line ~350 in _sampler.py. When the fix IS applied, it gets past that line.

Run (after installing the project in editable mode):

  python demo_gemma2_tool_bug.py
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from gemma.gm.nn import Output
from gemma.gm.text import _sampler as sampler_lib
from gemma.gm.text import _tokenizer as tokenizer_lib
from gemma.gm.utils import _types as types_lib


class _DummyInfo:
  tokenizer_version = 2
  default_ckpt = None


@dataclass
class _DummyConfig:
  input_config: types_lib.InputConfig = types_lib.InputConfig(
      support_images=False,
      num_tokens_per_image=0,
      special_tokens=tokenizer_lib._Gemma2SpecialTokens,
  )
  num_embed: int = 256000


class _DummyModel:
  INFO = _DummyInfo()
  config = _DummyConfig()

  def init_cache(self, *, batch_size, dtype, cache_length, sharding=None):
    return {}

  def apply(self, variables, *, tokens, images=None, cache=None, positions=None, attention_mask=None, return_last_only=None, return_hidden_states=None):
    batch_size, seq_len = tokens.shape[:2]
    vocab_size = self.config.num_embed
    logits = jnp.zeros((batch_size, seq_len, vocab_size), dtype=jnp.bfloat16)
    if return_last_only:
      logits = logits[:, -1, :]
    return Output(logits=logits, cache=cache, hidden_states=None)


def main() -> None:
  model = _DummyModel()
  params = {'dummy': jnp.array(0.0, dtype=jnp.bfloat16)}
  sampler = sampler_lib.Sampler(model=model, params=params)
  sampler.sample("test")


if __name__ == "__main__":
  main()
