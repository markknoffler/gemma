"""Minimal script to demonstrate Gemma2 tool-token crash.

This intentionally triggers the bug where `gm.text.Sampler` unconditionally
references `BEGIN_OF_TOOL_RESPONSE` even when using a Gemma2 tokenizer that
does not define this special token.

Run (after installing the project in editable mode):

  python demo_gemma2_tool_bug.py

The script is deliberately minimal and lets the raw traceback surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from gemma.gm.text import _sampler as sampler_lib
from gemma.gm.text import _tokenizer as tokenizer_lib
from gemma.gm.utils import _types as types_lib


@dataclass
class _DummyInputConfig:
  """Minimal InputConfig stub used by the dummy model."""

  # Gemma2 is text-only for this demo, so we disable images.
  support_images: bool = False
  num_tokens_per_image: int = 0
  # The Gemma2 special tokens enum does NOT define BEGIN_OF_TOOL_RESPONSE.
  special_tokens: type[tokenizer_lib._Gemma2SpecialTokens] = (
      tokenizer_lib._Gemma2SpecialTokens
  )


@dataclass
class _DummyConfig:
  """Minimal config stub exposing an InputConfig."""

  input_config: types_lib.InputConfig = types_lib.InputConfig(
      support_images=False,
      num_tokens_per_image=0,
      special_tokens=tokenizer_lib._Gemma2SpecialTokens,
  )


class _DummyInfo:
  """Minimal INFO stub indicating we are a Gemma2 model."""

  # This is what causes Sampler to pick Gemma2Tokenizer.
  tokenizer_version = 2
  default_ckpt = None


class _DummyModel:
  """Deliberately minimal TransformerLike-ish stub.

  The bug surfaces before any model methods are called; we only need the
  attributes that `gm.text.Sampler` touches before constructing SamplerLoop.
  """

  INFO = _DummyInfo()
  config = _DummyConfig()

  # The following methods exist only to satisfy the TransformerLike protocol.
  # They should never actually be reached before the AttributeError is raised.

  def init_cache(self, *, batch_size, dtype, cache_length, sharding=None):
    raise RuntimeError("init_cache should not be reached in this repro script")

  def apply(
      self,
      variables,
      *,
      tokens,
      images=None,
      cache=None,
      positions=None,
      attention_mask=None,
      return_last_only=None,
      return_hidden_states=None,
  ):
    raise RuntimeError("apply should not be reached in this repro script")


def main() -> None:
  # This will construct a Sampler that auto-selects Gemma2Tokenizer based on
  # _DummyModel.INFO.tokenizer_version == 2.
  dummy_model = _DummyModel()

  sampler = sampler_lib.Sampler(
      model=dummy_model,
      params=None,  # Not used before the bug triggers.
  )

  # The following call is expected to raise:
  #
  #   AttributeError: type object '_Gemma2SpecialTokens' has no attribute
  #   'BEGIN_OF_TOOL_RESPONSE'
  #
  # We do not catch it on purpose so that the raw traceback is printed.
  sampler.sample("Hello from Gemma2 without tool tokens!")


if __name__ == "__main__":
  main()
