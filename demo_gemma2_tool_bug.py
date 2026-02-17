"""Minimal script to demonstrate Gemma2 tool-token crash.

The bug: gm.text.Sampler always builds end_tokens using
tokenizer.special_tokens.BEGIN_OF_TOOL_RESPONSE, but Gemma2Tokenizer's
special-tokens enum does not define that attribute. So sampling with a
Gemma2 model (or tokenizer version 2) raises AttributeError.

This demo triggers the exact same line that fails in _sampler.py (around
line 350) without needing a real model or params, so the raw traceback
is short and clear.

Run (after installing the project in editable mode):

  python demo_gemma2_tool_bug.py
"""

from __future__ import annotations

from gemma.gm.text import _tokenizer as tokenizer_lib


def main() -> None:
  # Same tokenizer the Sampler uses when model.INFO.tokenizer_version == 2.
  tokenizer = tokenizer_lib.Tokenizer.from_version(2)

  # This is the same logic as in _sampler.py when building SamplerLoop.end_tokens.
  # Gemma2 special tokens do not define BEGIN_OF_TOOL_RESPONSE, so this raises.
  end_tokens = (
      tokenizer.special_tokens.EOS,
      tokenizer.special_tokens.END_OF_TURN,
      tokenizer.special_tokens.BEGIN_OF_TOOL_RESPONSE,  # AttributeError here
  )
  # Unreachable; just for clarity.
  assert end_tokens


if __name__ == "__main__":
  main()
