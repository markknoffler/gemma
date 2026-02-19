"""Demo script for bug #2: broken error messages in token normalization.

The ValueError message uses {token!r} as literal text instead of an f-string,
so the error shows {token!r} instead of the actual token value.

Run: python demo_bug2_fstring_token.py
"""

from gemma.gm.text import _sampler as sampler_lib
from gemma.gm.text import _tokenizer as tokenizer_lib

tokenizer = tokenizer_lib.Tokenizer.from_version(3)
# "hello world" tokenizes to multiple tokens, triggering the ValueError
sampler_lib._normalize_token(tokenizer, "hello world")
