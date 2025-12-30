#!/usr/bin/env python3
"""Demonstration script for cache_positions undefined variable issue.

NOTE: This is actually a code quality issue, not a runtime bug.
Python's short-circuit evaluation prevents a runtime error, but the code
is confusing and risky. The pylint warning (# pylint: disable=undefined-variable)
indicates this is a known code quality issue.

The problematic code at line 265 in _modules.py:
  cache_positions=cache_positions if cache else None

When cache is None, Python short-circuits and returns None without
evaluating cache_positions, so no runtime error occurs. However, this
is poor code quality and could cause issues in certain scenarios.
"""

import jax
import jax.numpy as jnp
from gemma.gm.nn import _modules

# Create an Attention module with LOCAL_SLIDING attention type
attention = _modules.Attention(
    num_heads=8,
    num_kv_heads=8,
    features=128,
    head_dim=16,
    attn_type=_modules.AttentionType.LOCAL_SLIDING,
    query_pre_attn_scalar=1.0,
    sliding_window_size=128,
)

# Initialize the attention module
batch_size = 2
seq_len = 64
embed_dim = 128

# Create dummy input data
rng = jax.random.PRNGKey(0)
x = jnp.ones((batch_size, seq_len, embed_dim), dtype=jnp.float32)
segment_pos = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
attn_mask = jnp.ones((batch_size, seq_len, seq_len), dtype=jnp.bool_)

# Initialize attention module parameters
params = attention.init(rng, x, segment_pos, None, attn_mask)

# This demonstrates the code quality issue: cache_positions is referenced
# but only defined inside 'if cache is not None:' block. Python's
# short-circuit evaluation prevents a runtime error, but this is
# confusing and risky code.
cache = None
cache_out, output = attention.apply(
    params,
    x,
    segment_pos,
    cache,
    attn_mask,
)

print("Script completed successfully.")
print("Note: No runtime error occurs due to Python's short-circuit evaluation.")
print("However, this is a code quality issue - cache_positions should be")
print("initialized before the conditional to avoid confusion and potential issues.")
