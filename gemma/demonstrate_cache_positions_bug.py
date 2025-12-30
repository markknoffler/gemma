#!/usr/bin/env python3
"""Demonstration script for cache_positions undefined variable bug.

This script demonstrates the bug where cache_positions is undefined when:
- cache is None (typical during training)
- attn_type is LOCAL_SLIDING (sliding window attention)

The bug occurs in gemma/gm/nn/_modules.py line 265 where cache_positions
is referenced but only defined inside `if cache is not None:` block.

Expected error: NameError: name 'cache_positions' is not defined
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

# This will trigger the bug: cache is None, but cache_positions is referenced
# at line 265 in _modules.py when attn_type is LOCAL_SLIDING
cache = None
cache_out, output = attention.apply(
    params,
    x,
    segment_pos,
    cache,
    attn_mask,
)
