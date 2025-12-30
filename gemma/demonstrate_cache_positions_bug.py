#!/usr/bin/env python3
"""Demonstration script for cache_positions undefined variable bug.

This script demonstrates the bug where cache_positions is undefined when:
- cache is None (typical during training)
- attn_type is LOCAL_SLIDING (sliding window attention)

The bug occurs in gemma/gm/nn/_modules.py line 265 where cache_positions
is referenced but only defined inside `if cache is not None:` block.

Expected error: NameError: name 'cache_positions' is not defined

NOTE: This script directly calls the Attention module and does NOT use
any Sampler, tokenization, or padding code paths to avoid triggering
other unrelated bugs (e.g., the pad_length tuple bug).
"""

import jax
import jax.numpy as jnp
from gemma.gm.nn import _modules

# IMPORTANT: We only import and use the Attention module directly.
# We do NOT use any Sampler, tokenizer, or data processing code that
# might trigger the padding bug (pad_length tuple issue).

# Create an Attention module with LOCAL_SLIDING attention type
# This simulates a model configuration that uses sliding window attention
attention = _modules.Attention(
    num_heads=8,
    num_kv_heads=8,
    features=128,
    head_dim=16,
    attn_type=_modules.AttentionType.LOCAL_SLIDING,  # This triggers the bug path
    query_pre_attn_scalar=1.0,
    sliding_window_size=128,  # Required for LOCAL_SLIDING
)

# Initialize the attention module
batch_size = 2
seq_len = 64
embed_dim = 128

# Create dummy input data (simulating training scenario)
rng = jax.random.PRNGKey(0)
x = jnp.ones((batch_size, seq_len, embed_dim), dtype=jnp.float32)
segment_pos = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
attn_mask = jnp.ones((batch_size, seq_len, seq_len), dtype=jnp.bool_)

# Initialize attention module parameters
params = attention.init(rng, x, segment_pos, None, attn_mask)

print("=" * 80)
print("Demonstrating cache_positions undefined variable bug")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  - Attention type: LOCAL_SLIDING")
print(f"  - Sliding window size: 128")
print(f"  - Cache: None (simulating training scenario)")
print(f"  - Input shape: {x.shape}")
print(f"\nAttempting forward pass with cache=None...")
print("-" * 80)

try:
    # This will trigger the bug: cache is None, but cache_positions is referenced
    # at line 265 in _modules.py
    # The bug occurs because:
    # 1. cache_positions is only defined inside `if cache is not None:` (line 234-238)
    # 2. But it's used at line 265: cache_positions=cache_positions if cache else None
    # 3. When cache is None, Python tries to evaluate 'cache_positions' first,
    #    but it was never defined, causing NameError
    #
    # NOTE: We call attention.apply() directly with cache=None.
    # This bypasses all Sampler/tokenization code to avoid the padding bug.
    cache = None  # This is None during training - triggers the bug!
    output = attention.apply(
        params,
        x,
        segment_pos,
        cache,  # None - this triggers the cache_positions bug!
        attn_mask=attn_mask,
    )
    print("ERROR: Expected NameError but forward pass succeeded!")
    print("This means the bug may have been fixed or the code path wasn't reached.")
except NameError as e:
    if "cache_positions" in str(e):
        print(f"\n✓ Bug successfully reproduced!")
        print(f"\nError message: {e}")
        print(f"\nError type: {type(e).__name__}")
        print("\n" + "=" * 80)
        print("BUG EXPLANATION:")
        print("=" * 80)
        print("""
The error occurs because:

1. In gemma/gm/nn/_modules.py, cache_positions is only defined inside
   the `if cache is not None:` block (lines 234-238).

2. However, at line 265, when attn_type is LOCAL_SLIDING, the code tries
   to use cache_positions:

   cache_positions=cache_positions if cache else None

3. When cache is None (training scenario), Python tries to evaluate
   'cache_positions' first before the conditional, but it was never
   defined, causing a NameError.

4. The same issue occurs at line 306 when creating new_cache, though
   that's inside the cache is not None block so it's less likely to
   trigger in practice.

FIX:
Initialize cache_positions = None before the if cache is not None block.
""")
        print("=" * 80)
    else:
        print(f"Unexpected NameError: {e}")
        raise
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
    print("\nThis might indicate the bug has been fixed or a different issue occurred.")
    import traceback
    traceback.print_exc()
