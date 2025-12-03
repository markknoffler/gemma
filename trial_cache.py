import jax
import jax.numpy as jnp
from gemma.gm.nn import _modules

# 1) Build a tiny attention layer that uses local sliding
attn = _modules.Attention(
    num_heads=2,
    num_kv_heads=2,
    features=8,
    head_dim=4,
    attn_type=_modules.AttentionType.LOCAL_SLIDING,
    sliding_window_size=2,
    query_pre_attn_scalar=1.0,
)

# 2) Create dummy inputs
rng = jax.random.PRNGKey(0)
batch_size, seq_len, features = 1, 4, 8
x = jnp.ones((batch_size, seq_len, features))
segment_pos = jnp.arange(seq_len)[None, :]
attn_mask = jnp.ones((batch_size, seq_len, seq_len))

# 3) Initialize parameters with some cache (init requires a cache)
init_cache = _modules.Attention.init_cache(
    cache_size=seq_len,
    num_heads=2,
    head_dim=4,
    batch_size=batch_size,
    dtype=jnp.float32,
)
params = attn.init(rng, x, segment_pos, init_cache, attn_mask)

# 4) Now call apply with cache=None -> this triggers the bug
attn.apply(params, x, segment_pos, None, attn_mask)
# NameError: name 'cache_positions' is not defined
