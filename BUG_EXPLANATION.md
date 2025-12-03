# Detailed Explanation of the `pad_length` Tuple Bug

## Overview

This is a **type signature mismatch bug** that creates an API inconsistency in the `Sampler` class. While it doesn't currently crash in normal usage, it represents a latent bug that can cause crashes if code is refactored or if methods are called directly.

## The Bug in Detail

### 1. Type Signature Mismatch

**Location**: `gemma/gm/text/_sampler.py`

```python
# Line 129: Sampler class definition
pad_length: None | int | tuple[int, ...] = (256, 512, 1024)  # Default is TUPLE

# Line 410: _tokenize_prompts method signature
def _tokenize_prompts(
    self,
    prompt: str | Sequence[str],
    *,
    add_bos: bool,
    pad_length: int | None = None,  # Only accepts INT or NONE, NOT tuple!
) -> Float['B L']:
```

**The Problem**:
- `Sampler.pad_length` can be a `tuple[int, ...]` (defaults to `(256, 512, 1024)`)
- `_tokenize_prompts.pad_length` only accepts `int | None`
- This creates an API inconsistency

### 2. The Problematic Code Path

**Location**: `gemma/gm/text/_sampler.py:418`

```python
def _tokenize_prompts(self, ..., pad_length: int | None = None):
    tokens = [self.tokenizer.encode(p, add_bos=add_bos) for p in prompt]

    # LINE 418: THE BUG IS HERE
    max_prompt_len = pad_length or max(len(t) for t in tokens)
    #                    ^^^^^^^^^
    #                    If pad_length is tuple (256, 512, 1024), this evaluates
    #                    to the tuple because tuples are truthy!
    #                    So max_prompt_len becomes (256, 512, 1024) instead of an int

    max_prompt_len = _max_across_hosts(max_prompt_len)  # Line 421
    #                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                This expects an int, but receives a tuple

    tokens = _functional.pad(tokens, max_length=max_prompt_len)  # Line 424
    #                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                       This expects max_length to be int, but receives tuple
```

### 3. Why It Crashes

When a tuple is passed to `_tokenize_prompts`:

1. **Line 418**: `pad_length or max(...)` evaluates to the tuple `(256, 512, 1024)` because:
   - Tuples are truthy in Python
   - The `or` operator returns the first truthy value
   - So `max_prompt_len = (256, 512, 1024)` (a tuple, not an int!)

2. **Line 421**: `_max_across_hosts(max_prompt_len)` receives a tuple but expects an int
   - Function signature: `def _max_across_hosts(x: int) -> int:`
   - This might work in single-process mode, but will fail in multi-host setups

3. **Line 424**: `_functional.pad(tokens, max_length=max_prompt_len)` receives a tuple
   - Inside `_pad()`, line 78 does: `if seq_length > max_length:`
   - Python tries to compare `int > tuple`, which causes:
   ```
   TypeError: '>' not supported between instances of 'int' and 'tuple'
   ```

### 4. Current Code Flow (Why It Doesn't Crash Normally)

```python
# In sampler.sample():
inputs = self._get_inputs(...)  # Line 302
    ↓
# In _get_inputs():
tokens = self._tokenize_prompts(
    prompt,
    add_bos=add_bos,
    # pad_length is NOT passed here! Uses default None
)  # Line 388-391
    ↓
# In _tokenize_prompts():
pad_length: int | None = None  # Uses default, so pad_length is None
max_prompt_len = None or max(...)  # Works fine, returns int
    ↓
# Later in sampler.sample():
init_state = _prefill.prefill(
    ...,
    pad_length=self.pad_length,  # Tuple IS passed here (line 317)
)
    ↓
# In _prefill.prefill():
pad_length: None | int | tuple[int, ...] = (256, 512, 1024)  # Accepts tuple!
# This works fine because _prefill.prefill() handles tuples correctly
```

**Key Point**: The bug doesn't manifest because `_get_inputs()` doesn't pass `pad_length` to `_tokenize_prompts()`. But this is still a bug because:

1. The type signatures are inconsistent
2. If someone refactors code to pass `pad_length`, it will crash
3. If someone calls `_tokenize_prompts()` directly with `sampler.pad_length`, it crashes
4. The API is confusing - `pad_length` can be a tuple but can't be used everywhere

### 5. The Purpose of `pad_length`

The `pad_length` parameter is used for **static shape optimization** to avoid JAX recompilation:

- **Without padding**: If prompts have different lengths, JAX recompiles for each unique length
- **With padding**: Prompts are padded to fixed "buckets" (256, 512, 1024), reducing recompilations
- **Tuple buckets**: Allows choosing the smallest bucket that fits, optimizing memory usage

The tuple `(256, 512, 1024)` represents different "bucket sizes" - the code picks the smallest bucket that fits the prompt length.

### 6. Why `_prefill.prefill()` Handles Tuples But `_tokenize_prompts()` Doesn't

**`_prefill.prefill()`** (line 59):
```python
pad_length: None | int | tuple[int, ...] = None

if isinstance(pad_length, int):
    pad_length = (pad_length,)  # Converts int to tuple

# Then uses _pad_to_bucket() which handles tuples:
token_length_padded = _pad_to_bucket(input.length_with_mm, pad_lengths)
```

**`_tokenize_prompts()`** (line 410):
```python
pad_length: int | None = None  # Only accepts int or None

# Directly uses pad_length without tuple handling:
max_prompt_len = pad_length or max(...)  # Assumes int or None
```

The inconsistency is that `_prefill.prefill()` was designed to handle bucket padding (tuples), but `_tokenize_prompts()` was not.

## Impact Assessment

### Current Impact: **LOW** (Latent Bug)
- Doesn't crash in normal usage
- Code works as-is

### Potential Impact: **MEDIUM-HIGH** (If Triggered)
- Crashes if `pad_length` tuple is passed to `_tokenize_prompts()`
- Crashes if code is refactored to pass `pad_length`
- Confusing API for developers
- Type safety violation

## Fix Options

### Option 1: Make `_tokenize_prompts()` Accept Tuples (Recommended)
```python
def _tokenize_prompts(
    self,
    prompt: str | Sequence[str],
    *,
    add_bos: bool,
    pad_length: int | tuple[int, ...] | None = None,  # Add tuple support
) -> Float['B L']:
    # ...
    if pad_length is None:
        max_prompt_len = max(len(t) for t in tokens)
    elif isinstance(pad_length, tuple):
        # Use bucket selection logic like _pad_to_bucket()
        max_prompt_len = _pad_to_bucket(max(len(t) for t in tokens), pad_length)
    else:
        max_prompt_len = pad_length
```

### Option 2: Change Default to `None`
```python
pad_length: None | int | tuple[int, ...] = None  # Change default
```

### Option 3: Always Pass `pad_length` to `_tokenize_prompts()` and Fix It
```python
# In _get_inputs():
tokens = self._tokenize_prompts(
    prompt,
    add_bos=add_bos,
    pad_length=self.pad_length,  # Pass it, but fix _tokenize_prompts first!
)
```

## Conclusion

This is a **real bug** that demonstrates:
1. Type signature inconsistency
2. API design flaw
3. Potential for future crashes
4. Confusing developer experience

While it doesn't crash in current usage, it's a valid bug that should be fixed to prevent future issues and improve API consistency.

