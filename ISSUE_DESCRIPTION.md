# Type mismatch: `_tokenize_prompts()` doesn't accept tuple `pad_length` but `Sampler.pad_length` defaults to tuple

## Problem

There's a type signature mismatch between `Sampler.pad_length` and `_tokenize_prompts()` that causes a `TypeError` when a tuple is passed. The `Sampler` class defaults `pad_length` to `(256, 512, 1024)` which is a tuple but `_tokenize_prompts()` only accepts `int | None` and doesn't handle tuples properly.

Looking at `gemma/gm/text/_sampler.py` line 129 shows `pad_length: None | int | tuple[int, ...] = (256, 512, 1024)` but at line 410 the method signature only accepts `pad_length: int | None = None` and the logic at line 418 uses `max_prompt_len = pad_length or max(len(t) for t in tokens)` which breaks when `pad_length` is a tuple because tuples are truthy so the `or` operator returns the tuple itself instead of calculating max length.

## Error Details

When calling `_tokenize_prompts()` directly with the tuple `pad_length` you get `TypeError: '>' not supported between instances of 'int' and 'tuple'`. The traceback shows the error originates from `_functional.py` line 78 where it tries to compare `seq_length > max_length` but `max_length` is a tuple `(256, 512, 1024)` instead of an integer.

## Solution

The fix updates the type signature and replaces the buggy logic:

1. Changed `pad_length: int | None = None` to `pad_length: int | tuple[int, ...] | None = None`
2. Replaced `max_prompt_len = pad_length or max(len(t) for t in tokens)` with proper handling for all three cases:
   - `None`: use actual max length
   - `int`: use that value directly
   - `tuple`: iterate through bucket sizes and pick the smallest one that fits

This makes `_tokenize_prompts()` consistent with `_prefill.prefill()` which already handles tuples correctly.

I'll submit a PR with this fix shortly.
