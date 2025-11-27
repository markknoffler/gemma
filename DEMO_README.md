# Pad Length Tuple Bug Demo

This demo script demonstrates a type signature mismatch bug in the Gemma `Sampler` class.

## Bug Description

The `Sampler` class has a default `pad_length` value of `(256, 512, 1024)` (a tuple), but the `_tokenize_prompts` method only accepts `int | None` for its `pad_length` parameter. This creates a type signature mismatch that can cause `TypeError` if the tuple is ever passed to `_tokenize_prompts`.

## How to Run

```bash
# Make sure you're in the gemma repository root
cd /path/to/gemma

# Run the demo script
python demo_pad_length_bug.py
```

Or if you have the gemma package installed:

```bash
python3 demo_pad_length_bug.py
```

## What the Demo Shows

1. **Default Configuration**: Shows that `Sampler.pad_length` defaults to a tuple `(256, 512, 1024)`
2. **Type Signature Mismatch**: Shows that `_tokenize_prompts` only accepts `int | None`
3. **Bug Demonstration**: Directly calls `_tokenize_prompts` with the tuple to show the `TypeError`
4. **Root Cause**: Explains why `pad_length or max(...)` fails when `pad_length` is a tuple
5. **Workaround**: Shows that setting `pad_length=None` avoids the issue

## Expected Output

The script will:
- Show the type mismatch between `Sampler.pad_length` (tuple) and `_tokenize_prompts.pad_length` (int|None)
- Demonstrate the bug by passing the tuple directly to `_tokenize_prompts`
- Catch and display the `TypeError` that occurs
- Explain the root cause (tuple is truthy, so `or` returns tuple instead of calculating max)
- Show a workaround

## Current Status

**Note**: The bug doesn't currently manifest in normal usage because `_get_inputs()` doesn't pass `pad_length` to `_tokenize_prompts`. However, this is still a bug because:

1. The type signatures are inconsistent
2. If code is refactored to pass `pad_length`, it will crash
3. If someone tries to use `_tokenize_prompts` directly with the default `pad_length`, it fails
4. The API is confusing - `pad_length` can be a tuple but can't be used everywhere

## Fix Suggestions

1. **Option 1**: Make `_tokenize_prompts` accept `tuple[int, ...] | None` and handle bucket selection
2. **Option 2**: Change `Sampler.pad_length` default to `None` instead of tuple
3. **Option 3**: Add logic in `_tokenize_prompts` to handle tuples by selecting the appropriate bucket

## Related Code Locations

- `gemma/gm/text/_sampler.py:129` - `pad_length` default value
- `gemma/gm/text/_sampler.py:410` - `_tokenize_prompts` signature
- `gemma/gm/text/_sampler.py:418` - Bug location: `pad_length or max(...)`
