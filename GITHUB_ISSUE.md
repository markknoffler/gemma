# Issue: `_decode_bytes` crashes on invalid UTF-8 sequences

## Description

The `_decode_bytes` helper function in `gemma/gm/data/_tasks.py` crashes with `UnicodeDecodeError` when encountering invalid UTF-8 byte sequences. This prevents data processing pipelines from handling datasets (especially TFDS datasets) that may contain corrupted or non-UTF-8 bytes.

## Current Behavior

The function attempts to decode bytes as UTF-8 without error handling:

```python
def _decode_bytes(element):
  if isinstance(element, bytes):
    return element.decode("utf-8")  # Crashes on invalid UTF-8
  else:
    return element
```

When invalid UTF-8 sequences are encountered (e.g., `bytes([0xFF, 0xFE, 0xFD])`), the function raises `UnicodeDecodeError`, causing the entire data processing pipeline to crash.

## Expected Behavior

The function should handle invalid UTF-8 sequences gracefully by:
1. Replacing invalid bytes with the Unicode replacement character (U+FFFD)
2. Issuing a warning to inform users about data quality issues
3. Allowing the data processing pipeline to continue

## Impact

- **Crashes entire data processing pipelines** when datasets contain invalid UTF-8 bytes
- **No graceful degradation** - valid data cannot be processed if any invalid bytes exist
- **Poor error messages** - `UnicodeDecodeError` doesn't clearly indicate the issue is with data encoding
- Affects both `Seq2SeqTask` and `ContrastiveTask` which use this helper function

## Reproduction

```python
from gemma.gm.data._tasks import _decode_bytes

# This crashes with UnicodeDecodeError
invalid_bytes = bytes([0xFF, 0xFE, 0xFD])
result = _decode_bytes(invalid_bytes)
```

**Error:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
```

## Proposed Solution

Add error handling with `errors='replace'` parameter and issue warnings:

```python
def _decode_bytes(element):
  if isinstance(element, bytes):
    try:
      return element.decode("utf-8")
    except UnicodeDecodeError as e:
      warnings.warn(
          f"Encountered invalid UTF-8 byte sequence at position {e.start}-{e.end}: "
          f"{e.object[e.start:e.end]!r}. Replacing with Unicode replacement "
          "character (U+FFFD).",
          UnicodeWarning,
          stacklevel=2,
      )
      return element.decode("utf-8", errors="replace")
  else:
    return element
```

**Benefits:**
- Preserves data pipeline flow
- Marks corrupted data with U+FFFD (visible to users)
- Issues warnings for debugging
- Maintains backward compatibility for valid inputs

## Environment

- Python version: Any
- Gemma version: Latest (as of issue creation)
- Affected file: `gemma/gm/data/_tasks.py`
