# Solution: Fix `_decode_bytes` to handle invalid UTF-8 gracefully

## Problem

The `_decode_bytes` function in `gemma/gm/data/_tasks.py` was crashing with `UnicodeDecodeError` when encountering invalid UTF-8 byte sequences, causing entire data processing pipelines to fail.

## Solution

Added error handling to gracefully handle invalid UTF-8 sequences by:

1. **Wrapping the decode operation in a try-except block** to catch `UnicodeDecodeError`
2. **Using `errors='replace'` parameter** to replace invalid bytes with the Unicode replacement character (U+FFFD)
3. **Issuing a `UnicodeWarning`** to inform users about data quality issues

## Implementation

### Changes Made

**File:** `gemma/gm/data/_tasks.py`

1. **Added import:**
   ```python
   import warnings
   ```

2. **Updated `_decode_bytes` function:**
   ```python
   def _decode_bytes(element):
     """Decode bytes to string, handling invalid UTF-8 gracefully.

     Some datasets (e.g., TFDS) return bytes instead of str. This function
     decodes bytes to UTF-8 strings, replacing invalid UTF-8 sequences with
     the Unicode replacement character (U+FFFD) rather than crashing.

     Args:
       element: Either bytes to decode or a non-bytes value to return as-is.

     Returns:
       Decoded string if element is bytes, otherwise the element unchanged.
     """
     if isinstance(element, bytes):
       try:
         return element.decode("utf-8")
       except UnicodeDecodeError as e:
         # Replace invalid UTF-8 sequences with the Unicode replacement character.
         # This is safer than 'ignore' as it preserves data flow while marking
         # corrupted bytes, and is better than crashing the entire pipeline.
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

## Design Decisions

### Why `errors='replace'` instead of `errors='ignore'`?

- **`errors='replace'`** replaces invalid bytes with U+FFFD (), which:
  - Preserves the data flow and length information
  - Makes corrupted data visible in the output
  - Allows users to identify and filter problematic data if needed

- **`errors='ignore'`** would silently remove invalid bytes, which:
  - Loses information about data corruption
  - Can cause subtle bugs if data length matters
  - Makes it harder to detect and debug data quality issues

### Why issue warnings?

- **Informs users** about data quality issues without breaking the pipeline
- **Helps with debugging** by providing context about where corruption occurred
- **Allows users to filter warnings** if they choose to ignore them
- **Follows Python best practices** for handling recoverable errors

## Behavior

### Before Fix

```python
from gemma.gm.data._tasks import _decode_bytes

invalid_bytes = bytes([0xFF, 0xFE, 0xFD])
result = _decode_bytes(invalid_bytes)
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
```

### After Fix

```python
from gemma.gm.data._tasks import _decode_bytes
import warnings

invalid_bytes = bytes([0xFF, 0xFE, 0xFD])
with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    result = _decode_bytes(invalid_bytes)
    # result: '\ufffd\ufffd\ufffd' (Unicode replacement characters)
    # Warning issued: UnicodeWarning with details about the error
```

### Mixed Valid/Invalid Bytes

```python
mixed_bytes = b"Valid text " + bytes([0xFF, 0xFE]) + b" more valid text"
result = _decode_bytes(mixed_bytes)
# result: 'Valid text \ufffd\ufffd more valid text'
# Valid parts are preserved, only invalid bytes are replaced
```

## Benefits

1. **Prevents pipeline crashes** - Data processing continues even with corrupted bytes
2. **Preserves valid data** - Valid UTF-8 sequences are decoded correctly
3. **Marks corrupted data** - Invalid bytes are replaced with visible U+FFFD characters
4. **User awareness** - Warnings inform users about data quality issues
5. **Backward compatible** - Valid inputs work exactly as before
6. **No performance impact** - Only adds overhead when invalid UTF-8 is encountered

## Testing

The fix handles:
- ✅ Valid UTF-8 bytes (works as before, no warnings)
- ✅ Invalid UTF-8 bytes (replaced with U+FFFD, warning issued)
- ✅ Mixed valid/invalid bytes (valid parts preserved, invalid replaced)
- ✅ Non-bytes input (returned as-is, no changes)

## Impact

- **Affected functions:** `Seq2SeqTask.map()` and `ContrastiveTask.map()` which use `_decode_bytes`
- **Backward compatibility:** Fully maintained - no breaking changes
- **Performance:** Negligible impact (only when invalid UTF-8 is encountered)
