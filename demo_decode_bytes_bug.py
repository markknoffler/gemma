#!/usr/bin/env python3
"""Demonstration script for the _decode_bytes UTF-8 error handling fix.

This script demonstrates that the _decode_bytes function in gemma/gm/data/_tasks.py
now handles invalid UTF-8 byte sequences gracefully with warnings instead of crashing.
"""

import warnings
from gemma.gm.data._tasks import _decode_bytes

# Unicode replacement character (U+FFFD)
REPLACEMENT_CHAR = '\ufffd'


def main():
    """Demonstrate the fixed _decode_bytes function with invalid UTF-8."""
    
    print("=" * 80)
    print("Testing _decode_bytes with various UTF-8 inputs")
    print("=" * 80)
    print()
    
    # Test 1: Valid UTF-8
    print("Test 1: Valid UTF-8 bytes")
    print("-" * 80)
    valid_bytes = "Hello, World!".encode('utf-8')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _decode_bytes(valid_bytes)
        print(f"Input: {valid_bytes}")
        print(f"Output: {result!r}")
        print(f"Warnings: {len(w)}")
    print()
    
    # Test 2: All invalid UTF-8
    print("Test 2: All invalid UTF-8 bytes")
    print("-" * 80)
    invalid_bytes = bytes([0xFF, 0xFE, 0xFD])
    print(f"Input: {invalid_bytes}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _decode_bytes(invalid_bytes)
        print(f"Output: {result!r}")
        print(f"Output (repr): {repr(result)}")
        print(f"Contains U+FFFD: {REPLACEMENT_CHAR in result}")
        print(f"Count of U+FFFD: {result.count(REPLACEMENT_CHAR)}")
        print(f"Length: {len(result)}")
        if w:
            print(f"Warning: {w[0].message}")
    print()
    
    # Test 3: Mixed valid and invalid
    print("Test 3: Mixed valid and invalid UTF-8 bytes")
    print("-" * 80)
    mixed_bytes = b"Valid text " + bytes([0xFF, 0xFE]) + b" more valid text"
    print(f"Input: {mixed_bytes}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _decode_bytes(mixed_bytes)
        print(f"Output: {result!r}")
        print(f"Output (repr): {repr(result)}")
        print(f"Starts with 'Valid text ': {result.startswith('Valid text ')}")
        print(f"Ends with ' more valid text': {result.endswith(' more valid text')}")
        print(f"Contains U+FFFD: {REPLACEMENT_CHAR in result}")
        print(f"Count of U+FFFD: {result.count(REPLACEMENT_CHAR)}")
        if w:
            print(f"Warning: {w[0].message}")
    print()
    
    # Test 4: Realistic example
    print("Test 4: Realistic example with corrupted bytes in middle")
    print("-" * 80)
    realistic_bytes = "Hello, this is valid text!".encode('utf-8') + bytes([0xFF, 0xFE]) + "And this continues.".encode('utf-8')
    print(f"Input bytes length: {len(realistic_bytes)}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _decode_bytes(realistic_bytes)
        print(f"Output: {result!r}")
        print(f"Output length: {len(result)}")
        print(f"First 30 chars: {result[:30]!r}")
        print(f"Last 20 chars: {result[-20:]!r}")
        print(f"Count of U+FFFD: {result.count(REPLACEMENT_CHAR)}")
        if w:
            print(f"Warning: {w[0].message}")
    print()


if __name__ == '__main__':
    main()
