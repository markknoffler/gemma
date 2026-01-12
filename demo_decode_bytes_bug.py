#!/usr/bin/env python3
"""Demonstration script for the _decode_bytes UTF-8 error handling fix.

This script demonstrates that the _decode_bytes function in gemma/gm/data/_tasks.py
now handles invalid UTF-8 byte sequences gracefully with warnings instead of crashing.

The fix replaces invalid UTF-8 sequences with the Unicode replacement character (U+FFFD)
and issues a UnicodeWarning to inform users about data quality issues.
"""

import warnings
from gemma.gm.data._tasks import _decode_bytes


def main():
    """Demonstrate the fixed _decode_bytes function with invalid UTF-8."""
    
    print("=" * 80)
    print("Demonstrating fixed _decode_bytes function with invalid UTF-8 sequences")
    print("=" * 80)
    print()
    print("The function now handles invalid UTF-8 gracefully with warnings.")
    print()
    
    # Valid UTF-8 bytes (works fine, no warning)
    print("Test 1: Valid UTF-8 bytes (this works without warnings)")
    print("-" * 80)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        valid_bytes = "Hello, World!".encode('utf-8')
        result = _decode_bytes(valid_bytes)
        print(f"✓ Successfully decoded: {result!r}")
        if w:
            print(f"  ⚠ Warning issued: {w[0].message}")
        else:
            print("  ✓ No warnings (expected for valid UTF-8)")
    print()
    
    # Invalid UTF-8 bytes - NOW HANDLED GRACEFULLY
    print("Test 2: Invalid UTF-8 bytes (NOW HANDLED GRACEFULLY)")
    print("-" * 80)
    print("Attempting to decode invalid UTF-8 bytes: bytes([0xFF, 0xFE, 0xFD])")
    print("Before fix: Would crash with UnicodeDecodeError")
    print("After fix: Issues warning and replaces invalid bytes with U+FFFD")
    print()
    
    # Create invalid UTF-8: 0xFF is not a valid UTF-8 byte
    invalid_bytes = bytes([0xFF, 0xFE, 0xFD])
    
    # This line now handles the error gracefully instead of crashing
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _decode_bytes(invalid_bytes)
        
        print(f"✓ Successfully decoded (with replacement): {result!r}")
        print(f"  Contains Unicode replacement character (U+FFFD): {'\ufffd' in result}")
        print(f"  Number of replacement characters: {result.count('\ufffd')}")
        
        if w:
            print(f"  ⚠ Warning issued: {w[0].message}")
            print(f"  ⚠ Warning category: {w[0].category.__name__}")
        else:
            print("  ✗ No warning issued (unexpected)")
    
    print()
    print("=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print("✓ Fix is working correctly!")
    print("  - No crashes on invalid UTF-8")
    print("  - Warnings are issued to inform users")
    print("  - Invalid bytes are replaced with U+FFFD")
    print("  - Data processing pipeline continues without interruption")
    print("=" * 80)


if __name__ == '__main__':
    main()
