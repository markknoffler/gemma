#!/usr/bin/env python3
"""Demonstration script for the _decode_bytes UTF-8 error handling bug.

This script demonstrates that the _decode_bytes function in gemma/gm/data/_tasks.py
crashes with a UnicodeDecodeError when encountering invalid UTF-8 byte sequences.

Run this script to see the bug in action - it will crash with a UnicodeDecodeError.
"""

from gemma.gm.data._tasks import _decode_bytes


def main():
    """Demonstrate the _decode_bytes bug with invalid UTF-8."""
    
    print("=" * 80)
    print("Demonstrating _decode_bytes bug with invalid UTF-8 sequences")
    print("=" * 80)
    print()
    print("This script will crash when _decode_bytes encounters invalid UTF-8 bytes.")
    print()
    
    # Valid UTF-8 bytes (works fine)
    print("Test 1: Valid UTF-8 bytes (this works)")
    valid_bytes = "Hello, World!".encode('utf-8')
    result = _decode_bytes(valid_bytes)
    print(f"✓ Successfully decoded: {result!r}")
    print()
    
    # Invalid UTF-8 bytes - THIS WILL CRASH
    print("Test 2: Invalid UTF-8 bytes (THIS WILL CRASH)")
    print("-" * 80)
    print("Attempting to decode invalid UTF-8 bytes: bytes([0xFF, 0xFE, 0xFD])")
    print("This will raise UnicodeDecodeError because 0xFF is not a valid UTF-8 byte.")
    print()
    
    # Create invalid UTF-8: 0xFF is not a valid UTF-8 byte
    invalid_bytes = bytes([0xFF, 0xFE, 0xFD])
    
    # This line will crash with UnicodeDecodeError
    # The error will be displayed with full traceback
    result = _decode_bytes(invalid_bytes)
    
    # This line will never be reached
    print(f"Unexpected: decoded to {result!r}")


if __name__ == '__main__':
    main()
