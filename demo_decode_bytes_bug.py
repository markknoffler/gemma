#!/usr/bin/env python3
"""Demonstration script for the _decode_bytes UTF-8 error handling bug.

This script demonstrates that the _decode_bytes function in gemma/gm/data/_tasks.py
crashes with a UnicodeDecodeError when encountering invalid UTF-8 byte sequences.

Expected behavior: The script should crash with a UnicodeDecodeError.
"""

import sys
from gemma.gm.data._tasks import _decode_bytes


def main():
    """Demonstrate the _decode_bytes bug with invalid UTF-8."""
    
    print("=" * 80)
    print("Demonstrating _decode_bytes bug with invalid UTF-8 sequences")
    print("=" * 80)
    print()
    
    # Test case 1: Valid UTF-8 bytes (should work)
    print("Test 1: Valid UTF-8 bytes")
    print("-" * 80)
    try:
        valid_bytes = "Hello, World!".encode('utf-8')
        result = _decode_bytes(valid_bytes)
        print(f"✓ Success: {result!r}")
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
    print()
    
    # Test case 2: Invalid UTF-8 bytes (should crash)
    print("Test 2: Invalid UTF-8 bytes - Single invalid byte sequence")
    print("-" * 80)
    try:
        # Create invalid UTF-8: 0xFF is not a valid UTF-8 byte
        invalid_bytes_1 = bytes([0xFF, 0xFE, 0xFD])
        result = _decode_bytes(invalid_bytes_1)
        print(f"✗ Unexpected success: {result!r}")
    except UnicodeDecodeError as e:
        print(f"✓ Bug reproduced! UnicodeDecodeError occurred:")
        print(f"  Error: {e}")
        print(f"  Encoding: {e.encoding}")
        print(f"  Start position: {e.start}")
        print(f"  End position: {e.end}")
        print(f"  Object: {e.object!r}")
    except Exception as e:
        print(f"✗ Unexpected error type: {type(e).__name__}: {e}")
    print()
    
    # Test case 3: Another invalid UTF-8 sequence
    print("Test 3: Invalid UTF-8 bytes - Truncated multi-byte sequence")
    print("-" * 80)
    try:
        # Create a truncated multi-byte UTF-8 sequence
        # 0xC0 starts a 2-byte sequence but is followed by an invalid byte
        invalid_bytes_2 = bytes([0xC0, 0x80])
        result = _decode_bytes(invalid_bytes_2)
        print(f"✗ Unexpected success: {result!r}")
    except UnicodeDecodeError as e:
        print(f"✓ Bug reproduced! UnicodeDecodeError occurred:")
        print(f"  Error: {e}")
        print(f"  Encoding: {e.encoding}")
        print(f"  Start position: {e.start}")
        print(f"  End position: {e.end}")
        print(f"  Object: {e.object!r}")
    except Exception as e:
        print(f"✗ Unexpected error type: {type(e).__name__}: {e}")
    print()
    
    # Test case 4: Simulate real-world scenario with mixed valid/invalid
    print("Test 4: Invalid UTF-8 bytes - Mixed valid and invalid")
    print("-" * 80)
    try:
        # Mix of valid and invalid bytes
        mixed_bytes = b"Valid text" + bytes([0xFF, 0xFE]) + b" more text"
        result = _decode_bytes(mixed_bytes)
        print(f"✗ Unexpected success: {result!r}")
    except UnicodeDecodeError as e:
        print(f"✓ Bug reproduced! UnicodeDecodeError occurred:")
        print(f"  Error: {e}")
        print(f"  Encoding: {e.encoding}")
        print(f"  Start position: {e.start}")
        print(f"  End position: {e.end}")
        print(f"  Object: {e.object!r}")
    except Exception as e:
        print(f"✗ Unexpected error type: {type(e).__name__}: {e}")
    print()
    
    # Test case 5: Simulate what happens in Seq2SeqTask
    print("Test 5: Simulating usage in Seq2SeqTask.map() method")
    print("-" * 80)
    try:
        # Simulate a dataset element with bytes (as might come from TFDS)
        element = {
            'prompt': bytes([0xFF, 0xFE]),  # Invalid UTF-8
            'response': b'Valid response'
        }
        
        # This is what happens inside Seq2SeqTask.map()
        prompt = element['prompt']
        prompt = _decode_bytes(prompt)  # This will crash
        
        print(f"✗ Unexpected success: prompt decoded to {prompt!r}")
    except UnicodeDecodeError as e:
        print(f"✓ Bug reproduced in realistic scenario!")
        print(f"  UnicodeDecodeError: {e}")
        print(f"  This would crash the entire data processing pipeline")
        print(f"  when processing datasets with invalid UTF-8 byte sequences.")
    except Exception as e:
        print(f"✗ Unexpected error type: {type(e).__name__}: {e}")
    print()
    
    print("=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print("The _decode_bytes function crashes with UnicodeDecodeError when")
    print("encountering invalid UTF-8 byte sequences. This demonstrates the")
    print("lack of error handling in the function.")
    print()
    print("Expected fix: Add error handling with errors='replace' or")
    print("errors='ignore', or wrap in try-except with informative messages.")
    print("=" * 80)
    
    # Return non-zero exit code to indicate the bug was demonstrated
    return 1


if __name__ == '__main__':
    sys.exit(main())

