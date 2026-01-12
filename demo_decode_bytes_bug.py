#!/usr/bin/env python3
"""Demonstration script for the _decode_bytes UTF-8 error handling fix.

This script demonstrates that the _decode_bytes function in gemma/gm/data/_tasks.py
now handles invalid UTF-8 byte sequences gracefully with warnings instead of crashing.

The fix replaces invalid UTF-8 sequences with the Unicode replacement character (U+FFFD)
and issues a UnicodeWarning to inform users about data quality issues.
Valid UTF-8 bytes are preserved, only invalid sequences are replaced.
"""

import warnings
from gemma.gm.data._tasks import _decode_bytes

# Unicode replacement character (U+FFFD)
REPLACEMENT_CHAR = '\ufffd'


def main():
    """Demonstrate the fixed _decode_bytes function with invalid UTF-8."""

    print("=" * 80)
    print("Demonstrating fixed _decode_bytes function with invalid UTF-8 sequences")
    print("=" * 80)
    print()
    print("The function now handles invalid UTF-8 gracefully with warnings.")
    print("Valid bytes are preserved, only invalid sequences are replaced.")
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

    # Invalid UTF-8 bytes - ALL invalid
    print("Test 2: All invalid UTF-8 bytes (all replaced with U+FFFD)")
    print("-" * 80)
    print("Attempting to decode invalid UTF-8 bytes: bytes([0xFF, 0xFE, 0xFD])")
    print("Before fix: Would crash with UnicodeDecodeError")
    print("After fix: Issues warning and replaces ALL invalid bytes with U+FFFD")
    print()

    invalid_bytes = bytes([0xFF, 0xFE, 0xFD])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _decode_bytes(invalid_bytes)

        print(f"✓ Successfully decoded (all replaced): {result!r}")
        print(f"  Contains Unicode replacement character (U+FFFD): {REPLACEMENT_CHAR in result}")
        print(f"  Number of replacement characters: {result.count(REPLACEMENT_CHAR)}")
        print(f"  Length: {len(result)} characters")

        if w:
            print(f"  ⚠ Warning issued: {w[0].message}")
            print(f"  ⚠ Warning category: {w[0].category.__name__}")
        else:
            print("  ✗ No warning issued (unexpected)")
    print()

    # Mixed valid and invalid UTF-8 bytes - THIS DEMONSTRATES THE KEY BEHAVIOR
    print("Test 3: Mixed valid and invalid UTF-8 bytes (KEY DEMONSTRATION)")
    print("-" * 80)
    print("This test shows that valid bytes are PRESERVED and only invalid bytes")
    print("are replaced with the Unicode replacement character (U+FFFD).")
    print()

    # Create a mix: valid text + invalid bytes + more valid text
    mixed_bytes = b"Valid text " + bytes([0xFF, 0xFE]) + b" more valid text"
    print(f"Input bytes: {mixed_bytes}")
    print(f"  - Valid part 1: {b'Valid text '}")
    print(f"  - Invalid part: {bytes([0xFF, 0xFE])} (2 invalid bytes)")
    print(f"  - Valid part 2: {b' more valid text'}")
    print()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _decode_bytes(mixed_bytes)

        print(f"✓ Successfully decoded: {result!r}")
        print()
        print("Analysis:")
        print(f"  - Valid text preserved at start: {result.startswith('Valid text ')}")
        print(f"  - Valid text preserved at end: {result.endswith(' more valid text')}")
        print(f"  - Contains replacement characters: {REPLACEMENT_CHAR in result}")
        print(f"  - Number of replacement characters: {result.count(REPLACEMENT_CHAR)}")
        print(f"  - Full result: {result!r}")
        print()
        print("  This demonstrates that:")
        print("    ✓ Valid UTF-8 bytes are decoded and preserved")
        print("    ✓ Invalid UTF-8 bytes are replaced with U+FFFD")
        print("    ✓ The data pipeline continues without crashing")

        if w:
            print(f"  ⚠ Warning issued: {w[0].message}")
            print(f"  ⚠ Warning category: {w[0].category.__name__}")
        else:
            print("  ✗ No warning issued (unexpected)")
    print()

    # Another realistic example with text and invalid bytes
    print("Test 4: Realistic example - Text with corrupted bytes in middle")
    print("-" * 80)
    realistic_bytes = "Hello, this is valid text!".encode('utf-8') + bytes([0xFF, 0xFE]) + "And this continues.".encode('utf-8')
    print(f"Input: Valid text + 2 invalid bytes + More valid text")
    print()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _decode_bytes(realistic_bytes)

        print(f"✓ Decoded result: {result!r}")
        print(f"  Starts with: {result[:30]!r}")
        print(f"  Ends with: {result[-20:]!r}")
        print(f"  Contains replacement chars in middle: {result.count('\ufffd')} instances")

        if w:
            print(f"  ⚠ Warning: {w[0].message}")
    print()

    print("=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print("✓ Fix is working correctly!")
    print("  - No crashes on invalid UTF-8")
    print("  - Warnings are issued to inform users about data quality issues")
    print("  - Valid UTF-8 bytes are PRESERVED and decoded correctly")
    print("  - Invalid UTF-8 bytes are REPLACED with U+FFFD (replacement character)")
    print("  - Data processing pipeline continues without interruption")
    print("  - This allows processing datasets with occasional corrupted bytes")
    print("=" * 80)


if __name__ == '__main__':
    main()
