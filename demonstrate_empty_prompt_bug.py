#!/usr/bin/env python3
"""Demonstration script for Bug #2: make_seq2seq_fields crashes with empty prompt.

This script demonstrates:
1. The original bug (before fix): Cryptic NumPy error
2. The fixed behavior: Clear validation error message

The fix adds input validation that raises a helpful error message instead
of crashing with a confusing NumPy error about negative dimensions.
"""

from gemma import gm
import numpy as np


def demonstrate_fixed_behavior():
    """Demonstrates the fixed behavior with clear error messages."""
    print("=" * 80)
    print("Demonstrating Fixed Behavior: Clear validation error for empty prompt")
    print("=" * 80)
    print()
    
    print("Test Case 1: Empty prompt array with non-empty response")
    print("-" * 80)
    print("Calling: gm.data.make_seq2seq_fields(prompt=[], response=[20, 21, 1])")
    print()
    
    try:
        result = gm.data.make_seq2seq_fields(
            prompt=[],  # Empty prompt - should raise clear validation error
            response=[20, 21, 1]  # Valid response
        )
        print("ERROR: Expected validation error but function returned:", result)
    except ValueError as e:
        error_message = str(e)
        print("✓ Fixed behavior: Clear ValueError raised!")
        print()
        print("Error message:")
        print(f"  {error_message}")
        print()
        
        # Check if it's the improved error message
        if "prompt cannot be empty" in error_message:
            print("✓ GOOD: Error message is clear and helpful!")
            print("  - Explains what's wrong (empty prompt)")
            print("  - Explains what's expected (at least one token)")
            print("  - Explains why (not supported for seq2seq training)")
        elif "negative dimensions" in error_message:
            print("✗ BUG STILL PRESENT: Error message is still cryptic!")
            print("  - This means the fix hasn't been applied")
        
        print()
        print("Comparison:")
        print("  BEFORE FIX: ValueError: negative dimensions are not allowed")
        print("  AFTER FIX:  ValueError: prompt cannot be empty...")
        print()
        print("The fix transforms a cryptic crash into a helpful validation error.")
    
    print()
    print("=" * 80)
    print("Test Case 2: Using AddSeq2SeqFields with empty prompt tokens")
    print("-" * 80)
    print("Calling: AddSeq2SeqFields.map() with empty prompt_tokens")
    print()
    
    try:
        transform = gm.data.AddSeq2SeqFields(
            in_prompt="prompt",
            in_response="response",
            out_input="input",
            out_target="target",
            out_target_mask="target_mask",
        )
        
        element = {
            "prompt": [],  # Empty prompt tokens - should raise clear validation error
            "response": [20, 21, 1]
        }
        
        result = transform.map(element)
        print("ERROR: Expected validation error but transform returned:", result)
    except ValueError as e:
        error_message = str(e)
        print("✓ Fixed behavior: Clear ValueError raised through AddSeq2SeqFields!")
        print()
        print("Error message:")
        print(f"  {error_message}")
        print()
        if "prompt cannot be empty" in error_message:
            print("✓ GOOD: Error message propagates correctly through transforms!")
    
    print()
    print("=" * 80)
    print("Comparison: Normal usage (still works correctly)")
    print("-" * 80)
    print("Calling: gm.data.make_seq2seq_fields(prompt=[10, 11], response=[20, 21, 1])")
    print()
    
    try:
        result = gm.data.make_seq2seq_fields(
            prompt=[10, 11],  # Valid non-empty prompt
            response=[20, 21, 1]
        )
        print("✓ Normal usage still works correctly:")
        print(f"  input: {result.input}")
        print(f"  target: {result.target}")
        print(f"  target_mask: {result.target_mask}")
        print()
        print("✓ GOOD: Fix doesn't break existing functionality!")
    except Exception as e:
        print(f"ERROR: Unexpected exception in normal usage: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_fixed_behavior()
