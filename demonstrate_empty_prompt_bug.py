#!/usr/bin/env python3
"""Demonstration script for Bug #2: make_seq2seq_fields crashes with empty prompt.

This script demonstrates the bug where make_seq2seq_fields crashes when
an empty prompt array is passed, due to attempting to create a NumPy array
with negative dimensions.

Expected behavior: Should raise a clear validation error
Actual behavior: Raises ValueError with confusing NumPy error message

Run this script to see the raw traceback demonstrating the bug.
"""

from gemma import gm
import numpy as np


def demonstrate_bug():
    """Demonstrates the bug by calling make_seq2seq_fields with empty prompt."""
    print("=" * 80)
    print("Demonstrating Bug: make_seq2seq_fields crashes with empty prompt")
    print("=" * 80)
    print()
    
    print("Test Case 1: Empty prompt array with non-empty response")
    print("-" * 80)
    print("Calling: gm.data.make_seq2seq_fields(prompt=[], response=[20, 21, 1])")
    print()
    
    try:
        result = gm.data.make_seq2seq_fields(
            prompt=[],  # Empty prompt - this is the bug trigger
            response=[20, 21, 1]  # Valid response
        )
        print("ERROR: Expected crash but function returned:", result)
    except ValueError as e:
        print("✓ Bug reproduced! ValueError raised:")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("Expected: Clear validation error message")
        print("Actual: NumPy ValueError about negative dimensions")
    
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
            "prompt": [],  # Empty prompt tokens - bug trigger
            "response": [20, 21, 1]
        }
        
        result = transform.map(element)
        print("ERROR: Expected crash but transform returned:", result)
    except ValueError as e:
        print("✓ Bug reproduced through AddSeq2SeqFields! ValueError raised:")
        print()
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("Comparison: Normal usage (works correctly)")
    print("-" * 80)
    print("Calling: gm.data.make_seq2seq_fields(prompt=[10, 11], response=[20, 21, 1])")
    print()
    
    try:
        result = gm.data.make_seq2seq_fields(
            prompt=[10, 11],  # Valid non-empty prompt
            response=[20, 21, 1]
        )
        print("✓ Normal usage works correctly:")
        print(f"  input: {result.input}")
        print(f"  target: {result.target}")
        print(f"  target_mask: {result.target_mask}")
    except Exception as e:
        print(f"ERROR: Unexpected exception in normal usage: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_bug()

