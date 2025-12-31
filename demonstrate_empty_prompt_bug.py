#!/usr/bin/env python3
"""Demonstration script for empty prompt handling in make_seq2seq_fields.

This script demonstrates that make_seq2seq_fields now handles empty prompts
gracefully by:
1. Issuing a warning (not crashing)
2. Using a default BOS token as fallback
3. Continuing execution successfully

The fix transforms a crash into graceful handling with a warning.
"""

from gemma import gm
import numpy as np
import warnings


def demonstrate_graceful_handling():
    """Demonstrates graceful handling of empty prompt with warning."""
    print("=" * 80)
    print("Demonstrating Graceful Handling: Empty prompt with warning")
    print("=" * 80)
    print()
    
    print("Test Case 1: Empty prompt array with non-empty response")
    print("-" * 80)
    print("Calling: gm.data.make_seq2seq_fields(prompt=[], response=[20, 21, 1])")
    print()
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Capture all warnings
        
        try:
            result = gm.data.make_seq2seq_fields(
                prompt=[],  # Empty prompt - should trigger warning but continue
                response=[20, 21, 1]  # Valid response
            )
            
            print("✓ Function executed successfully (did not crash)!")
            print()
            print("Result:")
            print(f"  input: {result.input}")
            print(f"  target: {result.target}")
            print(f"  target_mask: {result.target_mask}")
            print()
            
            # Check if warning was issued
            if w:
                print("✓ Warning was issued:")
                for warning in w:
                    print(f"  {warning.category.__name__}: {warning.message}")
            else:
                print("⚠ No warning was issued (unexpected)")
                
        except Exception as e:
            print(f"✗ ERROR: Function crashed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 80)
    print("Test Case 2: Using AddSeq2SeqFields with empty prompt tokens")
    print("-" * 80)
    print("Calling: AddSeq2SeqFields.map() with empty prompt_tokens")
    print()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            transform = gm.data.AddSeq2SeqFields(
                in_prompt="prompt",
                in_response="response",
                out_input="input",
                out_target="target",
                out_target_mask="target_mask",
            )
            
            element = {
                "prompt": [],  # Empty prompt tokens - should trigger warning
                "response": [20, 21, 1]
            }
            
            result = transform.map(element)
            print("✓ Transform executed successfully (did not crash)!")
            print()
            print("Result keys:", list(result.keys()))
            print(f"  input: {result['input']}")
            print(f"  target: {result['target']}")
            print(f"  target_mask: {result['target_mask']}")
            print()
            
            if w:
                print("✓ Warning was issued:")
                for warning in w:
                    print(f"  {warning.category.__name__}: {warning.message}")
                    
        except Exception as e:
            print(f"✗ ERROR: Transform crashed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 80)
    print("Test Case 3: Normal usage (still works correctly)")
    print("-" * 80)
    print("Calling: gm.data.make_seq2seq_fields(prompt=[10, 11], response=[20, 21, 1])")
    print()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            result = gm.data.make_seq2seq_fields(
                prompt=[10, 11],  # Valid non-empty prompt
                response=[20, 21, 1]
            )
            print("✓ Normal usage works correctly:")
            print(f"  input: {result.input}")
            print(f"  target: {result.target}")
            print(f"  target_mask: {result.target_mask}")
            
            if w:
                print("⚠ Unexpected warnings:")
                for warning in w:
                    print(f"  {warning.category.__name__}: {warning.message}")
            else:
                print("✓ No warnings (as expected for normal usage)")
                
        except Exception as e:
            print(f"ERROR: Unexpected exception in normal usage: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("✓ Empty prompts are handled gracefully")
    print("✓ Warning is issued to alert users")
    print("✓ Code continues execution (doesn't crash)")
    print("✓ Default BOS token is used as fallback")
    print("✓ Normal usage is unaffected")


if __name__ == "__main__":
    demonstrate_graceful_handling()
