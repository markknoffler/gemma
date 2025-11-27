#!/usr/bin/env python3
"""
Demo script to demonstrate the pad_length tuple bug in Sampler.

This script shows that when using the default Sampler configuration,
the pad_length tuple causes issues because _tokenize_prompts expects
an int | None but receives a tuple indirectly.

Bug: Sampler.pad_length defaults to (256, 512, 1024) but _tokenize_prompts
     signature only accepts int | None, causing type mismatch issues.
"""

import sys
from pathlib import Path

# Add the gemma package to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from gemma import gm
    import jax
    import jax.numpy as jnp
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease ensure you have installed the gemma package and its dependencies.")
    sys.exit(1)


def demonstrate_bug():
    """Demonstrate the pad_length tuple bug."""

    print("=" * 70)
    print("DEMONSTRATING PAD_LENGTH TUPLE BUG")
    print("=" * 70)
    print()

    # Create a dummy model and tokenizer for testing
    print("Step 1: Creating dummy model and tokenizer...")
    model = gm.testing.DummyGemma()
    tokenizer = gm.testing.DummyTokenizer()

    # Initialize model parameters
    print("Step 2: Initializing model parameters...")
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.zeros((5,), dtype=jnp.int32))
    params = params['params']

    print("Step 3: Checking default pad_length value...")
    # Check what the default pad_length is
    sampler = gm.text.Sampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
        cache_length=128,
        max_out_length=128,
        # Note: pad_length is NOT explicitly set, so it uses the default
    )

    print(f"   Default pad_length type: {type(sampler.pad_length)}")
    print(f"   Default pad_length value: {sampler.pad_length}")
    print()

    # Check the _tokenize_prompts signature
    print("Step 4: Checking _tokenize_prompts method signature...")
    import inspect
    sig = inspect.signature(sampler._tokenize_prompts)
    pad_length_param = sig.parameters.get('pad_length')
    if pad_length_param:
        print(f"   _tokenize_prompts.pad_length type annotation: {pad_length_param.annotation}")
        print(f"   _tokenize_prompts.pad_length default: {pad_length_param.default}")
    print()

    # Demonstrate the bug by directly calling _tokenize_prompts with tuple
    print("Step 5: Demonstrating bug by passing tuple to _tokenize_prompts...")
    print("   (This shows what happens if pad_length tuple is passed incorrectly)")
    print()

    try:
        # This demonstrates the bug - passing tuple where int|None is expected
        prompt = "Hello world"
        tokens = sampler._tokenize_prompts(
            prompt,
            add_bos=True,
            pad_length=sampler.pad_length  # Passing tuple (256, 512, 1024)
        )
        print(f"   ✗ Unexpected success! Tokens shape: {tokens.shape}")
        print("   This shouldn't happen - tuple should cause an error.")

    except TypeError as e:
        print(f"   ✓ TypeError caught (BUG CONFIRMED):")
        print(f"     {type(e).__name__}: {e}")
        print()
        print("   BUG CONFIRMED: When pad_length tuple is passed to")
        print("   _tokenize_prompts, it causes a TypeError because:")
        print("   - Line 418: max_prompt_len = pad_length or max(...)")
        print("   - If pad_length is tuple, max_prompt_len becomes the tuple")
        print("   - Line 421: _max_across_hosts(max_prompt_len) receives tuple")
        print("   - Line 424: _functional.pad(..., max_length=tuple) fails")

    except Exception as e:
        print(f"   ✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("Step 6: Showing what happens with the tuple in the problematic line...")
    pad_length_tuple = (256, 512, 1024)
    tokens_list = [[1, 2, 3, 4, 5], [1, 2, 3]]

    # Simulate line 418: max_prompt_len = pad_length or max(len(t) for t in tokens)
    max_prompt_len_buggy = pad_length_tuple or max(len(t) for t in tokens_list)
    print(f"   pad_length tuple: {pad_length_tuple}")
    print(f"   max(len(t) for t in tokens): {max(len(t) for t in tokens_list)}")
    print(f"   pad_length or max(...) = {max_prompt_len_buggy}")
    print(f"   Type of result: {type(max_prompt_len_buggy)}")
    print()
    print("   PROBLEM: The tuple is truthy, so 'or' returns the tuple,")
    print("   not the max length calculation!")

    print()
    print("=" * 70)
    print("DEMONSTRATING WORKAROUND")
    print("=" * 70)
    print()

    # Show the workaround
    print("Step 7: Demonstrating workaround (setting pad_length=None)...")
    sampler_fixed = gm.text.Sampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
        cache_length=128,
        max_out_length=128,
        pad_length=None,  # Workaround: explicitly set to None
    )

    try:
        result = sampler_fixed.sample("Hello world")
        print(f"   ✓ Sampling succeeded with pad_length=None: {result[:50]}...")
        print()
        print("   WORKAROUND: Set pad_length=None when creating the Sampler")
        print("   to avoid the tuple type mismatch issue.")

    except Exception as e:
        print(f"   ✗ Error even with workaround: {type(e).__name__}: {e}")
        print("   This suggests the bug might be deeper than expected.")


def demonstrate_type_mismatch():
    """Demonstrate the type mismatch more directly."""

    print()
    print("=" * 70)
    print("TYPE MISMATCH ANALYSIS")
    print("=" * 70)
    print()

    print("The issue:")
    print("  1. Sampler.pad_length has type: None | int | tuple[int, ...]")
    print("  2. Default value: (256, 512, 1024)  # This is a tuple")
    print("  3. _tokenize_prompts.pad_length has type: int | None")
    print("  4. _prefill.prefill.pad_length has type: None | int | tuple[int, ...]")
    print()
    print("The problem:")
    print("  - _prefill.prefill can handle tuples (it converts int to tuple)")
    print("  - But _tokenize_prompts cannot handle tuples")
    print("  - If pad_length tuple is passed to _tokenize_prompts, it will fail")
    print()
    print("Code flow:")
    print("  sampler.sample()")
    print("    -> _get_inputs()")
    print("       -> _tokenize_prompts(pad_length=???)  # Not passed currently")
    print("    -> _prefill.prefill(pad_length=self.pad_length)  # Tuple passed here")
    print()
    print("Current behavior:")
    print("  - _tokenize_prompts doesn't receive pad_length (uses default None)")
    print("  - _prefill.prefill receives the tuple and handles it")
    print("  - So the bug might not manifest in current code path")
    print()
    print("However, the type signature mismatch suggests:")
    print("  - Either _tokenize_prompts should accept tuple[int, ...] | None")
    print("  - Or pad_length should not default to a tuple")
    print("  - Or there's missing code that should pass pad_length to _tokenize_prompts")


if __name__ == "__main__":
    print()
    print("PAD_LENGTH TUPLE BUG DEMONSTRATION")
    print("=" * 70)
    print()

    try:
        demonstrate_bug()
        demonstrate_type_mismatch()

        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        print("BUG CONFIRMED: Type signature mismatch")
        print("  - Sampler.pad_length defaults to tuple: (256, 512, 1024)")
        print("  - _tokenize_prompts.pad_length only accepts: int | None")
        print("  - When tuple is passed to _tokenize_prompts, line 418 causes bug:")
        print("    max_prompt_len = pad_length or max(...)  # tuple is truthy!")
        print("  - This makes max_prompt_len a tuple instead of int")
        print("  - Then _max_across_hosts() and _functional.pad() receive tuple")
        print("  - Result: TypeError when tuple used as integer")
        print()
        print("Current code path:")
        print("  - _get_inputs() does NOT pass pad_length to _tokenize_prompts")
        print("  - So the bug doesn't manifest in normal usage YET")
        print("  - But the type signature mismatch is a latent bug")
        print("  - If code is refactored to pass pad_length, it will crash")
        print("  - Or if someone tries to use _tokenize_prompts directly with tuple")
        print()
        print("Impact:")
        print("  - API inconsistency: pad_length can be tuple but can't be used")
        print("  - Potential future crashes if code is refactored")
        print("  - Confusing for developers who expect pad_length to work everywhere")
        print()
        print("Fix needed:")
        print("  - Either: Make _tokenize_prompts accept tuple[int, ...] | None")
        print("  - Or: Change Sampler.pad_length default to None")
        print("  - Or: Add logic to handle tuple in _tokenize_prompts (pick bucket)")
        print()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during demo: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
