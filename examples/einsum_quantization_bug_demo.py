"""Demonstration of the SimulateQuantizedEinsum axis selection bug.

This script demonstrates that SimulateQuantizedEinsum.__call__ passes
self.wrapped.name (the module name) to get_axis_to_reduce_from_einsum_str
instead of the actual einsum_str, causing the pattern-specific axis
selection logic to never execute.

Context
-------

In `gemma.peft._quantization.SimulateQuantizedEinsum.__call__`, the
quantization code calls::

    kernel = simulate_quantize(
        kernel,
        self.method,
        axis_to_reduce=get_axis_to_reduce_from_einsum_str(
            einsum_str=self.wrapped.name,  # ← BUG: should be einsum_str
        ),
    )

However, `get_axis_to_reduce_from_einsum_str` is written to match on the
**einsum equation string**, e.g. "BTD,NDH->BTNH", "...H,HF->...F", etc.
Passing the module name (typically something like "einsum_0") means the
matcher always falls through to the default case and returns ``None``.

This script demonstrates the bug by:
1. Creating a spy that intercepts calls to get_axis_to_reduce_from_einsum_str
2. Instantiating SimulateQuantizedEinsum with a real einsum equation
3. Calling the module to trigger the quantization path
4. Showing that the spy received self.wrapped.name instead of einsum_str
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any

# Ensure we import from the local code, not an installed package
# Add the parent directory (gemma root) to Python path
_script_dir = Path(__file__).parent
_gemma_root = _script_dir.parent
if str(_gemma_root) not in sys.path:
  sys.path.insert(0, str(_gemma_root))

import jax.numpy as jnp
from flax import linen as nn

from gemma.peft import _quantization
from gemma.peft import _quantization_utils

# Reload the module to ensure we're using the latest code
import importlib
importlib.reload(_quantization)

# Print the code location to verify we're using local code
print(f"Using _quantization from: {_quantization.__file__}")
print(f"Expected location: {_gemma_root / 'gemma' / 'peft' / '_quantization.py'}")
print()

# Global variable to capture what argument was actually passed
_captured_argument: str | None = None
_original_function = _quantization.get_axis_to_reduce_from_einsum_str


def spy_get_axis_to_reduce_from_einsum_str(einsum_str: str) -> Any:
  """Spy function that captures the argument and calls the original."""
  global _captured_argument
  _captured_argument = einsum_str
  return _original_function(einsum_str)


def demonstrate_simulate_quantized_einsum_bug() -> None:
  """Demonstrates that SimulateQuantizedEinsum passes the wrong argument.

  This function:
  1. Patches get_axis_to_reduce_from_einsum_str with a spy
  2. Creates a SimulateQuantizedEinsum instance with a known einsum equation
  3. Calls the module to trigger quantization
  4. Shows that the spy received self.wrapped.name instead of einsum_str
  """
  global _captured_argument

  # The einsum equation we'll use - this is one that get_axis_to_reduce_from_einsum_str
  # knows how to handle and should return (1,) for
  einsum_equation = "BTD,NDH->BTNH"
  expected_axis = (1,)

  print("=" * 70)
  print("Demonstrating SimulateQuantizedEinsum axis selection bug")
  print("=" * 70)
  print()
  print(f"Einsum equation we'll use: {einsum_equation}")
  print(f"Expected axis when called with equation: {expected_axis}")
  print()

  # Patch the function with our spy
  _quantization.get_axis_to_reduce_from_einsum_str = spy_get_axis_to_reduce_from_einsum_str
  _captured_argument = None

  try:
    # Create a wrapped Einsum module with the einsum equation
    wrapped_einsum = nn.Einsum(
        einsum_str=einsum_equation,
        shape=(4, 8, 16),  # Example shape: (N, D, H)
        name="attention_proj",  # This is what self.wrapped.name will be
    )

    # Create SimulateQuantizedEinsum wrapper
    quantized_einsum = _quantization.SimulateQuantizedEinsum(
        wrapped=wrapped_einsum,
        method=_quantization_utils.QuantizationMethod.INT4,
    )

    # Initialize the module (Flax requires this)
    # For einsum "BTD,NDH->BTNH": input is (B, T, D), kernel is (N, D, H)
    # We'll use D=8 to match the kernel shape (4, 8, 16) where D=8
    key = jax.random.key(42)
    dummy_input = jnp.ones((2, 10, 8))  # Batch=2, Seq=10, Dim=8 (D dimension)

    # Initialize variables (einsum_str is already set in constructor, don't pass it again)
    variables = quantized_einsum.init(key, dummy_input)

    print("Created SimulateQuantizedEinsum with:")
    print(f"  - wrapped.einsum_str = '{wrapped_einsum.einsum_str}'")
    print(f"  - wrapped.name = '{wrapped_einsum.name}'")
    print()

    # Now call the module - this will trigger the quantization path
    # einsum_str is already set in constructor, don't pass it again
    print("Calling quantized_einsum(...) to trigger quantization...")
    print()

    _captured_argument = None  # Reset before the call
    output = quantized_einsum.apply(variables, dummy_input)

    print("=" * 70)
    print("BUG DEMONSTRATION RESULTS")
    print("=" * 70)
    print()
    print(f"What get_axis_to_reduce_from_einsum_str was called with:")
    print(f"  '{_captured_argument}'")
    print()
    print(f"What it SHOULD have been called with:")
    print(f"  '{einsum_equation}'")
    print()
    print(f"What it WAS called with (self.wrapped.name):")
    print(f"  '{wrapped_einsum.name}'")
    print()

    if _captured_argument == wrapped_einsum.name:
      print("❌ BUG CONFIRMED: The function was called with self.wrapped.name")
      print("   instead of the actual einsum_str!")
      print()
      print(f"   This means get_axis_to_reduce_from_einsum_str('{_captured_argument}')")
      print("   returns None, and the pattern-specific axis selection is never used.")
    elif _captured_argument == einsum_equation:
      print("✅ CORRECT: The function was called with the einsum_str")
      print("   (This would mean the bug is fixed)")
    else:
      print(f"⚠️  UNEXPECTED: The function was called with '{_captured_argument}'")
      print("   (Neither the name nor the equation)")

    print()
    print("=" * 70)
    print("Expected behavior vs actual behavior")
    print("=" * 70)
    print()
    print("Expected:")
    print(f"  get_axis_to_reduce_from_einsum_str('{einsum_equation}') -> {expected_axis}")
    print()
    print("Actual (due to bug):")
    actual_result = _original_function(_captured_argument)
    print(f"  get_axis_to_reduce_from_einsum_str('{_captured_argument}') -> {actual_result}")
    print()
    if actual_result is None:
      print("  Because None is returned, the quantization falls back to")
      print("  generic per-channel scaling along the last axis, instead of")
      print("  using the pattern-specific axis selection logic.")

  finally:
    # Restore the original function
    _quantization.get_axis_to_reduce_from_einsum_str = _original_function


if __name__ == "__main__":
  # Import jax here to avoid issues if jax is not available
  import jax

  demonstrate_simulate_quantized_einsum_bug()
