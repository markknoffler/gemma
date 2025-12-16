"""Demonstration of the SimulateQuantizedEinsum axis selection bug.

This script does NOT fix anything; it just shows the discrepancy
between what the code appears to intend and what it actually does
in the current implementation.

Context
-------

In `gemma.peft._quantization.SimulateQuantizedEinsum.__call__`, the
quantization code calls::

    kernel = simulate_quantize(
        kernel,
        self.method,
        axis_to_reduce=get_axis_to_reduce_from_einsum_str(
            einsum_str=self.wrapped.name,
        ),
    )

However, `get_axis_to_reduce_from_einsum_str` is written to match on the
**einsum equation string**, e.g. "BTD,NDH->BTNH", "...H,HF->...F", etc.
Passing the module name (typically something like "einsum_0") means the
matcher always falls through to the default case and returns ``None``.

That makes the pattern-specific axis handling effectively dead code,
and all einsums use the generic per-channel scaling path.

This script demonstrates that discrepancy without running any model:

* It shows that `get_axis_to_reduce_from_einsum_str` returns the
  expected axes when passed a known einsum equation.
* It also shows that if you pass something that looks like a realistic
  Flax module name (what SimulateQuantizedEinsum currently does), you
  always get ``None``.

You should NOT run this in production; it is solely a debugging/demo
helper for understanding the bug.
"""

from __future__ import annotations

from gemma.peft._quantization import get_axis_to_reduce_from_einsum_str


def demonstrate_axis_selection_mismatch() -> None:
  """Prints how axis selection behaves for equations vs names.

  This mirrors the key logic in SimulateQuantizedEinsum without actually
  instantiating the module or running any JAX/Flax code.
  """

  # This is one of the equations explicitly handled in
  # get_axis_to_reduce_from_einsum_str.
  einsum_str = "BTD,NDH->BTNH"

  # This string simulates what `self.wrapped.name` typically looks like
  # inside a Flax module hierarchy: a simple identifier that does NOT
  # match any of the patterns in get_axis_to_reduce_from_einsum_str.
  module_name_like = "einsum_0"

  axis_from_equation = get_axis_to_reduce_from_einsum_str(einsum_str)
  axis_from_name = get_axis_to_reduce_from_einsum_str(module_name_like)

  print("Einsum equation string:", einsum_str)
  print("Axis to reduce when called with the equation:", axis_from_equation)

  print()

  print("Module name-like string (what SimulateQuantizedEinsum passes today):",
        module_name_like)
  print("Axis to reduce when called with the module name:", axis_from_name)

  print()
  print(
      "Observation: get_axis_to_reduce_from_einsum_str knows how to choose",
      "axes for the equation string, but SimulateQuantizedEinsum currently",
      "passes the module name, so the matcher returns None and the",
      "pattern-specific logic is never used.",
  )


if __name__ == "__main__":
  # NOTE: The user requested that we do NOT actually run any code as
  # part of this contribution, so this block is intentionally only a
  # demonstration entry point. It is left here so that other contributors
  # can easily run it locally if they wish to inspect the behavior.
  demonstrate_axis_selection_mismatch()
