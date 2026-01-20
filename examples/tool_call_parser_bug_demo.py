r"""Repro: Tool-call parser false-positives on normal JSON and crashes.

This demonstrates a bug in `gemma.gm.tools._manager.OneShotToolManager`:

1) `_parse_tool_call()` uses a greedy regex `r'\{.*\}'` to extract a JSON
   substring from arbitrary model output.
2) `maybe_execute_tool()` assumes any parsed JSON dict contains "tool_name".

As a result, if the model outputs *any* JSON object (e.g. as an example),
the manager treats it as a tool call and crashes with:

    KeyError: 'tool_name'

Run:
  python examples/tool_call_parser_bug_demo.py
"""

from __future__ import annotations

# IMPORTANT: ensure we import the local checkout, not an installed `gemma`
# package from site-packages.
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from gemma.gm.tools import _calculator
from gemma.gm.tools import _manager as _manager_lib


def main() -> None:
  # A real tool isn't necessary to trigger the bug, but we include one to show
  # the realistic tool-manager flow.
  manager = _manager_lib.OneShotToolManager(tools=[_calculator.Calculator()])

  # This is *not* a tool call. It's a normal JSON snippet an LLM might output
  # when asked to provide structured data.
  model_output = """\
Here is the JSON you requested:

{"answer": 42, "reason": "because"}

Thanks!
"""

  # Before the fix, this crashed with KeyError('tool_name').
  # After the fix, non-tool JSON is ignored and this returns None.
  tool_out = manager.maybe_execute_tool(model_output)
  assert tool_out is None, tool_out


if __name__ == "__main__":
  main()

