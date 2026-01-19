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

  # This line crashes with KeyError('tool_name') in maybe_execute_tool().
  # It happens because the tool-call parser greedily extracts JSON from the
  # output, and the manager assumes the parsed JSON dict is a tool call.
  manager.maybe_execute_tool(model_output)


if __name__ == "__main__":
  main()

