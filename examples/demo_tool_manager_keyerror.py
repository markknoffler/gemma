"""Minimal demo to reproduce OneShotToolManager KeyError when `tool_name` is missing.

This script intentionally triggers the bug by passing a JSON payload without a
`tool_name` field to `OneShotToolManager.maybe_execute_tool`. The resulting
KeyError and full traceback are allowed to propagate.
"""

from gemma.gm.tools import _manager


def main() -> None:
  # No tools are required to trigger the bug; the failure occurs before any
  # lookup in `name_to_tool`.
  manager = _manager.OneShotToolManager(tools=[])

  # Valid JSON that is missing the required "tool_name" field.
  model_output = '{"expression": "1+1"}'

  # This call will raise `KeyError: 'tool_name'` inside maybe_execute_tool.
  # The exception is not caught, so the raw traceback will be printed.
  manager.maybe_execute_tool(model_output)


if __name__ == "__main__":
  main()
