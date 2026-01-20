"""Tests for tool manager parsing/execution."""

from __future__ import annotations

from gemma.gm.tools import _manager
from gemma.gm.tools import _tools


class DummyTool(_tools.Tool):
  DESCRIPTION = "Dummy tool."
  EXAMPLE = _tools.Example(
      query="q",
      tool_kwargs={"x": "1"},
      tool_kwargs_doc={"x": "<INT>"},
      result="ok",
      answer="a",
  )

  def call(self, x: int):  # pytype: disable=signature-mismatch
    return f"got {x}"


def test_non_tool_json_is_ignored() -> None:
  manager = _manager.OneShotToolManager(tools=[DummyTool()])
  model_output = 'Here is JSON:\n\n{"answer": 42}\n'
  assert manager.maybe_execute_tool(model_output) is None


def test_tool_call_json_is_executed() -> None:
  manager = _manager.OneShotToolManager(tools=[DummyTool()])
  model_output = '{"tool_name": "dummytool", "x": 123}'
  out = manager.maybe_execute_tool(model_output)
  assert out is not None
  assert "got 123" in out.text


def test_multiple_json_blocks_picks_tool_call() -> None:
  manager = _manager.OneShotToolManager(tools=[DummyTool()])
  model_output = (
      'Some json: {"answer": 1}\n'
      'Tool call: {"tool_name": "dummytool", "x": 7}\n'
  )
  out = manager.maybe_execute_tool(model_output)
  assert out is not None
  assert "got 7" in out.text

