### PR Title
Fix tool-call parsing to ignore non-tool JSON (fixes #511)

### PR Description
Fixes: #511

#### Summary
- Prevent `OneShotToolManager` from mis-parsing arbitrary JSON in model output as a tool call.
- Avoids `KeyError: 'tool_name'` when the model outputs normal JSON (e.g., structured responses) that is not a tool call.

#### What changed
- Updated `gemma/gm/tools/_manager.py` to:
  - Stop using greedy brace-regex extraction for JSON.
  - Only treat parsed JSON as a tool call when it is a JSON object containing a valid string `tool_name`.
  - Gracefully ignore non-tool JSON by returning `None` instead of raising.

#### Before / After
- **Before**: any `{...}` JSON in model output could be parsed as a tool call → `KeyError: 'tool_name'` if missing.
- **After**: non-tool JSON is ignored; only JSON objects containing `tool_name` are executed as tool calls.

#### Test plan
- Run the repro/demo and confirm no crash:
  - `python examples/tool_call_parser_bug_demo.py`
- (Optional) Run unit tests if included in your PR branch:
  - `python -m pytest -q`
