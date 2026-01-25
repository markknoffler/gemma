### Title
Tool manager mis-parses arbitrary JSON in model output as tool call, causing `KeyError: 'tool_name'`

### Issue description
#### Summary
`gemma.gm.tools._manager.OneShotToolManager` tries to detect tool calls by extracting the first `{` and the last `}` from the model output (via a greedy regex) and then `json.loads()` the result. This causes **false positives** whenever the model outputs normal JSON (which is common when asked to provide structured output).

If that JSON does **not** contain `"tool_name"`, `maybe_execute_tool()` assumes it’s a tool call and crashes with **`KeyError: 'tool_name'`**.

This is both:
- **Robustness issue**: hard crashes when the model outputs JSON for non-tool reasons.
- **Safety issue**: tool execution gating is based on brittle parsing rather than explicit framing/validation.

#### Affected code
- `gemma/gm/tools/_manager.py`
  - `_parse_tool_call()` uses greedy brace matching to extract a JSON substring.
  - `OneShotToolManager.maybe_execute_tool()` assumes parsed JSON is a tool call and does `pop("tool_name")` without validating the schema.

#### Steps to reproduce
1. Create a `OneShotToolManager` with any registered tool (e.g. `Calculator`).
2. Call `maybe_execute_tool()` with a model output string that contains a JSON object that does **not** include `"tool_name"`.

Example “model output” that triggers the crash:

```text
{"answer": 42, "reason": "because"}
```

#### Expected behavior
The tool manager should **ignore** non-tool JSON and return `None` (continue normally without executing a tool).

#### Actual behavior
The tool manager treats the JSON as a tool call and crashes with:

```text
KeyError: 'tool_name'
```

#### Suggested fix
- Avoid greedy regex extraction of `{...}` from arbitrary text.
- Parse more robustly and **validate** before executing:
  - Only treat JSON as a tool call if it’s a dict containing a valid string `tool_name`.
  - Otherwise return `None`.
- Optionally require explicit framing (e.g., tool call JSON must be the entire assistant output or inside a fenced block) to reduce false positives further.

