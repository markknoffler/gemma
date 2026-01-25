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

import os

# IMPORTANT: ensure we import the local checkout, not an installed `gemma`
# package from site-packages.
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from gemma.gm.tools import _file_explorer


def main() -> None:
  explorer = _file_explorer.FileExplorer()

  print("=" * 70)
  print("SECURITY VULNERABILITY DEMO: FileExplorer allows arbitrary file access")
  print("=" * 70)
  print()

  # Demo 1: Read user's home directory listing (could expose sensitive files)
  print("Demo 1: Listing user's home directory")
  print("-" * 70)
  home_path = os.path.expanduser("~")
  print(f"Attempting to list: {home_path}")
  result = explorer.call("ls", home_path)
  print(f"Result (first 500 chars):\n{result[:500]}")
  if len(result) > 500:
    print(f"... (truncated, total length: {len(result)} chars)")
  print()

  # Demo 2: Attempt to read common sensitive files (if they exist)
  print("Demo 2: Attempting to read potentially sensitive files")
  print("-" * 70)
  sensitive_paths = [
      os.path.expanduser("~/.ssh/config"),
      os.path.expanduser("~/.bashrc"),
      os.path.expanduser("~/.gitconfig"),
      "/etc/passwd",  # System file (readable on most Unix systems)
  ]

  for path in sensitive_paths:
    print(f"\nAttempting to read: {path}")
    result = explorer.call("cat", path)
    if "File not found" not in result and "OSError" not in result:
      print(f"⚠️  SUCCESSFULLY READ (security risk!):")
      print(f"   Content (first 200 chars): {result[:200]}")
      if len(result) > 200:
        print(f"   ... (truncated, total length: {len(result)} chars)")
    else:
      print(f"   (File not found or not accessible)")

  print()
  print("=" * 70)
  print("SECURITY IMPACT:")
  print("=" * 70)
  print(
      "The FileExplorer tool has NO path restrictions. In a tool-using agent,"
  )
  print(
      "an attacker could potentially use the model to exfiltrate sensitive data"
  )
  print("by asking it to read arbitrary files.")
  print()
  print(
      "SUGGESTED FIX: Restrict FileExplorer to a sandbox directory or require"
  )
  print("an explicit allowlist of permitted paths.")
  print("=" * 70)


if __name__ == "__main__":
  main()
