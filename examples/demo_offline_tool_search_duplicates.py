"""Minimal demo reproducing OfflineToolSearch duplicate tool registration.

This script intentionally triggers the duplicate-registration behavior by
calling `OfflineToolSearch` twice with the same query and applying the returned
`update_tools` callback each time. The bug is demonstrated by asserting tool
names remain unique; the assertion fails and prints a raw traceback.
"""

from gemma.gm import tools as gm_tools


def main() -> None:
  # OfflineToolSearch can "register" tools into a running tool list via the
  # ToolOutput.update_tools callback.
  search = gm_tools.OfflineToolSearch(
      tools=[gm_tools.Calculator(), gm_tools.FileExplorer()],
  )

  registered_tools = [search]

  # First search registers FileExplorer once.
  out1 = search.call("file")
  registered_tools = out1.update_tools(registered_tools)  # pytype: disable=attribute-error

  # Second identical search registers FileExplorer again (duplicate).
  out2 = search.call("file")
  registered_tools = out2.update_tools(registered_tools)  # pytype: disable=attribute-error

  names = [t.name for t in registered_tools]
  assert len(names) == len(set(names)), f"Duplicate tools registered: {names}"


if __name__ == "__main__":
  main()
