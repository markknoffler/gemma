"""Minimal demo to reproduce ChatSampler `_print_stream` NameError.

This script intentionally calls `_print_stream` with an empty iterator, which
exposes the bug where the `state` variable is referenced after the loop without
being defined when nothing is yielded.
"""

from gemma.gm.text import _chat_sampler


def main() -> None:
  # Empty iterator: the for-loop in `_print_stream` never executes,
  # so `state` is never bound, and the function crashes with:
  #   NameError: name 'state' is not defined
  empty_iterator = iter(())
  _chat_sampler._print_stream(empty_iterator)  # pytype: disable=protected-access


if __name__ == "__main__":
  main()

