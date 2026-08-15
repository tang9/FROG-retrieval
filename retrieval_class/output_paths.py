"""Shared helpers for locating generated FROG result files."""

from __future__ import annotations

import os


RETRIEVAL_RESULT_DIRNAME = "retrieval_result"


def build_retrieval_save_prefix(prefix: str) -> tuple[str, str]:
    """Return a stable result directory and output prefix.

    If ``prefix`` is already inside a ``retrieval_result`` directory, reuse the
    outermost such directory instead of creating another nested directory.
    """
    prefix_str = str(prefix).strip()
    if not prefix_str:
        raise ValueError("Save prefix cannot be empty.")

    absolute_prefix = os.path.abspath(prefix_str)
    source_dir = os.path.dirname(absolute_prefix)
    probe = source_dir
    retrieval_dir = None

    while True:
        if os.path.basename(probe).casefold() == RETRIEVAL_RESULT_DIRNAME.casefold():
            # Keep walking so an already nested path is repaired to the
            # outermost retrieval_result directory on the next save.
            retrieval_dir = probe
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent

    if retrieval_dir is None:
        retrieval_dir = os.path.join(source_dir, RETRIEVAL_RESULT_DIRNAME)

    save_prefix = os.path.join(retrieval_dir, os.path.basename(absolute_prefix))
    return retrieval_dir, save_prefix
