"""Shared argv-list parsing for the SMPL-H fitter CLI.

This module converges only the parsing genuinely common to the frozen
regression reader (``check_mainline_regression.fit_arguments``) and the
resource preflight reader (``preflight_smplh.fit_arguments``):

- reading a JSON array of CLI argument strings, and
- parsing it with the fitter's own ``build_parser`` while turning argparse's
  ``SystemExit`` into a ``ValueError``.

It deliberately does **not** append ``--fit-policy legacy``, trim leading
program tokens, reject duplicate options, or validate optimization ranges; those
remain in the respective adapters, whose public names and return values are
unchanged. The parser import stays lazy so importing this module never loads
the solver or any model.
"""
from __future__ import annotations

import json
from pathlib import Path


def read_argv_list(path: Path | str) -> list[str]:
    """Read a JSON array of CLI argument strings.

    The value must be a list whose every element is a string. No defaulting,
    trimming or option rewriting is performed here.
    """
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("args JSON must be an array of CLI argument strings")
    return values


def parse_fit_argv(values: list[str]):
    """Parse a fitter argv list into an argparse ``Namespace``.

    ``argparse`` exits the process on malformed options; that ``SystemExit`` is
    converted to a ``ValueError`` so callers can report it instead of aborting.
    """
    from walker_tools.body.fit_smplh_wilor_sequence import build_parser

    try:
        return build_parser().parse_args(values)
    except SystemExit as error:
        raise ValueError("invalid saved fit command") from error
