"""Shared argv-list parsing for the SMPL-H fitter CLI.

This module converges only the parsing genuinely common to the frozen
regression reader (``check_mainline_regression.fit_arguments``) and the
resource preflight reader (``preflight_smplh.fit_arguments``):

- reading a JSON array of CLI argument strings, and
- parsing it with the fitter's own ``build_parser``.

``parse_fit_argv`` deliberately keeps argparse's original behaviour: on invalid
options it lets ``SystemExit`` propagate, so a CLI that calls it exits with
argparse's own code (2 for usage errors) and message. Translating that
``SystemExit`` into a ``ValueError`` is part of the **historical regression
adapter** contract (``resolve_saved_fit_command``), not of the shared parser.

It also deliberately does **not** append ``--fit-policy legacy``, trim leading
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
    """Parse options while retaining argparse's original exit behavior."""
    from walker_tools.body.fit_smplh_wilor_sequence import build_parser

    return build_parser().parse_args(values)


def resolve_saved_fit_command(path):
    """Read and resolve a saved fitter command into an argparse ``Namespace``.

    Historical regression adapter. Accepts both a historical
    ``fit_command.json`` (with leading program tokens) and an options-only argv
    list; defaults a missing ``--fit-policy`` to ``legacy``; and rejects
    duplicate options. It requires ``--left-raw`` so an unrelated argv list
    cannot be silently accepted. Unlike the shared parser, it converts a
    malformed-option ``SystemExit`` into a ``ValueError``.
    """
    values = read_argv_list(path)
    if "--left-raw" not in values:
        raise ValueError("command has no --left-raw")
    # Accept both historical fit_command.json and an options-only argv list.
    values = values[next(i for i, value in enumerate(values) if value.startswith("--")):]
    if "--fit-policy" not in values:
        values.extend(["--fit-policy", "legacy"])
    if any(value.startswith("--") and values.count(value) != 1 for value in values):
        raise ValueError("duplicate options in command")
    try:
        return parse_fit_argv(values)
    except SystemExit as error:
        raise ValueError("invalid saved fit command") from error
