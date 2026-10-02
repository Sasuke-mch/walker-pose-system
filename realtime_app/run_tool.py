"""List and run the existing offline tools; all tool arguments pass unchanged."""

from __future__ import annotations

import runpy
import sys

from walker_tools._compat import prepare_imports
from walker_tools.catalog import command_path, commands


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"--help", "-h"}:
        print("Usage: python run_tool.py --list [domain]")
        print("       python run_tool.py domain.command [original arguments]")
        print("       python run_tool.py original_filename.py [original arguments]")
        print("Domains describe responsibilities, not method acceptance or priority.")
        return 0
    if args[0] == "--list":
        if len(args) > 2:
            raise SystemExit("--list accepts at most one domain")
        entries = commands()
        if len(args) == 2:
            entries = {key: value for key, value in entries.items()
                       if key.split(".", 1)[0] == args[1]}
            if not entries:
                raise SystemExit(f"Unknown domain: {args[1]}")
        for key, filename in sorted(entries.items()):
            print(f"{key}\ttools/{filename}")
        return 0
    try:
        path = command_path(args[0])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    prepare_imports()
    original = sys.argv
    try:
        sys.argv = [str(path), *args[1:]]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
