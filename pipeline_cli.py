"""Compatibility CLI for the relocated legacy pipeline command."""

from tools.legacy.pipeline_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
