"""Allow ``python -m foundry``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
