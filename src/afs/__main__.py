"""Entry point for python -m afs."""

import sys

if __name__ == "__main__":
    # Lifecycle hooks run on every tool call; skip the full CLI import for them.
    if sys.argv[1:3] == ["messages", "hook"]:
        from .message_doorbell import hook_main

        raise SystemExit(hook_main(sys.argv[3:]))
    from .cli import main

    raise SystemExit(main())
