"""Human-provenance gates for judgment-bearing CLI decisions.

Trust anchors — approval decisions, mission acceptance — must come from a
person, not an agent. These helpers implement the same pattern as the work
executor's external-write confirmation: prompt on and read from the
controlling terminal (``/dev/tty``) so that piped stdin cannot satisfy the
gate, and identify the reviewer from the operating-system user rather than a
claimable command-line flag. Headless callers fail closed: no terminal means
no confirmation, never a silent pass.
"""

from __future__ import annotations

import getpass
from collections.abc import Callable

TTY_PATH = "/dev/tty"

TtyReader = Callable[[str], "str | None"]


def default_tty_reader(tty_path: str = TTY_PATH) -> TtyReader:
    """Return a reader that prompts on and reads a line from the terminal.

    Reads from ``/dev/tty`` rather than stdin so an agent piping text into the
    command cannot satisfy the prompt. The reader returns ``None`` when no
    terminal is available (a headless or agent context), which callers must
    treat as a refusal.
    """

    def _read(prompt: str) -> str | None:
        try:
            with open(tty_path, "r+", encoding="utf-8") as tty:
                tty.write(prompt)
                tty.flush()
                line = tty.readline()
        except OSError:
            return None
        if line == "":
            return None
        return line.rstrip("\r\n")

    return _read


def os_reviewer() -> str:
    """Best-effort identity of the person at the terminal."""
    try:
        return getpass.getuser() or "human"
    except Exception:
        return "human"


def confirm_typed_token(
    token: str,
    prompt: str,
    *,
    reader: TtyReader | None = None,
) -> str | None:
    """Require the operator to re-type ``token`` on the controlling terminal.

    Returns the OS-level reviewer identity on success and ``None`` when no
    terminal is available or the typed value does not match, so headless
    callers fail closed. ``reader`` is injectable for testing.
    """
    read = reader or default_tty_reader()
    response = read(prompt)
    if response is None or response.strip() != token:
        return None
    return os_reviewer()


def read_human_line(
    prompt: str,
    *,
    reader: TtyReader | None = None,
) -> str | None:
    """Read one free-form line from the controlling terminal.

    Returns ``None`` when no terminal is available. Because the prompt is
    written to ``/dev/tty`` it never contaminates stdout, so callers emitting
    machine-readable output stay clean.
    """
    read = reader or default_tty_reader()
    return read(prompt)
