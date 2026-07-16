"""Cooperative human-provenance broker for judgment-bearing decisions.

The broker separates *collecting* a decision at a controlling terminal from
*persisting* it in an AFS store.  Stores only mark a decision as human-confirmed
when they receive a capability minted by :class:`HumanDecisionBroker`; caller
supplied strings such as ``reviewed_via="tty"`` are never authoritative.

This is a cooperative same-user boundary, not an operating-system security
sandbox.  A process running as the same account can import Python internals,
open the same terminal, or edit AFS state on disk.  The capability prevents
ordinary/public API misuse and accidental provenance forgery; process
isolation and hostile-code containment remain the host's responsibility.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

TtyReader = Callable[[str], "str | None"]

_AUTHORIZATION_SEAL = object()
_AUTHORIZATION_SIGNING_KEY = secrets.token_bytes(32)
_BROKER_INJECTION_SEAL = object()
_AUTHORIZATION_CONSUMPTION_LOCK = threading.Lock()
_CONSUMED_AUTHORIZATIONS: set[str] = set()


def decision_scope(surface: str, decision: str, subject: str) -> str:
    """Build a stable capability scope without embedding sensitive text."""

    digest = hashlib.sha256(subject.encode("utf-8")).hexdigest()
    return f"{surface}:{decision}:{digest}"


def decision_scope_parts(
    surface: str, decision: str, *parts: str
) -> str:
    """Build an unambiguous scope from caller-controlled components."""
    subject = json.dumps(
        list(parts),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return decision_scope(surface, decision, subject)


@dataclass(frozen=True)
class HumanIdentity:
    """OS-derived reviewer identity attached to terminal confirmation."""

    reviewer: str
    subject: str
    kind: str
    authenticated: bool


def _authorization_signature(
    *,
    identity: HumanIdentity,
    confirmed_via: str,
    scope: str,
    process_id: int,
    nonce: str,
) -> bytes:
    material = "\0".join(
        (
            identity.reviewer,
            identity.subject,
            identity.kind,
            "1" if identity.authenticated else "0",
            confirmed_via,
            scope,
            str(process_id),
            nonce,
        )
    ).encode("utf-8")
    return hmac.digest(_AUTHORIZATION_SIGNING_KEY, material, "sha256")


@dataclass(frozen=True, init=False)
class HumanAuthorization:
    """Opaque capability proving that the broker collected terminal input.

    Callers may pass this object to store ``*_human`` methods, but cannot mint
    one through the public constructor.  This deliberately is not a security
    token against hostile Python running as the same user; see the module
    threat boundary above.
    """

    _seal: object = field(repr=False, compare=False)
    identity: HumanIdentity
    confirmed_via: str
    scope: str
    process_id: int
    nonce: str
    _signature: bytes = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        _seal: object,
        identity: HumanIdentity,
        confirmed_via: str,
        scope: str,
        process_id: int,
        nonce: str,
        _signature: bytes,
    ) -> None:
        if _seal is not _AUTHORIZATION_SEAL:
            raise TypeError("HumanAuthorization values are minted by HumanDecisionBroker")
        if not scope.strip():
            raise TypeError("HumanAuthorization scope is required")
        if not nonce:
            raise TypeError("HumanAuthorization nonce is required")
        expected_signature = _authorization_signature(
            identity=identity,
            confirmed_via=confirmed_via,
            scope=scope,
            process_id=process_id,
            nonce=nonce,
        )
        if not hmac.compare_digest(_signature, expected_signature):
            raise TypeError("HumanAuthorization fields were not minted by the broker")
        object.__setattr__(self, "_seal", _seal)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "confirmed_via", confirmed_via)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "process_id", process_id)
        object.__setattr__(self, "nonce", nonce)
        object.__setattr__(self, "_signature", _signature)


def is_human_authorization(value: Any, *, scope: str) -> bool:
    """Return whether ``value`` is a broker-minted authorization capability."""

    return (
        isinstance(value, HumanAuthorization)
        and getattr(value, "_seal", None) is _AUTHORIZATION_SEAL
        and value.identity.authenticated
        and bool(value.identity.subject.strip())
        and value.confirmed_via == "controlling_terminal"
        and value.scope == scope
        and value.process_id == os.getpid()
        and bool(value.nonce)
        and hmac.compare_digest(
            value._signature,
            _authorization_signature(
                identity=value.identity,
                confirmed_via=value.confirmed_via,
                scope=value.scope,
                process_id=value.process_id,
                nonce=value.nonce,
            ),
        )
    )


def consume_human_authorization(value: Any, *, scope: str) -> bool:
    """Validate and consume a capability exactly once in this process.

    Capabilities are bound to the minting process, and holding consumed
    broker nonce here prevents replay even if ordinary Python copying creates
    another wrapper object. Persistent store/record scoping independently
    prevents a capability from authorizing a different decision.
    """
    if not is_human_authorization(value, scope=scope):
        return False
    key = value.nonce
    with _AUTHORIZATION_CONSUMPTION_LOCK:
        if key in _CONSUMED_AUTHORIZATIONS:
            return False
        _CONSUMED_AUTHORIZATIONS.add(key)
    return True


def _posix_identity() -> HumanIdentity | None:
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return None
    try:
        uid = int(getuid())
    except (OSError, TypeError, ValueError):
        return None
    reviewer = f"uid:{uid}"
    try:
        import pwd

        name = pwd.getpwuid(uid).pw_name.strip()
        if name:
            reviewer = name
    except (ImportError, KeyError, OSError):
        pass
    return HumanIdentity(
        reviewer=reviewer,
        subject=f"uid:{uid}",
        kind="uid",
        authenticated=True,
    )


def _windows_sid() -> str | None:
    """Return the current process token's user SID without trusting env vars."""

    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        token_query = 0x0008
        token_user = 1
        token = wintypes.HANDLE()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = [wintypes.LPVOID]
        kernel32.LocalFree.restype = wintypes.LPVOID
        advapi32.OpenProcessToken.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_uint,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.ConvertSidToStringSidW.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.LPWSTR),
        ]
        advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
        if not advapi32.OpenProcessToken(
            kernel32.GetCurrentProcess(), token_query, ctypes.byref(token)
        ):
            return None
        try:
            size = wintypes.DWORD()
            advapi32.GetTokenInformation(token, token_user, None, 0, ctypes.byref(size))
            if not size.value:
                return None
            buffer = ctypes.create_string_buffer(size.value)
            if not advapi32.GetTokenInformation(
                token,
                token_user,
                buffer,
                size.value,
                ctypes.byref(size),
            ):
                return None

            class SidAndAttributes(ctypes.Structure):
                _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]

            sid_pointer = ctypes.cast(buffer, ctypes.POINTER(SidAndAttributes)).contents.Sid
            sid_text = wintypes.LPWSTR()
            if not advapi32.ConvertSidToStringSidW(
                sid_pointer, ctypes.byref(sid_text)
            ):
                return None
            try:
                return str(sid_text.value or "").strip() or None
            finally:
                kernel32.LocalFree(ctypes.cast(sid_text, wintypes.LPVOID))
        finally:
            kernel32.CloseHandle(token)
    except (AttributeError, ImportError, OSError, ValueError):
        return None


def os_identity() -> HumanIdentity:
    """Resolve an OS UID/SID identity, or explicitly label it unavailable.

    Environment-derived usernames are intentionally not used: ``USER`` and
    similar variables are caller-controlled and therefore cannot establish
    provenance.
    """

    identity = _posix_identity()
    if identity is not None:
        return identity
    sid = _windows_sid()
    if sid:
        return HumanIdentity(
            reviewer=sid,
            subject=f"sid:{sid}",
            kind="sid",
            authenticated=True,
        )
    return HumanIdentity(
        reviewer="unauthenticated",
        subject="",
        kind="unavailable",
        authenticated=False,
    )


def os_reviewer() -> str:
    """Compatibility label for non-authoritative programmatic records.

    This value alone never proves human provenance.  Authoritative store
    writes require a :class:`HumanAuthorization` capability.
    """

    return os_identity().reviewer


def default_terminal_reader(terminal_path: str | None = None) -> TtyReader:
    """Return a cross-platform controlling-terminal line reader.

    POSIX uses ``os.ctermid()`` (normally ``/dev/tty``).  Windows uses the
    console devices ``CONOUT$`` and ``CONIN$``.  ``terminal_path`` remains an
    injectable compatibility seam for tests and callers with a known POSIX
    terminal.  Failure to obtain a controlling terminal returns ``None`` and
    callers fail closed.
    """

    def _read(prompt: str) -> str | None:
        try:
            if terminal_path:
                with open(terminal_path, "r+", encoding="utf-8") as terminal:
                    terminal.write(prompt)
                    terminal.flush()
                    line = terminal.readline()
            elif os.name == "nt":
                with open("CONOUT$", "w", encoding="utf-8") as output:
                    output.write(prompt)
                    output.flush()
                with open("CONIN$", encoding="utf-8") as input_stream:
                    line = input_stream.readline()
            else:
                ctermid = getattr(os, "ctermid", None)
                path = ctermid() if ctermid is not None else "/dev/tty"
                with open(path, "r+", encoding="utf-8") as terminal:
                    terminal.write(prompt)
                    terminal.flush()
                    line = terminal.readline()
        except (OSError, ValueError):
            return None
        if line == "":
            return None
        return line.rstrip("\r\n")

    return _read


# Backward-compatible name; the implementation is no longer /dev/tty-only.
default_tty_reader = default_terminal_reader


class HumanDecisionBroker:
    """Collect terminal input and mint store-verifiable capabilities."""

    def __init__(
        self,
        *,
        _reader: TtyReader | None = None,
        _identity_provider: Callable[[], HumanIdentity] | None = None,
        _injection_seal: object | None = None,
    ) -> None:
        if (_reader is not None or _identity_provider is not None) and (
            _injection_seal is not _BROKER_INJECTION_SEAL
        ):
            raise TypeError("custom broker backends are private test seams")
        self._reader = _reader or default_terminal_reader()
        self._identity_provider = _identity_provider or os_identity

    def _authorization(self, scope: str) -> HumanAuthorization | None:
        identity = self._identity_provider()
        if not identity.authenticated or not identity.subject.strip():
            return None
        process_id = os.getpid()
        nonce = secrets.token_hex(32)
        signature = _authorization_signature(
            identity=identity,
            confirmed_via="controlling_terminal",
            scope=scope,
            process_id=process_id,
            nonce=nonce,
        )
        return HumanAuthorization(
            _seal=_AUTHORIZATION_SEAL,
            identity=identity,
            confirmed_via="controlling_terminal",
            scope=scope,
            process_id=process_id,
            nonce=nonce,
            _signature=signature,
        )

    def confirm_token(
        self, token: str, prompt: str, *, scope: str
    ) -> HumanAuthorization | None:
        """Mint a capability only after the terminal re-types ``token``."""

        response = self._reader(prompt)
        if response is None or response.strip() != token:
            return None
        return self._authorization(scope)

    def read_line(
        self,
        prompt: str,
        *,
        scope: str | Callable[[str], str],
    ) -> tuple[str, HumanAuthorization] | None:
        """Read a human-authored line and return it with its capability."""

        response = self._reader(prompt)
        if response is None:
            return None
        resolved_scope = scope(response) if callable(scope) else scope
        authorization = self._authorization(resolved_scope)
        if authorization is None:
            return None
        return response, authorization


def _broker_for_reader(
    reader: TtyReader | None,
    *,
    identity_provider: Callable[[], HumanIdentity] | None = None,
) -> HumanDecisionBroker:
    """Private injection seam used by CLI unit tests."""

    if reader is None and identity_provider is None:
        return HumanDecisionBroker()
    return HumanDecisionBroker(
        _reader=reader or default_terminal_reader(),
        _identity_provider=identity_provider,
        _injection_seal=_BROKER_INJECTION_SEAL,
    )


def confirm_typed_token(
    token: str,
    prompt: str,
    *,
    scope: str,
) -> HumanAuthorization | None:
    """Compatibility helper backed by :class:`HumanDecisionBroker`."""

    return HumanDecisionBroker().confirm_token(token, prompt, scope=scope)


def read_human_line(
    prompt: str,
    *,
    scope: str,
) -> str | None:
    """Compatibility helper returning only the line, not authorization.

    Judgment-bearing store writes must use :class:`HumanDecisionBroker`
    directly so the capability is not discarded.
    """

    result = HumanDecisionBroker().read_line(prompt, scope=scope)
    return result[0] if result is not None else None
