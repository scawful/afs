from __future__ import annotations

import builtins
import copy
import dataclasses
import io

import pytest

import afs.human_provenance as provenance


def test_broker_mints_capability_only_for_matching_terminal_token() -> None:
    scope = provenance.decision_scope("test", "approve", "request-1")
    broker = provenance._broker_for_reader(lambda _prompt: "expected")
    authorization = broker.confirm_token("expected", "prompt", scope=scope)
    assert provenance.is_human_authorization(authorization, scope=scope)
    assert authorization is not None
    assert authorization.confirmed_via == "controlling_terminal"

    refused = provenance._broker_for_reader(lambda _prompt: "forged").confirm_token(
        "expected", "prompt", scope=scope
    )
    assert refused is None

    other_scope = provenance.decision_scope("test", "approve", "request-2")
    assert not provenance.is_human_authorization(authorization, scope=other_scope)


def test_structured_scope_components_cannot_collide_on_embedded_nul() -> None:
    left = provenance.decision_scope_parts("test", "create", "a\0b", "c")
    right = provenance.decision_scope_parts("test", "create", "a", "b\0c")
    assert left != right


def test_public_broker_rejects_custom_reader_injection() -> None:
    with pytest.raises(TypeError, match="private test seams"):
        provenance.HumanDecisionBroker(_reader=lambda _prompt: "forged")


def test_public_constructor_cannot_mint_authorization() -> None:
    with pytest.raises(TypeError, match="minted"):
        provenance.HumanAuthorization(
            _seal=object(),
            identity=provenance.HumanIdentity(
                reviewer="human",
                subject="uid:1",
                kind="uid",
                authenticated=True,
            ),
            confirmed_via="controlling_terminal",
            scope="test:approve:hash",
            process_id=provenance.os.getpid(),
            nonce="forged",
            _signature=b"forged",
        )


def test_copied_capability_cannot_bypass_one_shot_consumption() -> None:
    scope = provenance.decision_scope("test", "approve", "request-1")
    authorization = provenance._broker_for_reader(
        lambda _prompt: "expected"
    ).confirm_token("expected", "prompt", scope=scope)
    assert authorization is not None
    shallow_copy = copy.copy(authorization)
    replaced_copy = dataclasses.replace(authorization)
    with pytest.raises(TypeError, match="not minted"):
        dataclasses.replace(authorization, nonce="fresh")
    with pytest.raises(TypeError, match="not minted"):
        dataclasses.replace(authorization, scope="different")

    assert provenance.consume_human_authorization(authorization, scope=scope)
    assert not provenance.consume_human_authorization(shallow_copy, scope=scope)
    assert not provenance.consume_human_authorization(replaced_copy, scope=scope)


@pytest.mark.skipif(not hasattr(provenance.os, "fork"), reason="POSIX fork required")
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork.*:DeprecationWarning"
)
def test_capability_cannot_cross_a_fork_boundary() -> None:
    scope = provenance.decision_scope("test", "approve", "request-1")
    authorization = provenance._broker_for_reader(
        lambda _prompt: "expected"
    ).confirm_token("expected", "prompt", scope=scope)
    assert authorization is not None

    child = provenance.os.fork()
    if child == 0:
        accepted = provenance.is_human_authorization(
            authorization, scope=scope
        )
        provenance.os._exit(1 if accepted else 0)
    _pid, status = provenance.os.waitpid(child, 0)
    assert provenance.os.waitstatus_to_exitcode(status) == 0


def test_posix_identity_ignores_spoofable_user_environment(monkeypatch) -> None:
    monkeypatch.setenv("USER", "alleged-human")
    monkeypatch.setenv("LOGNAME", "alleged-human")
    identity = provenance.os_identity()
    if hasattr(provenance.os, "getuid"):
        assert identity.subject == f"uid:{provenance.os.getuid()}"
        assert identity.authenticated is True
        assert identity.reviewer != "alleged-human"


def test_unavailable_identity_is_explicitly_unauthenticated(monkeypatch) -> None:
    monkeypatch.setattr(provenance, "_posix_identity", lambda: None)
    monkeypatch.setattr(provenance, "_windows_sid", lambda: None)
    identity = provenance.os_identity()
    assert identity.reviewer == "unauthenticated"
    assert identity.subject == ""
    assert identity.authenticated is False


def test_broker_fails_closed_when_os_identity_is_unavailable() -> None:
    identity = provenance.HumanIdentity(
        reviewer="unauthenticated",
        subject="",
        kind="unavailable",
        authenticated=False,
    )
    broker = provenance._broker_for_reader(
        lambda _prompt: "expected", identity_provider=lambda: identity
    )
    scope = provenance.decision_scope("test", "approve", "request-1")

    assert broker.confirm_token("expected", "prompt", scope=scope) is None
    assert broker.read_line("prompt", scope=scope) is None


def test_windows_console_backend_uses_conin_and_conout(monkeypatch) -> None:
    class NoCloseStringIO(io.StringIO):
        def close(self) -> None:
            pass

    output = NoCloseStringIO()
    input_stream = NoCloseStringIO("confirmed\r\n")

    def fake_open(path, mode="r", encoding=None):  # noqa: ANN001, ANN202
        assert encoding == "utf-8"
        if path == "CONOUT$" and mode == "w":
            output.seek(0)
            return output
        if path == "CONIN$" and mode == "r":
            input_stream.seek(0)
            return input_stream
        raise AssertionError((path, mode))

    monkeypatch.setattr(provenance.os, "name", "nt")
    monkeypatch.setattr(builtins, "open", fake_open)
    assert provenance.default_terminal_reader()("prompt: ") == "confirmed"
    assert output.getvalue() == "prompt: "
