from __future__ import annotations

import pytest

from afs.feature_gates import feature_enabled, require_feature


def test_require_training_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AFS_ENABLE_TRAINING", raising=False)
    # Force markers off even if the host has torch installed.
    monkeypatch.setattr(
        "afs.feature_gates._marker_available",
        lambda _name: False,
    )
    assert feature_enabled("training") is False
    with pytest.raises(RuntimeError, match="AFS_ENABLE_TRAINING"):
        require_feature("training")


def test_env_enables_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "afs.feature_gates._marker_available",
        lambda _name: False,
    )
    monkeypatch.setenv("AFS_ENABLE_EMBEDDINGS", "1")
    assert feature_enabled("embeddings") is True
    require_feature("embeddings")
