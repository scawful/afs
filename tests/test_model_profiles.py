from __future__ import annotations

from afs.model_profiles import profile_for_client_model, resolve_model_client_profile


def test_resolve_modern_model_profiles() -> None:
    codex = resolve_model_client_profile("gpt-5.5")
    assert codex.family == "codex"
    assert "stable prefix" in codex.cache_strategy

    opus = resolve_model_client_profile("claude-opus-4-6")
    assert opus.family == "claude"
    assert "1024" in opus.cache_strategy
    assert resolve_model_client_profile("claude-opus-4-8").name == opus.name

    agy = resolve_model_client_profile("jetski")
    assert agy.name == "antigravity:agy"
    assert agy.family == "gemini"


def test_hcode_profile_is_provider_neutral() -> None:
    profile = profile_for_client_model("hcode", "generic")
    assert profile.family == "hcode"
    assert "provider-neutral" in " ".join(profile.notes)
