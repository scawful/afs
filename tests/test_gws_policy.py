"""Tests for GWS safety policy module."""

from __future__ import annotations

from afs.gws_policy import (
    GWSPolicyConfig,
    GWSPolicyEnforcer,
    RateLimiter,
    RiskTier,
    check_recipient,
    classify_action,
)

# -------------------------------------------------------------------
# classify_action
# -------------------------------------------------------------------


class TestClassifyAction:
    """Action classification maps gws commands to correct risk tiers."""

    def test_gmail_list_is_read(self) -> None:
        c = classify_action(["gmail", "users", "messages", "list", "--params", "{}"])
        assert c.tier == RiskTier.READ
        assert c.service == "gmail"

    def test_gmail_get_is_read(self) -> None:
        c = classify_action(["gmail", "users", "messages", "get", "--params", "{}"])
        assert c.tier == RiskTier.READ

    def test_gmail_send_is_mutate(self) -> None:
        c = classify_action(["gmail", "+send", "--to", "x@y.com"])
        assert c.tier == RiskTier.MUTATE
        assert c.requires_confirm is True

    def test_gmail_draft_is_notify(self) -> None:
        c = classify_action(["gmail", "+draft", "--to", "x@y.com"])
        assert c.tier == RiskTier.NOTIFY
        assert c.requires_confirm is False

    def test_gmail_delete_is_dangerous(self) -> None:
        c = classify_action(["gmail", "users", "messages", "delete", "--params", "{}"])
        assert c.tier == RiskTier.DANGEROUS

    def test_gmail_trash_is_dangerous(self) -> None:
        c = classify_action(["gmail", "users", "messages", "trash", "--params", "{}"])
        assert c.tier == RiskTier.DANGEROUS

    def test_calendar_agenda_is_read(self) -> None:
        c = classify_action(["calendar", "+agenda"])
        assert c.tier == RiskTier.READ

    def test_calendar_insert_is_mutate(self) -> None:
        c = classify_action(["calendar", "+insert", "--json", "{}"])
        assert c.tier == RiskTier.MUTATE

    def test_calendar_delete_is_dangerous(self) -> None:
        c = classify_action(["calendar", "events", "delete", "--params", "{}"])
        assert c.tier == RiskTier.DANGEROUS

    def test_drive_list_is_read(self) -> None:
        c = classify_action(["drive", "files", "list"])
        assert c.tier == RiskTier.READ

    def test_drive_upload_is_mutate(self) -> None:
        c = classify_action(["drive", "files", "create", "--upload", "test.txt"])
        assert c.tier == RiskTier.MUTATE

    def test_drive_delete_is_dangerous(self) -> None:
        c = classify_action(["drive", "files", "delete", "--params", "{}"])
        assert c.tier == RiskTier.DANGEROUS

    def test_drive_permission_change_is_dangerous(self) -> None:
        c = classify_action(["drive", "permissions", "create", "--params", "{}"])
        assert c.tier == RiskTier.DANGEROUS

    def test_sheets_read_is_read(self) -> None:
        c = classify_action(["sheets", "+read", "--spreadsheet-id", "abc"])
        assert c.tier == RiskTier.READ

    def test_sheets_append_is_mutate(self) -> None:
        c = classify_action(["sheets", "+append", "--spreadsheet-id", "abc"])
        assert c.tier == RiskTier.MUTATE

    def test_sheets_clear_is_dangerous(self) -> None:
        c = classify_action(["sheets", "spreadsheets", "values", "clear"])
        assert c.tier == RiskTier.DANGEROUS

    def test_unknown_command_defaults_to_mutate(self) -> None:
        c = classify_action(["admin", "users", "create"])
        assert c.tier == RiskTier.MUTATE
        assert c.requires_confirm is True

    def test_empty_args_defaults_to_mutate(self) -> None:
        c = classify_action([])
        assert c.tier == RiskTier.MUTATE


# -------------------------------------------------------------------
# Recipient filtering
# -------------------------------------------------------------------


class TestRecipientFiltering:
    def test_allowed_recipient(self) -> None:
        ok, _ = check_recipient("friend@example.com", ["*@blocked.com"])
        assert ok is True

    def test_blocked_recipient(self) -> None:
        ok, reason = check_recipient("ceo@competitor.com", ["*@competitor.com"])
        assert ok is False
        assert "competitor.com" in reason

    def test_case_insensitive(self) -> None:
        ok, _ = check_recipient("CEO@COMPETITOR.COM", ["*@competitor.com"])
        assert ok is False

    def test_empty_blocklist(self) -> None:
        ok, _ = check_recipient("anyone@anywhere.com", [])
        assert ok is True


# -------------------------------------------------------------------
# Rate limiter
# -------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_under_limit(self) -> None:
        rl = RateLimiter(max_per_hour=3)
        rl.record()
        rl.record()
        ok, _ = rl.check()
        assert ok is True

    def test_blocks_at_limit(self) -> None:
        rl = RateLimiter(max_per_hour=2)
        rl.record()
        rl.record()
        ok, reason = rl.check()
        assert ok is False
        assert "Rate limit" in reason


# -------------------------------------------------------------------
# GWSPolicyEnforcer
# -------------------------------------------------------------------


class TestGWSPolicyEnforcer:
    def test_read_always_allowed(self) -> None:
        enforcer = GWSPolicyEnforcer()
        decision = enforcer.evaluate(["gmail", "users", "messages", "list"])
        assert decision.allowed is True
        assert decision.needs_human_approval is False

    def test_mutate_requires_confirm_by_default(self) -> None:
        enforcer = GWSPolicyEnforcer()
        decision = enforcer.evaluate(["gmail", "+send", "--to", "x@y.com"])
        assert decision.allowed is True
        assert decision.needs_human_approval is True
        assert decision.requires_confirm is True

    def test_mutate_auto_approved_when_configured(self) -> None:
        config = GWSPolicyConfig(auto_approve_mutate=True)
        enforcer = GWSPolicyEnforcer(config=config)
        decision = enforcer.evaluate(["gmail", "+send", "--to", "x@y.com"])
        assert decision.allowed is True
        assert decision.needs_human_approval is False

    def test_dangerous_always_requires_confirm(self) -> None:
        config = GWSPolicyConfig(auto_approve_mutate=True)
        enforcer = GWSPolicyEnforcer(config=config)
        decision = enforcer.evaluate(["gmail", "users", "messages", "delete"])
        assert decision.allowed is True
        assert decision.needs_human_approval is True

    def test_service_allowlist_blocks_disallowed(self) -> None:
        config = GWSPolicyConfig(allowed_services=["gmail", "calendar"])
        enforcer = GWSPolicyEnforcer(config=config)
        decision = enforcer.evaluate(["drive", "files", "list"])
        assert decision.allowed is False
        assert "not in allowed list" in decision.reason

    def test_service_allowlist_permits_allowed(self) -> None:
        config = GWSPolicyConfig(allowed_services=["gmail", "calendar"])
        enforcer = GWSPolicyEnforcer(config=config)
        decision = enforcer.evaluate(["gmail", "users", "messages", "list"])
        assert decision.allowed is True

    def test_blocked_drive_action(self) -> None:
        config = GWSPolicyConfig(blocked_drive_actions=["permissions.delete"])
        enforcer = GWSPolicyEnforcer(config=config)
        decision = enforcer.evaluate(["drive", "permissions", "delete"])
        assert decision.allowed is False

    def test_disabled_policy_allows_everything(self) -> None:
        config = GWSPolicyConfig(enabled=False)
        enforcer = GWSPolicyEnforcer(config=config)
        decision = enforcer.evaluate(["gmail", "users", "messages", "delete"])
        assert decision.allowed is True

    def test_send_rate_limiting(self) -> None:
        config = GWSPolicyConfig(
            auto_approve_mutate=True,
            rate_limit_sends_per_hour=2,
        )
        enforcer = GWSPolicyEnforcer(config=config)

        # First two sends pass
        d1 = enforcer.evaluate(["gmail", "+send", "--to", "a@b.com"])
        assert d1.allowed is True
        enforcer.record_send()
        d2 = enforcer.evaluate(["gmail", "+send", "--to", "a@b.com"])
        assert d2.allowed is True
        enforcer.record_send()

        # Third send is blocked
        d3 = enforcer.evaluate(["gmail", "+send", "--to", "a@b.com"])
        assert d3.allowed is False
        assert "Rate limit" in d3.reason

    def test_audit_callback_fires(self) -> None:
        log: list[str] = []

        def audit(outcome, classification, decision):
            log.append(outcome)

        enforcer = GWSPolicyEnforcer(audit_callback=audit)
        enforcer.evaluate(["gmail", "users", "messages", "list"])
        assert len(log) >= 1

    def test_notify_auto_approved_by_default(self) -> None:
        enforcer = GWSPolicyEnforcer()
        decision = enforcer.evaluate(["gmail", "+draft", "--to", "x@y.com"])
        assert decision.allowed is True
        assert decision.needs_human_approval is False


# -------------------------------------------------------------------
# GWSPolicyConfig
# -------------------------------------------------------------------


class TestGWSPolicyConfig:
    def test_from_dict_defaults(self) -> None:
        config = GWSPolicyConfig.from_dict({})
        assert config.enabled is True
        assert config.auto_approve_mutate is False
        assert config.rate_limit_sends_per_hour == 10

    def test_from_dict_overrides(self) -> None:
        config = GWSPolicyConfig.from_dict({
            "enabled": True,
            "auto_approve_mutate": True,
            "rate_limit_sends_per_hour": 5,
            "blocked_recipients": ["*@evil.com"],
            "allowed_services": ["gmail"],
        })
        assert config.auto_approve_mutate is True
        assert config.rate_limit_sends_per_hour == 5
        assert config.blocked_recipients == ["*@evil.com"]
        assert config.allowed_services == ["gmail"]
