"""Agent guardrails: quota tracking, model fallback, worktree isolation, approval gates."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File locking for concurrent agent safety
# ---------------------------------------------------------------------------


def _nearest_existing_parent(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _prefer_writable_state_path(filename: str, *, env_var: str) -> Path:
    env_value = os.getenv(env_var, "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()

    registry_value = os.getenv("AFS_AGENT_REGISTRY_PATH", "").strip()
    if registry_value:
        return Path(registry_value).expanduser().resolve().with_name(filename)

    home_candidate = Path("~/.config/afs/agents").expanduser().resolve() / filename
    if os.access(_nearest_existing_parent(home_candidate), os.W_OK):
        return home_candidate

    return (Path.cwd() / ".afs" / "agents" / filename).expanduser().resolve()

@contextmanager
def _file_lock(path: Path, *, timeout: float = 5.0):
    """Advisory file lock using fcntl. Non-blocking with timeout."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    logger.warning("Lock timeout on %s after %.1fs", lock_path, timeout)
                    break  # proceed without lock rather than deadlocking
                time.sleep(0.05)
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


# ---------------------------------------------------------------------------
# Quota tracking
# ---------------------------------------------------------------------------

DEFAULT_QUOTA_FILE = Path("~/.config/afs/agents/quota.json")

# Defaults: conservative to protect wallet
DEFAULT_QUOTAS = {
    "claude": {"hourly": 20, "daily": 100, "cost_ceiling_usd": 5.0},
    "gemini": {"hourly": 60, "daily": 500, "cost_ceiling_usd": 2.0},
    "codex": {"hourly": 30, "daily": 200, "cost_ceiling_usd": 3.0},
    "local": {"hourly": 0, "daily": 0, "cost_ceiling_usd": 0.0},  # unlimited
}

# Model fallback chain: most capable → cheapest
DEFAULT_FALLBACK_CHAIN = ["claude", "gemini", "codex", "local"]


@dataclass
class QuotaEntry:
    provider: str
    calls_this_hour: int = 0
    calls_today: int = 0
    cost_today_usd: float = 0.0
    hour_window: str = ""  # ISO hour like "2026-03-24T14"
    day_window: str = ""  # ISO date like "2026-03-24"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "calls_this_hour": self.calls_this_hour,
            "calls_today": self.calls_today,
            "cost_today_usd": self.cost_today_usd,
            "hour_window": self.hour_window,
            "day_window": self.day_window,
        }


class QuotaTracker:
    """Track API call counts and costs per provider with automatic window rolling."""

    def __init__(self, path: Path | None = None, quotas: dict[str, Any] | None = None):
        self._path = (
            path.expanduser().resolve()
            if path is not None
            else _prefer_writable_state_path("quota.json", env_var="AFS_AGENT_QUOTA_PATH")
        )
        self._quotas = quotas or DEFAULT_QUOTAS
        self._entries: dict[str, QuotaEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with _file_lock(self._path):
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for key, val in data.items():
                    self._entries[key] = QuotaEntry(
                        provider=key,
                        calls_this_hour=val.get("calls_this_hour", 0),
                        calls_today=val.get("calls_today", 0),
                        cost_today_usd=val.get("cost_today_usd", 0.0),
                        hour_window=val.get("hour_window", ""),
                        day_window=val.get("day_window", ""),
                    )
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load quota file %s", self._path)

    def _save(self) -> None:
        with _file_lock(self._path):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Re-read before writing to merge concurrent updates
            fresh: dict[str, QuotaEntry] = {}
            if self._path.exists():
                try:
                    disk_data = json.loads(self._path.read_text(encoding="utf-8"))
                    for key, val in disk_data.items():
                        fresh[key] = QuotaEntry(
                            provider=key,
                            calls_this_hour=val.get("calls_this_hour", 0),
                            calls_today=val.get("calls_today", 0),
                            cost_today_usd=val.get("cost_today_usd", 0.0),
                            hour_window=val.get("hour_window", ""),
                            day_window=val.get("day_window", ""),
                        )
                except (json.JSONDecodeError, OSError):
                    pass
            # Merge: our in-memory values take precedence for providers we touched
            for key, entry in self._entries.items():
                disk_entry = fresh.get(key)
                if disk_entry and disk_entry.hour_window == entry.hour_window:
                    # Same window — take the max to avoid losing concurrent writes
                    entry.calls_this_hour = max(entry.calls_this_hour, disk_entry.calls_this_hour)
                if disk_entry and disk_entry.day_window == entry.day_window:
                    entry.calls_today = max(entry.calls_today, disk_entry.calls_today)
                    entry.cost_today_usd = max(entry.cost_today_usd, disk_entry.cost_today_usd)
                fresh[key] = entry
            data = {k: v.to_dict() for k, v in fresh.items()}
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _roll_windows(self, entry: QuotaEntry) -> QuotaEntry:
        now = datetime.now(timezone.utc)
        current_hour = now.strftime("%Y-%m-%dT%H")
        current_day = now.strftime("%Y-%m-%d")
        if entry.hour_window != current_hour:
            entry.calls_this_hour = 0
            entry.hour_window = current_hour
        if entry.day_window != current_day:
            entry.calls_today = 0
            entry.cost_today_usd = 0.0
            entry.day_window = current_day
        return entry

    def _get_entry(self, provider: str) -> QuotaEntry:
        if provider not in self._entries:
            self._entries[provider] = QuotaEntry(provider=provider)
        return self._roll_windows(self._entries[provider])

    def can_call(self, provider: str) -> bool:
        """Check if provider has remaining quota."""
        limits = self._quotas.get(provider, {})
        if not limits:
            return True
        # Local with zero limits = unlimited (no external API cost)
        if provider == "local":
            hourly = limits.get("hourly", 0)
            daily = limits.get("daily", 0)
            ceiling = limits.get("cost_ceiling_usd", 0.0)
            if hourly == 0 and daily == 0 and ceiling == 0.0:
                return True
        entry = self._get_entry(provider)
        hourly_limit = limits.get("hourly", 0)
        daily_limit = limits.get("daily", 0)
        cost_ceiling = limits.get("cost_ceiling_usd", 0.0)
        if hourly_limit > 0 and entry.calls_this_hour >= hourly_limit:
            logger.info("Quota exceeded for %s: %d/%d hourly calls",
                        provider, entry.calls_this_hour, hourly_limit)
            return False
        if daily_limit > 0 and entry.calls_today >= daily_limit:
            logger.info("Quota exceeded for %s: %d/%d daily calls",
                        provider, entry.calls_today, daily_limit)
            return False
        if cost_ceiling > 0 and entry.cost_today_usd >= cost_ceiling:
            logger.info("Cost ceiling hit for %s: $%.2f/$%.2f",
                        provider, entry.cost_today_usd, cost_ceiling)
            return False
        return True

    def record_call(self, provider: str, cost_usd: float = 0.0) -> None:
        """Record an API call for quota tracking."""
        entry = self._get_entry(provider)
        entry.calls_this_hour += 1
        entry.calls_today += 1
        entry.cost_today_usd += cost_usd
        self._save()

    def usage_summary(self) -> dict[str, Any]:
        """Return current usage across all providers."""
        summary = {}
        for provider in self._quotas:
            entry = self._get_entry(provider)
            limits = self._quotas.get(provider, {})
            summary[provider] = {
                "calls_this_hour": entry.calls_this_hour,
                "hourly_limit": limits.get("hourly", 0),
                "calls_today": entry.calls_today,
                "daily_limit": limits.get("daily", 0),
                "cost_today_usd": entry.cost_today_usd,
                "cost_ceiling_usd": limits.get("cost_ceiling_usd", 0.0),
            }
        return summary


# ---------------------------------------------------------------------------
# Model fallback chain
# ---------------------------------------------------------------------------

@dataclass
class ModelRoute:
    provider: str  # claude, gemini, codex, local
    model_id: str  # specific model identifier
    reason: str = ""  # why this provider was chosen


def resolve_model(
    preferred: str = "claude",
    fallback_chain: list[str] | None = None,
    quota_tracker: QuotaTracker | None = None,
    task_tier: str = "standard",  # critical, standard, background
) -> ModelRoute:
    """Resolve which model to use based on quota, availability, and task tier.

    Task tiers:
        critical  — must use Claude/Gemini, fail if unavailable
        standard  — prefer Claude, fall back through chain
        background — prefer cheapest available (local → codex → gemini → claude)
    """
    chain = fallback_chain or list(DEFAULT_FALLBACK_CHAIN)
    tracker = quota_tracker or QuotaTracker()

    # For background tasks, reverse the chain to prefer cheap models
    if task_tier == "background":
        chain = list(reversed(chain))

    model_map = {
        "claude": "claude-3-5-sonnet",
        "gemini": "gemini-1.5-pro",
        "codex": "codex",
        "local": "qwen2.5-coder:14b",
    }

    for provider in chain:
        if tracker.can_call(provider):
            return ModelRoute(
                provider=provider,
                model_id=model_map.get(provider, provider),
                reason=f"selected via fallback chain (tier={task_tier})",
            )

    # All quotas exhausted
    if task_tier == "critical":
        raise RuntimeError("All model providers exhausted for critical task")

    # Last resort: local always works
    return ModelRoute(
        provider="local",
        model_id="qwen2.5-coder:14b",
        reason="all quotas exhausted, falling back to local",
    )


# ---------------------------------------------------------------------------
# Worktree isolation
# ---------------------------------------------------------------------------

def ensure_worktree(repo_path: Path, branch_name: str) -> Path:
    """Create or reuse a git worktree for isolated agent work.

    Returns the worktree path. Does NOT modify the main repo.
    """
    worktrees_root = Path.home() / "src" / "worktrees"
    worktree_path = worktrees_root / branch_name

    if worktree_path.exists():
        # Verify it's still a valid worktree
        result = subprocess.run(
            ["git", "-C", str(worktree_path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return worktree_path
        # Stale worktree, remove and recreate
        subprocess.run(
            ["git", "-C", str(repo_path), "worktree", "remove", "--force", str(worktree_path)],
            capture_output=True, timeout=10,
        )

    worktrees_root.mkdir(parents=True, exist_ok=True)

    # Create branch if it doesn't exist
    subprocess.run(
        ["git", "-C", str(repo_path), "branch", branch_name],
        capture_output=True, timeout=10,
    )

    result = subprocess.run(
        ["git", "-C", str(repo_path), "worktree", "add", str(worktree_path), branch_name],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create worktree: {result.stderr.strip()}")

    logger.info("Created worktree at %s for branch %s", worktree_path, branch_name)
    return worktree_path


def cleanup_worktree(repo_path: Path, branch_name: str) -> bool:
    """Remove a worktree if it has no uncommitted changes."""
    worktree_path = Path.home() / "src" / "worktrees" / branch_name
    if not worktree_path.exists():
        return True

    # Check for uncommitted changes
    result = subprocess.run(
        ["git", "-C", str(worktree_path), "status", "--porcelain"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0 or result.stdout.strip():
        logger.warning("Worktree %s has uncommitted changes, skipping cleanup", worktree_path)
        return False

    subprocess.run(
        ["git", "-C", str(repo_path), "worktree", "remove", str(worktree_path)],
        capture_output=True, timeout=10,
    )
    return True


# ---------------------------------------------------------------------------
# Approval gates
# ---------------------------------------------------------------------------

APPROVAL_FILE = Path("~/.config/afs/agents/approvals.json")

# Actions that always require approval
ALWAYS_APPROVE = frozenset({
    "git_push",
    "git_force_push",
    "file_delete",
    "rom_edit",
    "deploy",
    "send_email",
    "create_pr",
    "merge_pr",
})

# Actions that can be auto-approved for background agents
AUTO_APPROVE = frozenset({
    "file_read",
    "file_write_scratchpad",
    "git_status",
    "git_diff",
    "git_log",
    "context_read",
    "context_write",
    "run_tests",
    "run_build",
    "embedding_update",
})


@dataclass
class ApprovalRequest:
    agent: str
    action: str
    detail: str
    timestamp: str
    status: str = "pending"  # pending, approved, rejected, auto_approved
    reviewed_by: str = ""
    reviewed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "action": self.action,
            "detail": self.detail,
            "timestamp": self.timestamp,
            "status": self.status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
        }


class ApprovalGate:
    """Gate that blocks dangerous actions until human review."""

    def __init__(self, path: Path | None = None):
        self._path = (
            path.expanduser().resolve()
            if path is not None
            else _prefer_writable_state_path(
                "approvals.json",
                env_var="AFS_AGENT_APPROVALS_PATH",
            )
        )
        self._pending: list[ApprovalRequest] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with _file_lock(self._path):
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for item in data:
                    self._pending.append(ApprovalRequest(**item))
            except (json.JSONDecodeError, OSError, TypeError):
                pass

    def _save(self) -> None:
        with _file_lock(self._path):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps([r.to_dict() for r in self._pending], indent=2),
                encoding="utf-8",
            )

    def check(self, agent: str, action: str, detail: str = "") -> bool:
        """Check if an action is allowed. Returns True if auto-approved or pre-approved."""
        if action in AUTO_APPROVE:
            return True
        if action in ALWAYS_APPROVE:
            # Check if there's a pre-approval for this specific request
            for req in self._pending:
                if (req.agent == agent and req.action == action
                        and req.status == "approved"):
                    return True
            # Queue for approval
            self._queue(agent, action, detail)
            return False
        # Unknown action — be conservative, queue it
        self._queue(agent, action, detail)
        return False

    def _queue(self, agent: str, action: str, detail: str) -> None:
        # Don't duplicate pending requests
        for req in self._pending:
            if req.agent == agent and req.action == action and req.status == "pending":
                return
        req = ApprovalRequest(
            agent=agent,
            action=action,
            detail=detail,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._pending.append(req)
        self._save()
        logger.info("Queued approval request: agent=%s action=%s", agent, action)

    def pending_requests(self) -> list[ApprovalRequest]:
        return [r for r in self._pending if r.status == "pending"]

    def approve(self, agent: str, action: str, reviewer: str = "human") -> bool:
        for req in self._pending:
            if req.agent == agent and req.action == action and req.status == "pending":
                req.status = "approved"
                req.reviewed_by = reviewer
                req.reviewed_at = datetime.now(timezone.utc).isoformat()
                self._save()
                return True
        return False

    def reject(self, agent: str, action: str, reviewer: str = "human") -> bool:
        for req in self._pending:
            if req.agent == agent and req.action == action and req.status == "pending":
                req.status = "rejected"
                req.reviewed_by = reviewer
                req.reviewed_at = datetime.now(timezone.utc).isoformat()
                self._save()
                return True
        return False


# ---------------------------------------------------------------------------
# Guardrailed agent context — combine all guardrails into one interface
# ---------------------------------------------------------------------------

@dataclass
class GuardrailConfig:
    """Configuration for agent guardrails."""
    enable_quota: bool = True
    enable_worktrees: bool = True
    enable_approvals: bool = True
    task_tier: str = "standard"
    fallback_chain: list[str] = field(default_factory=lambda: list(DEFAULT_FALLBACK_CHAIN))
    quotas: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_QUOTAS))
    max_iterations: int = 50  # RALPH loop iteration cap
    checkpoint_interval: int = 5  # write progress every N iterations
    dry_run: bool = False  # log actions without executing


class GuardrailedAgent:
    """Mixin / context wrapper for agents that need guardrails.

    Usage:
        guard = GuardrailedAgent("my-agent", config=GuardrailConfig())
        model = guard.resolve_model()
        if guard.can_do("git_push", "push changes to origin"):
            # do the push
            guard.record_call(model.provider, cost_usd=0.01)
    """

    def __init__(self, agent_name: str, config: GuardrailConfig | None = None):
        self.agent_name = agent_name
        self.config = config or GuardrailConfig()
        self._tracker = QuotaTracker(quotas=self.config.quotas) if self.config.enable_quota else None
        self._gate = ApprovalGate() if self.config.enable_approvals else None
        self._iteration = 0

    def resolve_model(self, task_tier: str | None = None) -> ModelRoute:
        """Get the best available model respecting quota limits."""
        tier = task_tier or self.config.task_tier
        return resolve_model(
            fallback_chain=self.config.fallback_chain,
            quota_tracker=self._tracker,
            task_tier=tier,
        )

    def can_do(self, action: str, detail: str = "") -> bool:
        """Check if action is allowed through approval gate."""
        if not self.config.enable_approvals:
            return True
        if self.config.dry_run:
            logger.info("[DRY RUN] Would request approval: %s — %s", action, detail)
            return False
        assert self._gate is not None
        return self._gate.check(self.agent_name, action, detail)

    def record_call(self, provider: str, cost_usd: float = 0.0) -> None:
        """Record an API call for quota tracking."""
        if self._tracker:
            self._tracker.record_call(provider, cost_usd)

    def should_continue(self) -> bool:
        """Check if RALPH-style loop should continue (iteration cap)."""
        self._iteration += 1
        if self._iteration > self.config.max_iterations:
            logger.info("Agent %s hit iteration cap (%d)", self.agent_name, self.config.max_iterations)
            return False
        return True

    def should_checkpoint(self) -> bool:
        """Check if this iteration should write a progress checkpoint."""
        return self._iteration % self.config.checkpoint_interval == 0

    def usage_summary(self) -> dict[str, Any]:
        """Return quota usage summary."""
        if self._tracker:
            return self._tracker.usage_summary()
        return {}

    def pending_approvals(self) -> list[ApprovalRequest]:
        """Return pending approval requests for this agent."""
        if self._gate:
            return [r for r in self._gate.pending_requests() if r.agent == self.agent_name]
        return []

    @property
    def iteration(self) -> int:
        return self._iteration
