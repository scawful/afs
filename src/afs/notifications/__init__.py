"""Notification system for training events.

Supports multiple notification channels:
- Desktop notifications (macOS)
- Email notifications (SMTP)
- Slack webhooks
- Discord webhooks

Usage:
    from afs.notifications import NotificationManager

    manager = NotificationManager()
    manager.notify("Training started", event_type="training_started")
"""

from __future__ import annotations

from .base import (
    EventType,
    NotificationEvent,
    NotificationHandler,
    NotificationLevel,
    NotificationManager,
)
from .desktop import DesktopNotifier
from .discord import DiscordNotifier
from .email import EmailNotifier
from .slack import SlackNotifier

__all__ = [
    "NotificationManager",
    "NotificationEvent",
    "NotificationHandler",
    "EventType",
    "NotificationLevel",
    "DesktopNotifier",
    "EmailNotifier",
    "SlackNotifier",
    "DiscordNotifier",
]
