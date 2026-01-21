"""Slack notification handler using webhooks."""

from __future__ import annotations

import json
import os
from typing import Optional

import requests

from afs.logging_config import get_logger

from .base import NotificationEvent, NotificationHandler, NotificationLevel

logger = get_logger(__name__)


class SlackNotifier(NotificationHandler):
    """Send notifications to Slack via webhooks."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        channel: Optional[str] = None,
        username: str = "AFS Notifier"
    ):
        """Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL (from env: SLACK_WEBHOOK_URL)
            channel: Target channel (optional, overrides webhook default)
            username: Bot username shown in Slack
        """
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.channel = channel or os.getenv("SLACK_CHANNEL")
        self.username = username

    def is_configured(self) -> bool:
        """Check if Slack notifier is properly configured."""
        return bool(self.webhook_url)

    def send(self, event: NotificationEvent) -> bool:
        """Send Slack notification.

        Args:
            event: Notification event

        Returns:
            True if successful, False otherwise
        """
        if not self.is_configured():
            logger.warning("Slack notifier not configured")
            return False

        try:
            payload = self._create_payload(event)
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.debug(f"Slack notification sent: {event.title}")
                return True
            else:
                logger.error(
                    f"Failed to send Slack notification: {response.status_code}"
                )
                return False

        except requests.Timeout:
            logger.error("Slack notification timeout")
            return False
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    def _create_payload(self, event: NotificationEvent) -> dict:
        """Create Slack webhook payload from event.

        Args:
            event: Notification event

        Returns:
            Slack payload dictionary
        """
        # Color based on level
        colors = {
            NotificationLevel.INFO: "#0099ff",
            NotificationLevel.SUCCESS: "#36a64f",
            NotificationLevel.WARNING: "#ff9900",
            NotificationLevel.ERROR: "#ff0000",
            NotificationLevel.CRITICAL: "#8B0000"
        }
        color = colors.get(event.level, "#0099ff")

        # Build fields
        fields = []

        if event.model_name:
            fields.append({
                "title": "Model",
                "value": event.model_name,
                "short": True
            })

        if event.run_id:
            fields.append({
                "title": "Run ID",
                "value": event.run_id,
                "short": True
            })

        if event.cost is not None:
            fields.append({
                "title": "Cost",
                "value": f"${event.cost:.2f}",
                "short": True
            })

        if event.epoch is not None:
            fields.append({
                "title": "Epoch",
                "value": str(event.epoch),
                "short": True
            })

        if event.batch is not None:
            fields.append({
                "title": "Batch",
                "value": str(event.batch),
                "short": True
            })

        # Add metrics as fields
        for key, value in event.metrics.items():
            fields.append({
                "title": key,
                "value": str(value),
                "short": True
            })

        # Build attachment
        attachment = {
            "color": color,
            "title": event.title,
            "text": event.message,
            "fields": fields,
            "ts": int(event.timestamp.timestamp())
        }

        # Add error details if present
        if event.error_details:
            attachment["fields"].append({
                "title": "Error",
                "value": f"```{event.error_details}```",
                "short": False
            })

        # Build payload
        payload = {
            "username": self.username,
            "attachments": [attachment]
        }

        if self.channel:
            payload["channel"] = self.channel

        return payload

    def send_batch(self, events: list[NotificationEvent]) -> int:
        """Send multiple notifications in a batch.

        Args:
            events: List of notification events

        Returns:
            Number of successfully sent messages
        """
        count = 0
        for event in events:
            if self.send(event):
                count += 1
        return count


def send_slack_message(
    message: str,
    title: Optional[str] = None,
    webhook_url: Optional[str] = None
) -> bool:
    """Convenience function to send a Slack message.

    Args:
        message: Message content
        title: Optional title
        webhook_url: Slack webhook URL (from env if not provided)

    Returns:
        True if successful
    """
    webhook = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        logger.error("No Slack webhook URL configured")
        return False

    try:
        payload = {
            "text": title or "AFS Notification",
            "attachments": [{
                "text": message,
                "color": "#0099ff"
            }]
        }

        response = requests.post(webhook, json=payload, timeout=10)
        return response.status_code == 200

    except Exception as e:
        logger.error(f"Failed to send Slack message: {e}")
        return False
