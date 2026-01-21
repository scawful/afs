"""Discord notification handler using webhooks."""

from __future__ import annotations

import json
import os
from typing import Optional

import requests

from afs.logging_config import get_logger

from .base import NotificationEvent, NotificationHandler, NotificationLevel

logger = get_logger(__name__)


class DiscordNotifier(NotificationHandler):
    """Send notifications to Discord via webhooks."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        username: str = "AFS"
    ):
        """Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL (from env: DISCORD_WEBHOOK_URL)
            username: Bot username shown in Discord
        """
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self.username = username

    def is_configured(self) -> bool:
        """Check if Discord notifier is properly configured."""
        return bool(self.webhook_url)

    def send(self, event: NotificationEvent) -> bool:
        """Send Discord notification.

        Args:
            event: Notification event

        Returns:
            True if successful, False otherwise
        """
        if not self.is_configured():
            logger.warning("Discord notifier not configured")
            return False

        try:
            payload = self._create_payload(event)
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 204:  # Discord returns 204 for success
                logger.debug(f"Discord notification sent: {event.title}")
                return True
            else:
                logger.error(
                    f"Failed to send Discord notification: {response.status_code}"
                )
                return False

        except requests.Timeout:
            logger.error("Discord notification timeout")
            return False
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def _create_payload(self, event: NotificationEvent) -> dict:
        """Create Discord webhook payload from event.

        Args:
            event: Notification event

        Returns:
            Discord payload dictionary
        """
        # Color based on level (Discord uses decimal colors)
        colors = {
            NotificationLevel.INFO: 0x0099FF,
            NotificationLevel.SUCCESS: 0x36A64F,
            NotificationLevel.WARNING: 0xFF9900,
            NotificationLevel.ERROR: 0xFF0000,
            NotificationLevel.CRITICAL: 0x8B0000
        }
        color = colors.get(event.level, 0x0099FF)

        # Build embed fields
        fields = []

        if event.model_name:
            fields.append({
                "name": "Model",
                "value": event.model_name,
                "inline": True
            })

        if event.run_id:
            fields.append({
                "name": "Run ID",
                "value": event.run_id,
                "inline": True
            })

        if event.cost is not None:
            fields.append({
                "name": "Cost",
                "value": f"${event.cost:.2f}",
                "inline": True
            })

        if event.epoch is not None:
            fields.append({
                "name": "Epoch",
                "value": str(event.epoch),
                "inline": True
            })

        if event.batch is not None:
            fields.append({
                "name": "Batch",
                "value": str(event.batch),
                "inline": True
            })

        # Add metrics as fields
        for key, value in event.metrics.items():
            # Format numeric values
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)

            fields.append({
                "name": key,
                "value": formatted_value,
                "inline": True
            })

        # Build embed
        embed = {
            "title": event.title,
            "description": event.message,
            "color": color,
            "fields": fields,
            "timestamp": event.timestamp.isoformat(),
            "footer": {
                "text": f"Level: {event.level.upper()}"
            }
        }

        # Add error details if present
        if event.error_details:
            error_text = event.error_details
            # Truncate if too long for Discord
            if len(error_text) > 1024:
                error_text = error_text[:1000] + "..."

            embed["fields"].append({
                "name": "Error Details",
                "value": f"```{error_text}```",
                "inline": False
            })

        # Build payload
        payload = {
            "username": self.username,
            "embeds": [embed]
        }

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


def send_discord_message(
    message: str,
    title: Optional[str] = None,
    webhook_url: Optional[str] = None,
    color: int = 0x0099FF
) -> bool:
    """Convenience function to send a Discord message.

    Args:
        message: Message content
        title: Optional title
        webhook_url: Discord webhook URL (from env if not provided)
        color: Message embed color (decimal)

    Returns:
        True if successful
    """
    webhook = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook:
        logger.error("No Discord webhook URL configured")
        return False

    try:
        embed = {
            "title": title or "AFS Notification",
            "description": message,
            "color": color
        }

        payload = {
            "username": "AFS",
            "embeds": [embed]
        }

        response = requests.post(webhook, json=payload, timeout=10)
        return response.status_code == 204

    except Exception as e:
        logger.error(f"Failed to send Discord message: {e}")
        return False
