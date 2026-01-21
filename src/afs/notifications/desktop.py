"""macOS desktop notification handler using osascript."""

from __future__ import annotations

import subprocess
import sys

from afs.logging_config import get_logger

from .base import NotificationEvent, NotificationHandler

logger = get_logger(__name__)


class DesktopNotifier(NotificationHandler):
    """Send notifications to macOS desktop using osascript."""

    def __init__(self, enable_sound: bool = True, app_name: str = "AFS"):
        """Initialize desktop notifier.

        Args:
            enable_sound: Play notification sound
            app_name: Application name shown in notification
        """
        self.enable_sound = enable_sound
        self.app_name = app_name
        self._check_macos()

    def _check_macos(self) -> None:
        """Check if running on macOS."""
        if sys.platform != "darwin":
            logger.warning("Desktop notifications only supported on macOS")

    def is_configured(self) -> bool:
        """Check if running on macOS."""
        return sys.platform == "darwin"

    def send(self, event: NotificationEvent) -> bool:
        """Send desktop notification via osascript.

        Args:
            event: Notification event

        Returns:
            True if successful, False otherwise
        """
        if not self.is_configured():
            return False

        try:
            # Build AppleScript
            script = self._build_applescript(event)

            # Execute via osascript
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                logger.debug(f"Desktop notification sent: {event.title}")
                return True
            else:
                logger.error(
                    f"Failed to send desktop notification: {result.stderr}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error("Desktop notification timeout")
            return False
        except Exception as e:
            logger.error(f"Failed to send desktop notification: {e}")
            return False

    def _build_applescript(self, event: NotificationEvent) -> str:
        """Build AppleScript for notification.

        Args:
            event: Notification event

        Returns:
            AppleScript string
        """
        # Escape quotes for AppleScript
        title = event.title.replace('"', '\\"')
        message = event.message.replace('"', '\\"')

        # Build notification with sound if enabled
        sound_part = 'sound name "Glass"' if self.enable_sound else ""

        # Map notification level to icon
        subtitle = f"[{event.level.upper()}]"

        script = (
            f'display notification "{message}" '
            f'with title "{title}" '
            f'subtitle "{subtitle}" '
        )

        if self.enable_sound:
            script += 'sound name "Glass"'

        return script

    def notify_simple(
        self,
        title: str,
        message: str,
        sound: bool | None = None
    ) -> bool:
        """Send a simple notification without event object.

        Args:
            title: Notification title
            message: Notification message
            sound: Override enable_sound setting

        Returns:
            True if successful
        """
        use_sound = self.enable_sound if sound is None else sound

        try:
            title_escaped = title.replace('"', '\\"')
            message_escaped = message.replace('"', '\\"')

            script = (
                f'display notification "{message_escaped}" '
                f'with title "{title_escaped}"'
            )

            if use_sound:
                script += ' sound name "Glass"'

            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )

            return result.returncode == 0

        except Exception as e:
            logger.error(f"Failed to send simple notification: {e}")
            return False


def send_notification(
    title: str,
    message: str,
    sound: bool = True
) -> bool:
    """Convenience function to send a desktop notification.

    Args:
        title: Notification title
        message: Notification message
        sound: Play notification sound

    Returns:
        True if successful
    """
    notifier = DesktopNotifier(enable_sound=sound)
    return notifier.notify_simple(title, message)
