"""Tests for notification system."""

import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from afs.notifications import (
    NotificationManager,
    NotificationEvent,
    EventType,
    NotificationLevel,
    DesktopNotifier,
    EmailNotifier,
    SlackNotifier,
    DiscordNotifier,
)


class TestNotificationEvent:
    """Test NotificationEvent dataclass."""

    def test_create_event(self):
        """Test creating a notification event."""
        event = NotificationEvent(
            event_type=EventType.TRAINING_COMPLETED,
            title="Training Complete",
            message="Model training finished successfully"
        )

        assert event.event_type == EventType.TRAINING_COMPLETED
        assert event.title == "Training Complete"
        assert event.level == NotificationLevel.INFO

    def test_event_with_context(self):
        """Test event with additional context."""
        event = NotificationEvent(
            event_type=EventType.TRAINING_COMPLETED,
            title="Training Complete",
            message="Done",
            model_name="test-model",
            run_id="run-123",
            metrics={"loss": 0.123, "accuracy": 0.95}
        )

        assert event.model_name == "test-model"
        assert event.run_id == "run-123"
        assert event.metrics["loss"] == 0.123

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = NotificationEvent(
            event_type=EventType.TRAINING_COMPLETED,
            title="Test",
            message="Message",
            level=NotificationLevel.SUCCESS,
            model_name="test-model"
        )

        data = event.to_dict()

        assert data["event_type"] == "training_completed"
        assert data["level"] == "success"
        assert data["model_name"] == "test-model"
        assert "timestamp" in data

    def test_event_str(self):
        """Test string representation."""
        event = NotificationEvent(
            event_type=EventType.ERROR_OCCURRED,
            title="Error",
            message="Something went wrong",
            level=NotificationLevel.ERROR
        )

        assert "ERROR" in str(event)
        assert "Error" in str(event)


class TestDesktopNotifier:
    """Test macOS desktop notifier."""

    def test_desktop_notifier_macos(self):
        """Test desktop notifier on macOS."""
        with patch("sys.platform", "darwin"):
            notifier = DesktopNotifier()
            assert notifier.is_configured()

    def test_desktop_notifier_non_macos(self):
        """Test desktop notifier on non-macOS."""
        with patch("sys.platform", "linux"):
            notifier = DesktopNotifier()
            assert not notifier.is_configured()

    @patch("subprocess.run")
    @patch("sys.platform", "darwin")
    def test_send_notification(self, mock_run):
        """Test sending desktop notification."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        notifier = DesktopNotifier()
        event = NotificationEvent(
            event_type=EventType.TRAINING_COMPLETED,
            title="Training Complete",
            message="Model trained"
        )

        result = notifier.send(event)

        assert result is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    @patch("sys.platform", "darwin")
    def test_send_notification_failure(self, mock_run):
        """Test failed desktop notification."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")

        notifier = DesktopNotifier()
        event = NotificationEvent(
            event_type=EventType.TRAINING_COMPLETED,
            title="Training Complete",
            message="Model trained"
        )

        result = notifier.send(event)

        assert result is False


class TestEmailNotifier:
    """Test email notifier."""

    def test_email_configured(self):
        """Test email configuration check."""
        with patch.dict(os.environ, {
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USER": "test@gmail.com",
            "SMTP_PASSWORD": "password",
            "FROM_EMAIL": "test@gmail.com",
            "TO_EMAILS": "recipient@example.com"
        }):
            notifier = EmailNotifier()
            assert notifier.is_configured()

    def test_email_not_configured(self):
        """Test unconfigured email."""
        with patch.dict(os.environ, {}, clear=True):
            notifier = EmailNotifier()
            assert not notifier.is_configured()

    @patch("smtplib.SMTP")
    def test_send_email(self, mock_smtp):
        """Test sending email notification."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        with patch.dict(os.environ, {
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "test@gmail.com",
            "SMTP_PASSWORD": "password",
            "FROM_EMAIL": "test@gmail.com",
            "TO_EMAILS": "recipient@example.com"
        }):
            notifier = EmailNotifier()
            event = NotificationEvent(
                event_type=EventType.TRAINING_COMPLETED,
                title="Training Complete",
                message="Model trained"
            )

            result = notifier.send(event)

            assert result is True
            mock_server.send_message.assert_called_once()

    def test_format_text_email(self):
        """Test text email formatting."""
        notifier = EmailNotifier(
            smtp_host="smtp.test.com",
            smtp_user="test@test.com",
            smtp_password="password",
            from_email="test@test.com",
            to_emails=["recipient@test.com"]
        )

        event = NotificationEvent(
            event_type=EventType.TRAINING_COMPLETED,
            title="Training Complete",
            message="Model trained",
            model_name="test-model",
            run_id="run-123",
            metrics={"loss": 0.123}
        )

        text = notifier._format_text(event)

        assert "Training Complete" not in text  # Title is not in body
        assert "Model trained" in text
        assert "test-model" in text
        assert "run-123" in text

    def test_format_html_email(self):
        """Test HTML email formatting."""
        notifier = EmailNotifier(
            smtp_host="smtp.test.com",
            smtp_user="test@test.com",
            smtp_password="password",
            from_email="test@test.com",
            to_emails=["recipient@test.com"]
        )

        event = NotificationEvent(
            event_type=EventType.TRAINING_COMPLETED,
            title="Training Complete",
            message="Model trained",
            model_name="test-model",
            metrics={"loss": 0.123}
        )

        html = notifier._format_html(event)

        assert "<html>" in html
        assert "Training Complete" in html
        assert "test-model" in html
        assert "loss" in html


class TestSlackNotifier:
    """Test Slack notifier."""

    def test_slack_configured(self):
        """Test Slack configuration check."""
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            notifier = SlackNotifier()
            assert notifier.is_configured()

    def test_slack_not_configured(self):
        """Test unconfigured Slack."""
        with patch.dict(os.environ, {}, clear=True):
            notifier = SlackNotifier()
            assert not notifier.is_configured()

    @patch("requests.post")
    def test_send_slack_notification(self, mock_post):
        """Test sending Slack notification."""
        mock_post.return_value = MagicMock(status_code=200)

        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            notifier = SlackNotifier()
            event = NotificationEvent(
                event_type=EventType.TRAINING_COMPLETED,
                title="Training Complete",
                message="Model trained",
                level=NotificationLevel.SUCCESS
            )

            result = notifier.send(event)

            assert result is True
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["json"]["attachments"][0]["color"] == "#36a64f"

    def test_slack_payload_structure(self):
        """Test Slack payload structure."""
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")

        event = NotificationEvent(
            event_type=EventType.TRAINING_COMPLETED,
            title="Training Complete",
            message="Model trained",
            model_name="test-model",
            run_id="run-123",
            metrics={"loss": 0.123, "accuracy": 0.95}
        )

        payload = notifier._create_payload(event)

        assert "attachments" in payload
        assert payload["username"] == "AFS Notifier"
        attachment = payload["attachments"][0]
        assert attachment["title"] == "Training Complete"
        assert attachment["text"] == "Model trained"

        # Check fields
        fields = {f["title"]: f["value"] for f in attachment["fields"]}
        assert "Model" in fields or any("test-model" in str(f) for f in attachment["fields"])


class TestDiscordNotifier:
    """Test Discord notifier."""

    def test_discord_configured(self):
        """Test Discord configuration check."""
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discordapp.com/test"}):
            notifier = DiscordNotifier()
            assert notifier.is_configured()

    def test_discord_not_configured(self):
        """Test unconfigured Discord."""
        with patch.dict(os.environ, {}, clear=True):
            notifier = DiscordNotifier()
            assert not notifier.is_configured()

    @patch("requests.post")
    def test_send_discord_notification(self, mock_post):
        """Test sending Discord notification."""
        mock_post.return_value = MagicMock(status_code=204)  # Discord returns 204

        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discordapp.com/test"}):
            notifier = DiscordNotifier()
            event = NotificationEvent(
                event_type=EventType.TRAINING_COMPLETED,
                title="Training Complete",
                message="Model trained",
                level=NotificationLevel.SUCCESS
            )

            result = notifier.send(event)

            assert result is True
            mock_post.assert_called_once()

    def test_discord_payload_structure(self):
        """Test Discord payload structure."""
        notifier = DiscordNotifier(webhook_url="https://discordapp.com/test")

        event = NotificationEvent(
            event_type=EventType.TRAINING_COMPLETED,
            title="Training Complete",
            message="Model trained",
            model_name="test-model",
            metrics={"accuracy": 0.95}
        )

        payload = notifier._create_payload(event)

        assert "embeds" in payload
        assert payload["username"] == "AFS"
        embed = payload["embeds"][0]
        assert embed["title"] == "Training Complete"
        assert embed["description"] == "Model trained"


class TestNotificationManager:
    """Test notification manager."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = NotificationManager()

        assert len(manager.handlers) == 0
        assert len(manager.enabled_channels) == 0

    def test_register_handler(self):
        """Test registering a handler."""
        manager = NotificationManager()

        notifier = DesktopNotifier()
        with patch("sys.platform", "darwin"):
            manager.register_handler("desktop", notifier)

        assert "desktop" in manager.handlers

    def test_notify_with_all_channels(self):
        """Test notify across multiple channels."""
        manager = NotificationManager()

        # Mock handlers
        mock_desktop = MagicMock()
        mock_desktop.is_configured.return_value = True
        mock_desktop.send.return_value = True

        mock_email = MagicMock()
        mock_email.is_configured.return_value = True
        mock_email.send.return_value = True

        manager.register_handler("desktop", mock_desktop)
        manager.register_handler("email", mock_email)

        result = manager.notify(
            title="Test",
            message="Test message",
            event_type=EventType.TRAINING_COMPLETED
        )

        assert result is True
        assert mock_desktop.send.called
        assert mock_email.send.called

    def test_notify_training_started(self):
        """Test training started notification."""
        manager = NotificationManager()

        mock_handler = MagicMock()
        mock_handler.is_configured.return_value = True
        mock_handler.send.return_value = True

        manager.register_handler("test", mock_handler)

        result = manager.notify_training_started(
            model_name="test-model",
            run_id="run-123",
            epochs=10,
            batch_size=4
        )

        assert result is True
        mock_handler.send.assert_called_once()

    def test_notify_training_completed(self):
        """Test training completed notification."""
        manager = NotificationManager()

        mock_handler = MagicMock()
        mock_handler.is_configured.return_value = True
        mock_handler.send.return_value = True

        manager.register_handler("test", mock_handler)

        result = manager.notify_training_completed(
            model_name="test-model",
            run_id="run-123",
            duration=3600.0,
            final_loss=0.1234,
            eval_metrics={"accuracy": 0.95}
        )

        assert result is True
        mock_handler.send.assert_called_once()

    def test_notify_cost_alert(self):
        """Test cost alert notification."""
        manager = NotificationManager()

        mock_handler = MagicMock()
        mock_handler.is_configured.return_value = True
        mock_handler.send.return_value = True

        manager.register_handler("test", mock_handler)

        result = manager.notify_cost_alert(
            current_cost=12.50,
            threshold=10.00,
            model_name="test-model"
        )

        assert result is True

    def test_get_status(self):
        """Test getting notification system status."""
        manager = NotificationManager()

        mock_handler = MagicMock()
        mock_handler.is_configured.return_value = True
        mock_handler.__class__.__name__ = "MockNotifier"

        manager.register_handler("test", mock_handler)

        status = manager.get_status()

        assert "registered_handlers" in status
        assert "enabled_channels" in status
        assert "channel_status" in status
        assert "test" in status["registered_handlers"]

    def test_no_handlers_registered(self):
        """Test notify with no handlers."""
        manager = NotificationManager()

        result = manager.notify(
            title="Test",
            message="Message",
            event_type=EventType.ERROR_OCCURRED
        )

        assert result is False

    def test_event_type_string_conversion(self):
        """Test converting string to EventType."""
        manager = NotificationManager()

        mock_handler = MagicMock()
        mock_handler.is_configured.return_value = True
        mock_handler.send.return_value = True

        manager.register_handler("test", mock_handler)

        result = manager.notify(
            title="Test",
            message="Message",
            event_type="training_started"  # String instead of enum
        )

        assert result is True


class TestNotificationIntegration:
    """Integration tests for notification system."""

    @patch("requests.post")
    @patch("smtplib.SMTP")
    @patch("subprocess.run")
    def test_multi_channel_notification(self, mock_run, mock_smtp, mock_post):
        """Test sending notification across multiple channels."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        mock_post.return_value = MagicMock(status_code=200)

        manager = NotificationManager()

        with patch("sys.platform", "darwin"):
            manager.register_handler("desktop", DesktopNotifier())

        with patch.dict(os.environ, {
            "SMTP_HOST": "smtp.test.com",
            "SMTP_USER": "test@test.com",
            "SMTP_PASSWORD": "password",
            "FROM_EMAIL": "test@test.com",
            "TO_EMAILS": "recipient@test.com"
        }):
            manager.register_handler("email", EmailNotifier())

        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            manager.register_handler("slack", SlackNotifier())

        event = NotificationEvent(
            event_type=EventType.TRAINING_COMPLETED,
            title="Training Complete",
            message="Model trained successfully",
            model_name="test-model",
            run_id="run-123",
            level=NotificationLevel.SUCCESS,
            metrics={"accuracy": 0.95, "loss": 0.123}
        )

        result = manager.send_event(event)

        assert result is True
        # Desktop notification would be called
        # Email would be sent
        # Slack webhook would be called
