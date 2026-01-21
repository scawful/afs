"""Email notification handler using SMTP."""

from __future__ import annotations

import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from afs.logging_config import get_logger

from .base import NotificationEvent, NotificationHandler

logger = get_logger(__name__)


class EmailNotifier(NotificationHandler):
    """Send notifications via email using SMTP."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None,
        to_emails: Optional[list[str]] = None,
        use_tls: bool = True
    ):
        """Initialize email notifier.

        Args:
            smtp_host: SMTP server hostname (from env: SMTP_HOST)
            smtp_port: SMTP server port (from env: SMTP_PORT, default 587)
            smtp_user: SMTP username (from env: SMTP_USER)
            smtp_password: SMTP password (from env: SMTP_PASSWORD)
            from_email: From email address (from env: FROM_EMAIL)
            to_emails: List of recipient emails (from env: TO_EMAILS, comma-separated)
            use_tls: Use TLS for connection
        """
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("FROM_EMAIL")
        self.use_tls = use_tls

        # Parse recipient emails
        to_emails_str = os.getenv("TO_EMAILS", "")
        self.to_emails = to_emails or (
            [e.strip() for e in to_emails_str.split(",") if e.strip()]
        )

    def is_configured(self) -> bool:
        """Check if email notifier is properly configured."""
        required = [
            self.smtp_host,
            self.smtp_user,
            self.smtp_password,
            self.from_email,
            self.to_emails
        ]
        return all(required)

    def send(self, event: NotificationEvent) -> bool:
        """Send email notification.

        Args:
            event: Notification event

        Returns:
            True if successful, False otherwise
        """
        if not self.is_configured():
            logger.warning("Email notifier not configured")
            return False

        try:
            # Create message
            msg = self._create_message(event)

            # Connect and send
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email notification sent: {event.title}")
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP authentication failed")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _create_message(self, event: NotificationEvent) -> MIMEMultipart:
        """Create email message from event.

        Args:
            event: Notification event

        Returns:
            MIMEMultipart message
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = event.title
        msg["From"] = self.from_email
        msg["To"] = ", ".join(self.to_emails)

        # Plain text version
        text_content = self._format_text(event)
        msg.attach(MIMEText(text_content, "plain"))

        # HTML version
        html_content = self._format_html(event)
        msg.attach(MIMEText(html_content, "html"))

        return msg

    def _format_text(self, event: NotificationEvent) -> str:
        """Format event as plain text email.

        Args:
            event: Notification event

        Returns:
            Plain text email body
        """
        lines = [
            f"Level: {event.level.upper()}",
            f"Time: {event.timestamp.isoformat()}",
            "",
            event.message,
            "",
        ]

        # Add context
        context = []
        if event.model_name:
            context.append(f"Model: {event.model_name}")
        if event.run_id:
            context.append(f"Run ID: {event.run_id}")
        if event.cost is not None:
            context.append(f"Cost: ${event.cost:.2f}")
        if event.epoch is not None:
            context.append(f"Epoch: {event.epoch}")
        if event.batch is not None:
            context.append(f"Batch: {event.batch}")

        if context:
            lines.extend(["Context:", *context, ""])

        # Add metrics
        if event.metrics:
            lines.append("Metrics:")
            for key, value in event.metrics.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        # Add error details
        if event.error_details:
            lines.extend(["Error Details:", event.error_details])

        return "\n".join(lines)

    def _format_html(self, event: NotificationEvent) -> str:
        """Format event as HTML email.

        Args:
            event: Notification event

        Returns:
            HTML email body
        """
        # Color based on level
        level_colors = {
            "info": "#0066CC",
            "success": "#228B22",
            "warning": "#FF8C00",
            "error": "#DC143C",
            "critical": "#8B0000"
        }
        color = level_colors.get(event.level.value, "#0066CC")

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .container {{ max-width: 600px; margin: 0 auto; }}
                .header {{
                    background-color: {color};
                    color: white;
                    padding: 20px;
                    border-radius: 5px 5px 0 0;
                }}
                .level-badge {{
                    background-color: white;
                    color: {color};
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-weight: bold;
                    display: inline-block;
                    margin-left: 10px;
                }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .section {{ margin-bottom: 15px; }}
                .section-title {{ font-weight: bold; margin-bottom: 5px; }}
                .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }}
                .metrics-table td {{
                    padding: 8px;
                    border: 1px solid #ddd;
                }}
                .metrics-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .error-box {{
                    background-color: #ffe6e6;
                    border-left: 4px solid #dc143c;
                    padding: 10px;
                    margin-top: 10px;
                }}
                .timestamp {{ color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2 style="margin: 0;">{event.title}</h2>
                    <span class="level-badge">{event.level.upper()}</span>
                </div>
                <div class="content">
                    <div class="section">
                        <p>{event.message.replace(chr(10), '<br>')}</p>
                    </div>
        """

        # Add context section
        context_items = []
        if event.model_name:
            context_items.append(("Model", event.model_name))
        if event.run_id:
            context_items.append(("Run ID", event.run_id))
        if event.cost is not None:
            context_items.append(("Cost", f"${event.cost:.2f}"))
        if event.epoch is not None:
            context_items.append(("Epoch", str(event.epoch)))
        if event.batch is not None:
            context_items.append(("Batch", str(event.batch)))

        if context_items:
            html += '<div class="section"><div class="section-title">Context:</div>'
            for key, value in context_items:
                html += f'<div>{key}: <strong>{value}</strong></div>'
            html += '</div>'

        # Add metrics table
        if event.metrics:
            html += '<div class="section"><div class="section-title">Metrics:</div>'
            html += '<table class="metrics-table">'
            for key, value in event.metrics.items():
                html += f'<tr><td>{key}</td><td><strong>{value}</strong></td></tr>'
            html += '</table></div>'

        # Add error details
        if event.error_details:
            html += f"""
                    <div class="error-box">
                        <strong>Error Details:</strong><br>
                        <pre style="margin: 10px 0 0 0;">{event.error_details}</pre>
                    </div>
            """

        html += f"""
                    <div style="margin-top: 20px; padding-top: 10px; border-top: 1px solid #ddd;">
                        <p class="timestamp">Sent: {event.timestamp.isoformat()}</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def send_batch(self, events: list[NotificationEvent]) -> int:
        """Send multiple notifications in a batch.

        Args:
            events: List of notification events

        Returns:
            Number of successfully sent emails
        """
        count = 0
        for event in events:
            if self.send(event):
                count += 1
        return count
