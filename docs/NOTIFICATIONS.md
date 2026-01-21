# AFS Notification System

Comprehensive notification system for training events with support for multiple notification channels.

## Features

- **Desktop Notifications** (macOS) - Native OS notifications
- **Email Notifications** - SMTP-based detailed reports with HTML formatting
- **Slack Integration** - Rich messages with metrics and formatting
- **Discord Integration** - Embed-based messages with color coding
- **Event-driven Architecture** - Flexible event types and severity levels
- **Configurable Thresholds** - Cost alerts, GPU monitoring, and more
- **Batch Notifications** - Send multiple notifications efficiently

## Quick Start

### 1. Installation

```bash
# Ensure requests library is installed (for Slack/Discord)
pip install requests python-dotenv
```

### 2. Configuration

Copy `.env.example` to `.env` and fill in your settings:

```bash
cp .env.example .env
```

### 3. Basic Usage

```python
from afs.notifications import NotificationManager, DesktopNotifier

# Create manager
manager = NotificationManager()

# Register handlers
manager.register_handler("desktop", DesktopNotifier())

# Send notification
manager.notify_training_completed(
    model_name="Qwen2.5-Coder-7B",
    run_id="training_123",
    duration=3600.0,
    final_loss=0.1234
)
```

## Notification Channels

### Desktop Notifications (macOS)

Native macOS notifications using osascript. Requires macOS.

```python
from afs.notifications import DesktopNotifier

notifier = DesktopNotifier(
    enable_sound=True,      # Play notification sound
    app_name="AFS"          # App name shown in notification
)

# Send simple notification
notifier.notify_simple(
    title="Training Complete",
    message="Model trained successfully"
)
```

**Configuration (.env):**
```
DESKTOP_NOTIFICATIONS_ENABLED=true
DESKTOP_NOTIFICATIONS_SOUND=true
```

### Email Notifications

SMTP-based email with HTML formatting. Supports Gmail, corporate mail servers, etc.

```python
from afs.notifications import EmailNotifier

notifier = EmailNotifier(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    smtp_user="your-email@gmail.com",
    smtp_password="your-app-password",
    from_email="your-email@gmail.com",
    to_emails=["recipient@example.com"]
)

event = NotificationEvent(
    event_type=EventType.TRAINING_COMPLETED,
    title="Training Complete",
    message="Model trained",
    model_name="test-model",
    metrics={"accuracy": 0.95}
)

notifier.send(event)
```

**Configuration (.env):**
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=your-email@gmail.com
TO_EMAILS=recipient1@example.com,recipient2@example.com
```

**Gmail Setup:**
1. Enable 2-Factor Authentication on Google Account
2. Create App Password: https://myaccount.google.com/apppasswords
3. Use 16-character App Password as SMTP_PASSWORD
4. Gmail SMTP: smtp.gmail.com:587 (with STARTTLS)

### Slack Integration

Send messages to Slack with rich formatting.

```python
from afs.notifications import SlackNotifier

notifier = SlackNotifier(
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    channel="#training-alerts"
)

manager.register_handler("slack", notifier)
```

**Configuration (.env):**
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SLACK_CHANNEL=#training-alerts
```

**Slack Setup:**
1. Go to https://api.slack.com/apps
2. Create New App → From scratch
3. Name your app and select workspace
4. Go to "Incoming Webhooks" and enable
5. Create New Webhook to Channel
6. Copy webhook URL to .env

### Discord Integration

Send messages to Discord with embed formatting.

```python
from afs.notifications import DiscordNotifier

notifier = DiscordNotifier(
    webhook_url="https://discordapp.com/api/webhooks/YOUR/WEBHOOK/URL"
)

manager.register_handler("discord", notifier)
```

**Configuration (.env):**
```
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/YOUR/WEBHOOK/URL
```

**Discord Setup:**
1. Open Discord server settings
2. Go to Integrations → Webhooks
3. Create New Webhook
4. Select target channel
5. Copy webhook URL to .env

## Event Types

The system supports the following event types:

### Training Events
- `TRAINING_STARTED` - Training begins
- `TRAINING_COMPLETED` - Training finished successfully
- `TRAINING_FAILED` - Training encountered an error
- `TRAINING_PAUSED` - Training paused
- `TRAINING_RESUMED` - Training resumed

### Checkpoint Events
- `CHECKPOINT_SAVED` - Model checkpoint saved
- `CHECKPOINT_LOADING` - Loading checkpoint

### Evaluation Events
- `EVALUATION_STARTED` - Evaluation begins
- `EVALUATION_COMPLETED` - Evaluation finished
- `EVALUATION_FAILED` - Evaluation error

### Milestone Events
- `EPOCH_COMPLETED` - Epoch finished
- `BATCH_PROCESSED` - Batch processed
- `LOSS_IMPROVED` - Loss value improved
- `LOSS_DEGRADED` - Loss value got worse

### Cost & Resource Events
- `COST_THRESHOLD_EXCEEDED` - Cost threshold alert
- `GPU_MEMORY_WARNING` - GPU memory usage high
- `GPU_UTILIZATION_LOW` - GPU underutilized
- `DISK_SPACE_WARNING` - Disk space running low

### Error Events
- `ERROR_OCCURRED` - Generic error
- `OUT_OF_MEMORY` - OOM error
- `NAN_DETECTED` - NaN values detected

## Notification Levels

Each notification has a severity level:

- `INFO` - Informational message
- `SUCCESS` - Successful operation
- `WARNING` - Warning, requires attention
- `ERROR` - Error occurred
- `CRITICAL` - Critical failure

## Usage Examples

### Notify Training Start

```python
manager.notify_training_started(
    model_name="Qwen2.5-Coder-7B",
    run_id="training_001",
    epochs=10,
    batch_size=4
)
```

### Notify Training Completion

```python
manager.notify_training_completed(
    model_name="Qwen2.5-Coder-7B",
    run_id="training_001",
    duration=3600.0,
    final_loss=0.1234,
    eval_metrics={
        "accuracy": 0.95,
        "f1_score": 0.92,
        "perplexity": 1.23
    }
)
```

### Cost Threshold Alert

```python
manager.notify_cost_alert(
    current_cost=12.50,
    threshold=10.00,
    model_name="Qwen2.5-Coder-7B"
)
```

### Evaluation Results

```python
manager.notify_evaluation_completed(
    model_name="Qwen2.5-Coder-7B",
    run_id="training_001",
    metrics={
        "accuracy": 0.9543,
        "f1": 0.9201,
        "perplexity": 1.234
    }
)
```

### Custom Notification

```python
from afs.notifications import EventType, NotificationLevel

manager.notify(
    title="Custom Event",
    message="Something happened",
    event_type=EventType.LOSS_IMPROVED,
    level=NotificationLevel.INFO,
    model_name="my-model",
    run_id="run-123",
    metrics={"new_loss": 0.08}
)
```

## CLI Tool

Send notifications from command line:

```bash
# Check notification system status
python scripts/notify_training_complete.py status

# Send training completion
python scripts/notify_training_complete.py complete \
  --model "Qwen2.5-Coder-7B" \
  --run-id "training_123" \
  --duration 3600 \
  --loss 0.1234 \
  --metrics "accuracy=0.95,f1=0.92"

# Send error notification
python scripts/notify_training_complete.py error \
  --model "Qwen2.5-Coder-7B" \
  --run-id "training_123" \
  --error "Out of memory error"

# Send cost alert
python scripts/notify_training_complete.py cost-alert \
  --cost 12.50 \
  --threshold 10.00

# Send evaluation results
python scripts/notify_training_complete.py evaluation \
  --model "Qwen2.5-Coder-7B" \
  --run-id "training_123" \
  --metrics "accuracy=0.95,f1=0.92,perplexity=1.23"

# Send custom notification
python scripts/notify_training_complete.py custom \
  --title "Custom Event" \
  --message "Something important" \
  --level warning \
  --model "my-model"
```

## Integration with Training Code

### In Training Loop

```python
from afs.training import Trainer
from afs.notifications import NotificationManager, DesktopNotifier, EmailNotifier

# Setup notifications
manager = NotificationManager()
manager.register_handler("desktop", DesktopNotifier())
manager.register_handler("email", EmailNotifier())

# Notify training start
manager.notify_training_started(
    model_name="my-model",
    run_id="training_001",
    epochs=10,
    batch_size=4
)

# In training loop
try:
    trainer = Trainer(config)
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(epoch)

        # Check for cost alerts
        if current_cost > cost_threshold:
            manager.notify_cost_alert(
                current_cost=current_cost,
                threshold=cost_threshold
            )

    # Notify completion
    manager.notify_training_completed(
        model_name="my-model",
        run_id="training_001",
        duration=elapsed_time,
        final_loss=final_loss,
        eval_metrics=eval_metrics
    )

except Exception as e:
    manager.notify_training_failed(
        model_name="my-model",
        run_id="training_001",
        error=str(e)
    )
```

### Cost Monitoring

```python
def check_cost_thresholds(current_cost, model_name, manager):
    """Check if cost exceeded thresholds."""
    thresholds = [5.0, 10.0, 20.0]

    for threshold in thresholds:
        if current_cost > threshold and not previously_alerted(threshold):
            manager.notify_cost_alert(
                current_cost=current_cost,
                threshold=threshold,
                model_name=model_name
            )
```

### GPU Monitoring

```python
def monitor_gpu(manager):
    """Monitor GPU metrics and send alerts."""
    import torch

    memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory

    if memory_used > 0.85:
        manager.notify(
            title="GPU Memory Warning",
            message=f"GPU memory usage: {memory_used*100:.1f}%",
            event_type=EventType.GPU_MEMORY_WARNING,
            level=NotificationLevel.WARNING
        )
```

## Advanced Configuration

### Batch Notifications

```python
from afs.notifications import EmailNotifier

notifier = EmailNotifier()

events = [event1, event2, event3]
sent_count = notifier.send_batch(events)
print(f"Sent {sent_count} notifications")
```

### Conditional Notifications

```python
# Only notify on certain events
if event.level in [NotificationLevel.ERROR, NotificationLevel.CRITICAL]:
    manager.send_event(event)

# Or notify only specific channels
if should_email_admin:
    email_handler = manager.handlers.get("email")
    if email_handler:
        email_handler.send(event)
```

### Notification Context

```python
# Add contextual information
event = NotificationEvent(
    event_type=EventType.TRAINING_COMPLETED,
    title="Training Complete",
    message="Model trained",
    model_name="my-model",
    run_id="run_123",
    epoch=10,
    batch=500,
    cost=15.25,
    metrics={"accuracy": 0.95, "loss": 0.12},
    tags=["production", "coder-model"]
)
```

## Troubleshooting

### Gmail SMTP Issues

```
Error: "SMTPAuthenticationError: (535, b'5.7.8 Username and password not accepted')"
```

Solution: Use App Password, not Gmail password. Generate at https://myaccount.google.com/apppasswords

### Slack/Discord Webhook Errors

```
Error: "404 Not Found"
```

Solution: Check webhook URL is correct and webhook hasn't been deleted.

### Desktop Notifications Not Working

```
Solution: Ensure running on macOS and osascript is available
```

### Email Not Sending

Check:
1. SMTP credentials are correct
2. Port is correct (usually 587 for TLS, 465 for SSL)
3. Firewall allows outbound SMTP connections
4. TO_EMAILS is comma-separated list

## Testing

Run tests:

```bash
pytest tests/test_notifications.py -v

# Run specific test
pytest tests/test_notifications.py::TestEmailNotifier -v

# With coverage
pytest tests/test_notifications.py --cov=afs.notifications
```

## Performance Considerations

- **Desktop notifications**: Instant, no network overhead
- **Email**: Takes ~1-2 seconds per message
- **Slack/Discord**: Takes ~500ms per message (depends on network)
- **Batch emails**: Use `send_batch()` to send multiple emails in one SMTP connection

## Security Notes

- Never commit `.env` file with credentials
- Use environment variables or secure credential storage
- SMTP passwords should be App Passwords, not account passwords
- Slack/Discord webhooks should be restricted to specific channels
- Consider rotating credentials regularly

## Future Enhancements

- SMS notifications via Twilio
- PagerDuty integration for critical alerts
- Webhook system for custom integrations
- Notification filtering and aggregation
- Real-time dashboard with notification history
- Mobile app push notifications
