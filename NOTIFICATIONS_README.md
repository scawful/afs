# AFS Notification System

Complete notification system for training events with 4 notification channels.

## What's Included

- **4 Notification Channels**: Desktop (macOS), Email (SMTP), Slack, Discord
- **21 Event Types**: Training, evaluation, milestones, costs, errors, and more
- **CLI Tool**: `scripts/notify_training_complete.py` for ad-hoc notifications
- **31 Tests**: Comprehensive test coverage, all passing
- **Examples**: Runnable example script with realistic scenarios
- **Documentation**: Complete API reference, quick start, and guides

## Quick Start (5 minutes)

### 1. Copy Configuration
```bash
cp .env.example .env
```

### 2. (Optional) Configure Email
Edit `.env` to add Gmail or SMTP:
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=your-email@gmail.com
TO_EMAILS=recipient@example.com
```

### 3. (Optional) Configure Slack/Discord
Add webhook URLs to `.env`:
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/...
```

### 4. Test It
```bash
# Check status
python scripts/notify_training_complete.py status

# Run example
python examples/notification_example.py
```

## Basic Usage

### In Your Code

```python
from afs.notifications import NotificationManager, DesktopNotifier, EmailNotifier

# Setup
manager = NotificationManager()
manager.register_handler("desktop", DesktopNotifier())
manager.register_handler("email", EmailNotifier())

# Notify training start
manager.notify_training_started(
    model_name="MyModel",
    run_id="run_001",
    epochs=10,
    batch_size=4
)

# ... training code ...

# Notify completion
manager.notify_training_completed(
    model_name="MyModel",
    run_id="run_001",
    duration=3600.0,
    final_loss=0.1234,
    eval_metrics={"accuracy": 0.95, "f1": 0.92}
)
```

### Via CLI

```bash
# Training completion
python scripts/notify_training_complete.py complete \
  --model "MyModel" \
  --run-id "run_001" \
  --duration 3600 \
  --loss 0.1234 \
  --metrics "accuracy=0.95,f1=0.92"

# Training error
python scripts/notify_training_complete.py error \
  --model "MyModel" \
  --run-id "run_001" \
  --error "Out of memory error"

# Cost alert
python scripts/notify_training_complete.py cost-alert \
  --cost 12.50 \
  --threshold 10.00

# Evaluation results
python scripts/notify_training_complete.py evaluation \
  --model "MyModel" \
  --run-id "run_001" \
  --metrics "accuracy=0.95,f1=0.92,perplexity=1.23"
```

## Event Types

### Training (5)
- `TRAINING_STARTED` - Training begins
- `TRAINING_COMPLETED` - Training finished
- `TRAINING_FAILED` - Training error
- `TRAINING_PAUSED` - Training paused
- `TRAINING_RESUMED` - Training resumed

### Evaluation (3)
- `EVALUATION_STARTED` - Evaluation begins
- `EVALUATION_COMPLETED` - Evaluation finished
- `EVALUATION_FAILED` - Evaluation error

### Milestones (4)
- `EPOCH_COMPLETED` - Epoch finished
- `BATCH_PROCESSED` - Batch processed
- `LOSS_IMPROVED` - Loss metric improved
- `LOSS_DEGRADED` - Loss metric worse

### Alerts (4)
- `COST_THRESHOLD_EXCEEDED` - Cost limit exceeded
- `GPU_MEMORY_WARNING` - GPU memory high
- `GPU_UTILIZATION_LOW` - GPU underutilized
- `DISK_SPACE_WARNING` - Disk space low

### Other (5)
- `CHECKPOINT_SAVED` - Checkpoint saved
- `CHECKPOINT_LOADING` - Loading checkpoint
- `ERROR_OCCURRED` - Generic error
- `OUT_OF_MEMORY` - OOM error
- `NAN_DETECTED` - NaN detected

## Notification Channels

### Desktop (macOS)
- Works out of the box
- Native OS notifications
- Optional sound alert
- No configuration needed

### Email (SMTP)
- Gmail, corporate mail, self-hosted
- HTML and plain text
- Rich formatting with metrics
- Requires SMTP configuration

### Slack
- Channel-based posting
- Color-coded messages
- Rich field formatting
- Requires webhook URL

### Discord
- Embed-based messages
- Color-coded by severity
- Clean formatting
- Requires webhook URL

## Files & Structure

```
src/afs/notifications/          Core module (6 files)
├── __init__.py
├── base.py                      Main classes
├── desktop.py                   macOS support
├── email.py                     SMTP support
├── slack.py                     Slack webhooks
└── discord.py                   Discord webhooks

scripts/
└── notify_training_complete.py  CLI tool

tests/
└── test_notifications.py        31 tests, all passing

docs/
├── NOTIFICATIONS.md             Complete API reference
├── NOTIFICATION_QUICKSTART.md   Quick start guide
└── IMPLEMENTATION_SUMMARY.md    Full technical details

examples/
└── notification_example.py      Runnable examples

.env.example                      Configuration template
NOTIFICATIONS_SETUP.md           Setup overview
NOTIFICATIONS_README.md          This file
```

## Documentation

- **NOTIFICATIONS.md** - Complete API reference with all details
- **NOTIFICATION_QUICKSTART.md** - 5-minute setup and recipes
- **IMPLEMENTATION_SUMMARY.md** - Technical implementation details
- **NOTIFICATIONS_SETUP.md** - Feature overview and architecture

## Testing

```bash
# Run all tests
pytest tests/test_notifications.py -v

# Run specific test class
pytest tests/test_notifications.py::TestEmailNotifier -v

# Run with coverage
pytest tests/test_notifications.py --cov=afs.notifications

# Quick status
python scripts/notify_training_complete.py status
```

## Common Recipes

### Training Loop with Notifications

```python
from afs.notifications import NotificationManager, DesktopNotifier, EmailNotifier

manager = NotificationManager()
manager.register_handler("desktop", DesktopNotifier())
manager.register_handler("email", EmailNotifier())

try:
    manager.notify_training_started(model, run_id, epochs, batch_size)

    for epoch in range(num_epochs):
        train_loss = train_epoch(epoch)

        if epoch % 5 == 0:
            manager.notify(
                title=f"Epoch {epoch} completed",
                message=f"Loss: {train_loss:.4f}",
                event_type="epoch_completed"
            )

    manager.notify_training_completed(
        model, run_id, duration, final_loss, eval_metrics
    )

except Exception as e:
    manager.notify_training_failed(model, run_id, str(e))
    raise
```

### Cost Monitoring

```python
thresholds = [5.0, 10.0, 20.0]
alerted = set()

for epoch in range(num_epochs):
    current_cost += cost_per_epoch

    for threshold in thresholds:
        if current_cost > threshold and threshold not in alerted:
            manager.notify_cost_alert(current_cost, threshold)
            alerted.add(threshold)
```

### GPU Monitoring

```python
import torch

memory_used = torch.cuda.memory_allocated()
memory_total = torch.cuda.get_device_properties(0).total_memory
memory_percent = memory_used / memory_total

if memory_percent > 0.85:
    manager.notify(
        title="GPU Memory Warning",
        message=f"GPU memory usage: {memory_percent:.0%}",
        event_type="gpu_memory_warning",
        level="warning"
    )
```

## API Overview

### NotificationManager

```python
manager = NotificationManager()

# Register handlers
manager.register_handler(name, handler)

# High-level methods
manager.notify_training_started(model_name, run_id, epochs, batch_size)
manager.notify_training_completed(model_name, run_id, duration, loss, eval_metrics)
manager.notify_training_failed(model_name, run_id, error)
manager.notify_cost_alert(current_cost, threshold, model_name)
manager.notify_evaluation_completed(model_name, run_id, metrics)

# Generic notification
manager.notify(title, message, event_type, level, **kwargs)

# Event-based
manager.send_event(notification_event)

# Status
manager.get_status()
```

## Configuration Options

See `.env.example` for full configuration. Key options:

```bash
# Desktop
DESKTOP_NOTIFICATIONS_ENABLED=true
DESKTOP_NOTIFICATIONS_SOUND=true

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=app-password
FROM_EMAIL=your-email@gmail.com
TO_EMAILS=recipient1@example.com,recipient2@example.com

# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_CHANNEL=#training-alerts

# Discord
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/...

# Cost thresholds
COST_ALERT_THRESHOLD_1=5.00
COST_ALERT_THRESHOLD_2=10.00
COST_ALERT_THRESHOLD_3=20.00
```

## Troubleshooting

### Email not sending?
1. Check SMTP credentials in `.env`
2. For Gmail: Use 16-character App Password, not account password
3. Verify port 587 is open (for TLS)
4. Check recipients list is comma-separated

### Slack/Discord not posting?
1. Verify webhook URL is correct and not expired
2. Check channel name is set (for Slack)
3. Ensure webhook hasn't been revoked in settings

### Desktop notifications not showing?
1. Ensure running on macOS
2. Check System Preferences → Notifications settings
3. Allow notifications for Terminal or your IDE

### Tests failing?
1. Run: `pytest tests/test_notifications.py -v`
2. Check Python 3.9+ is installed
3. Verify no .env file with invalid settings

## Support

1. **Quick Reference**: Read this file
2. **Setup Guide**: See `docs/NOTIFICATION_QUICKSTART.md`
3. **Full API**: See `docs/NOTIFICATIONS.md`
4. **Examples**: Run `python examples/notification_example.py`
5. **Tests**: Run `pytest tests/test_notifications.py -v`

## Next Steps

1. ✓ Read this README
2. ✓ Copy `.env.example` to `.env`
3. ✓ Configure your preferred channels
4. ✓ Run example: `python examples/notification_example.py`
5. ✓ Integrate into training code
6. ✓ Set up cost/GPU monitoring
7. ✓ Test in development
8. ✓ Deploy to training pipeline

## Features at a Glance

| Feature | Status | Notes |
|---------|--------|-------|
| Desktop (macOS) | ✓ Ready | Works out of box |
| Email (SMTP) | ✓ Ready | Requires config |
| Slack | ✓ Ready | Requires webhook |
| Discord | ✓ Ready | Requires webhook |
| 21 Event Types | ✓ Complete | Training, eval, alerts |
| 5 Severity Levels | ✓ Complete | info, success, warning, error, critical |
| CLI Tool | ✓ Ready | 7 commands |
| Tests | ✓ Complete | 31 tests, all passing |
| Documentation | ✓ Complete | 4 guides + examples |

## Summary

A production-ready notification system with:
- Multiple notification channels (4)
- Comprehensive event types (21)
- Full test coverage (31 tests)
- Complete documentation (4 guides)
- CLI tool for ad-hoc notifications
- Easy integration with training code
- Environment-based configuration
- Proper error handling and logging

Ready to use for training event monitoring!
