# Notification System Quick Start

## 5-Minute Setup

### 1. Copy Configuration Template

```bash
cp .env.example .env
```

### 2. Enable Desktop Notifications (Default on macOS)

No configuration needed! Desktop notifications work out of the box on macOS.

### 3. (Optional) Configure Email Notifications

Edit `.env` and add Gmail or SMTP settings:

```bash
# Gmail (recommended)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-16-char-app-password  # From https://myaccount.google.com/apppasswords
FROM_EMAIL=your-email@gmail.com
TO_EMAILS=recipient@example.com
```

### 4. (Optional) Configure Slack

```bash
# Create webhook at https://api.slack.com/apps/YOUR_APP_ID/incoming-webhooks
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SLACK_CHANNEL=#training-alerts
```

### 5. (Optional) Configure Discord

```bash
# Create webhook in Discord server settings > Integrations > Webhooks
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/YOUR/WEBHOOK/URL
```

## Basic Usage

### In Your Training Code

```python
from afs.notifications import NotificationManager, DesktopNotifier

# Setup
manager = NotificationManager()
manager.register_handler("desktop", DesktopNotifier())

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
    eval_metrics={"accuracy": 0.95}
)
```

### Via Command Line

```bash
# Check status
python scripts/notify_training_complete.py status

# Send completion notification
python scripts/notify_training_complete.py complete \
  --model "MyModel" \
  --run-id "run_001" \
  --duration 3600 \
  --loss 0.1234

# Send error notification
python scripts/notify_training_complete.py error \
  --model "MyModel" \
  --run-id "run_001" \
  --error "Out of memory"
```

## All Event Types

| Category | Events |
|----------|--------|
| **Training** | `TRAINING_STARTED`, `TRAINING_COMPLETED`, `TRAINING_FAILED`, `TRAINING_PAUSED`, `TRAINING_RESUMED` |
| **Checkpoints** | `CHECKPOINT_SAVED`, `CHECKPOINT_LOADING` |
| **Evaluation** | `EVALUATION_STARTED`, `EVALUATION_COMPLETED`, `EVALUATION_FAILED` |
| **Milestones** | `EPOCH_COMPLETED`, `BATCH_PROCESSED`, `LOSS_IMPROVED`, `LOSS_DEGRADED` |
| **Alerts** | `COST_THRESHOLD_EXCEEDED`, `GPU_MEMORY_WARNING`, `GPU_UTILIZATION_LOW`, `DISK_SPACE_WARNING` |
| **Errors** | `ERROR_OCCURRED`, `OUT_OF_MEMORY`, `NAN_DETECTED` |

## Common Recipes

### Training Loop with Notifications

```python
from afs.notifications import NotificationManager, DesktopNotifier, EmailNotifier

manager = NotificationManager()
manager.register_handler("desktop", DesktopNotifier())
manager.register_handler("email", EmailNotifier())

try:
    manager.notify_training_started("MyModel", "run_001", 10, 4)

    for epoch in range(num_epochs):
        train_loss = train_epoch()
        val_loss = validate()

        if val_loss < best_loss:
            manager.notify(
                title="Loss Improved",
                message=f"New best loss: {val_loss:.4f}",
                event_type="loss_improved",
                level="success"
            )

    manager.notify_training_completed(
        "MyModel", "run_001", duration, final_loss,
        eval_metrics=results
    )

except Exception as e:
    manager.notify_training_failed("MyModel", "run_001", str(e))
    raise
```

### Cost Monitoring

```python
cost_thresholds = [5.0, 10.0, 20.0]
current_cost = 0.0
alerted = set()

for epoch in range(num_epochs):
    current_cost += cost_per_epoch

    for threshold in cost_thresholds:
        if current_cost > threshold and threshold not in alerted:
            manager.notify_cost_alert(current_cost, threshold)
            alerted.add(threshold)
```

### GPU Monitoring

```python
import torch

def check_gpu_health(manager):
    memory_used = torch.cuda.memory_allocated()
    memory_total = torch.cuda.get_device_properties(0).total_memory
    memory_percent = memory_used / memory_total

    if memory_percent > 0.85:
        manager.notify(
            title="GPU Memory Warning",
            message=f"GPU memory: {memory_percent:.0%}",
            event_type="gpu_memory_warning",
            level="warning"
        )
```

## Notification Levels

- **info** - Informational (default)
- **success** - Operation succeeded
- **warning** - Needs attention
- **error** - Error occurred
- **critical** - Critical failure

## Troubleshooting

### Desktop notifications not showing?
- Ensure running on macOS
- Check System Preferences → Notifications

### Email not sending?
- Verify SMTP_HOST, SMTP_PORT, credentials in .env
- For Gmail: Use 16-character App Password, not account password
- Check firewall allows SMTP outbound

### Slack/Discord not posting?
- Verify webhook URL is correct
- Check webhook hasn't been deleted/revoked
- Verify channel name in SLACK_CHANNEL

### Run tests
```bash
pytest tests/test_notifications.py -v
```

## Examples

Run the full example:

```bash
python examples/notification_example.py
```

## API Reference

See [NOTIFICATIONS.md](NOTIFICATIONS.md) for complete API documentation.

## What's Next?

1. Configure your preferred notification channels in `.env`
2. Integrate `NotificationManager` into your training pipeline
3. Customize event types and thresholds for your needs
4. Use the CLI tool for ad-hoc notifications during development

For advanced usage, see [NOTIFICATIONS.md](NOTIFICATIONS.md).
