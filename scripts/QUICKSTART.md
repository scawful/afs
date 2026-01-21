# Deployment Pipeline - Quick Start Guide

## One-Minute Setup

1. **Configure vast.ai instance IDs:**
   ```bash
   # Edit deployment_config.yaml
   # Set instance_id for each model in vast.models section
   ```

2. **Run the pipeline:**
   ```bash
   ./deploy_pipeline.sh --all-models
   ```

3. **Monitor progress:**
   ```bash
   tail -f .logs/pipeline.log
   ```

Done! Models are deployed and ready.

## Common Commands

### Deploy All Models
```bash
./deploy_pipeline.sh --all-models
```

### Deploy Single Model
```bash
./deploy_pipeline.sh \
  --model majora \
  --instance-id 12345678
```

### Check Model Health
```bash
python3 health_check.py --all --detailed
```

### Rollback Previous Version
```bash
./deploy_pipeline.sh --rollback majora
```

### Download Only
```bash
python3 download_from_vastai.py \
  --model majora \
  --instance-id 12345678
```

### Merge & Quantize Only
```bash
python3 merge_and_quantize.py \
  --adapter models/majora_adapter_model.safetensors \
  --output models/majora_v1
```

### Run Specific Stage
```bash
./deploy_pipeline.sh --stage health_check
./deploy_pipeline.sh --stage quantize --model majora
```

## Configuration

Edit `/Users/scawful/src/lab/afs/scripts/deployment_config.yaml`:

**Critical settings:**
- `vast.models[].instance_id` - vast.ai instance IDs
- `deployment.lmstudio.home` - LMStudio path
- `deployment.models[].port` - Port assignments
- `quantization.formats` - Quantization levels

## Status & Logs

| Command | Purpose |
|---------|---------|
| `tail -f .logs/pipeline.log` | Real-time logs |
| `cat .logs/pipeline_summary.log` | Stage summary |
| `python3 health_check.py --all --json` | Detailed status |

## Model Status Codes

| Symbol | Meaning |
|--------|---------|
| ✓ | Healthy - Ready to use |
| ⚠ | Degraded - Working but issues |
| ✗ | Unhealthy - Not operational |

## Default Ports

| Model | Port |
|-------|------|
| majora | 5000 |
| nayru | 5001 |
| veran | 5002 |
| agahnim | 5003 |
| hylia | 5004 |

## Troubleshooting

### Models not responding?
```bash
# Restart LMStudio and wait
sleep 30
python3 health_check.py --all
```

### Out of memory?
```bash
# Check available GPU memory
nvidia-smi

# Use smaller quantization
./deploy_pipeline.sh --model majora --skip-evaluation
```

### Download timeout?
```bash
# Verify vast.ai instance is running
vastai show instance <id>

# Increase timeout in config
```

## Performance Tips

1. **Use Q4_K_M format** - Best speed/quality ratio
2. **Parallel downloads** - Download multiple models simultaneously
3. **Skip evaluation on CI/CD** - Saves time on automated deploys
4. **Use local adapters** - Skip download if already available

## Full Documentation

See `DEPLOYMENT.md` for complete documentation.

## Support

For issues, check:
1. `.logs/` directory for detailed logs
2. `DEPLOYMENT.md` troubleshooting section
3. Model-specific configuration in `deployment_config.yaml`
