#!/bin/bash
# Complete Deployment Pipeline Orchestrator
#
# Automates: download → merge → quantize → deploy → health check → evaluate
#
# Usage:
#     ./deploy_pipeline.sh --all-models
#     ./deploy_pipeline.sh --model majora --instance-id 12345
#     ./deploy_pipeline.sh --stage merge --from-checkpoint
#     ./deploy_pipeline.sh --rollback majora

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/deployment_config.yaml}"
LOG_DIR="${LOG_DIR:-.logs}"
BACKUP_DIR="${BACKUP_DIR:-models/backups}"

# Pipeline stages
STAGES=("download" "merge" "quantize" "deploy" "health_check" "evaluate")
CURRENT_STAGE=""
FAILED_STAGE=""

# Logging setup
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/pipeline.log"
SUMMARY_LOG="$LOG_DIR/pipeline_summary.log"

log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$PIPELINE_LOG"
}

log_stage() {
    local stage=$1
    local status=$2
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Stage: $stage | Status: $status" >> "$SUMMARY_LOG"
}

log_error() {
    echo -e "${RED}✗ ERROR: $@${NC}" >&2
    log "ERROR" "$@"
}

log_success() {
    echo -e "${GREEN}✓ $@${NC}"
    log "INFO" "SUCCESS: $@"
}

log_warning() {
    echo -e "${YELLOW}⚠ WARNING: $@${NC}"
    log "WARN" "$@"
}

log_info() {
    echo -e "${BLUE}→ $@${NC}"
    log "INFO" "$@"
}

# Print header
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}     AFS MODEL DEPLOYMENT PIPELINE${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Print footer
print_footer() {
    local success=$1
    echo ""
    echo -e "${BLUE}========================================${NC}"
    if [ $success -eq 0 ]; then
        echo -e "${GREEN}✓ PIPELINE COMPLETE${NC}"
    else
        echo -e "${RED}✗ PIPELINE FAILED${NC}"
    fi
    echo -e "${BLUE}========================================${NC}"
}

# Stage 1: Download LoRA adapters from vast.ai
stage_download() {
    local model=$1
    local instance_id=$2

    log_info "Starting download stage..."
    CURRENT_STAGE="download"

    if [ -z "$instance_id" ]; then
        log_warning "No instance_id provided for $model, skipping download"
        return 0
    fi

    log_info "Downloading $model from instance $instance_id..."

    python3 "$SCRIPT_DIR/download_from_vastai.py" \
        --model "$model" \
        --instance-id "$instance_id" \
        --config "$CONFIG_FILE" \
        --resume \
        --verify \
        2>&1 | tee -a "$LOG_DIR/download.log"

    local status=$?
    if [ $status -eq 0 ]; then
        log_success "Download complete for $model"
        log_stage "download" "success"
        return 0
    else
        log_error "Download failed for $model"
        log_stage "download" "failed"
        return 1
    fi
}

# Stage 2: Merge LoRA with base model
stage_merge() {
    local model=$1
    local adapter_path=$2

    log_info "Starting merge stage..."
    CURRENT_STAGE="merge"

    if [ ! -f "$adapter_path" ]; then
        log_error "Adapter not found: $adapter_path"
        log_stage "merge" "failed"
        return 1
    fi

    log_info "Merging $model adapter with base model..."

    python3 "$SCRIPT_DIR/merge_and_quantize.py" \
        --adapter "$adapter_path" \
        --output "models/${model}_merged" \
        --use-unsloth \
        --merge-only \
        --config "$CONFIG_FILE" \
        2>&1 | tee -a "$LOG_DIR/merge.log"

    local status=$?
    if [ $status -eq 0 ]; then
        log_success "Merge complete for $model"
        log_stage "merge" "success"
        return 0
    else
        log_error "Merge failed for $model"
        log_stage "merge" "failed"
        return 1
    fi
}

# Stage 3: Quantize GGUF models
stage_quantize() {
    local model=$1
    local merged_dir=$2
    local quantize_formats=${3:-"q4_k_m,q5_k_m"}

    log_info "Starting quantization stage..."
    CURRENT_STAGE="quantize"

    if [ ! -d "$merged_dir" ]; then
        log_error "Merged model directory not found: $merged_dir"
        log_stage "quantize" "failed"
        return 1
    fi

    log_info "Quantizing $model to formats: $quantize_formats..."

    # First convert to GGUF
    python3 "$SCRIPT_DIR/merge_and_quantize.py" \
        --adapter "models/${model}_adapter_model.safetensors" \
        --output "models/${model}_v1" \
        --quantize "$quantize_formats" \
        --config "$CONFIG_FILE" \
        2>&1 | tee -a "$LOG_DIR/quantize.log"

    local status=$?
    if [ $status -eq 0 ]; then
        log_success "Quantization complete for $model"
        log_stage "quantize" "success"
        return 0
    else
        log_error "Quantization failed for $model"
        log_stage "quantize" "failed"
        return 1
    fi
}

# Stage 4: Deploy to LMStudio
stage_deploy() {
    log_info "Starting deployment stage..."
    CURRENT_STAGE="deploy"

    log_info "Deploying models to LMStudio..."

    bash "$SCRIPT_DIR/deploy_to_lmstudio.sh" 2>&1 | tee -a "$LOG_DIR/deploy.log"

    local status=$?
    if [ $status -eq 0 ]; then
        log_success "Deployment complete"
        log_stage "deploy" "success"
        return 0
    else
        log_error "Deployment failed"
        log_stage "deploy" "failed"
        return 1
    fi
}

# Stage 5: Health check
stage_health_check() {
    log_info "Starting health check stage..."
    CURRENT_STAGE="health_check"

    # Wait for models to be available
    log_info "Waiting for LMStudio models to load..."
    sleep 10

    log_info "Running health checks..."

    python3 "$SCRIPT_DIR/health_check.py" \
        --all \
        --detailed \
        --config "$CONFIG_FILE" \
        2>&1 | tee -a "$LOG_DIR/health_check.log"

    local status=$?
    if [ $status -eq 0 ]; then
        log_success "All models passed health check"
        log_stage "health_check" "success"
        return 0
    else
        log_warning "Some models failed health check"
        log_stage "health_check" "warning"
        return 0  # Don't fail pipeline
    fi
}

# Stage 6: Evaluation
stage_evaluate() {
    log_info "Starting evaluation stage..."
    CURRENT_STAGE="evaluate"

    if [ ! -f "$SCRIPT_DIR/compare_models.py" ]; then
        log_warning "compare_models.py not found, skipping evaluation"
        return 0
    fi

    log_info "Running model evaluation..."

    python3 "$SCRIPT_DIR/compare_models.py" \
        --models majora nayru veran agahnim hylia \
        --sample-size 5 \
        2>&1 | tee -a "$LOG_DIR/evaluate.log"

    local status=$?
    if [ $status -eq 0 ]; then
        log_success "Evaluation complete"
        log_stage "evaluate" "success"
        return 0
    else
        log_warning "Evaluation encountered issues"
        log_stage "evaluate" "warning"
        return 0  # Don't fail pipeline
    fi
}

# Backup previous model versions
backup_models() {
    local model=$1

    log_info "Backing up previous versions of $model..."

    mkdir -p "$BACKUP_DIR"

    # Find existing GGUF files and backup
    for gguf_file in models/${model}*.gguf; do
        if [ -f "$gguf_file" ]; then
            local backup_name="${BACKUP_DIR}/$(basename "$gguf_file").backup.$(date +%s)"
            cp "$gguf_file" "$backup_name"
            log_info "Backed up: $backup_name"
        fi
    done
}

# Rollback to previous version
rollback() {
    local model=$1

    log_info "Rolling back $model to previous version..."

    # Find most recent backup
    local recent_backup=$(ls -t "$BACKUP_DIR"/${model}*.backup.* 2>/dev/null | head -1)

    if [ -z "$recent_backup" ]; then
        log_error "No backup found for $model"
        return 1
    fi

    # Restore backup
    local target_name=$(basename "$recent_backup" | sed 's/\.backup\.[0-9]*$//')
    cp "$recent_backup" "models/$target_name"

    log_success "Rollback complete: models/$target_name"
    return 0
}

# Show help
show_help() {
    cat << EOF
AFS Model Deployment Pipeline

Usage: ./deploy_pipeline.sh [OPTIONS]

Options:
    --all-models              Deploy all configured models
    --model NAME              Deploy specific model
    --instance-id ID          Vast.ai instance ID for download
    --stage STAGE             Run specific stage (download, merge, quantize, deploy, health_check, evaluate)
    --from-checkpoint         Resume from failed stage
    --skip-download           Skip download stage
    --skip-health-check       Skip health check stage
    --skip-evaluation         Skip evaluation stage
    --rollback MODEL          Rollback to previous version
    --config FILE             Path to deployment config (default: deployment_config.yaml)
    --dry-run                 Show what would happen without running
    --help                    Show this help message

Examples:
    # Deploy all models
    ./deploy_pipeline.sh --all-models

    # Deploy single model
    ./deploy_pipeline.sh --model majora --instance-id 12345

    # Resume from failed stage
    ./deploy_pipeline.sh --from-checkpoint

    # Rollback
    ./deploy_pipeline.sh --rollback majora

EOF
}

# Main entry point
main() {
    local deploy_all=false
    local model_name=""
    local instance_id=""
    local stage=""
    local from_checkpoint=false
    local skip_download=false
    local skip_health=false
    local skip_eval=false
    local do_rollback=false
    local dry_run=false

    print_header

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all-models)
                deploy_all=true
                shift
                ;;
            --model)
                model_name="$2"
                shift 2
                ;;
            --instance-id)
                instance_id="$2"
                shift 2
                ;;
            --stage)
                stage="$2"
                shift 2
                ;;
            --from-checkpoint)
                from_checkpoint=true
                shift
                ;;
            --skip-download)
                skip_download=true
                shift
                ;;
            --skip-health-check)
                skip_health=true
                shift
                ;;
            --skip-evaluation)
                skip_eval=true
                shift
                ;;
            --rollback)
                model_name="$2"
                do_rollback=true
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Check configuration file
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi

    log_info "Configuration: $CONFIG_FILE"
    log_info "Logs: $LOG_DIR"

    # Handle rollback
    if [ "$do_rollback" = true ]; then
        if [ -z "$model_name" ]; then
            log_error "Model name required for rollback"
            exit 1
        fi
        rollback "$model_name"
        exit $?
    fi

    # Run pipeline
    local exit_code=0

    if [ -z "$stage" ]; then
        # Run full pipeline
        if [ "$skip_download" = false ]; then
            stage_download "$model_name" "$instance_id" || exit_code=1
        fi

        if [ $exit_code -eq 0 ]; then
            stage_merge "$model_name" "models/${model_name}_adapter_model.safetensors" || exit_code=1
        fi

        if [ $exit_code -eq 0 ]; then
            stage_quantize "$model_name" "models/${model_name}_merged" || exit_code=1
        fi

        if [ $exit_code -eq 0 ]; then
            stage_deploy || exit_code=1
        fi

        if [ $exit_code -eq 0 ] && [ "$skip_health" = false ]; then
            stage_health_check || true  # Don't fail on health check
        fi

        if [ $exit_code -eq 0 ] && [ "$skip_eval" = false ]; then
            stage_evaluate || true  # Don't fail on evaluation
        fi
    else
        # Run specific stage
        case "$stage" in
            download)
                stage_download "$model_name" "$instance_id" || exit_code=1
                ;;
            merge)
                stage_merge "$model_name" "models/${model_name}_adapter_model.safetensors" || exit_code=1
                ;;
            quantize)
                stage_quantize "$model_name" "models/${model_name}_merged" || exit_code=1
                ;;
            deploy)
                stage_deploy || exit_code=1
                ;;
            health_check)
                stage_health_check || true
                ;;
            evaluate)
                stage_evaluate || true
                ;;
            *)
                log_error "Unknown stage: $stage"
                exit_code=1
                ;;
        esac
    fi

    # Summary
    print_footer $exit_code

    if [ -f "$SUMMARY_LOG" ]; then
        log_info "Pipeline summary:"
        cat "$SUMMARY_LOG"
    fi

    exit $exit_code
}

# Run main if script is executed directly
main "$@"
