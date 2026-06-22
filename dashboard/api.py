#!/usr/bin/env python3
"""
AFS Training Dashboard API
Real-time monitoring backend for training progress, costs, and performance.
"""

import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EVALUATIONS_DIR = PROJECT_ROOT / "evaluations"
RESULTS_DIR = EVALUATIONS_DIR / "results"
LIVE_STATUS_CACHE_TTL = 5.0
_LIVE_STATUS_CACHE: dict[str, Any] = {"expires_at": 0.0, "payload": None}

# Model configuration - 5 core models
MODELS_CONFIG = {
    "majora": {
        "name": "Majora v1",
        "description": "Oracle of Secrets expert",
        "gpu_hours": 4.0,
        "cost_per_hour": 0.24,
        "status": "completed",
        "progress": 100,
    },
    "veran": {
        "name": "Veran v5",
        "description": "Advanced verification",
        "gpu_hours": 3.5,
        "cost_per_hour": 0.24,
        "status": "pending",
        "progress": 0,
    },
    "din": {
        "name": "Din v4",
        "description": "Creative dialogue generation",
        "gpu_hours": 3.0,
        "cost_per_hour": 0.24,
        "status": "pending",
        "progress": 0,
    },
    "nayru": {
        "name": "Nayru v7",
        "description": "Assembly & architecture",
        "gpu_hours": 3.5,
        "cost_per_hour": 0.24,
        "status": "pending",
        "progress": 0,
    },
    "farore": {
        "name": "Farore v6",
        "description": "Task planning & decomposition",
        "gpu_hours": 3.0,
        "cost_per_hour": 0.24,
        "status": "pending",
        "progress": 0,
    },
}

# Simulated training state
TRAINING_STATE = {
    "session_start": datetime.now() - timedelta(hours=1),
    "total_cost": 0.0,
    "models_completed": 1,
    "models_in_progress": 0,
    "models_pending": 4,
}


def slugify_key(text: str) -> str:
    chars = []
    last_dash = False
    for char in text.lower():
        if char.isalnum():
            chars.append(char)
            last_dash = False
        elif not last_dash:
            chars.append("-")
            last_dash = True
    return "".join(chars).strip("-") or "run"


def resolve_live_status_script() -> Path:
    configured = os.environ.get("AFS_DASHBOARD_LIVE_STATUS_SCRIPT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT.parents[1] / "training" / "scripts" / "live_training_status.py"


def load_live_training_status(force: bool = False) -> dict[str, Any] | None:
    """Load shared live training status emitted by the training repo."""
    now = time.time()
    if not force and now < float(_LIVE_STATUS_CACHE["expires_at"]):
        cached = _LIVE_STATUS_CACHE.get("payload")
        if isinstance(cached, dict):
            return cached

    live_status_script = resolve_live_status_script()
    if not live_status_script.exists():
        return None

    try:
        proc = subprocess.run(
            ["python3", str(live_status_script)],
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        app.logger.error("Error running %s: %s", live_status_script, exc)
        return None

    if proc.returncode != 0:
        app.logger.error(
            "live_training_status.py failed: %s",
            proc.stderr.strip() or proc.stdout.strip() or proc.returncode,
        )
        return None

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        app.logger.error("Invalid training status JSON: %s", exc)
        return None

    if not isinstance(payload, dict):
        return None

    _LIVE_STATUS_CACHE["payload"] = payload
    _LIVE_STATUS_CACHE["expires_at"] = now + LIVE_STATUS_CACHE_TTL
    return payload


def build_live_model_status(run: dict[str, Any]) -> dict[str, Any]:
    """Map a shared live run into the dashboard model shape."""
    hourly = run.get("hourly_usd")
    elapsed = run.get("elapsed_hours")
    total = run.get("accumulated_cost_usd")
    key = str(run.get("source_id") or slugify_key(str(run.get("name") or "run")))

    return {
        "key": key,
        "name": str(run.get("name") or key),
        "description": " · ".join(
            part for part in [str(run.get("backend") or ""), str(run.get("host") or "")] if part
        ),
        "status": str(run.get("state") or "unknown"),
        "progress": 100 if run.get("running") else 0,
        "progress_text": str(run.get("progress") or ""),
        "gpu_hours": round(float(elapsed), 2) if isinstance(elapsed, (int, float)) else None,
        "cost_per_hour": float(hourly) if isinstance(hourly, (int, float)) else 0.0,
        "total_cost": round(float(total), 4) if isinstance(total, (int, float)) else 0.0,
        "training_samples": None,
        "file_size_mb": 0.0,
        "last_updated": run.get("started_at"),
        "evaluation_score": None,
        "evaluation_date": None,
        "backend": str(run.get("backend") or ""),
        "host": str(run.get("host") or ""),
        "gpu": str(run.get("gpu") or ""),
        "detail": str(run.get("detail") or ""),
        "note": str(run.get("note") or ""),
        "source_id": str(run.get("source_id") or ""),
        "running": bool(run.get("running")),
    }


def get_live_models() -> list[dict[str, Any]]:
    payload = load_live_training_status()
    if not payload:
        return []
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    return [build_live_model_status(run) for run in runs if isinstance(run, dict)]


def load_model_files() -> dict[str, Any]:
    """Load model metadata from JSONL files."""
    model_data = {}

    if not MODELS_DIR.exists():
        return model_data

    for model_key, _config in MODELS_CONFIG.items():
        # Look for training data files
        pattern = f"{model_key}*merged.jsonl"
        merged_files = list(MODELS_DIR.glob(pattern))

        if merged_files:
            file_path = merged_files[0]
            try:
                sample_count = sum(1 for _ in open(file_path))
                model_data[model_key] = {
                    "training_samples": sample_count,
                    "file_size": file_path.stat().st_size / 1024 / 1024,  # MB
                    "last_updated": datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat(),
                }
            except Exception as e:
                app.logger.error(f"Error loading {file_path}: {e}")

    return model_data


def load_evaluation_results() -> dict[str, Any]:
    """Load evaluation results from results directory."""
    results = {}

    if not RESULTS_DIR.exists():
        return results

    for result_file in RESULTS_DIR.glob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
                model_name = result_file.stem
                results[model_name] = data
        except Exception as e:
            app.logger.error(f"Error loading {result_file}: {e}")

    return results


def calculate_model_status(model_key: str) -> dict[str, Any]:
    """Calculate current status for a model."""
    config = MODELS_CONFIG[model_key]
    model_files = load_model_files()
    eval_results = load_evaluation_results()

    status = {
        "key": model_key,
        "name": config["name"],
        "description": config["description"],
        "status": config["status"],
        "progress": config["progress"],
        "gpu_hours": config["gpu_hours"],
        "cost_per_hour": config["cost_per_hour"],
        "total_cost": config["gpu_hours"] * config["cost_per_hour"],
        "training_samples": model_files.get(model_key, {}).get("training_samples", 0),
        "file_size_mb": round(
            model_files.get(model_key, {}).get("file_size", 0), 2
        ),
        "last_updated": model_files.get(model_key, {}).get("last_updated"),
        "evaluation_score": eval_results.get(f"{model_key}_results", {}).get(
            "overall_score", None
        ),
        "evaluation_date": None,
    }

    if f"{model_key}_results" in eval_results:
        eval_data = eval_results[f"{model_key}_results"]
        if "timestamp" in eval_data:
            status["evaluation_date"] = eval_data["timestamp"]

    return status


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/api/training/status", methods=["GET"])
def training_status():
    """Get overall training session status."""
    live = load_live_training_status()
    if live and isinstance(live.get("runs"), list):
        generated_at = str(live.get("generated_at") or datetime.now().isoformat())
        return jsonify(
            {
                "session_start": str(live.get("session_start") or generated_at),
                "current_time": generated_at,
                "total_cost": round(float(live.get("total_cost_usd") or 0.0), 4),
                "hourly_burn": round(float(live.get("live_hourly_usd") or 0.0), 4),
                "models_completed": int(live.get("stopped_count") or 0),
                "models_in_progress": int(live.get("running_count") or 0),
                "models_pending": 0,
                "total_runs": int(live.get("total_runs") or 0),
                "errors": list(live.get("errors") or []),
                "source": "live_training_status",
            }
        )

    return jsonify(
        {
            "session_start": TRAINING_STATE["session_start"].isoformat(),
            "current_time": datetime.now().isoformat(),
            "total_cost": round(TRAINING_STATE["total_cost"], 4),
            "models_completed": TRAINING_STATE["models_completed"],
            "models_in_progress": TRAINING_STATE["models_in_progress"],
            "models_pending": TRAINING_STATE["models_pending"],
            "estimated_completion": (
                TRAINING_STATE["session_start"]
                + timedelta(hours=18)  # Estimated total training time
            ).isoformat(),
        }
    )


@app.route("/api/models/status", methods=["GET"])
def models_status():
    """Get status of all models."""
    live_models = get_live_models()
    if live_models:
        payload = load_live_training_status() or {}
        return jsonify(
            {
                "models": live_models,
                "timestamp": str(payload.get("generated_at") or datetime.now().isoformat()),
                "source": "live_training_status",
            }
        )

    models = []
    for model_key in MODELS_CONFIG.keys():
        models.append(calculate_model_status(model_key))
    return jsonify({"models": models, "timestamp": datetime.now().isoformat()})


@app.route("/api/models/<model_key>/status", methods=["GET"])
def model_status(model_key: str):
    """Get status of a specific model."""
    live_models = get_live_models()
    if live_models:
        match = next((model for model in live_models if model["key"] == model_key), None)
        if not match:
            return jsonify({"error": "Model not found"}), 404
        return jsonify({**match, "timestamp": datetime.now().isoformat()})

    if model_key not in MODELS_CONFIG:
        return jsonify({"error": "Model not found"}), 404

    return jsonify({
        **calculate_model_status(model_key),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/costs/breakdown", methods=["GET"])
def cost_breakdown():
    """Get cost breakdown by model."""
    live_models = get_live_models()
    if live_models:
        breakdown = {}
        total = 0.0
        for model in live_models:
            breakdown[model["key"]] = {
                "model_name": model["name"],
                "gpu_hours": model["gpu_hours"],
                "cost_per_hour": model["cost_per_hour"],
                "total_cost": model["total_cost"],
                "backend": model["backend"],
                "status": model["status"],
            }
            total += float(model["total_cost"])

        payload = load_live_training_status() or {}
        return jsonify(
            {
                "breakdown": breakdown,
                "total_cost": round(total, 4),
                "hourly_rate": round(float(payload.get("live_hourly_usd") or 0.0), 4),
                "timestamp": str(payload.get("generated_at") or datetime.now().isoformat()),
                "source": "live_training_status",
            }
        )

    breakdown = {}
    total = 0

    for model_key, config in MODELS_CONFIG.items():
        cost = config["gpu_hours"] * config["cost_per_hour"]
        breakdown[model_key] = {
            "model_name": config["name"],
            "gpu_hours": config["gpu_hours"],
            "cost_per_hour": config["cost_per_hour"],
            "total_cost": round(cost, 4),
        }
        total += cost

    return jsonify({
        "breakdown": breakdown,
        "total_cost": round(total, 4),
        "hourly_rate": round(
            sum(c["cost_per_hour"] for c in MODELS_CONFIG.values()), 4
        ),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/metrics/gpu-utilization", methods=["GET"])
def gpu_utilization():
    """Get GPU utilization history."""
    # Simulate GPU utilization metrics
    now = datetime.now()
    metrics = []

    for i in range(24):  # Last 24 hours
        timestamp = now - timedelta(hours=24 - i)
        metrics.append({
            "timestamp": timestamp.isoformat(),
            "utilization_percent": 75 + (i % 15),  # Simulated data
            "memory_used_gb": 20 + (i % 8),
            "temperature_c": 65 + (i % 10),
        })

    return jsonify({
        "metrics": metrics,
        "current_utilization": 82,
        "current_memory": 24,
        "current_temperature": 71,
    })


@app.route("/api/metrics/training-loss", methods=["GET"])
def training_loss():
    """Get training loss curves."""
    # Simulate training loss metrics
    metrics = []
    base_loss = 2.5

    for step in range(0, 1000, 50):
        # Simulated loss curve (decreasing with noise)
        loss = base_loss * (1 - (step / 1000) ** 0.5) + (step % 10) * 0.01
        metrics.append({
            "step": step,
            "loss": round(loss, 4),
            "validation_loss": round(loss * 1.1, 4),
        })

    return jsonify({"metrics": metrics})


@app.route("/api/metrics/throughput", methods=["GET"])
def throughput():
    """Get training throughput metrics."""
    # Simulate throughput data
    metrics = []

    for i in range(24):
        timestamp = datetime.now() - timedelta(hours=24 - i)
        metrics.append({
            "timestamp": timestamp.isoformat(),
            "samples_per_second": 150 + (i % 30),
            "tokens_per_second": 3500 + (i % 500),
        })

    return jsonify({
        "metrics": metrics,
        "current_sps": 160,
        "current_tps": 3800,
    })


@app.route("/api/models/registry", methods=["GET"])
def model_registry():
    """Get complete model registry with deployment status."""
    live_models = get_live_models()
    if live_models:
        registry = []
        for model in live_models:
            registry.append({
                "key": model["key"],
                "name": model["name"],
                "version": model["source_id"] or "live",
                "status": model["status"],
                "training_samples": None,
                "file_size_mb": 0.0,
                "evaluation_score": None,
                "deployment_status": "active" if model["running"] else model["status"],
                "download_url": None,
                "last_updated": model["last_updated"],
                "backend": model["backend"],
                "host": model["host"],
                "gpu": model["gpu"],
                "detail": model["detail"],
                "note": model["note"],
            })

        payload = load_live_training_status() or {}
        return jsonify({
            "models": registry,
            "total_models": len(registry),
            "timestamp": str(payload.get("generated_at") or datetime.now().isoformat()),
            "source": "live_training_status",
        })

    registry = []

    for model_key, config in MODELS_CONFIG.items():
        model_files = load_model_files()
        status_info = calculate_model_status(model_key)

        registry.append({
            "key": model_key,
            "name": config["name"],
            "version": "1.0",
            "status": config["status"],
            "training_samples": model_files.get(model_key, {}).get("training_samples", 0),
            "file_size_mb": model_files.get(model_key, {}).get("file_size", 0),
            "evaluation_score": status_info.get("evaluation_score"),
            "deployment_status": "ready" if config["status"] == "completed" else "pending",
            "download_url": f"/api/models/{model_key}/download",
            "last_updated": model_files.get(model_key, {}).get("last_updated"),
        })

    return jsonify({
        "models": registry,
        "total_models": len(registry),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/export/csv", methods=["GET"])
def export_csv():
    """Export model status as CSV."""
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    live_models = get_live_models()
    if live_models:
        writer.writerow([
            "Run Name",
            "Backend",
            "State",
            "Host",
            "GPU",
            "Hours",
            "Rate/Hour",
            "Accrued Cost",
            "Note",
        ])

        for model in live_models:
            writer.writerow([
                model["name"],
                model["backend"],
                model["status"],
                model["host"],
                model["gpu"],
                model["gpu_hours"] if model["gpu_hours"] is not None else "",
                model["cost_per_hour"],
                model["total_cost"],
                model["note"],
            ])
    else:
    # Write header
        writer.writerow([
            "Model Name",
            "Status",
            "Progress %",
            "GPU Hours",
            "Cost/Hour",
            "Total Cost",
            "Training Samples",
            "File Size (MB)",
            "Evaluation Score",
        ])

    # Write data
        for model_key in MODELS_CONFIG.keys():
            status = calculate_model_status(model_key)
            writer.writerow([
                status["name"],
                status["status"],
                status["progress"],
                status["gpu_hours"],
                status["cost_per_hour"],
                status["total_cost"],
                status["training_samples"],
                status["file_size_mb"],
                status["evaluation_score"] or "N/A",
            ])

    response = app.response_class(
        response=output.getvalue(),
        status=200,
        mimetype="text/csv",
    )
    response.headers["Content-Disposition"] = (
        f'attachment; filename="afs-dashboard-{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv"'
    )
    return response


@app.route("/api/export/json", methods=["GET"])
def export_json():
    """Export all data as JSON."""
    live = load_live_training_status()
    if live and isinstance(live.get("runs"), list):
        data = {
            "timestamp": str(live.get("generated_at") or datetime.now().isoformat()),
            "training_status": {
                "session_start": str(live.get("session_start") or datetime.now().isoformat()),
                "total_cost": round(float(live.get("total_cost_usd") or 0.0), 4),
                "hourly_burn": round(float(live.get("live_hourly_usd") or 0.0), 4),
                "models_completed": int(live.get("stopped_count") or 0),
                "models_in_progress": int(live.get("running_count") or 0),
                "total_runs": int(live.get("total_runs") or 0),
                "errors": list(live.get("errors") or []),
            },
            "models": get_live_models(),
        }
    else:
        data = {
            "timestamp": datetime.now().isoformat(),
            "training_status": {
                "session_start": TRAINING_STATE["session_start"].isoformat(),
                "total_cost": round(TRAINING_STATE["total_cost"], 4),
                "models_completed": TRAINING_STATE["models_completed"],
                "models_in_progress": TRAINING_STATE["models_in_progress"],
                "models_pending": TRAINING_STATE["models_pending"],
            },
            "models": [],
        }

        for model_key in MODELS_CONFIG.keys():
            data["models"].append(calculate_model_status(model_key))

    response = app.response_class(
        response=json.dumps(data, indent=2),
        status=200,
        mimetype="application/json",
    )
    response.headers["Content-Disposition"] = (
        f'attachment; filename="afs-dashboard-{datetime.now().strftime("%Y%m%d-%H%M%S")}.json"'
    )
    return response


@app.route("/api/update-status", methods=["POST"])
def update_status():
    """Update model status (for manual updates)."""
    live = load_live_training_status()
    if live and isinstance(live.get("runs"), list):
        return jsonify({"error": "live training status is read-only"}), 405

    data = request.get_json()

    if not data or "model_key" not in data:
        return jsonify({"error": "Missing model_key"}), 400

    model_key = data["model_key"]
    if model_key not in MODELS_CONFIG:
        return jsonify({"error": "Model not found"}), 404

    # Update model configuration
    if "status" in data:
        MODELS_CONFIG[model_key]["status"] = data["status"]
    if "progress" in data:
        MODELS_CONFIG[model_key]["progress"] = data["progress"]

    # Update training state
    if "total_cost" in data:
        TRAINING_STATE["total_cost"] = data["total_cost"]

    return jsonify({
        "success": True,
        "model": model_key,
        "status": MODELS_CONFIG[model_key]["status"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
