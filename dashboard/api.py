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
from typing import Any, Dict, List, Optional

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


def load_model_files() -> Dict[str, Any]:
    """Load model metadata from JSONL files."""
    model_data = {}

    if not MODELS_DIR.exists():
        return model_data

    for model_key, config in MODELS_CONFIG.items():
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


def load_evaluation_results() -> Dict[str, Any]:
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


def calculate_model_status(model_key: str) -> Dict[str, Any]:
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
    models = []
    for model_key in MODELS_CONFIG.keys():
        models.append(calculate_model_status(model_key))
    return jsonify({"models": models, "timestamp": datetime.now().isoformat()})


@app.route("/api/models/<model_key>/status", methods=["GET"])
def model_status(model_key: str):
    """Get status of a specific model."""
    if model_key not in MODELS_CONFIG:
        return jsonify({"error": "Model not found"}), 404

    return jsonify({
        **calculate_model_status(model_key),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/costs/breakdown", methods=["GET"])
def cost_breakdown():
    """Get cost breakdown by model."""
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
