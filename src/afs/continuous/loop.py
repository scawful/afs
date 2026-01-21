"""Continuous learning main loop.

Orchestrates the full continuous improvement cycle:
1. Log usage and collect feedback
2. Monitor triggers for retraining
3. Generate quality training data
4. Execute retraining
5. Deploy as challenger in A/B test
6. Auto-promote if better
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from .logger import UsageLogger
from .generator import TrainingDataGenerator, DataGeneratorConfig
from .trigger import AutoRetrainer, TriggerConfig
from .ab_test import ABTestManager, ABTestConfig

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """Configuration for continuous learning loop."""

    # Paths
    db_path: Path = Path("~/.context/training/continuous/usage.db").expanduser()
    output_dir: Path = Path("~/.context/training/continuous").expanduser()
    models_dir: Path = Path("~/.context/training/continuous/models").expanduser()

    # Component configs
    trigger_config: TriggerConfig = field(default_factory=TriggerConfig)
    generator_config: DataGeneratorConfig = field(default_factory=DataGeneratorConfig)
    ab_test_config: ABTestConfig = field(default_factory=ABTestConfig)

    # Loop behavior
    check_interval_seconds: int = 3600  # Check every hour
    enable_ab_testing: bool = True
    enable_auto_promotion: bool = True


@dataclass
class LoopStatus:
    """Status of the continuous learning loop."""

    running: bool = False
    last_check: Optional[str] = None
    last_retrain: Optional[str] = None
    total_retrains: int = 0
    champion_model: Optional[str] = None
    challenger_model: Optional[str] = None
    errors: list[str] = field(default_factory=list)


class ContinuousLearningLoop:
    """Main continuous learning orchestrator."""

    def __init__(self, config: Optional[LoopConfig] = None):
        self.config = config or LoopConfig()

        # Initialize components
        self.usage_logger = UsageLogger(self.config.db_path)
        self.auto_retrainer = AutoRetrainer(
            self.usage_logger,
            trigger_config=self.config.trigger_config,
            generator_config=self.config.generator_config,
            output_dir=self.config.output_dir,
        )

        self.ab_test_manager: Optional[ABTestManager] = None
        if self.config.enable_ab_testing:
            self.ab_test_manager = ABTestManager(
                self.usage_logger,
                config=self.config.ab_test_config,
            )
        self.ab_manager = self.ab_test_manager

        self.status = LoopStatus()
        self._status_file = self.config.output_dir / "loop_status.json"

        # Create directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        self._load_status()

    def run_iteration(
        self,
        train_fn: Optional[Callable[[Path], dict]] = None,
    ) -> dict:
        """Run one iteration of the continuous learning loop.

        Args:
            train_fn: Training function (data_path) -> metrics dict

        Returns:
            Iteration result dict
        """
        self.status.last_check = datetime.now().isoformat()
        result = {
            "timestamp": self.status.last_check,
            "retrain_triggered": False,
            "ab_test_update": False,
        }

        logger.info("=== Continuous Learning Iteration ===")

        # Step 1: Check if retraining is needed
        logger.info("Step 1: Checking retrain triggers")
        retrain_result = self.auto_retrainer.check_and_retrain(train_fn)

        if retrain_result:
            result["retrain_triggered"] = True
            result["retrain"] = retrain_result
            self.status.total_retrains += 1
            self.status.last_retrain = datetime.now().isoformat()

            logger.info(f"Retrain completed: {retrain_result['status']}")

            # Step 2: Deploy new model as challenger (if A/B testing enabled)
            if self.config.enable_ab_testing and self.ab_test_manager:
                if retrain_result["status"] == "completed":
                    logger.info("Step 2: Deploying new model as challenger")
                    model_path = self._get_model_path_from_result(retrain_result)

                    if model_path:
                        model_name = f"model_{retrain_result['timestamp']}"
                        self.ab_test_manager.deploy_challenger(
                            model_name=model_name,
                            model_path=model_path,
                        )
                        result["ab_test_update"] = True
                        result["challenger_deployed"] = model_name

                        self.status.challenger_model = model_name
        else:
            logger.info("No retrain triggered")

        # Step 3: Check A/B test and auto-promote if ready
        if self.config.enable_ab_testing and self.ab_test_manager:
            logger.info("Step 3: Checking A/B test results")

            # Compare models
            comparison = self.ab_test_manager.compare_models()
            if comparison:
                result["ab_comparison"] = {
                    "winner": comparison.winner.value,
                    "improvement": comparison.improvement,
                    "reason": comparison.reason,
                }

                # Auto-promote if configured
                if self.config.enable_auto_promotion:
                    promoted = self.ab_test_manager.auto_promote_if_ready()
                    if promoted:
                        result["ab_test_update"] = True
                        result["champion_promoted"] = self.ab_test_manager.champion.name
                        self.status.champion_model = self.ab_test_manager.champion.name
                        self.status.challenger_model = None
                        logger.info(f"Auto-promoted champion: {self.status.champion_model}")

        # Update status
        if self.ab_test_manager:
            if self.ab_test_manager.champion:
                self.status.champion_model = self.ab_test_manager.champion.name
            if self.ab_test_manager.challenger:
                self.status.challenger_model = self.ab_test_manager.challenger.name

        self._save_status()

        logger.info("=== Iteration Complete ===")
        return result

    def run_loop(
        self,
        train_fn: Optional[Callable[[Path], dict]] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """Run the continuous learning loop indefinitely.

        Args:
            train_fn: Training function (data_path) -> metrics dict
            max_iterations: Optional maximum iterations (for testing)
        """
        self.status.running = True
        iteration = 0

        logger.info("Starting continuous learning loop")

        try:
            while self.status.running:
                iteration += 1

                if max_iterations and iteration > max_iterations:
                    logger.info(f"Reached max iterations: {max_iterations}")
                    break

                try:
                    result = self.run_iteration(train_fn)
                    logger.info(f"Iteration {iteration} result: {result}")
                except Exception as e:
                    error_msg = f"Error in iteration {iteration}: {e}"
                    logger.error(error_msg)
                    self.status.errors.append(error_msg)
                    self._save_status()

                # Sleep until next check
                logger.info(
                    f"Sleeping for {self.config.check_interval_seconds} seconds"
                )
                time.sleep(self.config.check_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Loop interrupted by user")
        finally:
            self.status.running = False
            self._save_status()
            logger.info("Continuous learning loop stopped")

    def stop(self) -> None:
        """Stop the continuous learning loop."""
        self.status.running = False
        logger.info("Stopping continuous learning loop")

    def log_usage(
        self,
        query: str,
        response: str,
        model: str,
        expert: Optional[str] = None,
        latency_ms: float = 0,
        quality_score: float = 0.0,
    ) -> str:
        """Log a model usage.

        Returns the record ID.
        """
        return self.usage_logger.log(
            query=query,
            response=response,
            model=model,
            expert=expert,
            latency_ms=latency_ms,
            quality_score=quality_score,
        )

    def record_feedback(
        self,
        record_id: str,
        feedback: int,
        feedback_text: Optional[str] = None,
    ) -> bool:
        """Record user feedback."""
        return self.usage_logger.record_feedback(record_id, feedback, feedback_text)

    def get_statistics(self) -> dict:
        """Get usage statistics."""
        return self.usage_logger.get_statistics()

    def get_status_summary(self) -> dict:
        """Get current status summary."""
        stats = self.usage_logger.get_statistics()

        summary = {
            "status": self.status.__dict__,
            "usage_stats": stats,
        }

        if self.ab_test_manager:
            summary["ab_test"] = {
                "champion": self.ab_test_manager.champion.to_dict()
                if self.ab_test_manager.champion
                else None,
                "challenger": self.ab_test_manager.challenger.to_dict()
                if self.ab_test_manager.challenger
                else None,
                "traffic_split": {
                    "champion": self.ab_test_manager.traffic_split.champion_weight,
                    "challenger": self.ab_test_manager.traffic_split.challenger_weight,
                },
            }

        return summary

    def _get_model_path_from_result(self, retrain_result: dict) -> Optional[Path]:
        """Extract model path from retrain result.

        This is a hook for the training function to communicate the model path.
        """
        # Check if training result includes model_path
        training = retrain_result.get("training", {})
        if "model_path" in training:
            return Path(training["model_path"])

        # Fallback: assume model saved next to data
        data_path = retrain_result.get("generation", {}).get("data_path")
        if data_path:
            # Convention: model saved as {data_path}.model
            return Path(data_path).with_suffix(".model")

        return None

    def _save_status(self) -> None:
        """Save status to disk."""
        with open(self._status_file, "w") as f:
            json.dump(self.status.__dict__, f, indent=2, default=str)

    def _load_status(self) -> None:
        """Load status from disk."""
        if not self._status_file.exists():
            return

        try:
            with open(self._status_file) as f:
                data = json.load(f)
                self.status = LoopStatus(**data)
        except Exception as e:
            logger.warning(f"Failed to load status: {e}")
