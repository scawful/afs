"""Tests for cost optimization system."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from afs.cost import (
    CostAnalyzer,
    CostOptimizer,
    GPUPrice,
    GPUPriceTracker,
    OptimizationRecommendation,
    PriceHistory,
    TrainingMetrics,
)


class TestGPUPriceTracker:
    """Test GPU price tracking."""

    def test_track_price(self):
        """Test tracking a GPU price."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = GPUPriceTracker(Path(tmpdir))

            price = GPUPrice(
                gpu_name="A100",
                provider="vast.ai",
                price_per_hour=1.50,
                vram_gb=40,
            )

            tracker.track_price(price)

            assert ("A100", "vast.ai") in tracker.current_prices
            assert tracker.current_prices[("A100", "vast.ai")] == 1.50

    def test_price_alert_drop(self):
        """Test price drop alert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = GPUPriceTracker(Path(tmpdir))

            # First price
            price1 = GPUPrice(
                gpu_name="RTX 4090",
                provider="vast.ai",
                price_per_hour=1.00,
            )
            tracker.track_price(price1)

            # Second price (11% drop)
            price2 = GPUPrice(
                gpu_name="RTX 4090",
                provider="vast.ai",
                price_per_hour=0.89,
                timestamp=datetime.now(timezone.utc) + timedelta(hours=1),
            )
            alert = tracker.track_price(price2)

            assert alert is not None
            assert alert.alert_type == "drop"
            assert alert.change_percent < -10

    def test_price_history(self):
        """Test price history calculation."""
        history = PriceHistory("A100", "vast.ai")

        now = datetime.now(timezone.utc)
        for i in range(5):
            history.add_price(1.0 + i * 0.1, now - timedelta(hours=4-i))

        avg = history.get_average(hours=24)
        assert avg > 0

        min_price, max_price = history.get_min_max(hours=24)
        assert min_price < max_price


class TestCostAnalyzer:
    """Test cost analysis."""

    def test_analyze_training_run(self):
        """Test analyzing a training run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CostAnalyzer(Path(tmpdir))

            metrics = TrainingMetrics(
                run_id="run_001",
                model_name="bert-base",
                num_samples=10000,
                num_epochs=3,
                batch_size=32,
                learning_rate=2e-5,
                total_duration_hours=2.5,
                gpu_name="A100",
                gpu_price_per_hour=1.50,
                test_accuracy=0.92,
                tokens_processed=5_000_000,
            )

            report = analyzer.analyze_training_run(metrics)

            assert report.total_cost == 3.75  # 2.5 * 1.50
            assert report.cost_per_sample == 3.75 / 10000
            assert report.cost_per_epoch == 3.75 / 3
            assert report.cost_per_token is not None
            assert report.efficiency_score > 0

    def test_budget_tracking(self):
        """Test budget tracking and alerts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CostAnalyzer(Path(tmpdir))

            analyzer.set_budget("test_model", 100.0)

            # 50% of budget
            alert1 = analyzer.check_budget("test_model", 50.0)
            assert alert1 is not None
            assert alert1.alert_type == "warning_50"

            # 75% of budget
            alert2 = analyzer.check_budget("test_model", 75.0)
            assert alert2 is not None
            assert alert2.alert_type == "warning_75"

            # 90% of budget
            alert3 = analyzer.check_budget("test_model", 90.0)
            assert alert3 is not None
            assert alert3.alert_type == "warning_90"

            # Exceeded budget
            alert4 = analyzer.check_budget("test_model", 101.0)
            assert alert4 is not None
            assert alert4.alert_type == "exceeded"

    def test_cost_comparison(self):
        """Test comparing costs across models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CostAnalyzer(Path(tmpdir))

            # Add multiple training runs
            for model_name in ["model_a", "model_b"]:
                for i in range(2):
                    metrics = TrainingMetrics(
                        run_id=f"{model_name}_run_{i}",
                        model_name=model_name,
                        num_samples=1000,
                        num_epochs=2,
                        batch_size=32,
                        learning_rate=2e-5,
                        total_duration_hours=1.0,
                        gpu_name="A100",
                        gpu_price_per_hour=1.0,
                        test_accuracy=0.90,
                    )
                    analyzer.analyze_training_run(metrics)

            comparison = analyzer.get_cost_comparison()

            assert "model_a" in comparison
            assert "model_b" in comparison
            assert comparison["model_a"]["runs"] == 2
            assert comparison["model_b"]["runs"] == 2

    def test_cost_forecast(self):
        """Test cost forecasting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CostAnalyzer(Path(tmpdir))

            analyzer.set_budget("test_model", 100.0)

            # Add a training run with known cost
            metrics = TrainingMetrics(
                run_id="run_001",
                model_name="test_model",
                num_samples=1000,
                num_epochs=2,
                batch_size=32,
                learning_rate=2e-5,
                total_duration_hours=1.0,
                gpu_name="A100",
                gpu_price_per_hour=10.0,
            )
            analyzer.analyze_training_run(metrics)

            # Forecast 5 more runs
            forecast = analyzer.forecast_cost("test_model", 5)

            assert forecast["model_name"] == "test_model"
            assert forecast["planned_runs"] == 5
            assert forecast["avg_cost_per_run"] == 10.0
            assert forecast["estimated_total"] == 50.0

    def test_roi_analysis(self):
        """Test ROI analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CostAnalyzer(Path(tmpdir))

            # First run: low accuracy
            metrics1 = TrainingMetrics(
                run_id="run_001",
                model_name="test_model",
                num_samples=1000,
                num_epochs=2,
                batch_size=32,
                learning_rate=2e-5,
                total_duration_hours=1.0,
                gpu_name="A100",
                gpu_price_per_hour=1.0,
                test_accuracy=0.80,
                timestamp=datetime.now(timezone.utc) - timedelta(days=1),
            )
            analyzer.analyze_training_run(metrics1)

            # Second run: higher accuracy
            metrics2 = TrainingMetrics(
                run_id="run_002",
                model_name="test_model",
                num_samples=1000,
                num_epochs=2,
                batch_size=32,
                learning_rate=2e-5,
                total_duration_hours=1.0,
                gpu_name="A100",
                gpu_price_per_hour=1.0,
                test_accuracy=0.92,
            )
            analyzer.analyze_training_run(metrics2)

            roi = analyzer.get_roi_analysis("test_model")

            assert roi["runs"] == 2
            assert roi["total_investment"] == 2.0
            assert roi["accuracy_improvement"] == 0.12


class TestCostOptimizer:
    """Test cost optimization."""

    def test_batch_size_recommendation(self):
        """Test batch size optimization recommendation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = CostOptimizer(Path(tmpdir))

            rec = optimizer.recommend_batch_size(
                current_batch_size=32,
                gpu_vram_gb=40,
                model_param_count=12_000_000,
                current_throughput=100,
                gpu_price_per_hour=1.0,
                epoch_hours=1.0,
            )

            if rec is not None:
                assert rec.category == "batch_size"
                assert rec.estimated_savings >= 0

    def test_early_stopping_recommendation(self):
        """Test early stopping recommendation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = CostOptimizer(Path(tmpdir))

            # Validation loss that plateaus
            val_loss_history = [0.50, 0.35, 0.25, 0.20, 0.18, 0.1801, 0.1799, 0.1798]

            rec = optimizer.recommend_early_stopping(
                validation_loss_history=val_loss_history,
                gpu_price_per_hour=1.0,
                hours_per_epoch=1.0,
                improvement_threshold=0.001,
                patience=3,
            )

            assert rec is not None
            assert rec.category == "early_stopping"
            assert rec.estimated_savings > 0

    def test_epoch_recommendation(self):
        """Test epoch count recommendation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = CostOptimizer(Path(tmpdir))

            # Accuracy that plateaus
            validation_accuracy = [0.70, 0.80, 0.88, 0.90, 0.91, 0.9101, 0.9102]

            rec = optimizer.recommend_epoch_count(
                validation_scores=validation_accuracy,
                gpu_price_per_hour=1.0,
                hours_per_epoch=1.0,
                score_type="accuracy",
            )

            if rec is not None:
                assert rec.category == "epochs"
                assert rec.estimated_savings >= 0

    def test_dataset_size_recommendation(self):
        """Test dataset size recommendation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = CostOptimizer(Path(tmpdir))

            accuracy_curve = {
                1000: 0.75,
                5000: 0.85,
                10000: 0.90,
                50000: 0.92,
                100000: 0.925,
            }

            rec = optimizer.recommend_dataset_size(
                dataset_size=100000,
                validation_accuracy_curve=accuracy_curve,
                gpu_price_per_hour=1.0,
                hours_per_epoch=1.0,
                num_epochs=3,
            )

            if rec is not None:
                assert rec.category == "dataset"
                assert rec.estimated_savings > 0

    def test_high_confidence_recommendations(self):
        """Test filtering high-confidence recommendations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = CostOptimizer(Path(tmpdir))

            rec1 = OptimizationRecommendation(
                category="test",
                title="Test 1",
                description="Test rec 1",
                estimated_savings=10.0,
                confidence=0.9,
            )
            rec2 = OptimizationRecommendation(
                category="test",
                title="Test 2",
                description="Test rec 2",
                estimated_savings=5.0,
                confidence=0.5,
            )

            optimizer.recommendations.append(rec1)
            optimizer.recommendations.append(rec2)

            high_conf = optimizer.get_high_confidence_recommendations(
                confidence_threshold=0.8
            )

            assert len(high_conf) == 1
            assert high_conf[0].confidence == 0.9


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        """Test complete workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize all components
            tracker = GPUPriceTracker(Path(tmpdir) / "tracker")
            analyzer = CostAnalyzer(Path(tmpdir) / "analyzer")
            optimizer = CostOptimizer(Path(tmpdir) / "optimizer")

            # Track prices
            price = GPUPrice(
                gpu_name="A100",
                provider="vast.ai",
                price_per_hour=1.50,
            )
            tracker.track_price(price)

            # Run training
            metrics = TrainingMetrics(
                run_id="test_run",
                model_name="test_model",
                num_samples=10000,
                num_epochs=3,
                batch_size=32,
                learning_rate=2e-5,
                total_duration_hours=2.5,
                gpu_name="A100",
                gpu_price_per_hour=1.50,
                test_accuracy=0.92,
            )

            # Analyze cost
            report = analyzer.analyze_training_run(metrics)

            assert report.total_cost > 0
            assert report.gpu_hours == 2.5

            # Set budget and check status
            analyzer.set_budget("test_model", 10.0)
            alert = analyzer.check_budget("test_model", 7.5)

            assert alert.alert_type == "warning_75"

            # Get recommendations
            recs = optimizer.recommend_batch_size(
                current_batch_size=32,
                gpu_vram_gb=40,
                model_param_count=12_000_000,
                current_throughput=100,
                gpu_price_per_hour=1.50,
                epoch_hours=2.5 / 3,
            )

            # Check statistics
            stats = tracker.get_price_statistics()
            assert "A100" in str(stats["gpus_by_type"])

            comparison = analyzer.get_cost_comparison()
            assert "test_model" in comparison


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
