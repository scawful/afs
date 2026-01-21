"""GPU price tracking from vast.ai and other cloud providers.

Scrapes GPU pricing, tracks trends, and alerts on price drops.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


@dataclass
class GPUPrice:
    """Represents a GPU instance price at a point in time."""

    gpu_name: str
    provider: str  # 'vast.ai', 'lambda', 'paperspace', etc.
    price_per_hour: float
    availability: int = 0
    cuda_version: str | None = None
    vram_gb: int | None = None
    instance_id: str | None = None
    region: str | None = None
    timestamp: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.timestamp = _ensure_utc(self.timestamp)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "GPUPrice":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = _ensure_utc(datetime.fromisoformat(data["timestamp"]))
        return cls(**data)


@dataclass
class PriceHistory:
    """Historical pricing data for a GPU type."""

    gpu_name: str
    provider: str
    prices: list[tuple[datetime, float]] = field(default_factory=list)

    def add_price(self, price: float, timestamp: datetime | None = None) -> None:
        """Add a price point to history."""
        if timestamp is None:
            timestamp = _utc_now()
        else:
            timestamp = _ensure_utc(timestamp)
        self.prices.append((timestamp, price))

    def get_average(self, hours: int = 24) -> float:
        """Get average price over the last N hours."""
        cutoff = _utc_now() - timedelta(hours=hours)
        recent = [p for t, p in self.prices if t >= cutoff]
        return sum(recent) / len(recent) if recent else 0.0

    def get_min_max(self, hours: int = 24) -> tuple[float, float]:
        """Get min/max price over the last N hours."""
        cutoff = _utc_now() - timedelta(hours=hours)
        recent = [p for t, p in self.prices if t >= cutoff]
        if not recent:
            return 0.0, 0.0
        return min(recent), max(recent)

    def get_trend(self, hours: int = 24) -> float:
        """Get price trend as percentage change over N hours.

        Positive = price increased, negative = price decreased.
        """
        cutoff = _utc_now() - timedelta(hours=hours)
        relevant = [(t, p) for t, p in self.prices if t >= cutoff]

        if len(relevant) < 2:
            return 0.0

        old_price = relevant[0][1]
        new_price = relevant[-1][1]

        if old_price == 0:
            return 0.0

        return ((new_price - old_price) / old_price) * 100


@dataclass
class PriceAlert:
    """Alert triggered by price changes."""

    gpu_name: str
    provider: str
    alert_type: str  # 'drop', 'surge', 'low'
    current_price: float
    previous_price: float | None
    change_percent: float
    timestamp: datetime = field(default_factory=_utc_now)
    message: str = ""

    def __post_init__(self) -> None:
        self.timestamp = _ensure_utc(self.timestamp)
        self._build_message()

    def _build_message(self) -> None:
        """Generate alert message."""
        if self.alert_type == "drop":
            self.message = (
                f"Price drop on {self.provider} {self.gpu_name}: "
                f"${self.current_price:.3f}/hr (was ${self.previous_price:.3f})"
            )
        elif self.alert_type == "surge":
            self.message = (
                f"Price surge on {self.provider} {self.gpu_name}: "
                f"${self.current_price:.3f}/hr (was ${self.previous_price:.3f})"
            )
        elif self.alert_type == "low":
            self.message = f"Low price opportunity: {self.gpu_name} @ ${self.current_price:.3f}/hr"


class GPUPriceTracker:
    """Tracks GPU prices from various cloud providers."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize price tracker.

        Args:
            data_dir: Directory to store price history. Defaults to ~/.context/training/prices
        """
        if data_dir is None:
            data_dir = Path.home() / ".context" / "training" / "prices"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.histories: dict[tuple[str, str], PriceHistory] = {}
        self.current_prices: dict[tuple[str, str], float] = {}
        self.alerts: list[PriceAlert] = []

        self._load_histories()

    def _load_histories(self) -> None:
        """Load price histories from disk."""
        history_file = self.data_dir / "price_history.json"
        if not history_file.exists():
            return

        try:
            with open(history_file) as f:
                data = json.load(f)

            for key, prices in data.items():
                gpu_name, provider = key.rsplit("|", 1)
                history = PriceHistory(gpu_name, provider)
                for price_str, timestamp_str in prices:
                    timestamp = _ensure_utc(datetime.fromisoformat(timestamp_str))
                    history.add_price(float(price_str), timestamp)
                self.histories[(gpu_name, provider)] = history

            logger.info(f"Loaded {len(self.histories)} price histories")
        except Exception as e:
            logger.error(f"Failed to load price histories: {e}")

    def _save_histories(self) -> None:
        """Save price histories to disk."""
        history_file = self.data_dir / "price_history.json"

        data = {}
        for (gpu_name, provider), history in self.histories.items():
            key = f"{gpu_name}|{provider}"
            data[key] = [
                [price, timestamp.isoformat()] for timestamp, price in history.prices
            ]

        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved {len(self.histories)} price histories")

    def track_price(self, gpu_price: GPUPrice) -> PriceAlert | None:
        """Track a GPU price and check for alerts.

        Args:
            gpu_price: GPU price data to track

        Returns:
            Alert if price threshold crossed, None otherwise
        """
        key = (gpu_price.gpu_name, gpu_price.provider)

        # Create history if needed
        if key not in self.histories:
            self.histories[key] = PriceHistory(
                gpu_price.gpu_name, gpu_price.provider
            )

        history = self.histories[key]
        old_price = self.current_prices.get(key)

        # Add to history
        history.add_price(gpu_price.price_per_hour, gpu_price.timestamp)
        self.current_prices[key] = gpu_price.price_per_hour

        # Check for alerts
        alert = None
        if old_price is not None:
            change = ((gpu_price.price_per_hour - old_price) / old_price) * 100

            # Price drop > 10%
            if change < -10:
                alert = PriceAlert(
                    gpu_name=gpu_price.gpu_name,
                    provider=gpu_price.provider,
                    alert_type="drop",
                    current_price=gpu_price.price_per_hour,
                    previous_price=old_price,
                    change_percent=change,
                    timestamp=gpu_price.timestamp,
                )
            # Price surge > 20%
            elif change > 20:
                alert = PriceAlert(
                    gpu_name=gpu_price.gpu_name,
                    provider=gpu_price.provider,
                    alert_type="surge",
                    current_price=gpu_price.price_per_hour,
                    previous_price=old_price,
                    change_percent=change,
                    timestamp=gpu_price.timestamp,
                )

        if alert:
            self.alerts.append(alert)
            logger.warning(alert.message)

        self._save_histories()
        return alert

    def fetch_vastai_prices(self) -> list[GPUPrice]:
        """Fetch current prices from vast.ai.

        Returns:
            List of GPU prices from vast.ai
        """
        try:
            response = requests.get(
                "https://api.vast.ai/api/v0/gpus",
                params={"order": "price"},
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            prices = []

            for gpu in data.get("gpus", [])[:50]:  # Top 50 by price
                price = GPUPrice(
                    gpu_name=gpu.get("gpu_name", "Unknown"),
                    provider="vast.ai",
                    price_per_hour=float(gpu.get("min_bid", 0)) or 0.0,
                    availability=int(gpu.get("available", 0)),
                    vram_gb=int(gpu.get("gpu_ram", 0)) // 1024 if gpu.get("gpu_ram") else None,
                    instance_id=str(gpu.get("id")),
                    region=gpu.get("region"),
                )
                prices.append(price)

            logger.info(f"Fetched {len(prices)} GPU prices from vast.ai")
            return prices

        except Exception as e:
            logger.error(f"Failed to fetch vast.ai prices: {e}")
            return []

    def get_cheapest_gpu(self, gpu_type: str | None = None) -> tuple[GPUPrice, float] | None:
        """Get cheapest GPU instance.

        Args:
            gpu_type: Filter to specific GPU type (e.g., 'RTX 4090'). If None, get cheapest overall.

        Returns:
            Tuple of (GPUPrice, average_price_24h) or None
        """
        candidates = []

        for (name, provider), history in self.histories.items():
            if gpu_type and gpu_type.lower() not in name.lower():
                continue

            current = self.current_prices.get((name, provider))
            if current is None:
                continue

            avg_24h = history.get_average(hours=24)
            candidates.append((GPUPrice(name, provider, current), avg_24h))

        if not candidates:
            return None

        return min(candidates, key=lambda x: x[0].price_per_hour)

    def get_price_recommendations(self, gpu_type: str, max_price: float) -> list[tuple[GPUPrice, str]]:
        """Get recommendations for GPU purchases.

        Args:
            gpu_type: GPU type to search for
            max_price: Maximum acceptable price per hour

        Returns:
            List of (GPUPrice, reason) tuples
        """
        recommendations = []

        for (name, provider), history in self.histories.items():
            if gpu_type.lower() not in name.lower():
                continue

            current = self.current_prices.get((name, provider))
            if current is None or current > max_price:
                continue

            price_obj = GPUPrice(name, provider, current)
            min_24h, max_24h = history.get_min_max(hours=24)
            avg_24h = history.get_average(hours=24)

            # Reason for recommendation
            if current <= min_24h * 1.05:
                reason = f"Near 24h low (${min_24h:.3f})"
            elif current < avg_24h * 0.9:
                reason = f"Below 24h average (${avg_24h:.3f})"
            else:
                reason = "Within price range"

            recommendations.append((price_obj, reason))

        # Sort by price
        return sorted(recommendations, key=lambda x: x[0].price_per_hour)

    def get_alerts(self, hours: int = 24) -> list[PriceAlert]:
        """Get recent alerts.

        Args:
            hours: Only return alerts from the last N hours

        Returns:
            List of recent alerts
        """
        cutoff = _utc_now() - timedelta(hours=hours)
        return [a for a in self.alerts if a.timestamp >= cutoff]

    def get_price_statistics(self) -> dict:
        """Get statistics on tracked prices.

        Returns:
            Dictionary with price statistics
        """
        stats = {
            "total_tracked": len(self.histories),
            "gpus_by_type": {},
            "current_prices": {},
            "average_prices_24h": {},
            "price_trends_24h": {},
        }

        for (gpu_name, provider), history in self.histories.items():
            key = f"{provider}:{gpu_name}"
            current = self.current_prices.get((gpu_name, provider), 0.0)

            stats["current_prices"][key] = current
            stats["average_prices_24h"][key] = history.get_average(hours=24)
            stats["price_trends_24h"][key] = history.get_trend(hours=24)

            if gpu_name not in stats["gpus_by_type"]:
                stats["gpus_by_type"][gpu_name] = []
            stats["gpus_by_type"][gpu_name].append(provider)

        return stats
