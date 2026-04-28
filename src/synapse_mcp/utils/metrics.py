"""
Performance metrics collector for Project Synapse.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MetricEntry:
    """A single metric measurement."""

    operation: str
    duration: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Singleton collector for performance metrics.

    Tracks execution time and operation frequency across components.
    """

    _instance: "MetricsCollector | None" = None
    _metrics: list[MetricEntry]
    _totals: dict[str, dict[str, Any]]

    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics = []
            cls._instance._totals = {}
        return cls._instance

    def record(self, operation: str, duration: float, **kwargs: Any) -> None:
        """Record an operation measurement."""
        entry = MetricEntry(operation=operation, duration=duration, metadata=kwargs)
        self._metrics.append(entry)

        # Update running totals
        if operation not in self._totals:
            self._totals[operation] = {
                "count": 0,
                "total_duration": 0.0,
                "max": 0.0,
                "min": float("inf"),
            }

        stats = self._totals[operation]
        stats["count"] += 1
        stats["total_duration"] += duration
        stats["max"] = max(stats["max"], duration)
        stats["min"] = min(stats["min"], duration)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for all recorded operations."""
        summary = {}
        for op, stats in self._totals.items():
            summary[op] = {
                "count": stats["count"],
                "avg_duration": stats["total_duration"] / stats["count"],
                "max_duration": stats["max"],
                "min_duration": stats["min"],
            }
        return summary

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self._metrics = []
        self._totals = {}


# Global instance
metrics = MetricsCollector()
