"""Progress reporting for the preprocessing pipeline.

Ops call a ProgressCallback with 0..1; the aggregator turns per-step reports
into the overall percentage + timing summary the API returns.
"""

from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class ProgressInfo:
    """One progress report from a single operation."""
    step: str
    percent: float
    message: str = ""
    step_index: int = 0
    total_steps: int = 0
    elapsed_ms: int = 0


class ProgressCallback:
    """Scales one operation's 0..1 progress into overall pipeline percent."""

    def __init__(
        self,
        on_progress: Optional[Callable[[ProgressInfo], None]] = None,
        step_name: str = "",
        step_index: int = 0,
        total_steps: int = 1,
    ):
        self.on_progress = on_progress
        self.step_name = step_name
        self.step_index = step_index
        self.total_steps = total_steps
        self.start_time = time.time()

    def __call__(self, percent: float, message: str = ""):
        """percent is 0.0–1.0 within this step; every step weighs the same."""
        elapsed_ms = int((time.time() - self.start_time) * 1000)

        step_contribution = 1.0 / self.total_steps if self.total_steps > 0 else 1.0
        overall_percent = (self.step_index + percent) * step_contribution * 100

        info = ProgressInfo(
            step=self.step_name,
            percent=min(100, max(0, overall_percent)),
            message=message,
            step_index=self.step_index,
            total_steps=self.total_steps,
            elapsed_ms=elapsed_ms,
        )

        if self.on_progress:
            self.on_progress(info)


class ProgressAggregator:
    """Collects per-step timing/status for the API's progress_info payload."""

    def __init__(self):
        self.steps: list[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.current_percent: float = 0

    def start(self):
        """Mark pipeline start"""
        self.start_time = time.time()
        self.steps = []

    def step_started(self, step_name: str, index: int, total: int):
        """Mark step started"""
        self.steps.append({
            "step": step_name,
            "index": index,
            "total": total,
            "start_time": time.time(),
            "end_time": None,
            "duration_ms": None,
            "success": None,
            "error": None,
        })

    def step_completed(self, success: bool = True, error: Optional[str] = None):
        """Mark current step completed"""
        if self.steps:
            step = self.steps[-1]
            step["end_time"] = time.time()
            step["duration_ms"] = int((step["end_time"] - step["start_time"]) * 1000)
            step["success"] = success
            step["error"] = error

    def update_progress(self, info: ProgressInfo):
        """Handle progress update from callback"""
        self.current_percent = info.percent

    def finish(self):
        """Mark pipeline finished"""
        self.end_time = time.time()

    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary for API response"""
        total_ms = 0
        if self.start_time and self.end_time:
            total_ms = int((self.end_time - self.start_time) * 1000)

        return {
            "total_duration_ms": total_ms,
            "steps": [
                {
                    "step": s["step"],
                    "duration_ms": s["duration_ms"],
                    "success": s["success"],
                    "error": s["error"],
                }
                for s in self.steps
            ],
            "final_percent": self.current_percent,
        }
