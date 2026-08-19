"""Runs an ordered list of preprocessing ops over one image.

Reports progress per step and either stops or keeps going on failure.
"""

import traceback
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
import numpy as np

from .operations import OP_REGISTRY, get_operation
from .progress import ProgressCallback, ProgressAggregator, ProgressInfo


@dataclass
class PipelineStep:
    """One step in the pipeline."""
    op: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    success: bool
    image: Optional[np.ndarray]
    progress_info: Dict[str, Any]
    errors: List[Dict[str, Any]] = field(default_factory=list)


class PipelineExecutor:
    """Runs the steps in order, timing each one and reporting progress."""

    def __init__(
        self,
        on_progress: Optional[Callable[[ProgressInfo], None]] = None,
        continue_on_error: bool = False,
    ):
        self.on_progress = on_progress
        self.continue_on_error = continue_on_error
        self.aggregator = ProgressAggregator()

    def execute(
        self,
        image: np.ndarray,
        steps: List[Dict[str, Any]],
        preview_mode: bool = False,
    ) -> PipelineResult:
        """Run `steps` ([{op, params, enabled}, ...]) over `image`.

        preview_mode swaps in faster, lower-quality variants of some ops.
        """
        self.aggregator.start()

        active_steps = [
            PipelineStep(op=s.get("op", ""), params=s.get("params", {}))
            for s in steps
            if s.get("enabled", True)
        ]

        if not active_steps:
            self.aggregator.finish()
            return PipelineResult(
                success=True,
                image=image,
                progress_info=self.aggregator.get_summary(),
            )

        total_steps = len(active_steps)
        current_image = image.copy()
        errors = []

        for i, step in enumerate(active_steps):
            step_name = step.op
            self.aggregator.step_started(step_name, i, total_steps)
            op_func = get_operation(step_name)

            if op_func is None:
                error_info = {
                    "step": step_name,
                    "index": i,
                    "error": f"Unknown operation: {step_name}",
                }
                errors.append(error_info)
                self.aggregator.step_completed(success=False, error=error_info["error"])

                if not self.continue_on_error:
                    self.aggregator.finish()
                    return PipelineResult(
                        success=False,
                        image=current_image,
                        progress_info=self.aggregator.get_summary(),
                        errors=errors,
                    )
                continue

            progress = ProgressCallback(
                on_progress=self._handle_step_progress,
                step_name=step_name,
                step_index=i,
                total_steps=total_steps,
            )

            try:
                params = step.params.copy()
                if preview_mode:
                    params = self._adjust_params_for_preview(step_name, params)

                current_image = op_func(current_image, params, progress)

                self.aggregator.step_completed(success=True)

            except Exception as e:
                error_msg = str(e)
                error_trace = traceback.format_exc()

                error_info = {
                    "step": step_name,
                    "index": i,
                    "error": error_msg,
                    "traceback": error_trace,
                }
                errors.append(error_info)
                self.aggregator.step_completed(success=False, error=error_msg)

                if not self.continue_on_error:
                    self.aggregator.finish()
                    return PipelineResult(
                        success=False,
                        image=current_image,
                        progress_info=self.aggregator.get_summary(),
                        errors=errors,
                    )

        self.aggregator.finish()

        return PipelineResult(
            success=len(errors) == 0,
            image=current_image,
            progress_info=self.aggregator.get_summary(),
            errors=errors,
        )

    def _handle_step_progress(self, info: ProgressInfo):
        """Handle progress from individual step"""
        self.aggregator.update_progress(info)
        if self.on_progress:
            self.on_progress(info)

    def _adjust_params_for_preview(
        self,
        op_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trade quality for speed in preview mode."""
        adjusted = params.copy()

        # NLM denoising is far too slow for live preview; bilateral is close enough.
        if op_name == "denoise":
            if adjusted.get("method") == "nlm":
                adjusted["method"] = "bilateral"
                adjusted["strength"] = min(adjusted.get("strength", 10), 10)

        return adjusted


def run_pipeline(
    image: np.ndarray,
    steps: List[Dict[str, Any]],
    on_progress: Optional[Callable[[ProgressInfo], None]] = None,
    continue_on_error: bool = False,
    preview_mode: bool = False,
) -> PipelineResult:
    """One-shot PipelineExecutor.

    steps looks like [{"op": "grayscale", "params": {}, "enabled": True}, ...].
    """
    executor = PipelineExecutor(
        on_progress=on_progress,
        continue_on_error=continue_on_error,
    )

    return executor.execute(image, steps, preview_mode=preview_mode)


def validate_pipeline_config(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check a step list before running it. Returns {valid, errors}."""
    errors = []

    if not isinstance(steps, list):
        return {"valid": False, "errors": ["Pipeline must be a list of steps"]}

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"Step {i}: must be a dictionary")
            continue

        op = step.get("op")
        if not op:
            errors.append(f"Step {i}: missing 'op' field")
        elif op not in OP_REGISTRY:
            errors.append(f"Step {i}: unknown operation '{op}'")

        params = step.get("params")
        if params is not None and not isinstance(params, dict):
            errors.append(f"Step {i}: 'params' must be a dictionary")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }
