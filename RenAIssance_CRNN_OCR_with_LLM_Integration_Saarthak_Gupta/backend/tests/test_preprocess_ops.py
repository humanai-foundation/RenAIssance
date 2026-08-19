"""Every preprocessing op and the PipelineExecutor must run on a synthetic
image and return a valid array — guards against op-registry regressions."""

import numpy as np
import pytest

from preprocessing.operations import OP_REGISTRY, list_operations
from preprocessing.pipeline import run_pipeline


def test_registry_matches_list():
    assert set(list_operations()) == set(OP_REGISTRY.keys())
    # 7 documented OpenCV ops at minimum (normalize, grayscale, deskew,
    # denoise, contrast, sharpen, binarize/threshold).
    assert len(OP_REGISTRY) >= 7


@pytest.mark.parametrize("op_name", sorted(OP_REGISTRY.keys()))
def test_each_op_runs_with_default_params(op_name, sample_color_image):
    fn = OP_REGISTRY[op_name]
    out = fn(sample_color_image, {})
    assert isinstance(out, np.ndarray)
    assert out.ndim in (2, 3)
    assert out.size > 0


def test_pipeline_executes_multi_step(sample_color_image):
    steps = [
        {"op": "grayscale", "params": {}, "enabled": True},
        {"op": "normalize", "params": {}, "enabled": True},
        {"op": "threshold", "params": {}, "enabled": True},
    ]
    result = run_pipeline(sample_color_image, steps)
    assert result.success
    assert isinstance(result.image, np.ndarray)
    assert result.image.size > 0


def test_pipeline_skips_disabled_steps(sample_color_image):
    steps = [{"op": "grayscale", "params": {}, "enabled": False}]
    result = run_pipeline(sample_color_image, steps)
    assert result.success
