"""OpenCV preprocessing ops for historical-document OCR.

The ops themselves live in operations.py and are reached through OP_REGISTRY;
pipeline.py runs an ordered list of them.
"""

from .operations import OP_REGISTRY
from .pipeline import run_pipeline, PipelineExecutor, validate_pipeline_config

__all__ = [
    'OP_REGISTRY',
    'run_pipeline',
    'PipelineExecutor',
    'validate_pipeline_config',
]
