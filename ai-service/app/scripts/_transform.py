"""
_transform.py — Re-exports Preprocessor from the single source of truth.

Single source of truth: app/core/preprocessor.py

Any logic change to the Preprocessor MUST be made in app/core/preprocessor.py only.
This file exists purely to maintain backward compatibility with existing import paths
inside the scripts/ package (run_pipeline.py, _wrapper.py).

Used by:
  - run_pipeline.py  : from ._transform import Preprocessor
  - _wrapper.py      : from ._transform import Preprocessor
"""
from app.core.preprocessor import Preprocessor

__all__ = ["Preprocessor"]
