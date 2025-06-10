"""Rear gait events and phases metrics."""
from .stance_phase_detector_rear import detect_stance_phase_rear
from .step_width import calculate_step_width
from .stride_symmetry import calculate_stride_symmetry
# ... import other rear lower body metric functions

__all__ = [
 'detect_stance_phase_rear',
 'calculate_step_width',
 'calculate_stride_symmetry',
    # ... add other function names here
]