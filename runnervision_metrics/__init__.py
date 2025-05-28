"""
RunnerVision Metrics Package.
Provides access to rear-view and side-view biomechanical metric calculations.
"""
from . import rear as rear_metrics    # Exposes 'rear_metrics' module
# from . import side as side_metrics    # Exposes 'side_metrics' module

__all__ = [
    'rear_metrics',
    # 'side_metrics',
]