"""Aggregates all rear-view metrics."""
from .lower_body import * # Imports all functions listed in lower_body/__all__
from .upper_body import * # Imports all functions listed in upper_body/__all__
from .gait_events_and_phases import *
# from .temporal import *
# Add any other rear subdirectories here in the same way

# Optional: Define __all__ for the 'rear' package itself if needed
# This would combine __all__ from lower_body and upper_body.
from .lower_body import __all__ as lower_all
from .upper_body import __all__ as upper_all
from .gait_events_and_phases import __all__ as gait_events_all
__all__ = lower_all + upper_all + gait_events_all