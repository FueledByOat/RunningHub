# exceiption_utils.py

"""
Custom Exceptions for RunningHub
"""

class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

class FormattingError(Exception):
    """Custom exception for formatting operations."""
    pass

class LanguageModelError(Exception):
    """Custom exception for language model operations."""
    pass