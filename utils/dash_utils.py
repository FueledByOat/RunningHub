# dash_utils.py
"""Utilities for preparing and rendering Dash application components.

This module provides functions for:
- Polyline decoding for mapping
- Database operations for activity streams
- Data interpolation for visualization
"""

import json
import logging
import sqlite3
from typing import List, Tuple, Optional, Union

import numpy as np
import polyline

from utils.db import db_utils
from utils import exception_utils
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

