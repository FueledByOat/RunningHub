# db_utils.py
"""Database utilities for general database queries and connection management."""


import logging
import sqlite3
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from threading import Lock
from queue import Queue, Empty

from utils import exception_utils

logger = logging.getLogger(__name__)

class ConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.pool = Queue(maxsize=max_connections)
        self.lock = Lock()
        
    def _create_connection(self):
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = dict_factory
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool or create new one."""
        try:
            conn = self.pool.get_nowait()
        except Empty:
            conn = self._create_connection()
            
        try:
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database operation failed: {e}")
            conn.rollback()
            raise exception_utils.DatabaseError(f"Database error: {e}") from e
        finally:
            try:
                self.pool.put_nowait(conn)
            except:
                conn.close()


# Global connection pool
_connection_pool = None

def get_connection_pool(db_path: str) -> ConnectionPool:
    """Get or create global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        if not db_path:
            raise exception_utils.DatabaseError("Database path not configured")
        _connection_pool = ConnectionPool(db_path)
    return _connection_pool


@contextmanager
def get_db_connection(db_path: str):
    """Context manager for pooled database connections."""
    pool = get_connection_pool(db_path)
    with pool.get_connection() as conn:
        yield conn


def dict_factory(cursor: sqlite3.Cursor, row: sqlite3.Row) -> Dict[str, Any]:
    """Convert database row objects to a dictionary.
    
    Args:
        cursor: Database cursor
        row: Database row
        
    Returns:
        Dictionary representation of the row
    """
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

def dict_from_row(row) -> Dict:
    """Convert sqlite3.Row to dictionary"""
    return dict(row) if row else {}

def dicts_from_rows(rows) -> List[Dict]:
    """Convert list of sqlite3.Row to list of dictionaries"""
    return [dict(row) for row in rows] if rows else []



