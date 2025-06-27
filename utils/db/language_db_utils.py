# language_db_utils.py
"""Database utilities supporting any language models."""

import logging
import sqlite3
from typing import Optional, Dict, Any, List, Tuple
import datetime

from utils import exception_utils

logger = logging.getLogger(__name__)

def update_daily_training_metrics(conn: sqlite3.Connection, df, user_id=1):
    """
    Inserts or updates daily training metrics into SQLite.

    Args:
        df (DataFrame): Must have columns ['date', 'tss', 'CTL', 'ATL', 'TSB']
        user_id (int): The athlete identifier
        db_path (str): Path to SQLite database
    """

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE UNIQUE,
            user_id INTEGER,
            total_tss FLOAT,
            ctl FLOAT,
            atl FLOAT,
            tsb FLOAT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    for _, row in df.iterrows():
        conn.execute("""
            INSERT INTO daily_training_metrics (
                date, user_id, total_tss, ctl, atl, tsb
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_tss = excluded.total_tss,
                ctl = excluded.ctl,
                atl = excluded.atl,
                tsb = excluded.tsb;
        """, (
            row['date'].date(),  # Ensure date only, no time
            user_id,
            row['tss'],
            row['CTL'],
            row['ATL'],
            row['TSB']
        ))

    conn.commit()

def get_latest_daily_training_metrics(conn: sqlite3.Connection) -> Optional[Dict]:
    """Retrieves the latest day's data as a single dictionary."""
    original_row_factory = conn.row_factory  # Store original factory
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = """
            SELECT date, total_tss, ctl, atl, tsb
            FROM daily_training_metrics
            ORDER BY date DESC
            LIMIT 1;
        """
        row = cursor.execute(query).fetchone()
        
        # Convert the sqlite3.Row to a standard dict before returning
        return dict(row) if row else None
        
    finally:
        # Guarantee that the row_factory is reset, even if an error occurs.
        conn.row_factory = original_row_factory

def initialize_conversation_database(conn: sqlite3.Connection):
    """
    Retreives latest day's data from the daily_training_metrics table.

    Args:
        conn
    """
    c = conn.cursor()
    c.execute('''
    CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT, -- "user" or "coach"
    message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')

def get_recent_messages(conn: sqlite3.Connection, session_id: str, max_tokens: int, tokenizer) -> List[Dict[str, str]]:
    """Retrieve the most recent conversation history within a token limit."""
    try:
        # --- FIX 1: Explicitly set the row_factory for this function ---
        # This guarantees we get dict-like rows, regardless of the connection's prior state.
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT role, message, timestamp FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 50
        """, (session_id,))
        rows = cursor.fetchall()
        
        # --- FIX 2: Reset row_factory immediately after use ---
        # This prevents this function from affecting other parts of the application.
        conn.row_factory = None

        if not rows:
            return []
        
        history = []
        total_tokens = 0
        
        for row in rows:
            # --- FIX 3 (The Critical One): Convert the Row object to a true dict ---
            # This creates a completely independent copy of the data.
            row_dict = dict(row)

            role = row_dict['role']
            message = str(row_dict.get('message', '')) # Safer access
            
            if not message:
                continue
            
            formatted = f"User: {message}\n" if role == 'user' else f"Coach G: {message}\n"
            
            try:
                token_count = len(tokenizer.encode(formatted))
            except Exception as e:
                logger.warning(f"Tokenizer failed, using fallback: {e}")
                token_count = len(formatted.split()) * 1.3 # Simple approximation
            
            if total_tokens + token_count > max_tokens:
                break
            
            # Now we append the independent dictionary copy.
            history.insert(0, row_dict)
            total_tokens += token_count
        
        logger.debug(f"Retrieved {len(history)} messages ({total_tokens} tokens) for session {session_id}")
        return history
    
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving messages: {e}", exc_info=True)
        # In case of error, ensure row_factory is reset
        if conn:
            conn.row_factory = None
        raise exception_utils.DatabaseError(f"Failed to retrieve messages: {e}") from e
        
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving messages: {e}")
        raise exception_utils.DatabaseError(f"Failed to retrieve messages: {e}") from e
    
    
def save_message(conn: sqlite3.Connection, session_id: str, role: str, message: str):
    """Store a single message (user or coach) in the conversation history."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (session_id, role, message, timestamp)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, message, datetime.datetime.utcnow().isoformat()))
        
        conn.commit()
        logger.debug(f"Saved {role} message for session {session_id}")
        
    except sqlite3.Error as e:
        logger.error(f"Database error saving message: {e}")
        conn.rollback()
        raise exception_utils.DatabaseError(f"Failed to save message: {e}") from e

def execute_generated_query(conn: sqlite3.Connection, sql_query: str) -> Tuple[List[Dict], List[str]]:
    """Executes a validated SELECT query and returns standard dicts."""
    original_row_factory = conn.row_factory
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        
        # Convert list of Row objects to a list of standard dicts
        return [dict(row) for row in rows], columns
        
    except sqlite3.Error as e:
        logger.error(f"Error executing generated SQL query '{sql_query}': {e}")
        raise exception_utils.DatabaseError(f"Failed to execute query: {e}") from e
    finally:
        conn.row_factory = original_row_factory

def get_running_summary_for_last_n_days(conn: sqlite3.Connection, days: int = 7) -> dict:
    """
    Fetches a summary of running activities from the last N days.
    (Corrected and Robust Version)
    """
    # Set the row_factory to sqlite3.Row to get dict-like rows
    conn.row_factory = sqlite3.Row

    start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    start_date_str = start_date.strftime('%Y-%m-%d')

    query = """
        SELECT
            SUM(distance) as total_distance,
            SUM(moving_time) as total_moving_time,
            COUNT(id) as num_runs,
            SUM(total_elevation_gain) as total_elevation
        FROM
            activities
        WHERE
            type = 'Run' AND start_date_local >= ?
    """
    params = (start_date_str,)
    cursor = conn.execute(query, params)
    summary_data = cursor.fetchone()
    
    # Reset row_factory if other parts of your app expect tuples (optional but good practice)
    conn.row_factory = None

    # Convert the sqlite3.Row object to a standard dict before returning
    return dict(summary_data) if summary_data and summary_data['num_runs'] is not None else {}


def get_daily_metrics_for_last_n_days(conn: sqlite3.Connection, days: int = 7) -> list[dict]:
    """
    Fetches the daily training metrics (CTL, ATL, TSB) for the last N days.
    (Corrected and Robust Version)
    """
    # Set the row_factory to sqlite3.Row to get dict-like rows
    conn.row_factory = sqlite3.Row

    start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    query = """
        SELECT date, ctl, atl, tsb
        FROM daily_training_metrics
        WHERE date >= ?
        ORDER BY date ASC
    """
    params = (start_date_str,)
    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    
    # Reset row_factory if other parts of your app expect tuples
    conn.row_factory = None

    # Convert the list of sqlite3.Row objects to a list of standard dicts
    return [dict(row) for row in rows]