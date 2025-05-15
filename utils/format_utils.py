# format_utils.py
"""Storing functions relating to front
end formatting of text values"""

from datetime import datetime

def format_pace(distance_miles, total_seconds):
        """
        Calculates and formats the average pace per mile.

        Args:
            distance_miles (float): Total distance run in miles.
            total_seconds (int): Total time taken in seconds.

        Returns:
            str: Formatted pace in minutes and seconds per mile (MM:SS).
                Returns "Invalid input" if inputs are invalid.
        """
        if not isinstance(distance_miles, (int, float)) or not isinstance(total_seconds, int) or distance_miles <= 0 or total_seconds < 0:
            return "Invalid input"

        seconds_per_mile = total_seconds / distance_miles
        minutes = int(seconds_per_mile // 60)
        seconds = int(seconds_per_mile % 60)
        return f"{minutes:02}:{seconds:02}"

def format_time(total_seconds):
        """
        Calculates and formats the average pace per mile.

        Args:
            distance_miles (float): Total distance run in miles.
            total_seconds (int): Total time taken in seconds.

        Returns:
            str: Formatted pace in minutes and seconds per mile (MM:SS).
                Returns "Invalid input" if inputs are invalid.
        """
        minutes = int(total_seconds // 60)
        # If event goes into hours
        if minutes < 60:
            seconds = int(total_seconds % 60)
            return f"{minutes:02}:{seconds:02}"
        else:
            hours = int(total_seconds // 60 // 60)
            minutes = int(total_seconds // 60 % 60)
            seconds = int(total_seconds % 60)
            return f"{hours:01}:{minutes:02}:{seconds:02}"
        
def date_format(input_string): 
    """
        Accepts a datetime value from the database in the following
        format - 2025-05-10T12:11:52Z and clean it to be a 
        standardized ISO 8601 date.

        Args:
            input_string (string): 2025-05-10T12:11:52Z

        Returns:
            start_date: ISO 8601 Formatted date.
            start_time: Formatted time.
        """
    dt = datetime.strptime(input_string, "%Y-%m-%dT%H:%M:%SZ")

    # Format the date and time parts separately
    start_date = dt.strftime("%Y-%m-%d")
    start_time = dt.strftime("%H:%M:%S")

    return start_date, start_time