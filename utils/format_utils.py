# format_utils.py
"""Storing functions relating to front
end formatting of text values"""

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
        seconds = int(total_seconds % 60)
        return f"{minutes:02}:{seconds:02}"