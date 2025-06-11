# rating_utils.py

def get_rating_from_score(score, thresholds):
    """
    Determines the rating category based on a score and defined thresholds.

    Args:
        score (float): The score to evaluate (0-100).
        thresholds (dict): A dictionary with keys 'optimal', 'good', 'fair' defining the lower bound of each category.

    Returns:
        tuple: A tuple containing the rating text (str) and rating key (str).
    """
    if score >= thresholds.get("optimal", 90):
        return "Optimal", "optimal"
    if score >= thresholds.get("good", 80):
        return "Good", "good"
    if score >= thresholds.get("fair", 70):
        return "Fair", "fair"
    return "Needs Work", "needs-work"

def rate_cadence(cadence):
    """Rates running cadence (in steps per minute)."""
    if 170 <= cadence <= 190:
        return "Optimal", "optimal"
    if 160 <= cadence < 170 or 190 < cadence <= 200:
        return "Good", "good"
    return "Needs Improvement", "needs-work"

def rate_trunk_angle(angle):
    """Rates forward trunk lean in degrees."""
    if 5 <= angle <= 15:
        return "Optimal", "optimal"
    if 0 <= angle < 5 or 15 < angle <= 20:
        return "Good", "good"
    return "Needs Improvement", "needs-work"

def rate_knee_symmetry(diff_percent):
    """Rates knee symmetry based on percentage difference."""
    if diff_percent < 5:
        return "Excellent", "optimal"
    if diff_percent < 10:
        return "Good", "good"
    if diff_percent < 15:
        return "Fair", "fair"
    return "Needs Improvement", "needs-work"

def rate_crossover(crossover_percent):
    """Rates foot crossover percentage (lower is better)."""
    if crossover_percent < 5:
        return "Optimal", "optimal"
    if crossover_percent < 10:
        return "Good", "good"
    if crossover_percent < 20:
        return "Fair", "fair"
    return "High", "needs-work"