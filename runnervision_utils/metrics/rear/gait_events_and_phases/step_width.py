# step_width.py

def calculate_step_width(landmarks):
    """Distance between both feet."""
    left_foot_x = landmarks['left_foot_index'][0]
    right_foot_x = landmarks['right_foot_index'][0]
    
    step_width = abs(left_foot_x - right_foot_x) * 100  # Rough cm conversion
    return step_width