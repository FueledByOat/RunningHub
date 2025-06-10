# stride_symmetry.py

def calculate_stride_symmetry(landmarks):
    """Compare stride or timing parameters over a cycle (requires frame history).
        Placeholder uses foot x-delta."""
    left_stride = landmarks['left_foot_index'][0] - landmarks['left_heel'][0]
    right_stride = landmarks['right_foot_index'][0] - landmarks['right_heel'][0]
    
    symmetry = (right_stride - left_stride) / max(abs(right_stride), abs(left_stride) + 1e-6)
    
    return symmetry