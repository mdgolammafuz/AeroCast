# Deprecated from V1. Kept for point anomaly test logs

def is_point_anomaly(current, previous, threshold=8.0):
    """Detects if current value deviates significantly from previous."""
    if previous is None:
        return False
    return abs(current - previous) > threshold
