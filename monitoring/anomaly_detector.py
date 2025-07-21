def is_anomalous(current, previous, threshold=8.0):
    """Detects if current value deviates significantly from previous."""
    if previous is None:
        return False
    return abs(current - previous) > threshold
