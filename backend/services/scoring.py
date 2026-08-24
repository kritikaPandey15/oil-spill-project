def calculate_suspicion_score(
    proximity_score: float,
    timing_score: float,
    trajectory_score: float,
    heading_score: float,
    ais_anomaly_score: float
):
    """
    Calculate a vessel's suspicion score.

    These weights are placeholders for the prototype.
    They can be adjusted after testing.
    """

    score = (
        proximity_score * 0.30
        + timing_score * 0.25
        + trajectory_score * 0.20
        + heading_score * 0.15
        + ais_anomaly_score * 0.10
    )

    return round(score, 2)