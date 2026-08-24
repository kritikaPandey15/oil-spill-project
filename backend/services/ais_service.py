from services.scoring import calculate_suspicion_score


def find_vessels(latitude: float, longitude: float):

    # Temporary synthetic vessel data.
    # Later, this will be replaced with real/synthetic AIS processing.
    vessels = [
        {
            "id": "V001",
            "proximity": 90,
            "timing": 95,
            "trajectory": 80,
            "heading": 85,
            "ais_anomaly": 70
        },
        {
            "id": "V002",
            "proximity": 50,
            "timing": 40,
            "trajectory": 45,
            "heading": 50,
            "ais_anomaly": 20
        }
    ]

    results = []

    for vessel in vessels:

        # Calculate overall suspicion score
        score = calculate_suspicion_score(
            vessel["proximity"],
            vessel["timing"],
            vessel["trajectory"],
            vessel["heading"],
            vessel["ais_anomaly"]
        )

        # Generate explanations dynamically
        reasons = []

        if vessel["proximity"] >= 70:
            reasons.append("close to estimated origin")

        if vessel["timing"] >= 70:
            reasons.append("within estimated spill time window")

        if vessel["trajectory"] >= 70:
            reasons.append("trajectory is consistent with spill drift")

        if vessel["heading"] >= 70:
            reasons.append("heading is consistent with spill location")

        if vessel["ais_anomaly"] >= 70:
            reasons.append("AIS anomaly detected near estimated spill time")

        results.append({
            "id": vessel["id"],
            "score": score,
            "reasons": reasons
        })

    # Highest suspicion first
    results.sort(
        key=lambda vessel: vessel["score"],
        reverse=True
    )

    return results