from datetime import datetime, timezone


def create_alert(data: dict) -> dict:
    """
    Process an Analysis Agent result and generate
    an alert when necessary.
    """

    message_type = data.get("type")
    zone = data.get("zone", "Unknown")
    payload = data.get("payload", {})

    alert = False
    severity = "normal"
    message = "No critical water management issue detected."

    # Handle anomaly analysis results
    if message_type == "analysis_result":

        is_anomaly = payload.get("is_anomaly", False)
        analysis_severity = payload.get("severity", "normal")

        if is_anomaly:
            alert = True
            severity = analysis_severity

            message = (
                f"{severity.capitalize()} water usage anomaly detected "
                f"in zone {zone}."
            )

    # Handle forecast results
    elif message_type == "forecast_result":

        shortage_risk = payload.get("shortage_risk", "low")

        if shortage_risk == "high":
            alert = True
            severity = "critical"
            message = (
                f"High water shortage risk detected in zone {zone}."
            )

        elif shortage_risk == "moderate":
            alert = True
            severity = "warning"
            message = (
                f"Moderate water shortage risk detected in zone {zone}."
            )

        else:
            severity = "normal"
            message = (
                f"Water shortage risk is low in zone {zone}."
            )

    else:
        return {
            "status": "error",
            "agent": "alert",
            "type": "alert_result",
            "zone": zone,
            "payload": {
                "error": f"Unsupported analysis type: {message_type}"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    return {
        "status": "ok",
        "agent": "alert",
        "type": "alert_result",
        "zone": zone,
        "payload": {
            "alert": alert,
            "severity": severity,
            "message": message
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
