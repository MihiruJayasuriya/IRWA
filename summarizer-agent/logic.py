import os
from typing import Dict, Any


# =========================================================
# LLM CONFIGURATION
# =========================================================

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


# =========================================================
# LLM SUMMARY FUNCTION
# =========================================================

def generate_llm_summary(prompt: str, fallback_summary: str) -> str:
    """
    Generate a natural-language summary using the local
    Llama 3.2 3B model through Ollama.

    Responsible AI rules:
    - Use only information provided in the prompt.
    - Do not invent causes or unsupported facts.
    - Do not invent measurements or events.
    - Clearly communicate uncertainty.
    - Do not make operational decisions.
    - Recommend human review for important decisions.

    If Ollama/Llama is unavailable, return the safe
    rule-based fallback summary.
    """

    try:

        from ollama import Client

        response = Client(timeout=4).chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a water-management reporting assistant. "

                        "Generate a concise, factual summary using ONLY "
                        "the information provided by the Analysis Agent. "

                        "Do not invent causes, measurements, events, "
                        "or other unsupported facts. "

                        "Do not claim that a specific cause exists unless "
                        "the provided data explicitly states the cause. "

                        "Do not make operational decisions or give commands "
                        "to automatically change water-management systems. "

                        "Clearly communicate when information is a forecast "
                        "or estimate. "

                        "Mention that human review is recommended for "
                        "important water-management decisions. "

                        "Return only the final natural-language summary."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        summary = response.message.content.strip()

        # Make sure we received usable text
        if summary:
            return summary

        # If the model returns an empty response
        return fallback_summary

    except Exception:
        # -------------------------------------------------
        # FALLBACK
        # -------------------------------------------------
        #
        # If Ollama is unavailable, the system does not fail.
        # Instead, it returns a safe rule-based summary.
        #

        return fallback_summary


# =========================================================
# MAIN SUMMARIZATION FUNCTION
# =========================================================

def create_summary(data: Dict[str, Any]) -> str:
    """
    Create a summary from Analysis Agent results.

    Supported result types:

    1. analysis_result
       - Water usage anomaly analysis

    2. forecast_result
       - Water demand forecast

    Responsible AI:
    - Uses only Analysis Agent information.
    - Avoids unsupported claims.
    - Communicates uncertainty.
    - Recommends human review.
    - Does not make automatic operational decisions.
    """

    # -----------------------------------------------------
    # Extract common fields
    # -----------------------------------------------------

    zone = data.get("zone", "Unknown")
    payload = data.get("payload", {})
    result_type = data.get("type", "")

    # =====================================================
    # ANALYSIS RESULT
    # =====================================================

    if result_type == "analysis_result":

        value = payload.get("value")
        severity = payload.get("severity")
        z_score = payload.get("z_score")
        baseline_mean = payload.get("baseline_mean")
        is_anomaly = payload.get("is_anomaly")

        # -------------------------------------------------
        # NORMAL WATER USAGE
        # -------------------------------------------------

        if not is_anomaly:

            fallback_summary = (
                f"Zone {zone} does not show a significant water "
                f"usage anomaly based on the provided analysis. "
                f"Current usage: {value}; "
                f"baseline average: {baseline_mean}. "
                "The result is based on the Analysis Agent's "
                "reported values. Human monitoring is recommended."
            )

            prompt = (
                f"Summarize the following water-usage analysis.\n\n"
                f"Zone: {zone}\n"
                f"Current usage: {value}\n"
                f"Baseline average: {baseline_mean}\n"
                f"Anomaly detected: {is_anomaly}\n\n"
                f"Explain that no significant anomaly was identified "
                f"based on the provided analysis. "
                f"Do not invent additional information."
            )

            return generate_llm_summary(
                prompt,
                fallback_summary
            )

        # -------------------------------------------------
        # ANOMALOUS WATER USAGE
        # -------------------------------------------------

        fallback_summary = (
            f"Water usage anomaly detected in Zone {zone}. "
            f"Severity: {severity}. "
            f"Current usage: {value}; "
            f"Baseline average: {baseline_mean}; "
            f"Z-score: {z_score}. "
            "The anomaly is identified from the statistical "
            "analysis provided by the Analysis Agent. "
            "The available data does not establish the specific "
            "cause of the anomaly. "
            "Human investigation is recommended before taking "
            "operational action."
        )

        prompt = (
            f"Create a concise natural-language summary of this "
            f"water-usage anomaly.\n\n"

            f"Zone: {zone}\n"
            f"Current water usage: {value}\n"
            f"Baseline average: {baseline_mean}\n"
            f"Z-score: {z_score}\n"
            f"Anomaly detected: {is_anomaly}\n"
            f"Severity: {severity}\n\n"

            f"Explain the abnormal usage using only these values. "

            f"Do NOT assume or invent a cause such as a pipe leak, "
            f"equipment failure, unusual customer behavior, or "
            f"any other cause unless explicitly provided. "

            f"State that the anomaly comes from the statistical "
            f"analysis provided by the Analysis Agent. "

            f"Recommend human investigation before operational "
            f"action."
        )

        return generate_llm_summary(
            prompt,
            fallback_summary
        )

    # =====================================================
    # FORECAST RESULT
    # =====================================================

    elif result_type == "forecast_result":

        days_ahead = payload.get("days_ahead")
        forecast = payload.get("forecast")
        trend_direction = payload.get("trend_direction")
        shortage_risk = payload.get("shortage_risk")
        reservoir = payload.get("reservoir_pct_full_used")

        # -------------------------------------------------
        # FALLBACK SUMMARY
        # -------------------------------------------------

        fallback_summary = (
            f"Zone {zone} is showing a "
            f"{trend_direction} water demand trend "
            f"over the next {days_ahead} days. "
            f"The current shortage risk is {shortage_risk}. "
            f"The reservoir level used for the forecast is "
            f"{reservoir}% full. "

            "This forecast is based on the data and prediction "
            "provided by the Analysis Agent. "
            "Forecasts are estimates and may change when new "
            "data becomes available. "

            "Human review is recommended before making major "
            "water-management decisions."
        )

        # -------------------------------------------------
        # LLM PROMPT
        # -------------------------------------------------

        prompt = (
            f"Create a concise natural-language summary of this "
            f"water-demand forecast.\n\n"

            f"Zone: {zone}\n"
            f"Forecast period: {days_ahead} days\n"
            f"Forecast values: {forecast}\n"
            f"Trend direction: {trend_direction}\n"
            f"Shortage risk: {shortage_risk}\n"
            f"Reservoir level: {reservoir}% full\n\n"

            f"Explain the expected trend and shortage risk. "

            f"Do not invent additional causes, events, "
            f"measurements, or information. "

            f"Clearly state that the forecast is an estimate "
            f"based on the Analysis Agent's prediction. "

            f"Explain that the forecast may change when new "
            f"data becomes available. "

            f"Recommend human review before major "
            f"water-management decisions."
        )

        return generate_llm_summary(
            prompt,
            fallback_summary
        )

    # =====================================================
    # UNSUPPORTED RESULT TYPE
    # =====================================================

    return (
        "The system could not generate a supported summary because "
        "the analysis result type was not recognized."
    )
