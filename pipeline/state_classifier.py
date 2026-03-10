"""
Rule-based mental state classifier from EEG band features.
Outputs: predicted_state, confidence, explanation, scores.
Designed to be replaceable later by a trained ML model.
"""

from typing import Any

STATES = ["Relaxed", "Focused", "Drowsy", "Stressed", "Neutral/Unclear"]
STATE_KEYS = ["relaxed", "focused", "drowsy", "stressed", "neutral"]


def classify(features: dict[str, float]) -> dict[str, Any]:
    """
    Deterministic rule-based classification.
    - Relaxed: alpha relatively high, beta/gamma not high.
    - Focused: beta elevated vs alpha/theta, theta_beta low.
    - Drowsy: theta/delta high, beta low.
    - Stressed: beta and gamma high, alpha low.
    - Neutral/Unclear: weak or tied evidence.
    """
    ra = features.get("relative_alpha", 0.0)
    rb = features.get("relative_beta", 0.0)
    rt = features.get("relative_theta", 0.0)
    rd = features.get("relative_delta", 0.0)
    rg = features.get("relative_gamma", 0.0)
    ba = features.get("beta_alpha_ratio", 1.0)
    tb = features.get("theta_beta_ratio", 1.0)
    abg = features.get("alpha_beta_gamma_ratio", 0.5)

    relaxed_score = max(0, ra * (1.2 - rb) * (1.2 - rg))
    focused_score = max(0, rb * (1.5 - tb) * min(2.0, 0.5 + abg))
    drowsy_score = max(0, (rt + rd) * (1.5 - rb))
    stressed_score = max(0, rb * rg * (1.5 - ra))

    scores_raw = {
        "relaxed": relaxed_score,
        "focused": focused_score,
        "drowsy": drowsy_score,
        "stressed": stressed_score,
        "neutral": 0.5,
    }
    total = sum(scores_raw.values()) + 1e-9
    scores = {k: round(v / total, 4) for k, v in scores_raw.items()}

    pred_key = max(scores, key=scores.get)
    confidence = scores[pred_key]
    if confidence < 0.35 or pred_key == "neutral":
        predicted_state = "Neutral/Unclear"
        confidence = max(0.2, min(0.5, confidence + 0.2))
        explanation = "Band activity does not strongly indicate one state; may be mixed or transitional."
    else:
        predicted_state = {"relaxed": "Relaxed", "focused": "Focused", "drowsy": "Drowsy", "stressed": "Stressed"}[pred_key]
        if predicted_state == "Relaxed":
            explanation = "Alpha activity is relatively dominant with moderate or low beta and gamma."
        elif predicted_state == "Focused":
            explanation = "Beta activity is elevated relative to alpha and theta."
        elif predicted_state == "Drowsy":
            explanation = "Theta and/or delta are elevated while beta is relatively low."
        elif predicted_state == "Stressed":
            explanation = "Beta and gamma are elevated with relatively low alpha."
        else:
            explanation = "Band pattern suggests this state."

    return {
        "predicted_state": predicted_state,
        "confidence": round(confidence, 4),
        "explanation": explanation,
        "scores": scores,
        "features": {k: round(v, 6) for k, v in features.items()},
    }
