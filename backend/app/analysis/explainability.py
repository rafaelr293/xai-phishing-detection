from __future__ import annotations

from app.analysis.risk_engine import RiskAssessment


class ExplainabilityEngine:
    def explain(self, assessment: RiskAssessment) -> dict[str, str]:
        score = "unknown" if assessment.final_score is None else f"{assessment.final_score}/100"
        level = (assessment.risk_level or "unknown").capitalize()
        indicators = sorted(
            assessment.indicators,
            key=lambda item: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(item.get("severity", "low"), 0),
            reverse=True,
        )
        evidence = [item["description"] for item in indicators[:4] if item.get("description")]
        coverage = assessment.coverage
        if evidence:
            technical = f"{level} phishing risk ({score}). " + " ".join(evidence) + "."
            user = "This email may be suspicious because " + "; ".join(evidence[:2]).lower() + "."
        else:
            technical = f"{level} phishing risk ({score}). No detector indicators were returned."
            user = "There is not enough detector evidence to explain this assessment confidently."
        if coverage.get("failed_detectors") or coverage.get("missing_detectors"):
            technical += " Analysis coverage was partial."
        return {"technical": technical, "user": user}
