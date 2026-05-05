"""Grounded explanation assistant for Vidyut Prajna.

The hosted LLM path is optional. When no API key is configured, the dashboard
uses deterministic local explanations from the same computed context.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional at import time
    def load_dotenv(*_: object, **__: object) -> bool:
        return False


load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 1.5

SYSTEM_INSTRUCTIONS = """
You are Vidyut Prajna's planner-facing explanation assistant for Bengaluru EV charging.

Rules:
1. Use only the computed JSON context provided by the dashboard.
2. Do not invent forecasts, percentages, zone names, or station locations.
3. Do not run prediction or optimization.
4. Recommendations are planning guidance, not grid-control commands.
5. Keep answers concise and useful for BESCOM planners and operators.
6. Mention uncertainty, utilization, capacity, tariff, or siting score when relevant.
7. If data is missing, say what is available and what is missing.
""".strip()


class VidyutLLM:
    """Gemini via OpenAI-compatible endpoint with deterministic fallback."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self.enabled = bool(self.api_key)
        self._client = None

        if self.enabled:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key, base_url=GEMINI_BASE_URL)
            except Exception:
                self.enabled = False
                self._client = None

    @staticmethod
    def _fallback_answer(question: str, context: Dict[str, object]) -> str:
        metrics = context.get("metrics", {}) if isinstance(context.get("metrics"), dict) else {}
        top_zones: List[Dict[str, object]] = context.get("top_predicted_demand_zones_at_selected_time", [])  # type: ignore[assignment]
        risk_zones: List[Dict[str, object]] = context.get("top_risk_zones_at_selected_time", [])  # type: ignore[assignment]
        station_recs: List[Dict[str, object]] = context.get("station_recommendations", [])  # type: ignore[assignment]
        siting_summary = context.get("siting_summary", {}) if isinstance(context.get("siting_summary"), dict) else {}

        lines = [
            "Local grounded explanation. Hosted LLM is not configured.",
            f"Peak reduction: {metrics.get('peak_reduction_pct', 'n/a')}%.",
            f"Overload events: {metrics.get('overload_events_before', 'n/a')} before, {metrics.get('overload_events_after', 'n/a')} after.",
            f"Grid stress after optimization: {metrics.get('stress_label_after', 'n/a')}.",
        ]

        if top_zones:
            top = top_zones[0]
            lines.append(
                f"Highest demand at the selected time is {top.get('zone_name')} "
                f"with {top.get('baseline_ev_load_kw')} kW unmanaged EV load."
            )
        if risk_zones:
            risk = risk_zones[0]
            util = risk.get("optimized_transformer_utilization", "n/a")
            lines.append(f"Highest post-optimization risk is {risk.get('zone_name')} at utilization {util}.")
        if station_recs:
            rec = station_recs[0]
            lines.append(
                f"Top infrastructure site is #{rec.get('rank')} {rec.get('zone_name')} "
                f"because: {rec.get('reason')}"
            )
        if siting_summary:
            lines.append(
                f"Recommended placement captures {siting_summary.get('capture_improvement_pct', 'n/a')}% "
                "more peak demand than the uniform baseline."
            )

        lines.append("This is decision support only; it does not modify grid systems.")
        return "\n".join(lines)

    def _call_hosted(self, user_input: str) -> str:
        assert self._client is not None
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": user_input},
                    ],
                    max_tokens=700,
                    temperature=0.25,
                )
                text = resp.choices[0].message.content if resp.choices else None
                return text.strip() if text else "The hosted model returned an empty response."
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY_S * (2 ** attempt))
        raise last_exc  # type: ignore[misc]

    def answer(self, question: str, context: Dict[str, object]) -> str:
        if not question.strip():
            return "Ask about the displayed forecast, optimizer result, or station recommendations."
        if not self.enabled or self._client is None:
            return self._fallback_answer(question, context)

        payload = {"question": question, "computed_context": context}
        user_input = (
            "Answer the planner's question using only this computed dashboard context.\n\n"
            + json.dumps(payload, indent=2, default=str)
        )
        try:
            return self._call_hosted(user_input)
        except Exception as exc:
            return (
                "Hosted LLM call failed. Falling back to local grounded explanation.\n"
                f"API error: {exc}\n\n"
                + self._fallback_answer(question, context)
            )

