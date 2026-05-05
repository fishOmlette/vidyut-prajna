"""Grounded LLM explanation layer for the Dash app.

Uses the Gemini API through its OpenAI-compatible endpoint so that the standard
``openai`` Python SDK can be reused with a different ``base_url``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

SYSTEM_INSTRUCTIONS = """
You are Vidyut Prajna's planner-facing explanation assistant for Bengaluru EV charging.

Domain context (never fabricate data from these):
- BESCOM operates the Bengaluru grid. Transformer ratings: 100-630 kVA.
- Monsoon (Jun-Sep) increases home charging, reduces outdoor station use.
- IT corridors (Whitefield, Electronic City, ORR) peak 10AM-8PM.
- Fleet mix: 2W (3.3kW), 3W (7.4kW), 4W (22kW), bus (60kW DC).
- BESCOM ToU tariffs: off-peak 22-06, mid-peak 06-10 & 14-18, on-peak 10-14 & 18-22.

Rules:
1. Use only the computed JSON context provided by the dashboard.
2. Do not invent data, forecasts, zone names, or percentages.
3. Do not run prediction or optimization.
4. If data is missing, say what is available and what is missing.
5. Keep answers concise and useful for infrastructure planners.
6. Reference utilisation %, stress labels, tariff multipliers, and CO2 estimates.
7. Recommendations are planning guidance, not grid-control commands.
""".strip()

MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 1.5


class VidyutLLM:
    """Thin wrapper around Gemini (via OpenAI-compatible endpoint) with fallback."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = os.getenv("GEMINI_API_KEY")
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

        peak_red = metrics.get("peak_reduction_pct", "n/a")
        var_red = metrics.get("variance_reduction_pct", "n/a")
        stress = metrics.get("stress_label_after", "n/a")
        cost = metrics.get("estimated_cost_savings_inr", "n/a")
        co2 = metrics.get("co2_reduction_kg", "n/a")
        sel_time = context.get("selected_time", "selected time")

        lines = [
            "Gemini API key not configured — local grounded explanation.",
            f"\nAt {sel_time}: peak reduction {peak_red}%, variance reduction {var_red}%.",
            f"Grid stress: {stress}.",
        ]
        if cost != "n/a":
            lines.append(f"Est. cost savings: ₹{cost}. CO₂ reduction: {co2} kg.")
        if top_zones:
            z = top_zones[0]
            lines.append(
                f"Top demand: {z.get('zone_name')} ({z.get('zone_type')}) "
                f"— {z.get('baseline_ev_load_kw')} kW unmanaged."
            )
        if risk_zones:
            r = risk_zones[0]
            lines.append(
                f"Top risk: {r.get('zone_name')} — utilisation "
                f"{r.get('optimized_transformer_utilization')}."
            )
        lines.append("Recommendations are planning guidance, not grid-control commands.")
        return "\n".join(lines)

    def _call_gemini(self, user_input: str) -> str:
        """Chat Completions call with exponential-backoff retry."""
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
                    temperature=0.3,
                )
                text = resp.choices[0].message.content if resp.choices else None
                return text.strip() if text else "The LLM returned an empty response."
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY_S * (2 ** attempt))
        raise last_exc  # type: ignore[misc]

    def answer(self, question: str, context: Dict[str, object]) -> str:
        if not question.strip():
            return "Please ask a question about the displayed forecast, optimizer result, or risk zones."
        if not self.enabled or self._client is None:
            return self._fallback_answer(question, context)

        payload = {"question": question, "computed_context": context}
        user_input = (
            "Answer the planner's question using only this computed dashboard context.\n\n"
            + json.dumps(payload, indent=2, default=str)
        )
        try:
            return self._call_gemini(user_input)
        except Exception as exc:
            return (
                "Gemini API call failed — local grounded explanation instead.\n"
                f"API error: {exc}\n\n"
                + self._fallback_answer(question, context)
            )