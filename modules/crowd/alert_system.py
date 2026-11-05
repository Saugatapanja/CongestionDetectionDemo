"""Festival Pandal Crowd Safety Alert System for Police Field Teams."""

from datetime import datetime
import json
import os
import time
from typing import Dict, List


class CrowdAlertSystem:
    def __init__(self, log_file: str = "outputs/crowd_alerts.json", alert_cooldown_sec: float = 30.0):
        self.log_file = log_file
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.alert_cooldown_sec = max(0.0, alert_cooldown_sec)
        self.alert_history: List[Dict] = []
        self._last_alert_time: Dict[str, float] = {}

    def evaluate_safety_risks(self, zone_stats: List[Dict]) -> List[Dict]:
        """Examine real-time zonal statistics and determine police action alerts.

        Rules:
        1. Escalate exit-corridor alerts only when its measured zone level rises.
        2. If Main Sanctum is critical, hold queue at entry barricades immediately.
        3. Scale entry-queue guidance with its measured level.
        """
        now = datetime.now()
        now_monotonic = time.monotonic()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        active_alerts: List[Dict] = []

        zone_dict = {z["id"]: z for z in zone_stats}

        def observed_load(zone: Dict) -> str:
            """Describe a density trigger without mislabelling a count limit."""
            if zone.get("count_limit_triggered"):
                return f"calibrated camera count limit reached ({zone['head_count']} detected people)"
            return f"density reached {zone['density_per_m2']} persons/m² ({zone['head_count']} people)"

        # Check Exit Corridor
        if "exit_corridor" in zone_dict:
            exit_zone = zone_dict["exit_corridor"]
            if exit_zone["level"] >= 3:
                alert = {
                    "timestamp": timestamp_str,
                    "severity": "CRITICAL",
                    "zone_id": "exit_corridor",
                    "title": "EMERGENCY: EXIT CORRIDOR BLOCKAGE",
                    "police_action": (
                        "Deploy rapid-response officers to clear the exit immediately; "
                        f"{observed_load(exit_zone)}."
                    ),
                    "details": f"Exit corridor {observed_load(exit_zone)}.",
                }
                active_alerts.append(alert)
            elif exit_zone["level"] == 2:
                alert = {
                    "timestamp": timestamp_str,
                    "severity": "WARNING",
                    "zone_id": "exit_corridor",
                    "title": "EXIT FLOW ADVISORY",
                    "police_action": (
                        f"Monitor exit flow and prevent queues from forming; {observed_load(exit_zone)}."
                    ),
                    "details": f"Exit corridor crowding detected: {observed_load(exit_zone)}.",
                }
                active_alerts.append(alert)

        # Check Main Sanctum / Idol viewing area
        if "main_sanctum" in zone_dict:
            sanctum = zone_dict["main_sanctum"]
            if sanctum["level"] >= 3:
                alert = {
                    "timestamp": timestamp_str,
                    "severity": "HIGH_ALERT",
                    "zone_id": "main_sanctum",
                    "title": "HOLD ENTRY QUEUE BARRICADE",
                    "police_action": (
                        "Temporarily hold intake at the outer barricade until the sanctum load falls; "
                        f"{observed_load(sanctum)}."
                    ),
                    "details": f"Main viewing sanctum load: {observed_load(sanctum)}.",
                }
                active_alerts.append(alert)
            elif sanctum["level"] == 2:
                alert = {
                    "timestamp": timestamp_str,
                    "severity": "WARNING",
                    "zone_id": "main_sanctum",
                    "title": "SANCTUM FLOW ADVISORY",
                    "police_action": (
                        "Encourage steady movement in front of the idols; "
                        f"{observed_load(sanctum)}."
                    ),
                    "details": f"Moderate sanctum congestion: {sanctum['density_per_m2']} persons/m².",
                }
                active_alerts.append(alert)

        # Check Entry Queue
        if "entry_queue" in zone_dict:
            queue = zone_dict["entry_queue"]
            if queue["level"] == 3:
                alert = {
                    "timestamp": timestamp_str,
                    "severity": "HIGH_ALERT",
                    "zone_id": "entry_queue",
                    "title": "ENTRY QUEUE SPILLOVER",
                    "police_action": (
                        "Open the secondary holding area and redirect arrivals; "
                        f"{observed_load(queue)}."
                    ),
                    "details": f"Entry queue is saturated: {observed_load(queue)}.",
                }
                active_alerts.append(alert)
            elif queue["level"] == 2:
                alert = {
                    "timestamp": timestamp_str,
                    "severity": "WARNING",
                    "zone_id": "entry_queue",
                    "title": "ENTRY QUEUE BUILDUP",
                    "police_action": (
                        f"Spread arrivals across the available queue lanes; {observed_load(queue)}."
                    ),
                    "details": f"Entry queue is building: {observed_load(queue)}.",
                }
                active_alerts.append(alert)

        # Select the live highest-risk condition first instead of always showing
        # whichever zone's rule happened to run first.
        severity_rank = {"CRITICAL": 3, "HIGH_ALERT": 2, "WARNING": 1}
        active_alerts.sort(
            key=lambda alert: (
                severity_rank.get(alert["severity"], 0),
                zone_dict[alert["zone_id"]]["level"],
                zone_dict[alert["zone_id"]]["head_count"],
            ),
            reverse=True,
        )

        # Keep the live result current, but avoid writing the same sustained alert
        # on every processed frame.
        new_alerts = []
        for alert in active_alerts:
            alert_key = f"{alert['zone_id']}:{alert['title']}"
            if now_monotonic - self._last_alert_time.get(alert_key, float("-inf")) >= self.alert_cooldown_sec:
                self._last_alert_time[alert_key] = now_monotonic
                new_alerts.append(alert)

        if new_alerts:
            self._save_alerts(new_alerts)

        return active_alerts

    def _save_alerts(self, alerts: List[Dict]):
        self.alert_history.extend(alerts)
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.alert_history[-100:], f, indent=2)
        except Exception:
            pass
