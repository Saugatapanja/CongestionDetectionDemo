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
        1. Exit corridor must NEVER be blocked (highest stampede risk).
        2. If Main Sanctum is critical, hold queue at entry barricades immediately.
        3. If Entry queue is backing up, advise deployment of secondary queue lines.
        """
        now = datetime.now()
        now_monotonic = time.monotonic()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        active_alerts: List[Dict] = []

        zone_dict = {z["id"]: z for z in zone_stats}

        # Check Exit Corridor
        if "exit_corridor" in zone_dict:
            exit_zone = zone_dict["exit_corridor"]
            if exit_zone["level"] >= 2:
                alert = {
                    "timestamp": timestamp_str,
                    "severity": "CRITICAL",
                    "zone_id": "exit_corridor",
                    "title": "EMERGENCY: EXIT CORRIDOR RESTRICTION",
                    "police_action": "Deploy rapid-response officers to clear exit bottlenecks immediately to avoid stampede pressure.",
                    "details": f"Exit corridor density reached {exit_zone['density_per_m2']} persons/m² ({exit_zone['head_count']} people).",
                }
                active_alerts.append(alert)

        # Check Main Sanctum / Idol viewing area
        if "main_sanctum" in zone_dict:
            sanctum = zone_dict["main_sanctum"]
            if sanctum["level"] == 3 or sanctum["head_count"] > 5:
                alert = {
                    "timestamp": timestamp_str,
                    "severity": "HIGH_ALERT",
                    "zone_id": "main_sanctum",
                    "title": "HOLD ENTRY QUEUE BARRICADE",
                    "police_action": "Temporarily stop intake at outer entry barricades for 3-5 minutes until sanctum clears.",
                    "details": f"Main viewing sanctum at maximum capacity: {sanctum['density_per_m2']} persons/m² ({sanctum['head_count']} people inside).",
                }
                active_alerts.append(alert)
            elif sanctum["level"] == 2:
                alert = {
                    "timestamp": timestamp_str,
                    "severity": "WARNING",
                    "zone_id": "main_sanctum",
                    "title": "SANCTUM FLOW ADVISORY",
                    "police_action": "Speed up crowd flow in front of idols via public address announcements.",
                    "details": f"Moderate congestion: {sanctum['density_per_m2']} persons/m².",
                }
                active_alerts.append(alert)

        # Check Entry Queue
        if "entry_queue" in zone_dict:
            queue = zone_dict["entry_queue"]
            if queue["level"] == 3:
                alert = {
                    "timestamp": timestamp_str,
                    "severity": "WARNING",
                    "zone_id": "entry_queue",
                    "title": "ENTRY QUEUE SPILLOVER",
                    "police_action": "Open secondary barricaded zigzag holding area on approach road.",
                    "details": f"Entry queue is saturated: {queue['head_count']} people waiting.",
                }
                active_alerts.append(alert)

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
