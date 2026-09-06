"""Traffic Congestion and Route Diversion Package."""
from modules.traffic.vehicle_detector import VehicleDetector
from modules.traffic.congestion_meter import CongestionMeter
from modules.traffic.route_recommender import RouteRecommender

__all__ = ["VehicleDetector", "CongestionMeter", "RouteRecommender"]
