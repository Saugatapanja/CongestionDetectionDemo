"""Unit tests for Dynamic Route Recommender, Multi-Factor Confidence Scoring, and Folium Map Generation."""

import os
from modules.traffic.route_recommender import RouteRecommender


def test_route_recommender_uncongested():
    recommender = RouteRecommender()
    routes = recommender.find_alternate_routes("Shyambazar_5Point", "Park_Circus_7Point", top_k=2)
    assert len(routes) >= 1
    # Without congestion, primary route through Central Avenue / Girish Park is fastest
    best = routes[0]
    assert "Central_Ave_GirishPark" in best["path_nodes"]
    assert best["contains_congested_link"] is False


def test_route_recommender_under_congestion_with_confidence():
    recommender = RouteRecommender()
    # Congest monitored CCTV link between Shyambazar and Central Ave
    recommender.update_segment_congestion(("Shyambazar_5Point", "Central_Ave_GirishPark"), is_congested=True)

    routes = recommender.find_alternate_routes("Shyambazar_5Point", "Park_Circus_7Point", top_k=2)
    assert len(routes) >= 1

    # Under congestion, the top recommended route must avoid Central Avenue
    rec_route = next(r for r in routes if r["is_recommended"])
    assert "Central_Ave_GirishPark" not in rec_route["path_nodes"]
    assert ("Sealdah_Flyover" in rec_route["path_nodes"]) or ("EM_Bypass_ScienceCity" in rec_route["path_nodes"])

    # Multi-factor confidence score validation
    assert 50.0 <= rec_route["confidence_score"] <= 100.0
    assert "time_score" in rec_route["confidence_breakdown"]
    assert "capacity_score" in rec_route["confidence_breakdown"]
    assert rec_route["time_saved_min"] > 0
    assert "KOLKATA POLICE DIRECTIVE" in rec_route["police_action"]
    assert rec_route["suitability_tag"] in ["ALL_CLASSES", "LIGHT_ONLY"]

    # Test Folium Map output
    map_path = "outputs/test_diversion_map.html"
    out_file = recommender.generate_folium_map(routes, ("Shyambazar_5Point", "Central_Ave_GirishPark"), map_path)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 500


def test_vehicle_class_routing():
    recommender = RouteRecommender()
    recommender.update_segment_congestion(("Shyambazar_5Point", "Central_Ave_GirishPark"), is_congested=True)

    # Route specifically for heavy trucks
    truck_routes = recommender.find_alternate_routes("Shyambazar_5Point", "Park_Circus_7Point", top_k=2, target_vehicle_class="truck")
    assert len(truck_routes) >= 1
    best_truck_route = truck_routes[0]
    # Heavy trucks must be routed via the Eastern Metropolitan Bypass (EM_Bypass_ScienceCity) which permits trucks
    assert "EM_Bypass_ScienceCity" in best_truck_route["path_nodes"]
    assert best_truck_route["confidence_score"] > 60.0


def test_route_configuration_and_node_validation():
    recommender = RouteRecommender(penalty_factor=20.0)
    recommender.update_segment_congestion(("Shyambazar_5Point", "Central_Ave_GirishPark"), True)
    edge = recommender.graph["Shyambazar_5Point"]["Central_Ave_GirishPark"]
    assert edge["current_time"] == edge["base_time"] * 20.0

    try:
        recommender.find_alternate_routes("missing", "Park_Circus_7Point")
    except ValueError as error:
        assert "Unknown route node" in str(error)
    else:
        raise AssertionError("Missing route nodes should raise ValueError")


def test_video_occupancy_updates_bpr_route_cost():
    recommender = RouteRecommender(penalty_factor=15.0)
    edge = recommender.graph["Shyambazar_5Point"]["Central_Ave_GirishPark"]
    base_time = edge["base_time"]
    recommender.update_segment_congestion(
        ("Shyambazar_5Point", "Central_Ave_GirishPark"), True, congestion_ratio=0.8
    )
    assert edge["current_time"] > base_time
    assert edge["current_time"] < base_time * 15.0
    routes = recommender.find_alternate_routes("Shyambazar_5Point", "Park_Circus_7Point", top_k=2)
    assert routes[0]["is_recommended"] is True


def test_filename_only_map_output(tmp_path, monkeypatch):
    recommender = RouteRecommender()
    routes = recommender.find_alternate_routes()
    monkeypatch.chdir(tmp_path)
    assert recommender.generate_folium_map(routes, output_path="map.html") == "map.html"
    assert (tmp_path / "map.html").exists()
