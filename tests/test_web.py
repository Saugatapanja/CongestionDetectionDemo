"""Unit tests for FastAPI Command Center Web Endpoints."""

import pytest
from fastapi.testclient import TestClient
from web.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Police Command Center" in response.text
    assert "videoFeed" in response.text


def test_telemetry_endpoint(client):
    response = client.get("/api/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "mode" in data
    assert "traffic" in data
    assert "crowd" in data
    assert "level_of_service" in data["traffic"]
    assert "zones" in data["crowd"]


def test_mode_switch(client):
    # Switch to crowd
    res_crowd = client.post("/api/set_mode", json={"mode": "crowd"})
    assert res_crowd.status_code == 200
    assert res_crowd.json()["mode"] == "crowd"

    # Switch back to traffic
    res_traffic = client.post("/api/set_mode", json={"mode": "traffic"})
    assert res_traffic.status_code == 200
    assert res_traffic.json()["mode"] == "traffic"


def test_map_endpoint(client):
    response = client.get("/api/map")
    assert response.status_code == 200


def test_kolkata_nodes_endpoint(client):
    response = client.get("/api/kolkata_nodes")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert len(data["nodes"]) >= 5
    assert any(n["id"] == "Shyambazar_5Point" for n in data["nodes"])
    assert any(n["id"] == "Park_Circus_7Point" for n in data["nodes"])
    assert "popular_roads" in data


def test_video_upload_with_kolkata_options(client, tmp_path):
    fake_video = tmp_path / "test_clip.mp4"
    fake_video.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

    with open(fake_video, "rb") as f:
        res = client.post(
            "/api/upload",
            files={"file": ("test_clip.mp4", f, "video/mp4")},
            data={
                "mode": "traffic",
                "road_name": "Eastern Metropolitan (EM) Bypass",
                "source_junction": "Ultadanga_Junction",
                "dest_junction": "EM_Bypass_ScienceCity",
                "roi_mode": "full",
            },
        )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["road_name"] == "Eastern Metropolitan (EM) Bypass"
    assert data["source_junction"] == "Ultadanga_Junction"
    assert data["dest_junction"] == "EM_Bypass_ScienceCity"
    assert data["roi_mode"] == "full"
