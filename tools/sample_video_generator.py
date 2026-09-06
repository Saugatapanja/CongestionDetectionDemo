"""Synthetic Sample Video Generator for Traffic and Festival Pandal Crowds.

Generates realistic animated test videos (.mp4) that simulate:
1. Traffic flow transitioning into a severe vehicle bottleneck.
2. Overhead top-down pandal crowd gathering at the main sanctum.
"""

import os
import cv2
import numpy as np


def generate_traffic_sample(
    output_path: str = "data/sample_videos/traffic_demo.mp4",
    duration_sec: int = 8,
    fps: int = 20,
    width: int = 1024,
    height: int = 576,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    # Generate vehicle objects (x, y, speed, length, width, color)
    vehicles = []
    colors = [(40, 40, 220), (200, 150, 40), (60, 180, 60), (230, 230, 230), (20, 120, 200)]

    for i in range(14):
        v_type = "car" if i % 4 != 0 else "bus"
        vw = 75 if v_type == "car" else 120
        vh = 40 if v_type == "car" else 48
        lane = i % 3
        y = 300 + lane * 75
        x = -150 - i * 140
        speed = 5.0 + (i % 3) * 1.5
        vehicles.append({
            "x": float(x),
            "y": float(y),
            "base_speed": speed,
            "speed": speed,
            "w": vw,
            "h": vh,
            "color": colors[i % len(colors)],
            "type": v_type,
        })

    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background: asphalt road
        frame[:] = (70, 70, 70)

        # Sidewalks
        frame[:220, :] = (40, 90, 40)
        frame[530:, :] = (50, 50, 50)

        # Road boundaries
        cv2.line(frame, (0, 220), (width, 220), (255, 255, 255), 4)
        cv2.line(frame, (0, 530), (width, 530), (255, 255, 255), 4)

        # Lane dividers (dashed)
        dash_w = 40
        dash_gap = 40
        for y_lane in [320, 425]:
            for x_start in range(0, width, dash_w + dash_gap):
                cv2.line(frame, (x_start, y_lane), (x_start + dash_w, y_lane), (255, 255, 255), 2)

        # Bottleneck scenario: After frame 60, lead vehicle slows down and stops
        is_jam_time = f > (total_frames * 0.35)

        for i, v in enumerate(vehicles):
            if is_jam_time:
                # Cars slow down near x=750 to simulate red light / obstacle
                dist_to_block = 750 - v["x"]
                if 0 < dist_to_block < 350:
                    v["speed"] = max(0.2, v["speed"] * 0.92)
                elif v["x"] >= 750:
                    v["speed"] = 0.0

            v["x"] += v["speed"]
            vx = int(v["x"])
            vy = int(v["y"])

            # Draw vehicle body
            if -150 < vx < width + 150:
                cv2.rectangle(frame, (vx, vy), (vx + v["w"], vy + v["h"]), v["color"], -1)
                # Windshield & windows
                cv2.rectangle(frame, (vx + int(v["w"] * 0.65), vy + 4), (vx + int(v["w"] * 0.85), vy + v["h"] - 4), (220, 220, 220), -1)
                cv2.rectangle(frame, (vx, vy), (vx + v["w"], vy + v["h"]), (20, 20, 20), 2)
                # Headlights
                cv2.circle(frame, (vx + v["w"], vy + 8), 4, (0, 255, 255), -1)
                cv2.circle(frame, (vx + v["w"], vy + v["h"] - 8), 4, (0, 255, 255), -1)

        cv2.putText(frame, "SIMULATED TRAFFIC CCTV STREAM", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    print(f"Generated traffic sample video: {output_path}")
    return output_path


def generate_crowd_sample(
    output_path: str = "data/sample_videos/crowd_demo.mp4",
    duration_sec: int = 8,
    fps: int = 20,
    width: int = 1024,
    height: int = 576,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    # Generate crowd individuals (overhead view: heads and shoulders)
    np.random.seed(42)
    people = []
    for _ in range(65):
        # Entry queue or sanctum initial positions
        zone_choice = np.random.choice(["queue", "sanctum", "exit"], p=[0.4, 0.45, 0.15])
        if zone_choice == "queue":
            px = np.random.uniform(70, 320)
            py = np.random.uniform(340, 520)
            vx, vy = np.random.uniform(0.5, 1.5), np.random.uniform(-0.2, 0.2)
        elif zone_choice == "sanctum":
            px = np.random.uniform(380, 880)
            py = np.random.uniform(220, 550)
            vx, vy = np.random.uniform(-0.2, 0.5), np.random.uniform(-0.3, 0.3)
        else:
            px = np.random.uniform(920, 1000)
            py = np.random.uniform(320, 520)
            vx, vy = np.random.uniform(0.8, 1.8), np.random.uniform(-0.2, 0.2)

        hair_color = (np.random.randint(10, 40), np.random.randint(10, 40), np.random.randint(10, 40))
        shirt_color = (np.random.randint(50, 240), np.random.randint(50, 240), np.random.randint(50, 240))
        people.append({
            "x": px,
            "y": py,
            "vx": vx,
            "vy": vy,
            "hair": hair_color,
            "shirt": shirt_color,
            "radius": np.random.randint(9, 13),
        })

    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Pandal flooring / carpet
        frame[:] = (40, 30, 50)

        # Draw Pandal Structure Outline
        # Sanctum stage
        cv2.rectangle(frame, (420, 30), (840, 190), (30, 60, 150), -1)
        cv2.putText(frame, "MAA DURGA SANCTUM STAGE", (460, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 100), 2)

        # Barricades
        cv2.line(frame, (350, 200), (350, 560), (0, 215, 255), 4)  # Left barricade
        cv2.line(frame, (900, 200), (900, 560), (0, 215, 255), 4)  # Right barricade

        # Zone labels
        cv2.putText(frame, "ENTRY QUEUE", (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "MAIN VIEWING AREA", (520, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "EXIT PASSAGE", (910, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Surge simulation: As time advances, more people crowd into sanctum and slow down
        surge_active = f > (total_frames * 0.4)

        for p in people:
            if surge_active and 380 <= p["x"] <= 880:
                # Slow down to take darshan/photos
                p["vx"] *= 0.98
                p["vy"] *= 0.98

            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # Wrap around from exit back to queue to maintain continuous flow
            if p["x"] > width + 20:
                p["x"] = -10
                p["y"] = np.random.uniform(340, 500)
                p["vx"] = np.random.uniform(1.0, 2.0)

            cx = int(p["x"])
            cy = int(p["y"])

            # Draw overhead person: Head (circle) and shoulders (ellipse)
            cv2.ellipse(frame, (cx, cy), (int(p["radius"] * 1.6), int(p["radius"] * 1.1)), 0, 0, 360, p["shirt"], -1)
            cv2.circle(frame, (cx, cy), p["radius"], p["hair"], -1)

        cv2.putText(frame, "DURGA PUJA PANDAL - OVERHEAD CCTV CAM 04", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    print(f"Generated crowd sample video: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_traffic_sample()
    generate_crowd_sample()
