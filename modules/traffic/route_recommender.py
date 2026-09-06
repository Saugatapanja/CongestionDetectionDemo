"""Dynamic Alternate Route Recommendation Engine for Kolkata Traffic Police.

Features:
- Authentic Kolkata urban road graph covering 10 major transit nodes
- Multi-factor route recommendation confidence scoring (0-100%)
- Road name matching & BPR congestion penalization (e.g., Central Ave, EM Bypass, AJC Bose Rd)
- Vehicle-class-aware routing (Heavy Commercial & Buses vs Light Vehicles & Cars)
- Actionable Kolkata Traffic Police (KTP) junction deployment directives
- Interactive Folium GIS map centered on Kolkata
"""

import os
from typing import Dict, List, Optional, Tuple
import folium
import networkx as nx


class RouteRecommender:
    # Backwards compatibility aliases mapping old synthetic IDs to Kolkata nodes
    NODE_ALIASES = {
        "J1_North": "Shyambazar_5Point",
        "J2_Central": "Central_Ave_GirishPark",
        "J3_South": "Park_Circus_7Point",
        "J4_West_Link": "Sealdah_Flyover",
        "J5_East_Link": "Ultadanga_Junction",
        "J6_Mid_West": "Esplanade_Crossing",
    }

    def __init__(
        self,
        city_name: str = "Kolkata, West Bengal, India",
        penalty_factor: float = 12.0,
    ):
        self.city_name = city_name
        self.penalty_factor = penalty_factor
        self.graph = nx.DiGraph()
        self._build_kolkata_network()

    def resolve_node(self, node_id: str) -> str:
        """Resolve aliases to standard Kolkata node keys."""
        return self.NODE_ALIASES.get(node_id, node_id)

    def _build_kolkata_network(self):
        """Constructs an authentic Kolkata road graph with 10 major transit junctions."""
        # Kolkata Nodes (Lat, Lon, Label, Junction Type, Zone)
        nodes = {
            "Shyambazar_5Point": (22.6015, 88.3712, "Shyambazar 5-Point Crossing (North Gateway)", "rotary_hub", "North"),
            "Ultadanga_Junction": (22.5925, 88.3962, "Ultadanga Crossing / Hudco (North-East Hub)", "signalized_crossing", "North-East"),
            "Central_Ave_GirishPark": (22.5830, 88.3630, "Central Avenue (CR Ave / Girish Park)", "arterial_corridor", "Central"),
            "Howrah_Bridge_Approach": (22.5855, 88.3470, "Howrah Bridge / Station Approach", "bridge_terminal", "West"),
            "Esplanade_Crossing": (22.5650, 88.3520, "Esplanade / Dharmatala (Central Kolkata)", "major_crossing", "Central"),
            "Sealdah_Flyover": (22.5670, 88.3710, "Sealdah Station Hub & Flyover", "flyover_hub", "East-Central"),
            "Park_Circus_7Point": (22.5440, 88.3680, "Park Circus 7-Point Crossing (South-Central Hub)", "rotary_hub", "South-Central"),
            "Rabindra_Sadan_Exide": (22.5360, 88.3470, "Rabindra Sadan / Exide Crossing (South Gateway)", "major_junction", "South"),
            "EM_Bypass_ScienceCity": (22.5390, 88.3970, "EM Bypass / Science City Expressway", "expressway_interchange", "East"),
            "Gariahat_Junction": (22.5180, 88.3650, "Gariahat Crossing (South Kolkata Hub)", "commercial_hub", "South"),
        }

        for node_id, (lat, lon, label, jtype, zone) in nodes.items():
            self.graph.add_node(
                node_id, lat=lat, lon=lon, label=label, junction_type=jtype, zone=zone
            )

        # Edges (u, v, length_m, speed_kmh, capacity_vph, road_type, name, allowed_classes, bidirectional)
        raw_edges = [
            # 1. Central Avenue (CR Avenue) Corridor (Monitored Central Spine)
            ("Shyambazar_5Point", "Central_Ave_GirishPark", 2100, 52, 2800, "primary", "Central Avenue North (CR Ave)", ["car", "motorcycle", "bus", "truck"], True),
            ("Central_Ave_GirishPark", "Esplanade_Crossing", 2200, 50, 2800, "primary", "Central Avenue South (CR Ave)", ["car", "motorcycle", "bus", "truck"], True),
            ("Esplanade_Crossing", "Park_Circus_7Point", 2400, 48, 2600, "primary", "Park Street / Shakespeare Sarani", ["car", "motorcycle", "bus"], True),
            ("Esplanade_Crossing", "Rabindra_Sadan_Exide", 2500, 48, 2800, "primary", "Chowringhee Road (JL Nehru Rd)", ["car", "motorcycle", "bus", "truck"], True),

            # 2. Eastern Metropolitan Bypass Expressway (High-Speed Outer Bypass for All Vehicles)
            ("Shyambazar_5Point", "Ultadanga_Junction", 2600, 45, 2400, "secondary", "Ultadanga Link Road (Canal Circular)", ["car", "motorcycle", "bus", "truck"], True),
            ("Ultadanga_Junction", "EM_Bypass_ScienceCity", 6200, 60, 3600, "trunk", "Eastern Metropolitan (EM) Bypass North", ["car", "motorcycle", "bus", "truck"], True),
            ("EM_Bypass_ScienceCity", "Park_Circus_7Point", 3100, 55, 3400, "trunk", "Maa Flyover (Park Circus Connector)", ["car", "motorcycle", "bus", "truck"], True),
            ("EM_Bypass_ScienceCity", "Gariahat_Junction", 3800, 50, 3000, "trunk", "EM Bypass Connector to Ruby / Gariahat", ["car", "motorcycle", "bus", "truck"], True),

            # 3. APC Road & AJC Bose Road Corridor (Eastern Arterial Bypass)
            ("Shyambazar_5Point", "Sealdah_Flyover", 4600, 40, 2200, "secondary", "Acharya Prafulla Chandra (APC) Road", ["car", "motorcycle", "bus"], True),
            ("Sealdah_Flyover", "Park_Circus_7Point", 2800, 40, 2400, "secondary", "AJC Bose Road (Sealdah to Park Circus)", ["car", "motorcycle", "bus"], True),
            ("Park_Circus_7Point", "Rabindra_Sadan_Exide", 2900, 52, 3000, "trunk", "AJC Bose Road Flyover", ["car", "motorcycle", "bus"], True),
            ("Park_Circus_7Point", "Gariahat_Junction", 3200, 40, 2000, "secondary", "Syed Amir Ali Avenue / Gariahat Road", ["car", "motorcycle", "bus"], True),

            # 4. Riverbank & West Corridor
            ("Howrah_Bridge_Approach", "Central_Ave_GirishPark", 1600, 35, 1800, "secondary", "Mahatma Gandhi (MG) Road Link", ["car", "motorcycle", "bus", "truck"], True),
            ("Howrah_Bridge_Approach", "Esplanade_Crossing", 2400, 42, 2400, "secondary", "Strand Road (Riverfront Corridor)", ["car", "motorcycle", "bus", "truck"], True),
            ("Rabindra_Sadan_Exide", "Esplanade_Crossing", 2600, 55, 3000, "primary", "Red Road / Maidan Corridor", ["car", "motorcycle"], True),
        ]

        for u, v, length, speed, capacity, rtype, name, allowed, bidi in raw_edges:
            travel_time_sec = length / (speed * 1000.0 / 3600.0)
            self.graph.add_edge(
                u, v,
                length=length, speed=speed,
                base_time=travel_time_sec, current_time=travel_time_sec,
                capacity=capacity, road_type=rtype, name=name,
                allowed_classes=allowed, is_congested=False, congestion_ratio=0.0,
            )
            if bidi:
                self.graph.add_edge(
                    v, u,
                    length=length, speed=speed,
                    base_time=travel_time_sec, current_time=travel_time_sec,
                    capacity=capacity, road_type=rtype, name=name,
                    allowed_classes=allowed, is_congested=False, congestion_ratio=0.0,
                )

    def penalize_road_by_name(
        self,
        road_name: str,
        is_congested: bool,
        congestion_ratio: float = 1.0,
    ) -> List[Tuple[str, str]]:
        """Finds all edges matching road_name and applies BPR congestion penalty."""
        matched_edges = []
        name_clean = road_name.strip().lower()

        for u, v, data in self.graph.edges(data=True):
            edge_name = data.get("name", "").lower()
            # Match road name substring (e.g. "central", "em bypass", "ajc bose", "park street")
            if any(term in edge_name for term in name_clean.split()) or name_clean in edge_name:
                data["is_congested"] = is_congested
                if is_congested:
                    load = min(1.0, max(0.1, congestion_ratio))
                    data["current_time"] = data["base_time"] * (1.0 + (self.penalty_factor - 1.0) * (load ** 4))
                    data["congestion_ratio"] = load
                else:
                    data["current_time"] = data["base_time"]
                    data["congestion_ratio"] = 0.0
                matched_edges.append((u, v))

        # If no edge matched by name, default to Central Avenue spine
        if not matched_edges and is_congested:
            u, v = "Shyambazar_5Point", "Central_Ave_GirishPark"
            self.update_segment_congestion((u, v), True, congestion_ratio=congestion_ratio)
            matched_edges.append((u, v))

        return matched_edges

    def update_segment_congestion(
        self,
        segment: Tuple[str, str],
        is_congested: bool,
        penalty_factor: Optional[float] = None,
        congestion_ratio: Optional[float] = None,
    ):
        """Adjust specific segment travel time from live camera congestion."""
        raw_u, raw_v = segment
        u = self.resolve_node(raw_u)
        v = self.resolve_node(raw_v)

        if self.graph.has_edge(u, v):
            edge = self.graph[u][v]
            edge["is_congested"] = is_congested
            if is_congested:
                factor = self.penalty_factor if penalty_factor is None else max(1.0, penalty_factor)
                if congestion_ratio is None:
                    edge["current_time"] = edge["base_time"] * factor
                else:
                    load = min(1.0, max(0.0, congestion_ratio))
                    edge["current_time"] = edge["base_time"] * (1.0 + (factor - 1.0) * (load ** 4))
                edge["congestion_ratio"] = congestion_ratio or 1.0
            else:
                edge["current_time"] = edge["base_time"]
                edge["congestion_ratio"] = 0.0

    def calculate_route_confidence(
        self,
        route_time_sec: float,
        congested_corridor_time_sec: float,
        route_dist_m: float,
        direct_dist_m: float,
        min_capacity: int,
        target_class: str = "all",
        allowed_classes: Optional[List[str]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute an algorithmic confidence score (0-100%) for an alternate route."""
        time_saved_sec = max(0.0, congested_corridor_time_sec - route_time_sec)
        if congested_corridor_time_sec > 0:
            time_savings_ratio = time_saved_sec / congested_corridor_time_sec
            time_score = min(100.0, time_savings_ratio * 125.0)
        else:
            time_score = 50.0

        capacity_score = min(100.0, max(25.0, (min_capacity / 3600.0) * 100.0))

        dist_ratio = route_dist_m / max(1.0, direct_dist_m)
        if dist_ratio <= 1.25:
            dist_score = 100.0
        elif dist_ratio <= 1.6:
            dist_score = 80.0
        elif dist_ratio <= 2.0:
            dist_score = 60.0
        else:
            dist_score = 40.0

        if target_class == "all":
            class_score = 100.0 if ("truck" in (allowed_classes or [])) else 80.0
        elif target_class in ["truck", "bus"]:
            class_score = 100.0 if (target_class in (allowed_classes or [])) else 0.0
        else:
            class_score = 100.0

        total_confidence = (
            0.40 * time_score
            + 0.30 * capacity_score
            + 0.15 * dist_score
            + 0.15 * class_score
        )
        total_confidence = round(min(99.0, max(20.0, total_confidence)), 1)

        breakdown = {
            "time_score": round(time_score, 1),
            "capacity_score": round(capacity_score, 1),
            "distance_score": round(dist_score, 1),
            "class_score": round(class_score, 1),
        }
        return total_confidence, breakdown

    def find_alternate_routes(
        self,
        start_node: str = "Shyambazar_5Point",
        target_node: str = "Park_Circus_7Point",
        top_k: int = 2,
        target_vehicle_class: str = "all",
    ) -> List[Dict]:
        """Compute primary and alternate Kolkata routes with multi-factor confidence scoring."""
        s = self.resolve_node(start_node)
        t = self.resolve_node(target_node)

        if s not in self.graph or t not in self.graph:
            raise ValueError(
                f"Unknown route node(s): start={start_node!r}, target={target_node!r}. "
                f"Available Kolkata nodes: {', '.join(self.graph.nodes)}"
            )

        # Direct distance calculation
        try:
            direct_path = nx.shortest_path(self.graph, s, t, weight="length")
            direct_dist_m = sum(self.graph[direct_path[i]][direct_path[i + 1]]["length"] for i in range(len(direct_path) - 1))
        except Exception:
            direct_dist_m = 3500.0

        # Find travel time of default central path if available
        default_congested_time_sec = 0.0
        try:
            base_shortest = nx.shortest_path(self.graph, s, t, weight="base_time")
            default_congested_time_sec = sum(self.graph[base_shortest[i]][base_shortest[i + 1]]["current_time"] for i in range(len(base_shortest) - 1))
        except Exception:
            default_congested_time_sec = 1800.0

        routing_graph = self.graph
        if target_vehicle_class != "all":
            valid_edges = [
                (u, v) for u, v, d in self.graph.edges(data=True)
                if target_vehicle_class in d.get("allowed_classes", [])
            ]
            routing_graph = self.graph.edge_subgraph(valid_edges)

        try:
            paths = list(
                nx.shortest_simple_paths(
                    routing_graph, s, t, weight="current_time"
                )
            )[:top_k + 1]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        route_options = []
        for rank, path in enumerate(paths, 1):
            total_time_sec = 0.0
            total_dist_m = 0.0
            contains_congested_link = False
            step_instructions = []
            min_capacity = 99999
            route_allowed_classes = ["car", "motorcycle", "bus", "truck"]

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = self.graph[u][v]
                total_time_sec += edge["current_time"]
                total_dist_m += edge["length"]
                min_capacity = min(min_capacity, edge.get("capacity", 2000))
                edge_allowed = edge.get("allowed_classes", ["car"])
                route_allowed_classes = [c for c in route_allowed_classes if c in edge_allowed]

                if edge.get("is_congested", False):
                    contains_congested_link = True

                step_instructions.append(
                    f"Follow {edge['name']} from {self.graph.nodes[u]['label']} to {self.graph.nodes[v]['label']} ({edge['length']}m)"
                )

            time_saved_sec = max(0.0, default_congested_time_sec - total_time_sec)
            time_saved_min = round(time_saved_sec / 60.0, 1)

            confidence_score, conf_breakdown = self.calculate_route_confidence(
                route_time_sec=total_time_sec,
                congested_corridor_time_sec=default_congested_time_sec,
                route_dist_m=total_dist_m,
                direct_dist_m=direct_dist_m,
                min_capacity=min_capacity,
                target_class=target_vehicle_class,
                allowed_classes=route_allowed_classes,
            )

            # Determine Vehicle Suitability
            if "truck" in route_allowed_classes and "bus" in route_allowed_classes:
                suitability = "All Vehicle Classes (Heavy Freight & Buses Permitted)"
                suitability_tag = "ALL_CLASSES"
            else:
                suitability = "Light Vehicles & Cars Only (Bypass Narrow for Heavy Multi-Axle Trucks)"
                suitability_tag = "LIGHT_ONLY"

            corridor_name = self.graph[path[0]][path[1]]["name"]

            # Kolkata Traffic Police Tactical Directives
            first_junction_label = self.graph.nodes[path[0]]["label"]
            if contains_congested_link:
                police_action = f"Monitored corridor along {corridor_name} is currently experiencing severe congestion. Standard signal timings congested."
            elif "EM_Bypass" in corridor_name or "Ultadanga" in corridor_name:
                police_action = (
                    f"KOLKATA POLICE DIRECTIVE: Post traffic officers at {first_junction_label} to divert vehicles "
                    f"onto Eastern Metropolitan Bypass Expressway via Maa Flyover. High-capacity corridor (3,600 vph) saves approx. {time_saved_min} mins."
                )
            elif "AJC Bose" in corridor_name or "APC" in corridor_name:
                police_action = (
                    f"KOLKATA POLICE DIRECTIVE: Divert light vehicles and cars onto APC Road / AJC Bose Road Flyover at {first_junction_label}. "
                    f"Restricts heavy commercial vehicles to maintain flyover velocity. Saves approx. {time_saved_min} mins."
                )
            else:
                police_action = (
                    f"KOLKATA POLICE DIRECTIVE: Divert vehicles via {corridor_name} from {first_junction_label}. "
                    f"Saves approx. {time_saved_min} mins."
                )

            route_options.append({
                "rank": rank,
                "path_nodes": path,
                "corridor_name": corridor_name,
                "total_time_min": round(total_time_sec / 60.0, 1),
                "total_dist_km": round(total_dist_m / 1000.0, 2),
                "time_saved_min": time_saved_min,
                "confidence_score": confidence_score,
                "confidence_breakdown": conf_breakdown,
                "contains_congested_link": contains_congested_link,
                "is_recommended": not contains_congested_link,
                "min_capacity_vph": min_capacity,
                "vehicle_suitability": suitability,
                "suitability_tag": suitability_tag,
                "police_action": police_action,
                "instructions": step_instructions,
            })

        # Sort non-congested routes by confidence descending, take top_k
        route_options.sort(key=lambda r: (r["contains_congested_link"], -r["confidence_score"]))
        final_routes = route_options[:top_k]
        for idx, r in enumerate(final_routes, 1):
            r["rank"] = idx

        return final_routes

    def generate_folium_map(
        self,
        routes: List[Dict],
        congested_segment: Optional[Tuple[str, str]] = None,
        output_path: str = "outputs/traffic_diversion_map.html",
    ) -> str:
        """Render interactive Kolkata Police operations map with real landmark pins and styled diversion paths."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Center map at Kolkata heart
        center_lat = 22.5650
        center_lon = 88.3680

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

        # Mark Kolkata Junction Nodes
        for node, data in self.graph.nodes(data=True):
            # Collect connected road names
            connected_roads = set()
            for u, v, edge_data in self.graph.edges(data=True):
                if u == node or v == node:
                    connected_roads.add(edge_data.get("name", "Unknown Road"))
            road_names_str = ", ".join(list(connected_roads)[:3])
            if len(connected_roads) > 3:
                road_names_str += "..."

            folium.CircleMarker(
                location=[data["lat"], data["lon"]],
                radius=8,
                popup=folium.Popup(
                    f"<div style='font-family:sans-serif; min-width:180px;'>"
                    f"<b style='color:#1e3a8a; font-size:13px;'>{data['label']}</b><br>"
                    f"<span style='font-size:11px; color:#475569;'>Zone: {data.get('zone', 'Kolkata')} | Type: {data.get('junction_type')}</span>"
                    f"</div>",
                    max_width=260,
                ),
                tooltip=f"Node: {data['label']} | Roads: {road_names_str}",
                color="#1d4ed8",
                fill=True,
                fill_color="#3b82f6",
                fill_opacity=0.9,
            ).add_to(m)

        # Draw Base Network
        for u, v, edge in self.graph.edges(data=True):
            pt1 = [self.graph.nodes[u]["lat"], self.graph.nodes[u]["lon"]]
            pt2 = [self.graph.nodes[v]["lat"], self.graph.nodes[v]["lon"]]
            is_cong = edge.get("is_congested", False)
            color = "#dc2626" if is_cong else "#94a3b8"
            weight = 8 if is_cong else 3
            dash = "6, 12" if is_cong else None

            folium.PolyLine(
                [pt1, pt2],
                color=color,
                weight=weight,
                dash_array=dash,
                tooltip=f"{edge['name']} ({'CONGESTED BOTTLE-NECK' if is_cong else 'Normal Flow'})",
            ).add_to(m)

        # Draw Alternate Diversion Routes in Vibrant Green & Sky Blue
        colors = ["#16a34a", "#0284c7"]
        for idx, route in enumerate(routes):
            if route["is_recommended"]:
                path = route["path_nodes"]
                line_points = [
                    [self.graph.nodes[n]["lat"], self.graph.nodes[n]["lon"]]
                    for n in path
                ]
                color = colors[idx % len(colors)]
                conf = route["confidence_score"]
                saved = route["time_saved_min"]

                popup_html = (
                    f"<div style='font-family:sans-serif; min-width:260px;'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>"
                    f"<b style='color:{color}; font-size:14px;'>KOLKATA DIVERSION #{route['rank']}</b>"
                    f"<span style='background:#dcfce7; color:#166534; font-weight:bold; font-size:11px; padding:2px 6px; border-radius:4px;'>"
                    f"{conf}% CONFIDENCE</span>"
                    f"</div>"
                    f"<b>Corridor:</b> {route['corridor_name']}<br>"
                    f"<b>Est. Travel Time:</b> {route['total_time_min']} min "
                    f"(<span style='color:#16a34a; font-weight:bold;'>Saves {saved} min</span>)<br>"
                    f"<b>Distance:</b> {route['total_dist_km']} km<br>"
                    f"<b>Capacity:</b> {route['min_capacity_vph']} veh/hr<br>"
                    f"<b>Suitability:</b> {route['vehicle_suitability']}<br><br>"
                    f"<div style='background:#f1f5f9; padding:6px; border-left:3px solid {color}; font-size:11px;'>"
                    f"<b>Police Action:</b> {route['police_action']}"
                    f"</div>"
                    f"</div>"
                )

                folium.PolyLine(
                    line_points,
                    color=color,
                    weight=6,
                    opacity=0.9,
                    popup=folium.Popup(popup_html, max_width=330),
                    tooltip=f"Kolkata Diversion #{route['rank']}: {route['corridor_name']} ({conf}% Conf | Saves {saved}m)",
                ).add_to(m)

        m.save(output_path)
        return output_path
