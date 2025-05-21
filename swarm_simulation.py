import networkx as nx
import math
import random
import csv
import numpy as np
import rf_simulation as rf
# import uuid # No longer explicitly used for drone IDs as they are predefined
from enum import Enum

# --- Constants and Configuration ---

# Network Capacity Types
class NetworkCapacity(Enum):
    SMALL = {"name": "small", "value": 50e3} # 50 kbps
    MEDIUM = {"name": "medium", "value": 1e6} # 1 Mbps
    LARGE = {"name": "large", "value": 5e6} # 5 Mbps

CAPACITY_TO_BW_HZ = {
        NetworkCapacity.SMALL : 0.05e6,   #  50 kHz  → “narrow”
        NetworkCapacity.MEDIUM: 0.5e6,   # 500 kHz  → “medium”
        NetworkCapacity.LARGE : 2.5e6,   # 300 kHz  → “wide”
    }

# Relay Connectivity Configuration
class RelayConnectivityConfig(Enum):
    MINIMAL = 1     # Only M-R and R-S/A links
    CROSS_ROW = 2   # MINIMAL + R1-R3, R2-R4
    ALL_CONNECT = 3 # CROSS_ROW + R1-R2, R3-R4, R1-R4, R2-R3 (full inter-relay mesh)


# Simulation Parameters
SIM_PARAMS = {
    "network_capacity_type": NetworkCapacity.MEDIUM,
    "LINK_LENGTH_METERS": 250.0,
    "relay_connectivity_config": RelayConnectivityConfig.MINIMAL, # Enum member
    "ew_location": (10000.0, 0.0),
    "ew_power_W": 100.0,
    "BANDWIDTH_MHZ_RANGE": (400, 1200),
    "EW_JAMMER_BW_AREA_SELECTION": "random",
    "start_point": (0.0, 0.0),
    "end_point": (10000.0, 0.0),
    "step_size_m": 10.0,
    "max_steps": 1000,
    "logging_enabled": True,
    "csv_output_enabled": True,
    "csv_filename": "swarm_simulation_log_v3.csv",
}

# --- Formation Unit Coordinates ---
K_FACTOR = 1.0 / math.sqrt(5.0)
UNIT_COORDS_RELATIVE_TO_LEADER = {
    "M1": (0.0, 0.0), "R1": (K_FACTOR * 1, K_FACTOR * 2), "R2": (K_FACTOR * 1, K_FACTOR * -2),
    "S1": (K_FACTOR * 3, K_FACTOR * 3), "S2": (K_FACTOR * 3, K_FACTOR * 1),
    "S3": (K_FACTOR * 3, K_FACTOR * -1),"S4": (K_FACTOR * 3, K_FACTOR * -3),
    "R3": (K_FACTOR * -1, K_FACTOR * 2),"R4": (K_FACTOR * -1, K_FACTOR * -2),
    "A1": (K_FACTOR * -3, K_FACTOR * 3),"A2": (K_FACTOR * -3, K_FACTOR * 1),
    "A3": (K_FACTOR * -3, K_FACTOR * -1),"A4": (K_FACTOR * -3, K_FACTOR * -3),
}

# --- Edge Properties by Scenario ---
# Defines 'bw_area' (1-4) and 'safety_factor' (multiplier for base_capacity) for each edge.
# Edges are defined as sorted tuples of drone IDs.
EDGE_PROPERTIES_BY_SCENARIO = {
    RelayConnectivityConfig.MINIMAL: {
        ("M1", "R1"): {"bw_area": 1, "safety_factor": 1.0},
        ("M1", "R2"): {"bw_area": 2, "safety_factor": 1.0},
        ("M1", "R3"): {"bw_area": 4, "safety_factor": 1.0},
        ("M1", "R4"): {"bw_area": 3, "safety_factor": 1.0},
        ("R1", "S1"): {"bw_area": 2, "safety_factor": 0.5},
        ("R1", "S2"): {"bw_area": 3, "safety_factor": 0.5},
        ("R2", "S3"): {"bw_area": 1, "safety_factor": 0.5},
        ("R2", "S4"): {"bw_area": 4, "safety_factor": 0.5},
        ("A1", "R3"): {"bw_area": 2, "safety_factor": 0.5},
        ("A2", "R3"): {"bw_area": 3, "safety_factor": 0.5},
        ("A3", "R4"): {"bw_area": 1, "safety_factor": 0.5},
        ("A4", "R4"): {"bw_area": 4, "safety_factor": 0.5},
    },
    RelayConnectivityConfig.CROSS_ROW: {
        ("M1", "R1"): {"bw_area": 1, "safety_factor": 1.0},
        ("M1", "R2"): {"bw_area": 2, "safety_factor": 1.0},
        ("M1", "R3"): {"bw_area": 4, "safety_factor": 1.0},
        ("M1", "R4"): {"bw_area": 3, "safety_factor": 1.0},
        ("R1", "S1"): {"bw_area": 2, "safety_factor": 0.25},
        ("R1", "S2"): {"bw_area": 3, "safety_factor": 0.25},
        ("R2", "S3"): {"bw_area": 1, "safety_factor": 0.25},
        ("R2", "S4"): {"bw_area": 4, "safety_factor": 0.25},
        ("A1", "R3"): {"bw_area": 2, "safety_factor": 0.25},
        ("A2", "R3"): {"bw_area": 3, "safety_factor": 0.25},
        ("A3", "R4"): {"bw_area": 1, "safety_factor": 0.25},
        ("A4", "R4"): {"bw_area": 4, "safety_factor": 0.25},
        ("R1", "R3"): {"bw_area": 2, "safety_factor": 0.5}, # Cross-row
        ("R2", "R4"): {"bw_area": 4, "safety_factor": 0.5}, # Cross-row
    },
    RelayConnectivityConfig.ALL_CONNECT: {
        ("M1", "R1"): {"bw_area": 1, "safety_factor": 1.0},
        ("M1", "R2"): {"bw_area": 2, "safety_factor": 1.0},
        ("M1", "R3"): {"bw_area": 4, "safety_factor": 1.0},
        ("M1", "R4"): {"bw_area": 3, "safety_factor": 1.0},
        ("R1", "S1"): {"bw_area": 2, "safety_factor": 3.0/8.0},
        ("R1", "S2"): {"bw_area": 3, "safety_factor": 3.0/8.0},
        ("R2", "S3"): {"bw_area": 1, "safety_factor": 3.0/8.0},
        ("R2", "S4"): {"bw_area": 4, "safety_factor": 3.0/8.0},
        ("A1", "R3"): {"bw_area": 2, "safety_factor": 3.0/8.0},
        ("A2", "R3"): {"bw_area": 3, "safety_factor": 3.0/8.0},
        ("A3", "R4"): {"bw_area": 1, "safety_factor": 3.0/8.0},
        ("A4", "R4"): {"bw_area": 4, "safety_factor": 3.0/8.0},
        ("R1", "R2"): {"bw_area": 3, "safety_factor": 0.5},
        ("R3", "R4"): {"bw_area": 1, "safety_factor": 0.5},
        ("R1", "R3"): {"bw_area": 2, "safety_factor": 0.5},
        ("R2", "R4"): {"bw_area": 4, "safety_factor": 0.5},
        ("R1", "R4"): {"bw_area": 2, "safety_factor": 0.5},
        ("R2", "R3"): {"bw_area": 4, "safety_factor": 0.5},
    }
}
# Helper to get edge properties, ensuring sorted tuple for key
def get_edge_property_value(u, v, scenario_props, property_key, default_value=None):
    key = tuple(sorted((u, v)))
    if key in scenario_props:
        return scenario_props[key][property_key]
    # This block should ideally not be reached if graph construction aligns with scenario_props
    print(f"Critical Error: Edge {key} was constructed but not found in EDGE_PROPERTIES_BY_SCENARIO for property '{property_key}'. Review graph construction and scenario definitions.")
    if property_key == "bw_area":
        # Returning a random value here can mask underlying issues.
        # It's better to ensure all constructed edges are defined.
        raise ValueError(f"Undefined bw_area for constructed edge {key}")
    if property_key == "safety_factor":
        raise ValueError(f"Undefined safety_factor for constructed edge {key}")
    return default_value

# Drone Types
class DroneType(Enum):
    LEADER = "leader"; RELAY = "relay"; SENSOR = "sensor"; ATTACK = "attack"

# --- Drone Class ---
class Drone:
    def __init__(self, id_str, drone_type, initial_pos_abs):
        self.id = id_str
        self.drone_type = drone_type
        self.pos = list(initial_pos_abs)
        self.is_connected_to_leader = True
        self.initial_relative_pos_to_leader = [0.0, 0.0]

    def __repr__(self):
        return f"Drone({self.id}, {self.drone_type.value}, pos={self.pos}, connected={self.is_connected_to_leader})"
    def move(self, dx, dy):
        if self.is_connected_to_leader: self.pos[0] += dx; self.pos[1] += dy
    def move_to(self, new_pos):
        if self.is_connected_to_leader: self.pos[0] = new_pos[0]; self.pos[1] = new_pos[1]

# --- Swarm Simulation Class ---
class SwarmSimulation:
    def __init__(self, params):
        self.params = params
        self.graph = nx.Graph()
        self.drones = {}
        self.leader_id = "M1"
        self.current_step = 0
        self.log_data = []
        self.drone_ids_map = {
            "M1": DroneType.LEADER, "R1": DroneType.RELAY, "R2": DroneType.RELAY,
            "R3": DroneType.RELAY, "R4": DroneType.RELAY, "S1": DroneType.SENSOR,
            "S2": DroneType.SENSOR, "S3": DroneType.SENSOR, "S4": DroneType.SENSOR,
            "A1": DroneType.ATTACK, "A2": DroneType.ATTACK, "A3": DroneType.ATTACK,
            "A4": DroneType.ATTACK,}
        self.fixed_connections = [
            ("M1", "R1"), ("M1", "R2"), ("M1", "R3"), ("M1", "R4"),
            ("R1", "S1"), ("R1", "S2"), ("R2", "S3"), ("R2", "S4"),
            ("R3", "A1"), ("R3", "A2"), ("R4", "A3"), ("R4", "A4"),]
        self.relay_ids_ordered = ["R1", "R2", "R3", "R4"]
        if self.params["EW_JAMMER_BW_AREA_SELECTION"] == "random":
            self.jam_band_idx = random.randint(1, 4)
        else:
            self.jam_band_idx = int(self.params["EW_JAMMER_BW_AREA_SELECTION"])
        print(f"Simulation Jammer Target BW Area: {self.jam_band_idx}")
        self._setup_swarm_formation_and_graph()
        self._log_initial_parameters()

        self._init_rf_world_and_swarm()

        ew_system_edge_list = []
        base_cap_value = self.params["network_capacity_type"].value["value"]
        for u, v, data in self.graph.edges(data=True):
            ew_system_edge_list.append( (u, v, data['bw_area'], base_cap_value) )

    def _setup_swarm_formation_and_graph(self):
        leader_start_pos = list(self.params["start_point"])
        link_length = self.params["LINK_LENGTH_METERS"]
        base_network_capacity_value = self.params["network_capacity_type"].value["value"]
        # Use the enum member directly as the key for EDGE_PROPERTIES_BY_SCENARIO
        current_scenario_props = EDGE_PROPERTIES_BY_SCENARIO[self.params["relay_connectivity_config"]]

        for drone_id_str, drone_type in self.drone_ids_map.items():
            unit_rel_x, unit_rel_y = UNIT_COORDS_RELATIVE_TO_LEADER[drone_id_str]
            actual_rel_x = unit_rel_x * link_length
            actual_rel_y = unit_rel_y * link_length
            initial_pos_x = leader_start_pos[0] + actual_rel_x
            initial_pos_y = leader_start_pos[1] + actual_rel_y
            drone = Drone(drone_id_str, drone_type, (initial_pos_x, initial_pos_y))
            drone.initial_relative_pos_to_leader = [actual_rel_x, actual_rel_y]
            self.drones[drone_id_str] = drone
            self.graph.add_node(drone_id_str, drone=drone)

        # Add fixed connections (M-R, R-S/A)
        for u, v in self.fixed_connections:
            self.graph.add_edge(u,v)

        # Add inter-relay connections based on the connectivity config
        config = self.params["relay_connectivity_config"] # This is the enum member

        # R1 (Front-Left), R2 (Front-Right), R3 (Back-Left), R4 (Back-Right)
        if config == RelayConnectivityConfig.CROSS_ROW or config == RelayConnectivityConfig.ALL_CONNECT:
            # These edges exist in CROSS_ROW and ALL_CONNECT
            self.graph.add_edge(self.relay_ids_ordered[0], self.relay_ids_ordered[2]) # R1-R3
            self.graph.add_edge(self.relay_ids_ordered[1], self.relay_ids_ordered[3]) # R2-R4

        if config == RelayConnectivityConfig.ALL_CONNECT:
            # These additional edges only exist in ALL_CONNECT
            self.graph.add_edge(self.relay_ids_ordered[0], self.relay_ids_ordered[1]) # R1-R2 (Front row)
            self.graph.add_edge(self.relay_ids_ordered[2], self.relay_ids_ordered[3]) # R3-R4 (Back row)
            self.graph.add_edge(self.relay_ids_ordered[0], self.relay_ids_ordered[3]) # R1-R4 (Diagonal)
            self.graph.add_edge(self.relay_ids_ordered[1], self.relay_ids_ordered[2]) # R2-R3 (Diagonal)
        # For MINIMAL, no additional inter-relay edges are added here.

        # Initialize edge attributes: bw_area, required_safety_capacity, base_capacity
        for u, v in self.graph.edges(): # Iterate over edges that NOW correctly exist for the scenario
            edge_bw_area = get_edge_property_value(u,v,current_scenario_props, "bw_area")
            edge_safety_factor = get_edge_property_value(u,v,current_scenario_props, "safety_factor")

            self.graph.edges[u,v]['bw_area'] = edge_bw_area
            self.graph.edges[u,v]['required_safety_capacity'] = edge_safety_factor * base_network_capacity_value
            self.graph.edges[u,v]['base_capacity'] = base_network_capacity_value
            self.graph.edges[u,v]['current_capacity'] = base_network_capacity_value
            self.graph.edges[u,v]['is_active'] = True
            self.graph.edges[u,v]['is_ew_susceptible'] = False

    def _log_initial_parameters(self):
        if not self.params["logging_enabled"]: return
        param_log = {"type": "parameters", "data": {}}
        for key, value in self.params.items():
            if isinstance(value, Enum): param_log["data"][key] = value.name # Log enum name
            else: param_log["data"][key] = value
        param_log["data"]["EW_JAMMER_ACTUAL_BW_AREA"] = self.jam_band_idx
        self.log_data.append(param_log)

    def _init_rf_world_and_swarm(self):
        # find the bounding box of *all* drones in *relative* metres
        rel_xy = np.array(
            [self.drones[d_id].initial_relative_pos_to_leader            # metres
            for d_id in self.drones]
        )
        x_min_rel, y_min_rel = rel_xy.min(axis=0)
        x_max_rel, y_max_rel = rel_xy.max(axis=0)

        swarm_width  = x_max_rel - x_min_rel       # in metres
        swarm_height = y_max_rel - y_min_rel

        # leader moves from start-point → end-point (both in params)
        sx, sy = self.params["start_point"]
        ex, ey = self.params["end_point"]

        # build the RF world 
        NX = int(ex - sx + 2*swarm_width)
        NY = int(ey - sy + 2*swarm_height)                    
        self.rf_world = rf.World2D((sx - swarm_height, ex + swarm_height), (sy - swarm_width, ey + swarm_width), nx=NX, ny=NY)

        centres, sub_bw_hz = rf.build_subbands()   # all Hz now
        band_idx   = self.jam_band_idx - 1     

        # convert jammer world-coords → nearest grid index
        jx, jy  = self.params["ew_location"]
        ix = int(np.argmin(np.abs(self.rf_world.xs - jx)))
        iy = int(np.argmin(np.abs(self.rf_world.ys - jy)))

        jammer = rf.Tx2D(self.rf_world, (ix, iy),
                 freq_hz = centres[band_idx],
                 p_tx_w  = self.params["ew_power_W"],
                 bw_hz   = sub_bw_hz,
                 fspl_thresh_dbm = -200)
        
        self.rf_net = rf.Network2D([jammer])

        # ---------- build rf drones with SAME relative pattern -----------
        rf_drones: list[rf.Drone2D] = []
        rf_drone_map: dict[str, rf.Drone2D] = {}
        for d_id, sim_drone in self.drones.items():
            rel = np.array(sim_drone.initial_relative_pos_to_leader, dtype=float)
            rf_d = rf.Drone2D(d_id, rel_xy=rel / self.params["LINK_LENGTH_METERS"])
            rf_drones.append(rf_d); rf_drone_map[d_id] = rf_d
        rf_master = rf_drone_map[self.leader_id]


        # ---------- RF links for every swarm edge -------------------------
        tier       = self.params["network_capacity_type"]          # Enum member
        link_bw_hz = CAPACITY_TO_BW_HZ[tier]                       # Hz

        LINK_POWER_W = 0.1            # 100 mW per intra-swarm radio (example)
        self.rf_links: dict[tuple, rf.Link2D] = {}
        for u, v, data in self.graph.edges(data=True):
            band_idx = data['bw_area'] - 1
            lk = rf.Link2D(
                tx  = rf_drone_map[u],
                rx  = rf_drone_map[v],
                freq_hz = centres[band_idx],
                p_tx_w  = LINK_POWER_W,
                bw_hz   = link_bw_hz,             # already a Hz value
                is_susceptible = (band_idx == (self.jam_band_idx - 1))
        )
            self.rf_links[tuple(sorted((u, v)))] = lk

        # one canonical swarm object
        self.rf_swarm = rf.Swarm2D(rf_master, rf_drones,
                                    list(self.rf_links.values()),
                                    scale   = self.params["LINK_LENGTH_METERS"])

    def _calculate_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def step(self):
        self.current_step += 1
        leader_drone = self.drones[self.leader_id]
        current_pos, target_pos = leader_drone.pos, self.params["end_point"]
        dist_to_target = self._calculate_distance(current_pos, target_pos)
        dx, dy = 0.0, 0.0
        if dist_to_target < self.params["step_size_m"]:
            if dist_to_target > 1e-6 : dx, dy = target_pos[0] - current_pos[0], target_pos[1] - current_pos[1]
        elif dist_to_target > 1e-6:
            dir_x, dir_y = (target_pos[0] - current_pos[0]) / dist_to_target, (target_pos[1] - current_pos[1]) / dist_to_target
            dx, dy = dir_x * self.params["step_size_m"], dir_y * self.params["step_size_m"]
        leader_drone.move(dx, dy)
        for drone_id_str, drone_obj in self.drones.items():
            if drone_id_str == self.leader_id: continue
            if drone_obj.is_connected_to_leader:
                new_pos_x = leader_drone.pos[0] + drone_obj.initial_relative_pos_to_leader[0]
                new_pos_y = leader_drone.pos[1] + drone_obj.initial_relative_pos_to_leader[1]
                drone_obj.move_to((new_pos_x, new_pos_y))

        ew_outputs_for_edges = self.rf_swarm.update(
            np.array(self.drones[self.leader_id].pos),
            self.rf_net)

        active_graph_edges = []
        for u, v, edge_data in self.graph.edges(data=True):
            edge_key = tuple(sorted((u,v)))
            ew_result = ew_outputs_for_edges.get(edge_key)
            if ew_result:
                edge_data.update({
                    'is_ew_susceptible': ew_result['is_susceptible'],
                    'current_capacity': ew_result['current_max_capacity_after_ew'],
                    'is_active': ew_result['current_max_capacity_after_ew'] >= edge_data['required_safety_capacity']})
            else: # Should not happen if ew_system.swarm_edges_info covers all graph edges
                print(f"Warning: No EW result for edge {edge_key}. Defaulting to active/not susceptible.")
                edge_data.update({'is_ew_susceptible': False, 'current_capacity': edge_data['base_capacity'],'is_active': True})
            if edge_data['is_active']: active_graph_edges.append((u,v))
        temp_active_graph = nx.Graph()
        temp_active_graph.add_nodes_from(self.graph.nodes())
        temp_active_graph.add_edges_from(active_graph_edges)
        for drone_obj in self.drones.values(): drone_obj.is_connected_to_leader = False
        if self.leader_id in temp_active_graph:
            try:
                for node_id in nx.node_connected_component(temp_active_graph, self.leader_id):
                    if node_id in self.drones: self.drones[node_id].is_connected_to_leader = True
            except (nx.NetworkXError, KeyError):
                if self.leader_id in self.drones: self.drones[self.leader_id].is_connected_to_leader = True
        if self.params["logging_enabled"]: self._log_step_data()
        num_connected = sum(1 for d in self.drones.values() if d.is_connected_to_leader)
        leader_at_target = self._calculate_distance(leader_drone.pos, self.params["end_point"]) < 1e-3
        if leader_at_target: print(f"End: Leader reached target at step {self.current_step}."); return True
        if num_connected == 0 and len(self.drones) > 0: print(f"End: All drones disconnected at step {self.current_step}."); return True
        if num_connected == 1 and self.drones[self.leader_id].is_connected_to_leader and len(self.drones) > 1:
            print(f"End: Only leader M1 connected at step {self.current_step}."); return True
        if self.current_step >= self.params["max_steps"]: print(f"End: Max steps ({self.params['max_steps']}) reached."); return True
        return False

    def _log_step_data(self):
        leader_drone = self.drones[self.leader_id]
        step_log = {"type": "step_data", "step": self.current_step,
                    "leader_pos": list(leader_drone.pos),
                    "connected_drones_count": sum(1 for d in self.drones.values() if d.is_connected_to_leader),
                    "disabled_drones_ids": [id_str for id_str,d in self.drones.items() if not d.is_connected_to_leader],
                    "edges": []}
        for u,v,data in self.graph.edges(data=True):
            step_log["edges"].append({"u":u,"v":v,"is_active":data['is_active'],
                                     "current_capacity":data['current_capacity'],
                                     "required_safety_capacity":data['required_safety_capacity'],
                                     "bw_area":data['bw_area'],
                                     "is_ew_susceptible":data['is_ew_susceptible']})
        self.log_data.append(step_log)

    def run_simulation(self):
        print(f"Starting simulation...\nLeader: {self.leader_id}, Target: {self.params['end_point']}")
        print(f"Total Drones: {len(self.drones)}, EW Location: {self.params['ew_location']}")
        if not self.graph.number_of_edges() > 0: print("No edges in the graph after setup.")
        for _ in range(self.params["max_steps"]):
            if self.step(): break
        print("Simulation finished.")
        if self.params["logging_enabled"] and self.params["csv_output_enabled"]: self.write_log_to_csv()

    def write_log_to_csv(self):
        if not self.log_data: print("No log data to write."); return
        filename = self.params["csv_filename"]; print(f"Writing log to {filename}...")
        param_data = next((item["data"] for item in self.log_data if item["type"] == "parameters"),{})
        edge_fieldnames = ['step','leader_pos_x','leader_pos_y','connected_drones_count',
                           'edge_u','edge_v','is_active','current_capacity',
                           'required_safety_capacity','bw_area','is_ew_susceptible']
        with open(filename,'w',newline='') as csvfile:
            writer = csv.writer(csvfile); writer.writerow(["Parameter","Value"])
            for k,v in param_data.items(): writer.writerow([k,v])
            writer.writerow([]); writer.writerow(edge_fieldnames)
            for log_entry in self.log_data:
                if log_entry["type"] == "step_data":
                    lp_x = log_entry["leader_pos"][0] if len(log_entry["leader_pos"]) > 0 else 'N/A'
                    lp_y = log_entry["leader_pos"][1] if len(log_entry["leader_pos"]) > 1 else 'N/A'
                    for edge_info in log_entry["edges"]:
                        writer.writerow([log_entry["step"],lp_x,lp_y,log_entry["connected_drones_count"],
                                         edge_info['u'],edge_info['v'],edge_info['is_active'],
                                         edge_info['current_capacity'],edge_info['required_safety_capacity'],
                                         edge_info['bw_area'],edge_info['is_ew_susceptible']])
        print(f"Log written to {filename}")

# --- Main Execution ---
if __name__ == "__main__":
    sim = SwarmSimulation(SIM_PARAMS)
    sim.step()
    sim.rf_swarm.plot_snapshot()
    simulation = SwarmSimulation(SIM_PARAMS)
    simulation.run_simulation()
