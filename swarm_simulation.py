import networkx as nx
import math
import random
import csv
import numpy as np
import rf_simulation as rf
from enum import Enum
import copy # For deep copying params

# --- Constants and Configuration ---

# Network Capacity Types
class NetworkCapacity(Enum):
    SMALL = {"name": "small", "value": 50e3} # 50 kbps
    MEDIUM = {"name": "medium", "value": 0.25e6} # 250 kbps
    LARGE = {"name": "large", "value": 5e6} # 5 Mbps

CAPACITY_TO_BW_HZ = {
        NetworkCapacity.SMALL : 0.2e6,   #  200 kHz  → “narrow”
        NetworkCapacity.MEDIUM: 1.0e6,   # 1000 kHz  → “medium”
        NetworkCapacity.LARGE : 20.0e6,   # 20000 kHz  → “wide”
    }

# List of the avalable modulation types as defined in RF_simulation
modulation_types = [
    "THEORETICAL", "BPSK", "QPSK", "16QAM", "64QAM"
]

# Relay Connectivity Configuration
class RelayConnectivityConfig(Enum):
    MINIMAL = 1     # Only M-R and R-S/A links
    CROSS_ROW = 2   # MINIMAL + R1-R3, R2-R4
    ALL_CONNECT = 3 # CROSS_ROW + R1-R2, R3-R4, R1-R4, R2-R3 (full inter-relay mesh)


# Default Simulation Parameters (can be overridden by data_collector.py)
DEFAULT_SIM_PARAMS = {
    "network_capacity_type": NetworkCapacity.LARGE,
    "LINK_LENGTH_METERS": 50.0,
    "relay_connectivity_config": RelayConnectivityConfig.CROSS_ROW,
    "ew_location": (10000.0, 0.0),
    "ew_power_W": 100.0,
    "BANDWIDTH_MHZ_RANGE": (400, 1200),
    "EW_JAMMER_BW_AREA_SELECTION": "random", # 1, 2, 3, 4, or "random"
    "start_point": (0.0, 0.0),
    "end_point": (10000.0, 0.0), # Target for leader, sim might end sooner/later
    "step_size_m": 10.0,
    "FLIGHT_MARGIN_PAST_EW_M": 0, # How far the rearmost drone must pass EW_X for sim to end
    "logging_enabled": True, # For single detailed runs
    "csv_output_enabled": True, # For single detailed runs
    "csv_filename_prefix": "swarm_log_detailed_",
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
EDGE_PROPERTIES_BY_SCENARIO = {
    RelayConnectivityConfig.MINIMAL: {
        ("M1", "R1"): {"bw_area": 1, "safety_factor": 1.0}, ("M1", "R2"): {"bw_area": 2, "safety_factor": 1.0},
        ("M1", "R3"): {"bw_area": 4, "safety_factor": 1.0}, ("M1", "R4"): {"bw_area": 3, "safety_factor": 1.0},
        ("R1", "S1"): {"bw_area": 2, "safety_factor": 0.5}, ("R1", "S2"): {"bw_area": 3, "safety_factor": 0.5},
        ("R2", "S3"): {"bw_area": 1, "safety_factor": 0.5}, ("R2", "S4"): {"bw_area": 4, "safety_factor": 0.5},
        ("A1", "R3"): {"bw_area": 2, "safety_factor": 0.5}, ("A2", "R3"): {"bw_area": 3, "safety_factor": 0.5},
        ("A3", "R4"): {"bw_area": 1, "safety_factor": 0.5}, ("A4", "R4"): {"bw_area": 4, "safety_factor": 0.5},
    },
    RelayConnectivityConfig.CROSS_ROW: {
        ("M1", "R1"): {"bw_area": 1, "safety_factor": 1.0}, ("M1", "R2"): {"bw_area": 2, "safety_factor": 1.0},
        ("M1", "R3"): {"bw_area": 4, "safety_factor": 1.0}, ("M1", "R4"): {"bw_area": 3, "safety_factor": 1.0},
        ("R1", "S1"): {"bw_area": 2, "safety_factor": 0.25}, ("R1", "S2"): {"bw_area": 3, "safety_factor": 0.25},
        ("R2", "S3"): {"bw_area": 1, "safety_factor": 0.25}, ("R2", "S4"): {"bw_area": 4, "safety_factor": 0.25},
        ("A1", "R3"): {"bw_area": 2, "safety_factor": 0.25}, ("A2", "R3"): {"bw_area": 3, "safety_factor": 0.25},
        ("A3", "R4"): {"bw_area": 1, "safety_factor": 0.25}, ("A4", "R4"): {"bw_area": 4, "safety_factor": 0.25},
        ("R1", "R3"): {"bw_area": 2, "safety_factor": 0.5}, ("R2", "R4"): {"bw_area": 4, "safety_factor": 0.5},
    },
    RelayConnectivityConfig.ALL_CONNECT: {
        ("M1", "R1"): {"bw_area": 1, "safety_factor": 1.0}, ("M1", "R2"): {"bw_area": 2, "safety_factor": 1.0},
        ("M1", "R3"): {"bw_area": 4, "safety_factor": 1.0}, ("M1", "R4"): {"bw_area": 3, "safety_factor": 1.0},
        ("R1", "S1"): {"bw_area": 2, "safety_factor": 3.0/8.0}, ("R1", "S2"): {"bw_area": 3, "safety_factor": 3.0/8.0},
        ("R2", "S3"): {"bw_area": 1, "safety_factor": 3.0/8.0}, ("R2", "S4"): {"bw_area": 4, "safety_factor": 3.0/8.0},
        ("A1", "R3"): {"bw_area": 2, "safety_factor": 3.0/8.0}, ("A2", "R3"): {"bw_area": 3, "safety_factor": 3.0/8.0},
        ("A3", "R4"): {"bw_area": 1, "safety_factor": 3.0/8.0}, ("A4", "R4"): {"bw_area": 4, "safety_factor": 3.0/8.0},
        ("R1", "R2"): {"bw_area": 3, "safety_factor": 0.5}, ("R3", "R4"): {"bw_area": 1, "safety_factor": 0.5},
        ("R1", "R3"): {"bw_area": 2, "safety_factor": 0.5}, ("R2", "R4"): {"bw_area": 4, "safety_factor": 0.5},
        ("R1", "R4"): {"bw_area": 2, "safety_factor": 0.5}, ("R2", "R3"): {"bw_area": 4, "safety_factor": 0.5},
    }
}
def get_edge_property_value(u, v, scenario_props, property_key):
    key = tuple(sorted((u, v)))
    if key in scenario_props:
        if property_key in scenario_props[key]:
            return scenario_props[key][property_key]
        else:
            raise ValueError(f"Property '{property_key}' not found for edge {key} in scenario definitions.")
    raise ValueError(f"Edge {key} not found in EDGE_PROPERTIES_BY_SCENARIO. Ensure all constructed edges are defined for the current scenario: {key}")

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
    def __init__(self, params_override=None):
        self.params = copy.deepcopy(DEFAULT_SIM_PARAMS) # Start with defaults
        if params_override:
            self.params.update(params_override) # Apply overrides

        # This block ensures that capacity_requirement and link_bw_hz are always consistent.
        # The data_collector can now EITHER provide a specific 'capacity_requirement'
        # OR fall back to the high-level 'network_capacity_type'.

        if "capacity_requirement" in self.params and self.params["capacity_requirement"] is not None:
            # If a specific capacity requirement is given (e.g., from a grid search),
            # use it to derive the link bandwidth. This overrides any default.
            cap_req = self.params["capacity_requirement"]
            self.params["link_bw_hz"] = int(cap_req * 4)
            # Ensure network_capacity_type is not used to avoid conflicts later
            if "network_capacity_type" in self.params:
                del self.params["network_capacity_type"]
        else:
            # If no specific capacity requirement is given, derive both values
            # from the high-level network_capacity_type enum.
            capacity_type = self.params["network_capacity_type"]
            self.params["capacity_requirement"] = capacity_type.value["value"]
            self.params["link_bw_hz"] = CAPACITY_TO_BW_HZ[capacity_type]

        self.graph = nx.Graph()
        self.drones = {}; self.leader_id = "M1"; self.current_step = 0; self.log_data = []
        self.sim_intended_directed_flows = []
        self.drone_ids_map = {
            "M1": DroneType.LEADER, "R1": DroneType.RELAY, "R2": DroneType.RELAY, "R3": DroneType.RELAY, "R4": DroneType.RELAY,
            "S1": DroneType.SENSOR, "S2": DroneType.SENSOR, "S3": DroneType.SENSOR, "S4": DroneType.SENSOR,
            "A1": DroneType.ATTACK, "A2": DroneType.ATTACK, "A3": DroneType.ATTACK, "A4": DroneType.ATTACK,}
        self.base_physical_links = [
            ("M1", "R1"), ("M1", "R2"), ("M1", "R3"), ("M1", "R4"),
            ("R1", "S1"), ("R1", "S2"), ("R2", "S3"), ("R2", "S4"),
            ("R3", "A1"), ("R3", "A2"), ("R4", "A3"), ("R4", "A4"),]
        self.relay_ids_ordered = ["R1", "R2", "R3", "R4"]

        if self.params["EW_JAMMER_BW_AREA_SELECTION"] == "random":
            self.jam_band_idx = random.randint(1, 4)
        else: self.jam_band_idx = int(self.params["EW_JAMMER_BW_AREA_SELECTION"])
        
        # Final metrics to be returned
        self.rs1_leader_dist_to_ew = {name: np.nan for name in modulation_types}
        self.rs2_leader_dist_to_ew = {name: np.nan for name in modulation_types}
        self.ra1_leader_dist_to_ew = {name: np.nan for name in modulation_types}
        self.ra2_leader_dist_to_ew = {name: np.nan for name in modulation_types}
        
        # Intermediate dictionaries to log disconnection events for each drone individually
        self._sensor_disconnect_distances = {mod: {} for mod in modulation_types}
        self._attack_disconnect_distances = {mod: {} for mod in modulation_types}

        self.all_drones_passed_ew_x = False
        self.initial_susceptible_physical_links = set()
        self.disconnected_susceptible_physical_links = {name: set() for name in modulation_types}
        self.r1_equiv  = {mod: np.nan for mod in modulation_types}
        self.r2_equiv  = {mod: np.nan for mod in modulation_types}

        # Drone IDs by type
        self.sensor_drone_ids = [k for k, v in self.drone_ids_map.items() if v == DroneType.SENSOR]
        self.attack_drone_ids = [k for k, v in self.drone_ids_map.items() if v == DroneType.ATTACK]

        self._link_break_normed = { name: {} for name in modulation_types }

        self._setup_swarm_formation_and_graph()
        if self.params["logging_enabled"]:
            self._log_initial_parameters()
        self._init_rf_world_and_swarm() # This also populates initial_susceptible_physical_links

    def _setup_swarm_formation_and_graph(self):
        leader_start_pos = list(self.params["start_point"]); link_length = self.params["LINK_LENGTH_METERS"]

        if "capacity_requirement" in self.params:
            base_cap_val = self.params["capacity_requirement"]
        else:
            base_cap_val = self.params["network_capacity_type"].value["value"]
        current_scenario_props = EDGE_PROPERTIES_BY_SCENARIO[self.params["relay_connectivity_config"]]

        for drone_id_str, drone_type in self.drone_ids_map.items():
            unit_rel_x, unit_rel_y = UNIT_COORDS_RELATIVE_TO_LEADER[drone_id_str]
            actual_rel_x, actual_rel_y = unit_rel_x * link_length, unit_rel_y * link_length
            initial_pos_x, initial_pos_y = leader_start_pos[0] + actual_rel_x, leader_start_pos[1] + actual_rel_y
            drone = Drone(drone_id_str, drone_type, (initial_pos_x, initial_pos_y))
            drone.initial_relative_pos_to_leader = [actual_rel_x, actual_rel_y]
            self.drones[drone_id_str] = drone
            self.graph.add_node(drone_id_str, drone=drone)

        for u, v in self.base_physical_links:
            self.graph.add_edge(u,v)

        config = self.params["relay_connectivity_config"] 
        r1, r2, r3, r4 = self.relay_ids_ordered
        if config == RelayConnectivityConfig.CROSS_ROW or config == RelayConnectivityConfig.ALL_CONNECT:
            self.graph.add_edge(r1, r3)
            self.graph.add_edge(r2, r4)
        if config == RelayConnectivityConfig.ALL_CONNECT:
            self.graph.add_edge(r1, r2) 
            self.graph.add_edge(r3, r4)

        for u, v in self.graph.edges():
            edge_bw_area = get_edge_property_value(u,v,current_scenario_props, "bw_area")
            edge_safety_factor = get_edge_property_value(u,v,current_scenario_props, "safety_factor")
            pos_u = np.array(self.drones[u].initial_relative_pos_to_leader)
            pos_v = np.array(self.drones[v].initial_relative_pos_to_leader)
            L_uv = float(np.linalg.norm(pos_u - pos_v))

            self.graph.edges[u, v].update({
                'bw_area': edge_bw_area,
                'required_safety_capacity': edge_safety_factor * base_cap_val,
                'base_capacity': base_cap_val,
                'current_capacities': {name: base_cap_val for name in modulation_types},
                'is_active': {name: True for name in modulation_types},
                'is_ew_susceptible': False,
                'link_length': L_uv
            })

    def _log_initial_parameters(self):
        if not self.params["logging_enabled"]: return
        param_log = {"type": "parameters", "data": {k: (v.name if isinstance(v, Enum) else v) for k, v in self.params.items()}}
        param_log["data"]["EW_JAMMER_ACTUAL_BW_AREA"] = self.jam_band_idx
        self.log_data.append(param_log)

    def _init_rf_world_and_swarm(self):
        rel_xy = np.array(
            [self.drones[d_id].initial_relative_pos_to_leader
            for d_id in self.drones]
        )
        x_min_rel, y_min_rel = rel_xy.min(axis=0)
        x_max_rel, y_max_rel = rel_xy.max(axis=0)

        swarm_width  = x_max_rel - x_min_rel
        swarm_height = y_max_rel - y_min_rel

        sx, sy = self.params["start_point"]
        ex, ey = self.params["end_point"]

        NX = int(ex - sx + 2*swarm_width)
        NY = int(ey - sy + 2*swarm_height)                    
        self.rf_world = rf.World2D((sx - swarm_height, ex + swarm_height), (sy - swarm_width, ey + swarm_width), nx=NX, ny=NY)

        centres, sub_bw_hz = rf.build_subbands()
        band_idx   = self.jam_band_idx - 1     

        jx, jy  = self.params["ew_location"]
        ix = int(np.argmin(np.abs(self.rf_world.xs - jx)))
        iy = int(np.argmin(np.abs(self.rf_world.ys - jy)))

        jammer = rf.Tx2D(self.rf_world, (ix, iy),
                 freq_hz = centres[band_idx],
                 p_tx_w  = self.params["ew_power_W"],
                 bw_hz   = sub_bw_hz,
                 fspl_thresh_dbm = -200)
        
        self.rf_net = rf.Network2D([jammer])

        rf_drones: list[rf.Drone2D] = []
        rf_drone_map: dict[str, rf.Drone2D] = {}
        for d_id, sim_drone in self.drones.items():
            rel = np.array(sim_drone.initial_relative_pos_to_leader, dtype=float)
            rf_d = rf.Drone2D(d_id, rel_xy=rel / self.params["LINK_LENGTH_METERS"])
            rf_drones.append(rf_d); rf_drone_map[d_id] = rf_d
        rf_master = rf_drone_map[self.leader_id]

        intended_directed_flows = []
        intended_directed_flows.extend([("S1", "R1"), ("S2", "R1"), ("S3", "R2"), ("S4", "R2")])
        intended_directed_flows.extend([("R1", "M1"), ("R2", "M1"), ("M1", "R3"), ("M1", "R4")])
        intended_directed_flows.extend([("R3", "A1"), ("R3", "A2"), ("R4", "A3"), ("R4", "A4")])

        config = self.params["relay_connectivity_config"]
        r1, r2, r3, r4 = self.relay_ids_ordered
        if config == RelayConnectivityConfig.CROSS_ROW or config == RelayConnectivityConfig.ALL_CONNECT:
            intended_directed_flows.append((r1, r3))
            intended_directed_flows.append((r2, r4))
        if config == RelayConnectivityConfig.ALL_CONNECT:
            intended_directed_flows.append((r1, r2))
            intended_directed_flows.append((r3, r4))

        self.sim_intended_directed_flows = []
        all_rf_link_objects = []
        current_scenario_props = EDGE_PROPERTIES_BY_SCENARIO[self.params["relay_connectivity_config"]]
        centres, sub_bw_hz = rf.build_subbands()
        if "link_bw_hz" in self.params:
            link_bw_hz = self.params["link_bw_hz"]
        else:
            link_bw_hz = CAPACITY_TO_BW_HZ[self.params["network_capacity_type"]]
        LINK_POWER_W = 0.1
        band_idx_jammer = self.jam_band_idx - 1


        for tx_id, rx_id in intended_directed_flows:
            if tx_id not in rf_drone_map or rx_id not in rf_drone_map:
                print(f"Warning: Drone ID {tx_id} or {rx_id} not in rf_drone_map. Skipping Link2D.")
                continue

            edge_bw_area = get_edge_property_value(tx_id, rx_id, current_scenario_props, "bw_area")
            band_idx_edge = edge_bw_area - 1
            capacity_requirement = self.graph.edges[tx_id, rx_id]['required_safety_capacity']

            lk = rf.Link2D(
                tx=rf_drone_map[tx_id],
                rx=rf_drone_map[rx_id],
                freq_hz=centres[band_idx_edge],
                p_tx_w=LINK_POWER_W,
                bw_hz=link_bw_hz,
                is_susceptible=(band_idx_edge == band_idx_jammer),
                capacity_requirement=capacity_requirement,
            )
            all_rf_link_objects.append(lk)
            self.sim_intended_directed_flows.append((tx_id, rx_id))

            if lk.is_susceptible:
                self.initial_susceptible_physical_links.add(tuple(sorted((tx_id, rx_id))))

        self.rf_swarm = rf.Swarm2D(rf_master, rf_drones,
                                    all_rf_link_objects,
                                    scale=self.params["LINK_LENGTH_METERS"])

    def _calculate_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def step(self):
        self.current_step += 1
        leader_drone = self.drones[self.leader_id]
        current_pos, target_pos = leader_drone.pos, self.params["end_point"]
        dist_to_tar = self._calculate_distance(current_pos, target_pos); dx, dy = 0.0, 0.0
        if dist_to_tar > 1e-6 :
            step_dist = min(self.params["step_size_m"], dist_to_tar)
            dir_x,dir_y = (target_pos[0]-current_pos[0])/dist_to_tar, (target_pos[1]-current_pos[1])/dist_to_tar
            dx,dy = dir_x*step_dist, dir_y*step_dist
        leader_drone.move(dx,dy)

        for d_id, d_obj in self.drones.items():
            if d_id == self.leader_id: continue
            npx = leader_drone.pos[0]+d_obj.initial_relative_pos_to_leader[0]
            npy = leader_drone.pos[1]+d_obj.initial_relative_pos_to_leader[1]
            d_obj.move_to((npx,npy))
        
        ew_outputs_for_physical_links = self.rf_swarm.update(np.array(leader_drone.pos), self.rf_net)
        
        active_graph_edges_undirected = []
        current_step_disconnected_susceptible_links = {name: set() for name in modulation_types}

        for u, v, edge_data_undirected in self.graph.edges(data=True):
            physical_link_key = tuple(sorted((u, v)))
            ew_result = ew_outputs_for_physical_links.get(physical_link_key)

            if ew_result:
                edge_data_undirected.update({
                    'is_ew_susceptible': ew_result['is_susceptible'],
                    'current_capacities': ew_result['throughput_dict'],
                    'is_active': ew_result['is_active'],
                })

                if ew_result['is_susceptible']:
                    for mod_name in modulation_types:
                        if not ew_result['is_active'].get(mod_name, True):
                            if physical_link_key not in self.disconnected_susceptible_physical_links[mod_name]:
                                current_step_disconnected_susceptible_links[mod_name].add(physical_link_key)
                            self.disconnected_susceptible_physical_links[mod_name].add(physical_link_key)

            else:
                edge_data_undirected.update({
                    'is_ew_susceptible': False,
                    'current_capacities': {name: edge_data_undirected['base_capacity'] for name in modulation_types},
                    'is_active': {name: True for name in modulation_types},
                })

            if edge_data_undirected['is_active'].get("THEORETICAL"):
                active_graph_edges_undirected.append((u, v))

        R_abs = abs(leader_drone.pos[0] - self.params["ew_location"][0])

        for mod_name in modulation_types:
            active_graph_for_mod = nx.Graph()
            active_graph_for_mod.add_nodes_from(self.graph.nodes())
            active_edges_for_mod = [
                (u, v) for u, v, data in self.graph.edges(data=True) 
                if data['is_active'].get(mod_name, True)
            ]
            active_graph_for_mod.add_edges_from(active_edges_for_mod)

            connected_nodes_for_mod = set()
            if self.leader_id in active_graph_for_mod:
                try:
                    connected_nodes_for_mod = nx.node_connected_component(active_graph_for_mod, self.leader_id)
                except (nx.NetworkXError, KeyError):
                    connected_nodes_for_mod = {self.leader_id}

            current_disconnected_sensors = set(self.sensor_drone_ids) - connected_nodes_for_mod
            for sensor_id in current_disconnected_sensors:
                if sensor_id not in self._sensor_disconnect_distances[mod_name]:
                    self._sensor_disconnect_distances[mod_name][sensor_id] = R_abs
            
            current_disconnected_attackers = set(self.attack_drone_ids) - connected_nodes_for_mod
            for attacker_id in current_disconnected_attackers:
                if attacker_id not in self._attack_disconnect_distances[mod_name]:
                    self._attack_disconnect_distances[mod_name][attacker_id] = R_abs
            
            newly_broken = current_step_disconnected_susceptible_links[mod_name]
            for link in newly_broken:
                if link not in self._link_break_normed[mod_name]:
                    L_uv = self.graph.edges[link]['link_length']
                    self._link_break_normed[mod_name][link] = R_abs #/ L_uv no normalization needed

        final_connected_nodes = nx.node_connected_component(
            nx.Graph(
                [(u, v) for u, v, data in self.graph.edges(data=True) if data['is_active'].get("THEORETICAL", True)]
            ), self.leader_id
        ) if self.leader_id in self.graph else {self.leader_id}
        
        for drone_id, drone_obj in self.drones.items():
            drone_obj.is_connected_to_leader = drone_id in final_connected_nodes

        if self.params["logging_enabled"]: self._log_step_data()
        
        num_conn = sum(1 for d in self.drones.values() if d.is_connected_to_leader)
        
        ew_x = self.params["ew_location"][0]
        rearmost_drone_x = min(d.pos[0] for d in self.drones.values()) if self.drones else leader_drone.pos[0]
        
        if rearmost_drone_x > (ew_x + self.params["FLIGHT_MARGIN_PAST_EW_M"]):
            self.all_drones_passed_ew_x = True
            print(f"End: All drones passed EW jammer X-coordinate at step {self.current_step}.")
            return True
            
        if num_conn == 0 and len(self.drones)>0: print(f"End: All disconnected step {self.current_step}."); return True
        if num_conn == 1 and self.drones[self.leader_id].is_connected_to_leader and len(self.drones)>1:
            print(f"End: Only M1 connected step {self.current_step}."); return True
        
        max_expected_steps = (self._calculate_distance(self.params["start_point"], self.params["end_point"]) + self.params["FLIGHT_MARGIN_PAST_EW_M"] + 200) / self.params["step_size_m"]
        if self.current_step > max_expected_steps * 1.5:
            print(f"End: Exceeded safety max steps ({self.current_step} > {max_expected_steps * 1.5:.0f}).")
            return True
        
        return False

    def _log_step_data(self):
        leader_drone = self.drones[self.leader_id]
        step_log = {"type": "step_data", "step": self.current_step,
                    "leader_pos": list(leader_drone.pos),
                    "connected_drones_count": sum(1 for d in self.drones.values() if d.is_connected_to_leader),
                    "disabled_drones_ids": [id_str for id_str,d in self.drones.items() if not d.is_connected_to_leader],
                    "edges": []}
        if hasattr(self, 'sim_intended_directed_flows'):
            for tx_id, rx_id in self.sim_intended_directed_flows:
                physical_link_key = tuple(sorted((tx_id, rx_id)))
                if self.graph.has_edge(*physical_link_key):
                    data = self.graph.edges[physical_link_key]
                    step_log["edges"].append({
                        "u": tx_id, "v": rx_id, "is_active": data['is_active'],
                        "current_capacities": dict(data['current_capacities']),
                        "required_safety_capacity": data['required_safety_capacity'],
                        "bw_area": data['bw_area'],
                        "is_ew_susceptible": data['is_ew_susceptible']})
        self.log_data.append(step_log)

    def run_simulation(self):
        print(f"Starting simulation for config: LINK_LENGTH_METERS={self.params['LINK_LENGTH_METERS']}, JAMMER_BW_AREA={self.jam_band_idx}, CONNECTIVITY={self.params['relay_connectivity_config'].name}")
        
        while True:
            if self.step():
                break
        
        print(f"Simulation finished at step {self.current_step}.")
        if self.params["logging_enabled"] and self.params["csv_output_enabled"]:
            self.write_log_to_csv()
        
        r1_normed = {mod: np.nan for mod in modulation_types}
        r2_normed = {mod: np.nan for mod in modulation_types}
        for mod_name in modulation_types:
            if not self._link_break_normed[mod_name]:
                continue

            all_normed_values = list(self._link_break_normed[mod_name].values())
            r2_normed[mod_name] = min(all_normed_values)
            r1_normed[mod_name] = max(all_normed_values)

        L_ref = self.params["LINK_LENGTH_METERS"]
        for mod_name in modulation_types:
            self.r1_equiv[mod_name] = r1_normed[mod_name] #* L_ref no normalization needed
            self.r2_equiv[mod_name] = r2_normed[mod_name] #* L_ref

        # This logic robustly defines RS1/RA1 as the first failure event (max distance) and RS2/RA2 as the last failure event among those that failed (min distance).
        for mod_name in modulation_types:
            # --- Sensor Metrics (RS1, RS2) ---
            sensor_distances = list(self._sensor_disconnect_distances[mod_name].values())
            if sensor_distances:
                # RS1: First sensor to disconnect. This happens at the largest distance from the jammer.
                self.rs1_leader_dist_to_ew[mod_name] = max(sensor_distances)
                
                # RS2: Last of the disconnected sensors to disconnect. This happens at the smallest distance.
                # This value is meaningful even if not all sensors disconnect. It represents the point
                # at which the last failure event occurred for the sensor sub-swarm.
                self.rs2_leader_dist_to_ew[mod_name] = min(sensor_distances)
            # If sensor_distances is empty, both RS1 and RS2 remain NaN, indicating no sensor ever failed.
            
            # --- Attack Metrics (RA1, RA2) ---
            attack_distances = list(self._attack_disconnect_distances[mod_name].values())
            if attack_distances:
                # RA1: First attack drone to disconnect is at the MAX distance.
                self.ra1_leader_dist_to_ew[mod_name] = max(attack_distances)
                # RA2: Last of the disconnected attack drones to disconnect is at the MIN distance.
                self.ra2_leader_dist_to_ew[mod_name] = min(attack_distances)
            # If attack_distances is empty, both RA1 and RA2 remain NaN.
        
        return {
            "r1_leader_dist_to_ew": self.r1_equiv,
            "r2_leader_dist_to_ew": self.r2_equiv,
            "rs1_leader_dist_to_ew": self.rs1_leader_dist_to_ew,
            "rs2_leader_dist_to_ew": self.rs2_leader_dist_to_ew,
            "ra1_leader_dist_to_ew": self.ra1_leader_dist_to_ew,
            "ra2_leader_dist_to_ew": self.ra2_leader_dist_to_ew,
            "all_drones_passed_ew_x": self.all_drones_passed_ew_x,
            "final_step": self.current_step,
            "num_initial_susceptible_links": len(self.initial_susceptible_physical_links),
            "num_disconnected_susceptible_links": len(self.disconnected_susceptible_physical_links["THEORETICAL"])
        }


    def write_log_to_csv(self):
        if not self.log_data: print("No log data to write."); return
        
        if self.params.get("run_id_for_filename"):
            filename = f"{self.params['csv_filename_prefix']}{self.params['run_id_for_filename']}.csv"
        else:
            timestamp = random.randint(1000,9999)
            filename = f"{self.params['csv_filename_prefix']}{self.params['relay_connectivity_config'].name}_len{int(self.params['LINK_LENGTH_METERS'])}_jam{self.jam_band_idx}_{timestamp}.csv"
        
        print(f"Writing detailed log to {filename}...")
        param_data = next((item["data"] for item in self.log_data if item["type"] == "parameters"),{})
        param_data["All_Drones_Passed_EW_X"] = self.all_drones_passed_ew_x

        edge_fieldnames = ['step','leader_pos_x','leader_pos_y','connected_drones_count',
                           'edge_u','edge_v','required_safety_capacity',
                           'bw_area','is_ew_susceptible']
        edge_fieldnames.extend([f"{name.lower()}_cap" for name in modulation_types])

        with open(filename,'w',newline='') as csvfile:
            writer = csv.writer(csvfile); writer.writerow(["Parameter","Value"])
            for k,v in param_data.items(): writer.writerow([k,v])
            writer.writerow([]); writer.writerow(edge_fieldnames)
            for log_entry in self.log_data:
                if log_entry["type"] == "step_data":
                    lp_x = log_entry["leader_pos"][0] if len(log_entry["leader_pos"]) > 0 else 'N/A'
                    lp_y = log_entry["leader_pos"][1] if len(log_entry["leader_pos"]) > 1 else 'N/A'
                    for edge_info in log_entry["edges"]:
                        row = [
                            log_entry["step"],
                            lp_x,
                            lp_y,
                            log_entry["connected_drones_count"],
                            edge_info['u'],
                            edge_info['v'],
                            edge_info['required_safety_capacity'],
                            edge_info['bw_area'],
                            edge_info['is_ew_susceptible']
                        ]

                        capacities = edge_info.get("current_capacities", {})
                        row.extend([
                            capacities.get(mod, 0.0)
                            for mod in modulation_types
                        ])

                        writer.writerow(row)
        print(f"Log written to {filename}")

# --- Main Execution (for testing this file directly) ---
if __name__ == "__main__":
    print("Running a single test simulation from swarm_simulation.py...")
    test_params = {
        "LINK_LENGTH_METERS": 300.0,
        "relay_connectivity_config": RelayConnectivityConfig.CROSS_ROW,
        "EW_JAMMER_BW_AREA_SELECTION": 2,
        "logging_enabled": True,
        "csv_output_enabled": True
    }
    simulation = SwarmSimulation(params_override=test_params)
    results = simulation.run_simulation()
    print("\n--- Test Run Results ---")
    print(f"R1 (First Susceptible Disconnect): {results['r1_leader_dist_to_ew']['THEORETICAL']:.2f} m (or NaN)")
    print(f"R2 (Last Susceptible Disconnect): {results['r2_leader_dist_to_ew']['THEORETICAL']:.2f} m (or NaN)")
    print(f"All Drones Passed EW X-coord: {results['all_drones_passed_ew_x']}")
    print(f"Simulation ended at step: {results['final_step']}")
    print(f"Initial susceptible links: {results['num_initial_susceptible_links']}")
    print(f"Disconnected susceptible links: {results['num_disconnected_susceptible_links']}")
