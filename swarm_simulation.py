import networkx as nx
import math
import random
import csv
# import uuid # No longer explicitly used for drone IDs as they are predefined
from enum import Enum

# --- Constants and Configuration ---

# Network Capacity Types
class NetworkCapacity(Enum):
    SMALL = {"name": "small", "value": 50e3} # 50 kbps
    MEDIUM = {"name": "medium", "value": 1e6} # 1 Mbps
    LARGE = {"name": "large", "value": 5e6} # 5 Mbps

# Relay Connectivity Configuration
class RelayConnectivityConfig(Enum):
    MINIMAL = 1     # Only M-R and R-S/A links
    CROSS_ROW = 2   # MINIMAL + R1-R3, R2-R4
    ALL_CONNECT = 3 # CROSS_ROW + R1-R2, R3-R4, R1-R4, R2-R3 (full inter-relay mesh)


# Simulation Parameters
SIM_PARAMS = {
    "network_capacity_type": NetworkCapacity.MEDIUM,
    "LINK_LENGTH_METERS": 25.0,
    "relay_connectivity_config": RelayConnectivityConfig.MINIMAL, # Enum member
    "ew_location": (1050.0, 0.0),
    "ew_strength_factor": 100000.0,
    "BANDWIDTH_MHZ_RANGE": (400, 1200),
    "EW_JAMMER_BW_AREA_SELECTION": "random",
    "start_point": (0.0, 0.0),
    "end_point": (1000.0, 0.0),
    "step_size_m": 10.0,
    "max_steps": 200,
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

# --- EWSystem Class ---
class EWSystem:
    def __init__(self, ew_jammer_location, ew_jammer_bw_area, ew_strength_factor,
                 link_length_meters, all_swarm_edges_info):
        self.jammer_location = ew_jammer_location
        self.jammer_bw_area = ew_jammer_bw_area
        self.strength_factor = ew_strength_factor
        self.link_length_meters = link_length_meters
        self.swarm_edges_info = all_swarm_edges_info
        print(f"EW System Initialized: Jammer BW Area {self.jammer_bw_area}, Strength {self.strength_factor}")

    def _calculate_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _get_edge_midpoint(self, drone1_pos, drone2_pos):
        return ((drone1_pos[0] + drone2_pos[0]) / 2, (drone1_pos[1] + drone2_pos[1]) / 2)

    def update_ew_effects_on_edges(self, current_drone_objects):
        edge_ew_outputs = {}
        for u_id, v_id, edge_bw_area, edge_base_capacity in self.swarm_edges_info:
            edge_key = tuple(sorted((u_id, v_id)))
            if u_id not in current_drone_objects or v_id not in current_drone_objects:
                edge_ew_outputs[edge_key] = {
                    'is_susceptible': False, 'raw_ew_effect': 0.0,
                    'current_max_capacity_after_ew': edge_base_capacity}
                continue
            drone1, drone2 = current_drone_objects[u_id], current_drone_objects[v_id]
            is_susceptible = (edge_bw_area == self.jammer_bw_area)
            raw_ew_effect, current_max_capacity = 0.0, edge_base_capacity
            if is_susceptible:
                edge_actual_length = self._calculate_distance(drone1.pos, drone2.pos)
                edge_midpoint = self._get_edge_midpoint(drone1.pos, drone2.pos)
                distance_to_ew = self._calculate_distance(edge_midpoint, self.jammer_location)
                if distance_to_ew < 1e-6: raw_ew_effect = 1.0
                else:
                    edge_length_factor = (edge_actual_length / self.link_length_meters) if self.link_length_meters > 1e-6 else 1.0
                    raw_ew_effect = min(1.0, max(0.0, (self.strength_factor * edge_length_factor) / (distance_to_ew**2 + 1)))
                current_max_capacity = max(0.0, edge_base_capacity * (1.0 - raw_ew_effect))
            edge_ew_outputs[edge_key] = {
                'is_susceptible': is_susceptible, 'raw_ew_effect': raw_ew_effect,
                'current_max_capacity_after_ew': current_max_capacity}
        return edge_ew_outputs

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
            self.ew_jammer_target_bw_area = random.randint(1, 4)
        else:
            self.ew_jammer_target_bw_area = int(self.params["EW_JAMMER_BW_AREA_SELECTION"])
        print(f"Simulation Jammer Target BW Area: {self.ew_jammer_target_bw_area}")
        self._setup_swarm_formation_and_graph()
        self._log_initial_parameters()
        ew_system_edge_list = []
        base_cap_value = self.params["network_capacity_type"].value["value"]
        for u, v, data in self.graph.edges(data=True):
            ew_system_edge_list.append( (u, v, data['bw_area'], base_cap_value) )
        self.ew_system = EWSystem(
            ew_jammer_location=self.params["ew_location"],
            ew_jammer_bw_area=self.ew_jammer_target_bw_area,
            ew_strength_factor=self.params["ew_strength_factor"],
            link_length_meters=self.params["LINK_LENGTH_METERS"],
            all_swarm_edges_info=ew_system_edge_list)

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
            self.graph.edges[u,v]['last_ew_effect'] = 0.0

    def _log_initial_parameters(self):
        if not self.params["logging_enabled"]: return
        param_log = {"type": "parameters", "data": {}}
        for key, value in self.params.items():
            if isinstance(value, Enum): param_log["data"][key] = value.name # Log enum name
            else: param_log["data"][key] = value
        param_log["data"]["EW_JAMMER_ACTUAL_BW_AREA"] = self.ew_jammer_target_bw_area
        self.log_data.append(param_log)

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
        ew_outputs_for_edges = self.ew_system.update_ew_effects_on_edges(self.drones)
        active_graph_edges = []
        for u, v, edge_data in self.graph.edges(data=True):
            edge_key = tuple(sorted((u,v)))
            ew_result = ew_outputs_for_edges.get(edge_key)
            if ew_result:
                edge_data.update({
                    'is_ew_susceptible': ew_result['is_susceptible'],
                    'current_capacity': ew_result['current_max_capacity_after_ew'],
                    'last_ew_effect': ew_result['raw_ew_effect'],
                    'is_active': ew_result['current_max_capacity_after_ew'] >= edge_data['required_safety_capacity']})
            else: # Should not happen if ew_system.swarm_edges_info covers all graph edges
                print(f"Warning: No EW result for edge {edge_key}. Defaulting to active/not susceptible.")
                edge_data.update({'is_ew_susceptible': False, 'current_capacity': edge_data['base_capacity'],
                                  'last_ew_effect': 0.0, 'is_active': True})
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
                                     "is_ew_susceptible":data['is_ew_susceptible'],
                                     "ew_effect":data.get('last_ew_effect',0.0)})
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
                           'required_safety_capacity','bw_area','is_ew_susceptible','ew_effect']
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
                                         edge_info['bw_area'],edge_info['is_ew_susceptible'],edge_info['ew_effect']])
        print(f"Log written to {filename}")

# --- Main Execution ---
if __name__ == "__main__":
    simulation = SwarmSimulation(SIM_PARAMS)
    simulation.run_simulation()
