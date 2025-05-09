import networkx as nx
import math
import random
import csv
import uuid # For unique drone IDs
from enum import Enum

# --- Constants and Configuration ---
# For easily changing parameters

# Network Capacity Types
class NetworkCapacity(Enum):
    SMALL = {"name": "small", "value": 50e3}  # 50 kbps
    MEDIUM = {"name": "medium", "value": 1e6} # 1 Mbps
    LARGE = {"name": "large", "value": 10e6} # 10 Mbps

# Relay Edge Configuration (Interpreted as desired degree for relays regarding L and other R)
# A relay always connects to the Leader (1 edge) and its 2 assigned drones (2 edges).
# So, base degree is 3.
# MINIMAL (3 in old naming): Base: L-R, R-Subordinates
# CROSS_ROW (4 in old naming): Base + Cross-row relays connect
# ALL_CONNECT (5 in old naming): Base + Same-row + cross-row relays connect
class RelayConnectivityConfig(Enum):
    MINIMAL = 3
    CROSS_ROW = 4
    ALL_CONNECT = 5

# Simulation Parameters
SIM_PARAMS = {
    "network_capacity_type": NetworkCapacity.MEDIUM,
    "LINK_LENGTH_METERS": 25.0,  # Target length for M-R, R-S, R-A links & overall scale
    "relay_connectivity_config": RelayConnectivityConfig.CROSS_ROW,
    "ew_disruption_percentage": 0.3,  # 30% of edges are susceptible to EW
    "ew_location": (1050.0, 0.0), # (x,y) slightly after the end point
    "ew_strength_factor": 100000.0, # Higher value means stronger EW effect at same distance
    "ew_disable_threshold": 0.8, # If calculated EW effect > this, edge is disabled
    "start_point": (0.0, 0.0),
    "end_point": (1000.0, 0.0),
    "step_size_m": 10.0, # How much the leader moves each step
    "max_steps": 200,
    "logging_enabled": True,
    "csv_output_enabled": True,
    "csv_filename": "swarm_simulation_log.csv",
}

# --- New Formation Unit Coordinates ---
# These coordinates define the swarm shape where primary links (M-R, R-S, R-A)
# would have a length of 1 unit. The LINK_LENGTH_METERS parameter then scales this.
# Swarm's local frame: +X is forward, +Y is left (when looking forward).
K_FACTOR = 1.0 / math.sqrt(5.0)  # Approx 0.4472

# Drone names are now 1-indexed, Leader is M1
UNIT_COORDS_RELATIVE_TO_LEADER = {
    "M1": (0.0, 0.0),
    # Front Relays (R1, R2)
    "R1": (K_FACTOR * 1, K_FACTOR * 2),   # Front-Left Relay (was R0)
    "R2": (K_FACTOR * 1, K_FACTOR * -2),  # Front-Right Relay (was R1)
    # Sensors (S1-S4)
    # S1, S2 on R1; S3, S4 on R2
    "S1": (K_FACTOR * 3, K_FACTOR * 3),   # Outer-Left Sensor (on R1, was S0)
    "S2": (K_FACTOR * 3, K_FACTOR * 1),   # Inner-Left Sensor (on R1, was S1)
    "S3": (K_FACTOR * 3, K_FACTOR * -1),  # Inner-Right Sensor (on R2, was S2)
    "S4": (K_FACTOR * 3, K_FACTOR * -3),  # Outer-Right Sensor (on R2, was S3)
    # Back Relays (R3, R4)
    "R3": (K_FACTOR * -1, K_FACTOR * 2),  # Back-Left Relay (was R2)
    "R4": (K_FACTOR * -1, K_FACTOR * -2), # Back-Right Relay (was R3)
    # Attack Drones (A1-A4)
    # A1, A2 on R3; A3, A4 on R4
    "A1": (K_FACTOR * -3, K_FACTOR * 3),  # Outer-Left Attack (on R3, was A0)
    "A2": (K_FACTOR * -3, K_FACTOR * 1),  # Inner-Left Attack (on R3, was A1)
    "A3": (K_FACTOR * -3, K_FACTOR * -1), # Inner-Right Attack (on R4, was A2)
    "A4": (K_FACTOR * -3, K_FACTOR * -3), # Outer-Right Attack (on R4, was A3)
}

# Drone Types
class DroneType(Enum):
    LEADER = "leader" # M-type
    RELAY = "relay"   # R-type
    SENSOR = "sensor" # S-type
    ATTACK = "attack" # A-type

# --- Drone Class ---
class Drone:
    def __init__(self, id_str, drone_type, initial_pos_abs):
        self.id = id_str # Changed from id to id_str for clarity
        self.drone_type = drone_type
        self.pos = list(initial_pos_abs) # Store as list for mutability [x,y]
        self.is_connected_to_leader = True # Initially all are connected
        self.initial_relative_pos_to_leader = [0.0, 0.0] # Will be set during setup

    def __repr__(self):
        return f"Drone({self.id}, {self.drone_type.value}, pos={self.pos}, connected={self.is_connected_to_leader})"

    def move(self, dx, dy):
        if self.is_connected_to_leader:
            self.pos[0] += dx
            self.pos[1] += dy

    def move_to(self, new_pos):
        if self.is_connected_to_leader:
            self.pos[0] = new_pos[0]
            self.pos[1] = new_pos[1]

# --- Swarm Simulation Class ---
class SwarmSimulation:
    def __init__(self, params):
        self.params = params
        self.graph = nx.Graph()
        self.drones = {} # Stores Drone objects, keyed by ID
        self.leader_id = "M1" # Changed from L0
        self.current_step = 0
        self.log_data = []
        self.ew_susceptible_edges = set()
        
        # Define drone IDs for clarity in setup, now 1-indexed
        self.drone_ids_map = {
            "M1": DroneType.LEADER,
            "R1": DroneType.RELAY, "R2": DroneType.RELAY, "R3": DroneType.RELAY, "R4": DroneType.RELAY,
            "S1": DroneType.SENSOR, "S2": DroneType.SENSOR, "S3": DroneType.SENSOR, "S4": DroneType.SENSOR,
            "A1": DroneType.ATTACK, "A2": DroneType.ATTACK, "A3": DroneType.ATTACK, "A4": DroneType.ATTACK,
        }
        # Define fixed connections with new 1-indexed names
        self.fixed_connections = [
            ("M1", "R1"), ("M1", "R2"), ("M1", "R3"), ("M1", "R4"), # Leader to Relays
            ("R1", "S1"), ("R1", "S2"), # R1 (Front-Left) to its Sensors
            ("R2", "S3"), ("R2", "S4"), # R2 (Front-Right) to its Sensors
            ("R3", "A1"), ("R3", "A2"), # R3 (Back-Left) to its Attacks
            ("R4", "A3"), ("R4", "A4"), # R4 (Back-Right) to its Attacks
        ]
        # For inter-relay connections, R1,R2 are front; R3,R4 are back
        self.relay_ids_ordered = ["R1", "R2", "R3", "R4"]


        self._setup_swarm_formation_and_graph()
        self._select_ew_susceptible_edges()
        self._log_initial_parameters()

    # Removed _get_drone_id as IDs are now predefined strings

    def _setup_swarm_formation_and_graph(self):
        leader_start_pos = list(self.params["start_point"])
        link_length = self.params["LINK_LENGTH_METERS"] # This is our scalar

        # Create and position all drones
        for drone_id_str, drone_type in self.drone_ids_map.items():
            # Fetch unit coordinates relative to the leader (M1)
            unit_rel_x, unit_rel_y = UNIT_COORDS_RELATIVE_TO_LEADER[drone_id_str]

            # Calculate actual relative position in meters from M1
            actual_rel_x_from_leader = unit_rel_x * link_length
            actual_rel_y_from_leader = unit_rel_y * link_length
            
            # Calculate absolute initial position based on leader's start_point
            # Assuming swarm's local +X (forward) aligns with global +X,
            # and swarm's local +Y (left) aligns with global +Y at the start.
            initial_pos_x = leader_start_pos[0] + actual_rel_x_from_leader
            initial_pos_y = leader_start_pos[1] + actual_rel_y_from_leader
            
            drone = Drone(drone_id_str, drone_type, (initial_pos_x, initial_pos_y))
            drone.initial_relative_pos_to_leader = [actual_rel_x_from_leader, actual_rel_y_from_leader]
            
            self.drones[drone_id_str] = drone
            self.graph.add_node(drone_id_str, drone=drone) # Add drone object as node attribute

        # Add fixed connections (Leader-Relay, Relay-Subordinate)
        for u, v in self.fixed_connections:
            if u in self.drones and v in self.drones:
                 self.graph.add_edge(u,v)
            else:
                # This should not happen if drone_ids_map and fixed_connections are correct
                print(f"Warning: Drone ID not found for fixed connection: {u} or {v}")


        # Add inter-relay connections based on relay_connectivity_config
        # self.relay_ids_ordered = ["R1", "R2", "R3", "R4"]
        # R1 (Front-Left), R2 (Front-Right), R3 (Back-Left), R4 (Back-Right)
        config = self.params["relay_connectivity_config"]
        
        # MINIMAL config implies only fixed connections already made (M-R, R-S/A)

        if config.value >= RelayConnectivityConfig.CROSS_ROW.value:
            # Connect R1-R3 (Front-Left to Back-Left) & R2-R4 (Front-Right to Back-Right)
            if not self.graph.has_edge(self.relay_ids_ordered[0], self.relay_ids_ordered[2]): # R1-R3
                self.graph.add_edge(self.relay_ids_ordered[0], self.relay_ids_ordered[2])
            if not self.graph.has_edge(self.relay_ids_ordered[1], self.relay_ids_ordered[3]): # R2-R4
                self.graph.add_edge(self.relay_ids_ordered[1], self.relay_ids_ordered[3])

        if config.value >= RelayConnectivityConfig.ALL_CONNECT.value:
            # Add same-row connections (intra-row)
            if not self.graph.has_edge(self.relay_ids_ordered[0], self.relay_ids_ordered[1]): # R1-R2 (Front row)
                self.graph.add_edge(self.relay_ids_ordered[0], self.relay_ids_ordered[1])
            if not self.graph.has_edge(self.relay_ids_ordered[2], self.relay_ids_ordered[3]): # R3-R4 (Back row)
                self.graph.add_edge(self.relay_ids_ordered[2], self.relay_ids_ordered[3])
            
            # Add remaining diagonal connections for full inter-relay mesh
            # R1-R4 (Front-Left to Back-Right) and R2-R3 (Front-Right to Back-Left)
            if not self.graph.has_edge(self.relay_ids_ordered[0], self.relay_ids_ordered[3]): # R1-R4
                self.graph.add_edge(self.relay_ids_ordered[0], self.relay_ids_ordered[3])
            if not self.graph.has_edge(self.relay_ids_ordered[1], self.relay_ids_ordered[2]): # R2-R3
                self.graph.add_edge(self.relay_ids_ordered[1], self.relay_ids_ordered[2])


        # Initialize edge attributes
        # Ensure network_capacity_type.value is the dict, then access its "value" key
        base_capacity = self.params["network_capacity_type"].value["value"]
        for u, v in self.graph.edges():
            self.graph.edges[u,v]['base_capacity'] = base_capacity
            self.graph.edges[u,v]['current_capacity'] = base_capacity
            self.graph.edges[u,v]['is_active'] = True
            self.graph.edges[u,v]['is_ew_susceptible'] = False # Will be set later


    def _select_ew_susceptible_edges(self):
        all_edges = list(self.graph.edges())
        if not all_edges: # Check if there are any edges
            print("Warning: No edges in graph to select for EW susceptibility.")
            return
        num_susceptible_edges = int(len(all_edges) * self.params["ew_disruption_percentage"])
        # Ensure num_susceptible_edges does not exceed available edges
        num_susceptible_edges = min(num_susceptible_edges, len(all_edges))
        
        susceptible_edges_list = random.sample(all_edges, num_susceptible_edges)
        for u, v in susceptible_edges_list:
            # Store as sorted tuple for consistent lookup if order isn't guaranteed
            self.ew_susceptible_edges.add(tuple(sorted((u,v)))) 
            self.graph.edges[u,v]['is_ew_susceptible'] = True

    def _log_initial_parameters(self):
        if not self.params["logging_enabled"]:
            return
        param_log = {"type": "parameters", "data": {}}
        for key, value in self.params.items():
            if isinstance(value, Enum): # Log enum name for readability
                param_log["data"][key] = value.name
            else:
                param_log["data"][key] = value
        self.log_data.append(param_log)

    def _calculate_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _get_edge_midpoint(self, drone1_pos, drone2_pos):
        return ((drone1_pos[0] + drone2_pos[0]) / 2, (drone1_pos[1] + drone2_pos[1]) / 2)

    def _calculate_ew_effect_on_edge(self, u, v):
        """
        Calculates EW disruption effect on an edge.
        Effect increases with edge actual_length and decreases with distance to EW source.
        """
        drone1 = self.drones[u]
        drone2 = self.drones[v]
        
        # Edge properties
        edge_actual_length = self._calculate_distance(drone1.pos, drone2.pos)
        edge_midpoint = self._get_edge_midpoint(drone1.pos, drone2.pos)
        
        distance_to_ew = self._calculate_distance(edge_midpoint, self.params["ew_location"])
        
        if distance_to_ew < 1e-6: # Avoid division by zero if directly on EW source
            return 1.0 # Max disruption
            
        # This simple model: effect proportional to edge_length / distance_to_ew^2
        # and scaled by ew_strength_factor
        # The 'susceptible' flag for the edge determines if this calculation matters.
        
        # Normalize edge length effect, e.g. relative to configured LINK_LENGTH_METERS
        # Longer edges are inherently more susceptible.
        if self.params["LINK_LENGTH_METERS"] > 1e-6 : # Avoid division by zero
            edge_length_factor = edge_actual_length / self.params["LINK_LENGTH_METERS"]
        else:
            edge_length_factor = 1.0 # Default factor if LINK_LENGTH_METERS is effectively zero

        # Example: Inverse square law for distance, linear for edge length factor
        # Add a small constant to distance_to_ew to prevent extreme values when close
        ew_effect = (self.params["ew_strength_factor"] * edge_length_factor) / (distance_to_ew**2 + 1)
        
        return min(1.0, max(0.0, ew_effect)) # Clamp between 0 and 1

    def _get_current_edge_capacity(self, u, v, ew_effect_value):
        """
        Calculates the current capacity of an edge based on its base capacity
        and the EW effect.
        """
        base_capacity = self.graph.edges[u,v]['base_capacity']
        
        # Simplistic: capacity reduces linearly with ew_effect.
        # If ew_effect is 1.0, capacity is 0.
        reduced_capacity = base_capacity * (1.0 - ew_effect_value)
        
        return max(0.0, reduced_capacity)


    def step(self):
        self.current_step += 1
        leader_drone = self.drones[self.leader_id] # Leader is M1

        # 1. Move Leader (if not at end_point)
        current_pos = leader_drone.pos
        target_pos = self.params["end_point"]
        dist_to_target = self._calculate_distance(current_pos, target_pos)

        dx, dy = 0.0, 0.0 # Initialize movement deltas
        if dist_to_target < self.params["step_size_m"]: # If closer than one step
            if dist_to_target > 1e-6 : # Move the remaining small distance if not exactly at target
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
            # else, leader is effectively at target, dx/dy remain 0
        elif dist_to_target > 1e-6: # If further than tolerance and not within a step
            direction_x = (target_pos[0] - current_pos[0]) / dist_to_target
            direction_y = (target_pos[1] - current_pos[1]) / dist_to_target
            dx = direction_x * self.params["step_size_m"]
            dy = direction_y * self.params["step_size_m"]
        
        leader_drone.move(dx, dy) # Leader always moves if able (or stays if at target)

        # 2. Move other connected drones to maintain formation relative to leader
        for drone_id_str, drone_obj in self.drones.items():
            if drone_id_str == self.leader_id: # Skip leader drone itself
                continue 
            if drone_obj.is_connected_to_leader:
                # Swarm local X (forward) maps to global X, local Y (left) maps to global Y
                # (No independent swarm rotation relative to global axes is modeled here)
                new_pos_x = leader_drone.pos[0] + drone_obj.initial_relative_pos_to_leader[0]
                new_pos_y = leader_drone.pos[1] + drone_obj.initial_relative_pos_to_leader[1]
                drone_obj.move_to((new_pos_x, new_pos_y))
            # else: drone does not move if disconnected

        # 3. Update edge status and capacities based on EW
        active_graph_edges = []
        for u, v, edge_data in self.graph.edges(data=True):
            ew_effect_on_this_edge = 0.0
            if edge_data['is_ew_susceptible']:
                ew_effect_on_this_edge = self._calculate_ew_effect_on_edge(u,v)

            if ew_effect_on_this_edge > self.params["ew_disable_threshold"]:
                edge_data['is_active'] = False
                edge_data['current_capacity'] = 0.0
            else:
                edge_data['is_active'] = True
                # Capacity might be reduced even if not fully disabled
                edge_data['current_capacity'] = self._get_current_edge_capacity(u, v, ew_effect_on_this_edge)

            edge_data['last_ew_effect'] = ew_effect_on_this_edge # For logging

            if edge_data['is_active']:
                active_graph_edges.append((u,v))
        
        # 4. Check connectivity and update drone status
        # Create a temporary graph with only active edges for connectivity checks
        temp_active_graph = nx.Graph()
        temp_active_graph.add_nodes_from(self.graph.nodes()) # Add all drone IDs as nodes
        temp_active_graph.add_edges_from(active_graph_edges) # Add only active edges

        # Assume all drones disconnected initially for this step's check
        for drone_obj in self.drones.values():
            drone_obj.is_connected_to_leader = False
        
        if self.leader_id in temp_active_graph: # Check if leader node exists in temp graph
            try:
                # Find all nodes in the same connected component as the leader
                connected_component_nodes = nx.node_connected_component(temp_active_graph, self.leader_id)
                for node_id in connected_component_nodes:
                    if node_id in self.drones:
                        self.drones[node_id].is_connected_to_leader = True
            except nx.NetworkXError: 
                # Happens if leader is isolated or graph is empty (though caught by 'in temp_active_graph')
                # If leader is isolated but exists, it's connected only to itself (implicitly handled by loop start)
                if self.leader_id in self.drones: # Ensure leader drone object exists
                    self.drones[self.leader_id].is_connected_to_leader = True 
            except KeyError: # leader_id not in temp_active_graph (should be caught by outer if)
                 print(f"CRITICAL: Leader {self.leader_id} not found in temp_active_graph during component search!")
        else:
             print(f"CRITICAL: Leader {self.leader_id} not in temp_active_graph (no active edges connected or node missing)!")
             # All drones remain disconnected as per initial assumption for this step
        
        # 5. Log data for this step
        if self.params["logging_enabled"]:
            self._log_step_data()

        # 6. Check simulation end conditions
        num_connected_drones = sum(1 for d in self.drones.values() if d.is_connected_to_leader)
        
        # Check if leader has reached the target position (within a small tolerance)
        leader_at_target = self._calculate_distance(leader_drone.pos, self.params["end_point"]) < 1e-3 # Tolerance for float comparison


        if leader_at_target:
            print(f"Simulation ended: Leader reached end point at step {self.current_step}.")
            return True
        if num_connected_drones == 0 and len(self.drones) > 0: # All drones disconnected
             print(f"Simulation ended: All drones disconnected at step {self.current_step}.")
             return True
        if num_connected_drones == 1 and self.drones[self.leader_id].is_connected_to_leader and len(self.drones) > 1: # Only leader connected
             print(f"Simulation ended: Only leader M1 remains connected at step {self.current_step}.")
             return True
        if self.current_step >= self.params["max_steps"]:
            print(f"Simulation ended: Reached max steps ({self.params['max_steps']}).")
            return True
        
        return False # Continue simulation

    def _log_step_data(self):
        leader_drone = self.drones[self.leader_id] # Leader is M1
        step_log = {
            "type": "step_data",
            "step": self.current_step,
            "leader_pos": list(leader_drone.pos), # Ensure it's a list for JSON/CSV
            "connected_drones_count": sum(1 for d in self.drones.values() if d.is_connected_to_leader),
            "disabled_drones_ids": [id_str for id_str, d in self.drones.items() if not d.is_connected_to_leader],
            "edges": []
        }
        for u, v, data in self.graph.edges(data=True):
            step_log["edges"].append({
                "u": u,
                "v": v,
                "is_active": data['is_active'],
                "current_capacity": data['current_capacity'],
                "is_ew_susceptible": data['is_ew_susceptible'],
                "ew_effect": data.get('last_ew_effect', 0.0) # Get last calculated EW effect
            })
        self.log_data.append(step_log)

    def run_simulation(self):
        print("Starting simulation...")
        print(f"Leader: {self.leader_id}, Target: {self.params['end_point']}")
        print(f"Total Drones: {len(self.drones)}, EW Location: {self.params['ew_location']}")
        if self.graph.number_of_edges() > 0: # Only print if there are edges
            print(f"Susceptible edges ({len(self.ew_susceptible_edges)}): {self.ew_susceptible_edges}")
        else:
            print("No edges in the graph after setup.") # More informative

        for _ in range(self.params["max_steps"]): # Use _ if loop variable i is not used
            if self.step(): # step() returns True if simulation should end
                break
        
        print("Simulation finished.")
        if self.params["logging_enabled"] and self.params["csv_output_enabled"]:
            self.write_log_to_csv()

    def write_log_to_csv(self):
        if not self.log_data:
            print("No log data to write.")
            return

        filename = self.params["csv_filename"]
        print(f"Writing log to {filename}...")

        # Extract parameter data for header rows in CSV
        param_data = next((item["data"] for item in self.log_data if item["type"] == "parameters"), {})
        
        # Define fieldnames for step data - focusing on edges for detailed CSV
        # Added leader_pos_y to match a previous version's CSV output for analysis.
        edge_fieldnames = ['step', 'leader_pos_x', 'leader_pos_y', 'connected_drones_count', 
                           'edge_u', 'edge_v', 'is_active', 'current_capacity', 
                           'is_ew_susceptible', 'ew_effect']

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write parameters first
            writer.writerow(["Parameter", "Value"])
            for key, value in param_data.items():
                writer.writerow([key, value])
            writer.writerow([]) # Blank line separator

            # Write header for step data (focused on edges)
            writer.writerow(edge_fieldnames)

            # Write step data
            for log_entry in self.log_data:
                if log_entry["type"] == "step_data":
                    # Ensure leader_pos always has two elements for consistent indexing
                    leader_pos_x = log_entry["leader_pos"][0] if len(log_entry["leader_pos"]) > 0 else 'N/A'
                    leader_pos_y = log_entry["leader_pos"][1] if len(log_entry["leader_pos"]) > 1 else 'N/A'
                    connected_count = log_entry["connected_drones_count"]
                    
                    for edge_info in log_entry["edges"]:
                        row = [
                            log_entry["step"], leader_pos_x, leader_pos_y, 
                            connected_count,
                            edge_info['u'], edge_info['v'],
                            edge_info['is_active'], edge_info['current_capacity'],
                            edge_info['is_ew_susceptible'], edge_info['ew_effect']
                        ]
                        writer.writerow(row)
        print(f"Log written to {filename}")

# --- Main Execution ---
if __name__ == "__main__":
    # Example: Override a parameter for a specific run if needed
    # SIM_PARAMS["LINK_LENGTH_METERS"] = 20
    # SIM_PARAMS["ew_disruption_percentage"] = 0.8
    # SIM_PARAMS["relay_connectivity_config"] = RelayConnectivityConfig.MINIMAL
    # SIM_PARAMS["network_capacity_type"] = NetworkCapacity.SMALL
    
    simulation = SwarmSimulation(SIM_PARAMS)
    simulation.run_simulation()
