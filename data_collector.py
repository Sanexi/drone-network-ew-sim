import csv
import itertools
import os
import time
import numpy as np # For np.nan
import swarm_simulation as sim # Import the simulation module
from enum import Enum

# Import Enums from the simulation module to be used in param_grid
from swarm_simulation import NetworkCapacity, RelayConnectivityConfig

# List of the avalable modulation types as defined in RF_simulation
modulation_types = [
    "THEORETICAL", "BPSK", "QPSK", "16QAM", "64QAM"
]

def run_single_simulation(params_override=None, detailed_log_filename_id="single_run"):
    """
    Runs a single simulation with optional parameter overrides and detailed logging.
    """
    print(f"\n--- Running Single Detailed Simulation (ID: {detailed_log_filename_id}) ---")
    
    current_sim_params = sim.DEFAULT_SIM_PARAMS.copy()
    if params_override:
        current_sim_params.update(params_override)
    
    # Ensure detailed logging is enabled for single runs
    current_sim_params["logging_enabled"] = True
    current_sim_params["csv_output_enabled"] = True
    current_sim_params["run_id_for_filename"] = detailed_log_filename_id # Pass ID for unique filename

    simulation_instance = sim.SwarmSimulation(params_override=current_sim_params)
    results = simulation_instance.run_simulation()

    r1_theoretical = results.get('r1_leader_dist_to_ew', {}).get("THEORETICAL", np.nan)
    r2_theoretical = results.get('r2_leader_dist_to_ew', {}).get("THEORETICAL", np.nan)

    print(f"\n--- Results for Single Run (ID: {detailed_log_filename_id}) ---")
    print(f"  R1 (First Susceptible Disconnect at Leader-EW X-Dist): {r1_theoretical:.2f} m")
    print(f"  R2 (Last Susceptible Disconnect at Leader-EW X-Dist): {r2_theoretical:.2f} m")
    print(f"  All Drones Passed EW X-coord: {results.get('all_drones_passed_ew_x', False)}")
    print(f"  Simulation ended at step: {results.get('final_step', 0)}")
    print(f"  Initial susceptible links: {results.get('num_initial_susceptible_links',0)}")
    print(f"  Disconnected susceptible links: {results.get('num_disconnected_susceptible_links',0)}")
    print("  Detailed log saved to CSV (if enabled in params).")
    return results

def run_grid_search(param_grid, grid_log_filename="grid_search_summary.csv"):
    """
    Runs multiple simulations based on a grid of parameters and logs summary results (R1, R2).
    
    Args:
        param_grid (dict): Keys are parameter names (matching SIM_PARAMS keys),
                           values are lists of values to test for that parameter.
        grid_log_filename (str): Name of the CSV file to save summary results.
    """
    print(f"\n--- Starting Grid Search ---")
    param_names = list(param_grid.keys()) # These are the parameters being varied
    param_value_lists = list(param_grid.values())
    
    all_combinations = list(itertools.product(*param_value_lists))
    total_runs = len(all_combinations)
    print(f"Total simulation runs in grid search: {total_runs}")

    # Construct summary_fieldnames carefully to include all columns
    # Start with parameters varied in the grid
    fieldnames_list = list(param_names) # Keep order from param_grid
    fieldnames_set = set(param_names)

    # Add other fixed parameters or result columns that will be logged
    # These include parameters set by default for grid runs and the actual results
    additional_and_result_fields = [
        "logging_enabled", 
        "csv_output_enabled",
        "EW_JAMMER_ACTUAL_BW_AREA",
        "all_drones_passed_ew_x", 
        "final_step",
        "num_initial_susceptible_links", 
        "num_disconnected_susceptible_links"
    ]

    # Add per-modulation R1 and R2 fields
    for mod in modulation_types:
        additional_and_result_fields.append(f"r1_{mod.lower()}_m")
        additional_and_result_fields.append(f"r2_{mod.lower()}_m")

    for field in additional_and_result_fields:
        if field not in fieldnames_set: # Avoid duplicates if already in param_names
            fieldnames_list.append(field)
            # fieldnames_set.add(field) # Not strictly necessary to add to set here

    summary_fieldnames = fieldnames_list
    
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    full_grid_log_path = os.path.join(log_dir, grid_log_filename)

    with open(full_grid_log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_fieldnames)
        writer.writeheader()

        for i, param_values in enumerate(all_combinations):
            # Start with a fresh copy of defaults for each run's parameters
            current_run_specific_params = sim.DEFAULT_SIM_PARAMS.copy() 
            
            # Apply the current combination of grid parameters
            grid_specific_overrides = dict(zip(param_names, param_values))
            current_run_specific_params.update(grid_specific_overrides)
            
            # Ensure logging and detailed CSV are off for grid search runs for speed
            # unless explicitly overridden by the grid_specific_overrides
            if "logging_enabled" not in grid_specific_overrides:
                current_run_specific_params["logging_enabled"] = False
            if "csv_output_enabled" not in grid_specific_overrides:
                current_run_specific_params["csv_output_enabled"] = False
            
            # Prepare the log_row with all parameters that will be passed to simulation
            # and eventually logged.
            log_row = current_run_specific_params.copy() 
            # Convert enums to names now, as these are the values used/logged for input params
            for key, val in log_row.items():
                if isinstance(val, Enum):
                    log_row[key] = val.name

            print(f"\nRunning grid search {i+1}/{total_runs} with effective params: {current_run_specific_params}")
            
            try:
                # Pass current_run_specific_params which has all necessary keys
                simulation_instance = sim.SwarmSimulation(params_override=current_run_specific_params)
                results = simulation_instance.run_simulation()
                                
                # Add results to the log_row
                log_row["EW_JAMMER_ACTUAL_BW_AREA"] = simulation_instance.jam_band_idx
                # log_row["R1_leader_dist_to_ew"] = results.get('r1_leader_dist_to_ew', np.nan)
                # log_row["R2_leader_dist_to_ew"] = results.get('r2_leader_dist_to_ew', np.nan)
                r1_results = results.get('r1_leader_dist_to_ew', {})
                r2_results = results.get('r2_leader_dist_to_ew', {})
                for mod in modulation_types:
                    log_row[f"r1_{mod.lower()}_m"] = r1_results.get(mod, np.nan)
                    log_row[f"r2_{mod.lower()}_m"] = r2_results.get(mod, np.nan)

                log_row["all_drones_passed_ew_x"] = results.get('all_drones_passed_ew_x', False)
                log_row["final_step"] = results.get('final_step', 0)
                log_row["num_initial_susceptible_links"] = results.get('num_initial_susceptible_links',0)
                log_row["num_disconnected_susceptible_links"] = results.get('num_disconnected_susceptible_links',0)
                
                # Before writing, ensure log_row only contains keys in summary_fieldnames
                # This is critical if current_run_specific_params (copied to log_row)
                # had other keys not in summary_fieldnames (e.g. from DEFAULT_SIM_PARAMS
                # that are not part of param_grid nor additional_and_result_fields).
                final_log_row_for_csv = {sf: log_row.get(sf) for sf in summary_fieldnames}

                writer.writerow(final_log_row_for_csv)
                csvfile.flush()

            except Exception as e:
                print(f"!!! ERROR during simulation run {i+1} with params {current_run_specific_params}: {e}")
                error_row_data = current_run_specific_params.copy()
                for key, val in error_row_data.items():
                    if isinstance(val, Enum): error_row_data[key] = val.name
                
                error_row_data["EW_JAMMER_ACTUAL_BW_AREA"] = "ERROR"
                for mod in modulation_types:
                    error_row_data[f"r1_{mod.lower()}_m"] = "ERROR"
                    error_row_data[f"r2_{mod.lower()}_m"] = "ERROR"
                error_row_data["all_drones_passed_ew_x"] = "ERROR"
                error_row_data["final_step"] = "ERROR"
                error_row_data["num_initial_susceptible_links"] = "ERROR"
                error_row_data["num_disconnected_susceptible_links"] = "ERROR"

                final_error_row_for_csv = {sf: error_row_data.get(sf, "ERROR_FIELD_MISSING") for sf in summary_fieldnames}
                writer.writerow(final_error_row_for_csv)
                csvfile.flush()

    print(f"\n--- Grid Search Finished ---")
    print(f"Summary results saved to: {full_grid_log_path}")

if __name__ == "__main__":
    # --- Example: Run a Single Detailed Simulation ---
    single_run_params = {
        "LINK_LENGTH_METERS": 200.0,
        "relay_connectivity_config": RelayConnectivityConfig.CROSS_ROW,
        "EW_JAMMER_BW_AREA_SELECTION": 3, # Fixed jammer band for this run
        "ew_power_W": 150.0,
        # "csv_filename_prefix": "my_special_run_" # Optional: custom prefix
    }
    run_single_simulation(params_override=single_run_params, detailed_log_filename_id="cross_row_200m_jam3_p150")

    # Wait a moment before starting grid search if desired
    # time.sleep(2) 

    # --- Example: Run a Grid Search ---
    # Define the parameter grid
    # Note: Use Enum members directly, not their names as strings, for type consistency.
    grid_parameters = {
        "LINK_LENGTH_METERS": [50.0, 150.0, 250.0],
        "relay_connectivity_config": [
            RelayConnectivityConfig.MINIMAL,
            RelayConnectivityConfig.CROSS_ROW,
            RelayConnectivityConfig.ALL_CONNECT
        ],
        # Add other parameters you want to vary, e.g.:
        "network_capacity_type": [NetworkCapacity.SMALL, NetworkCapacity.MEDIUM, NetworkCapacity.LARGE],
    }
    run_grid_search(param_grid=grid_parameters, grid_log_filename="grid_summary_main_test.csv")
