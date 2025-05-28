import csv
import itertools
import os
import time
import numpy as np
import swarm_simulation as sim
from enum import Enum
from multiprocessing import Pool, cpu_count

# Import Enums from the simulation module to be used in param_grid
from swarm_simulation import NetworkCapacity, RelayConnectivityConfig

# List of the available modulation types as defined in RF_simulation
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
    current_sim_params["run_id_for_filename"] = detailed_log_filename_id

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

def run_single_grid_simulation(args):
    """
    Helper function to run a single simulation for the grid search.
    Args:
        args: Tuple of (index, total_runs, param_names, param_values, default_params)
    Returns:
        Dictionary containing the log row data for CSV writing.
    """
    idx, total_runs, param_names, param_values, default_params = args
    print(f"\nRunning grid search {idx+1}/{total_runs} with params: {dict(zip(param_names, param_values))}")
    
    # Start with default parameters
    current_run_specific_params = default_params.copy()
    
    # Apply the current combination of grid parameters
    grid_specific_overrides = dict(zip(param_names, param_values))
    current_run_specific_params.update(grid_specific_overrides)
    
    # Ensure logging and detailed CSV are off for grid search runs for speed
    if "logging_enabled" not in grid_specific_overrides:
        current_run_specific_params["logging_enabled"] = False
    if "csv_output_enabled" not in grid_specific_overrides:
        current_run_specific_params["csv_output_enabled"] = False
    
    # Prepare the log_row with all parameters
    log_row = current_run_specific_params.copy()
    for key, val in log_row.items():
        if isinstance(val, Enum):
            log_row[key] = val.name

    try:
        simulation_instance = sim.SwarmSimulation(params_override=current_run_specific_params)
        results = simulation_instance.run_simulation()
        
        # Add results to the log_row
        log_row["EW_JAMMER_ACTUAL_BW_AREA"] = simulation_instance.jam_band_idx
        r1_results = results.get('r1_leader_dist_to_ew', {})
        r2_results = results.get('r2_leader_dist_to_ew', {})
        for mod in modulation_types:
            log_row[f"r1_{mod.lower()}_m"] = r1_results.get(mod, np.nan)
            log_row[f"r2_{mod.lower()}_m"] = r2_results.get(mod, np.nan)
        log_row["all_drones_passed_ew_x"] = results.get('all_drones_passed_ew_x', False)
        log_row["final_step"] = results.get('final_step', 0)
        log_row["num_initial_susceptible_links"] = results.get('num_initial_susceptible_links', 0)
        log_row["num_disconnected_susceptible_links"] = results.get('num_disconnected_susceptible_links', 0)
        
    except Exception as e:
        print(f"!!! ERROR during simulation run {idx+1} with params {current_run_specific_params}: {e}")
        log_row["EW_JAMMER_ACTUAL_BW_AREA"] = "ERROR"
        for mod in modulation_types:
            log_row[f"r1_{mod.lower()}_m"] = "ERROR"
            log_row[f"r2_{mod.lower()}_m"] = "ERROR"
        log_row["all_drones_passed_ew_x"] = "ERROR"
        log_row["final_step"] = "ERROR"
        log_row["num_initial_susceptible_links"] = "ERROR"
        log_row["num_disconnected_susceptible_links"] = "ERROR"
    
    return log_row

def run_grid_search(param_grid, grid_log_filename="grid_search_summary.csv", parallel=False):
    """
    Runs multiple simulations based on a grid of parameters, with option for parallel or sequential execution.
    
    Args:
        param_grid (dict): Keys are parameter names (matching SIM_PARAMS keys),
                           values are lists of values to test for that parameter.
        grid_log_filename (str): Name of the CSV file to save summary results.
        parallel (bool): If True, use multiprocessing for parallel execution; if False, run sequentially.
    """
    execution_mode = "Parallel" if parallel else "Sequential"
    print(f"\n--- Starting {execution_mode} Grid Search ---")
    param_names = list(param_grid.keys())
    param_value_lists = list(param_grid.values())
    
    all_combinations = list(itertools.product(*param_value_lists))
    total_runs = len(all_combinations)
    print(f"Total simulation runs in grid search: {total_runs}")

    # Construct summary_fieldnames
    fieldnames_list = list(param_names)
    fieldnames_set = set(param_names)
    additional_and_result_fields = [
        "logging_enabled",
        "csv_output_enabled",
        "EW_JAMMER_ACTUAL_BW_AREA",
        "all_drones_passed_ew_x",
        "final_step",
        "num_initial_susceptible_links",
        "num_disconnected_susceptible_links"
    ]
    for mod in modulation_types:
        additional_and_result_fields.append(f"r1_{mod.lower()}_m")
        additional_and_result_fields.append(f"r2_{mod.lower()}_m")
    
    for field in additional_and_result_fields:
        if field not in fieldnames_set:
            fieldnames_list.append(field)
    
    summary_fieldnames = fieldnames_list
    
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    full_grid_log_path = os.path.join(log_dir, grid_log_filename)
    
    # Prepare arguments and default parameters
    default_params = sim.DEFAULT_SIM_PARAMS.copy()
    results = []

    if parallel:
        # Parallel execution using multiprocessing
        num_processes = min(cpu_count(), 16)  # Cap at 16 to prevent excessive resource use
        print(f"Using {num_processes} processes for parallel execution")
        run_args = [(i, total_runs, param_names, param_values, default_params)
                    for i, param_values in enumerate(all_combinations)]
        
        with Pool(processes=num_processes) as pool:
            results = pool.map(run_single_grid_simulation, run_args)
    else:
        # Sequential execution
        for i, param_values in enumerate(all_combinations):
            log_row = run_single_grid_simulation((i, total_runs, param_names, param_values, default_params))
            results.append(log_row)
    
    # Write results to CSV
    with open(full_grid_log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_fieldnames)
        writer.writeheader()
        for log_row in results:
            # Ensure only summary_fieldnames keys are written
            final_log_row = {sf: log_row.get(sf, "MISSING") for sf in summary_fieldnames}
            writer.writerow(final_log_row)
        csvfile.flush()
    
    print(f"\n--- {execution_mode} Grid Search Finished ---")
    print(f"Summary results saved to: {full_grid_log_path}")

if __name__ == "__main__":
    # --- Example: Run a Single Detailed Simulation ---
    single_run_params = {
        "LINK_LENGTH_METERS": 200.0,
        "relay_connectivity_config": RelayConnectivityConfig.CROSS_ROW,
        "EW_JAMMER_BW_AREA_SELECTION": 3,
        "ew_power_W": 150.0,
    }
    # run_single_simulation(params_override=single_run_params, detailed_log_filename_id="cross_row_200m_jam3_p150")

    # --- Example: Run a Parallel Grid Search ---
    grid_parameters = {
        "EW_JAMMER_BW_AREA_SELECTION": [1, 2, 3, 4],
        "LINK_LENGTH_METERS": list(range(10, 301, 10)),
        "relay_connectivity_config": [
            RelayConnectivityConfig.MINIMAL,
            RelayConnectivityConfig.CROSS_ROW,
            RelayConnectivityConfig.ALL_CONNECT
        ],
        "network_capacity_type": [NetworkCapacity.SMALL, NetworkCapacity.MEDIUM, NetworkCapacity.LARGE],
    }
    print("\nRunning grid search in parallel mode:")
    run_grid_search(param_grid=grid_parameters, grid_log_filename="grid_summary_parallel.csv", parallel=True)
