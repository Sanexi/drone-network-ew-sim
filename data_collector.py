import csv
import itertools
import os
import time
import numpy as np
import pandas as pd
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
        rs1_results = results.get('rs1_leader_dist_to_ew', {})
        rs2_results = results.get('rs2_leader_dist_to_ew', {})
        ra1_results = results.get('ra1_leader_dist_to_ew', {})
        ra2_results = results.get('ra2_leader_dist_to_ew', {})
        for mod in modulation_types:
            log_row[f"r1_{mod.lower()}_m"] = r1_results.get(mod, np.nan)
            log_row[f"r2_{mod.lower()}_m"] = r2_results.get(mod, np.nan)
            log_row[f"rs1_{mod.lower()}_m"] = rs1_results.get(mod, np.nan)
            log_row[f"rs2_{mod.lower()}_m"] = rs2_results.get(mod, np.nan)
            log_row[f"ra1_{mod.lower()}_m"] = ra1_results.get(mod, np.nan)
            log_row[f"ra2_{mod.lower()}_m"] = ra2_results.get(mod, np.nan)
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
            log_row[f"rs1_{mod.lower()}_m"] = "ERROR"
            log_row[f"rs2_{mod.lower()}_m"] = "ERROR"
            log_row[f"ra1_{mod.lower()}_m"] = "ERROR"
            log_row[f"ra2_{mod.lower()}_m"] = "ERROR"
        log_row["all_drones_passed_ew_x"] = "ERROR"
        log_row["final_step"] = "ERROR"
        log_row["num_initial_susceptible_links"] = "ERROR"
        log_row["num_disconnected_susceptible_links"] = "ERROR"
    
    return log_row

def run_grid_search(param_grid={}, grid_log_filename="grid_search_summary.csv", parallel=False):
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
        additional_and_result_fields.append(f"rs1_{mod.lower()}_m")
        additional_and_result_fields.append(f"rs2_{mod.lower()}_m")
        additional_and_result_fields.append(f"ra1_{mod.lower()}_m")
        additional_and_result_fields.append(f"ra2_{mod.lower()}_m")
    
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



# -----------------------------------------------------------
# 1)  Per-run worker  –  returns one DataFrame
# -----------------------------------------------------------
def run_surface_one(args):
    """
    args = (topology_enum, jammer_band, cap_req_bits_per_sec)

    For every step this records, per modulation, the *average*
    normalised-throughput across **all susceptible links** in that
    jammer band.
    """
    topology, band, cap_req = args

    params = sim.DEFAULT_SIM_PARAMS.copy()
    params.update({
        "relay_connectivity_config":   topology,
        "EW_JAMMER_BW_AREA_SELECTION": band,
        "link_bw_hz":                  cap_req * 4,
        "capacity_requirement":        cap_req,
        "logging_enabled":             False,
        "csv_output_enabled":          False,
    })

    L_ref = params["LINK_LENGTH_METERS"]
    sim_inst = sim.SwarmSimulation(params_override=params)

    dist_trace  = []
    traces_mean = {m: [] for m in modulation_types}   # time-series

    # ------------------------------------------------- run the sim
    while True:
        done = sim_inst.step()

        # distance leader ↔ jammer
        leader_x = sim_inst.drones[sim_inst.leader_id].pos[0]
        dist = abs(leader_x - params["ew_location"][0])
        dist_trace.append(dist)

        # accumulate normalised throughput over all susceptible links
        acc = {m: [] for m in modulation_types}
        for u, v, e in sim_inst.graph.edges(data=True):
            if not e["is_ew_susceptible"]:
                continue
            L_uv = e["link_length"]
            scale = (L_uv / L_ref) ** 2
            for mod in modulation_types:
                acc[mod].append(e["current_capacities"][mod] * scale)

        # store step-average (mean over susceptible links, NaN if none)
        for mod in modulation_types:
            traces_mean[mod].append(
                np.nan if not acc[mod] else np.mean(acc[mod])
            )

        if done:
            break

    # ------------------------------------------------- build DataFrame
    rows = []
    for k, dist in enumerate(dist_trace):
        row = {
            "topology":   topology.name,
            "band":       band,
            "cap_req":    cap_req,
            "bw_hz":      params["link_bw_hz"],
            "distance_m": dist,
        }
        for mod in modulation_types:
            row[f"tp_{mod}"] = traces_mean[mod][k]
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------
# 2)  Sweep builder  –  averages the four bands
# -----------------------------------------------------------
def build_throughput_surface(parallel: bool = True):
    """
    Sweeps
        • topology ∈ {MINIMAL, CROSS_ROW, ALL_CONNECT}
        • band     ∈ {1,2,3,4}
        • cap_req  ∈ 50 points 0.05 → 5 Mb/s
    Records the *average* normalised-throughput across susceptible
    links (see run_surface_one), then averages those curves over the
    four bands and writes results/throughput_surface.csv
    """
    cap_reqs = np.linspace(0.05e6, 5e6, 50)

    combos = list(itertools.product(
        RelayConnectivityConfig,       # 3 topologies
        [1, 2, 3, 4],                  # 4 jammer bands
        cap_reqs                       # 50 capacities
    ))

    if parallel:
        with Pool(min(cpu_count(), 16)) as pool:
            dfs = pool.map(run_surface_one, combos)
    else:
        dfs = [run_surface_one(c) for c in combos]

    big = pd.concat(dfs, ignore_index=True)

    # long format: one row per modulation
    long = big.melt(
        id_vars=["topology", "band", "cap_req", "bw_hz", "distance_m"],
        value_vars=[f"tp_{m}" for m in modulation_types],
        var_name="modulation",
        value_name="throughput_bps"
    )
    long["modulation"] = long["modulation"].str.replace("tp_", "", regex=False)

    # average over the four jammer bands
    surface = (
        long
        .groupby(["topology", "modulation", "bw_hz", "distance_m"], as_index=False)
        .throughput_bps
        .mean()
    )

    surface.to_csv("results/throughput_surface.csv", index=False)
    print("→ Saved averaged-band surface to results/throughput_surface.csv")
    return surface


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
    capacity_requirements = [100000]#np.linspace(50e3, 5e6, 100).astype(int)
    grid_parameters = {
        "EW_JAMMER_BW_AREA_SELECTION": [1],#, 2, 3, 4],
        "LINK_LENGTH_METERS": [100],#list(range(10, 501, 20)),
        "relay_connectivity_config": [
            RelayConnectivityConfig.MINIMAL,
            RelayConnectivityConfig.CROSS_ROW,
            RelayConnectivityConfig.ALL_CONNECT
        ],
        "capacity_requirement": capacity_requirements#.tolist()
    }
    print("\nRunning grid search in parallel mode:")
    run_grid_search(grid_parameters, grid_log_filename="grid_summary_parallelv4.csv", parallel=True)

    # Example usage for collecting throughput surface:
    # print("\nRunning continuous-capacity throughput-surface sweep …")
    # build_throughput_surface(parallel=True)
    # print("✓ Done building throughput surface.  File: results/throughput_surface.csv")
