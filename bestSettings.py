#---- Librerias ----#
import multiprocessing as mp
import pandas as pd
import re
import cplex

#---- Modulos ----#
import instanceData as ID
import Solver as SLV
import Metodo_Nuevo as MN
import Metodo_Actual as MA
import results as RS

def search_best_params(instance_file):

    params = {
            
        #   ns0 | ns1 | ns2 → nodeselect
        #   hf0 | hf1 | hf5 | hfA → heuristicfreq (A = automático = -1)
        #   cuts0 | cuts1 | cuts2
        #   ps0 | ps1

        # ======================================================
        # BASELINES (control experimental)
        # ======================================================
        "ns1_hfA_cuts0_ps1": {
            "threads": 1,
            "mip.strategy.nodeselect": 1,
            "mip.strategy.heuristicfreq": -1,
            "preprocessing.presolve": 1,
        },

        "ns1_hfA_cuts0_ps0": {
            "threads": 1,
            "mip.strategy.nodeselect": 1,
            "mip.strategy.heuristicfreq": -1,
            "preprocessing.presolve": 0,
        },

        "ns2_hfA_cuts0_ps1": {
            "threads": 1,
            "mip.strategy.nodeselect": 2,
            "mip.strategy.heuristicfreq": -1,
            "preprocessing.presolve": 1,
        },

        # ======================================================
        # CONSERVADORAS (estabilidad / diagnóstico)
        # ======================================================
        "ns0_hf0_cuts0_ps1": {
            "threads": 1,
            "mip.strategy.nodeselect": 0,
            "mip.strategy.heuristicfreq": 0,
            "preprocessing.presolve": 1,
        },

        # ======================================================
        # HEURÍSTICAS (exploración)
        # ======================================================
        "ns0_hf1_cuts0_ps1": {
            "threads": 1,
            "mip.strategy.nodeselect": 0,
            "mip.strategy.heuristicfreq": 1,
            "preprocessing.presolve": 1,
        },

        "ns2_hf5_cuts0_ps1": {
            "threads": 1,
            "mip.strategy.nodeselect": 2,
            "mip.strategy.heuristicfreq": 5,
            "preprocessing.presolve": 1,
        },

        "ns0_hf10_cuts0_ps1": {
            "threads": 1,
            "mip.strategy.nodeselect": 0,
            "mip.strategy.heuristicfreq": 10,
            "preprocessing.presolve": 1,
        },

        # ======================================================
        # CORTES
        # ======================================================
        "ns1_hfA_cuts1_ps1": {
            "threads": 1,
            "mip.strategy.nodeselect": 1,
            "mip.cuts.covers": 1,
            "mip.cuts.flowcovers": 1,
            "mip.cuts.mircut": 1,
            "preprocessing.presolve": 1,
        },

        # ======================================================
        # MIXTOS
        # ======================================================
        "ns2_hf5_cuts1_ps1": {
            "threads": 1,
            "mip.strategy.nodeselect": 2,
            "mip.strategy.heuristicfreq": 5,
            "mip.cuts.covers": 1,
            "mip.cuts.flowcovers": 1,
            "mip.cuts.mircut": 1,
            "preprocessing.presolve": 1,
        },

        # ======================================================
        # RÁPIDOS
        # ======================================================
        "ns0_hf1_cuts0_ps0": {
            "threads": 1,
            "mip.strategy.nodeselect": 0,
            "mip.strategy.heuristicfreq": 1,
            "preprocessing.presolve": 0,
        },

        "ns0_hf1_cuts0_ps1": {
            "threads": 1,
            "mip.strategy.nodeselect": 0,
            "mip.strategy.heuristicfreq": 1,
            "preprocessing.presolve": 1,
        },
    }
    
    res_new = []
    res_current = []
    
    tasks = [
    (param_set, param_values, instance_file)
    for param_set, param_values in params.items()
    ]

    n_cores = mp.cpu_count()

    with mp.Pool(processes=n_cores) as pool:
        res_current = pool.map(process_current, tasks)

    with mp.Pool(processes=n_cores) as pool:
        res_new = pool.map(process_new, tasks)

    
    df_new = pd.DataFrame(res_new)
    df_current = pd.DataFrame(res_current)
    df = pd.concat([df_new, df_current], ignore_index=True)
    df.to_csv("Results/settings_results.csv", index=False)

    best_params_new = top_CPLEX_settings(df_new)
    params_new = {k: v for k, v in params.items() if k in best_params_new}
    print("Best parameters for New Method:", params_new)

    best_params_current = top_CPLEX_settings(df_current)
    params_current = {k: v for k, v in params.items() if k in best_params_current}
    print("Best parameters for Current Method:", params_current)
    
    return params_new, params_current

def process_new(args):
    param_set,param_values,instance_file = args
    print(f"Processing {param_set} for {instance_file} with new method.")
    LOG_FILE = f'logs/CPLEX_Settings/NewMethod_{param_set}.log'
    currentInstance = ID.InstanceData(f"instances/{instance_file}.json")

    param_values["timelimit"] = 1800  # 30 minutes

    probe = MN.prob_metodo_nuevo(currentInstance)
    SLV.apply_params(probe, param_values)
    res = SLV.solve(probe, log_file=LOG_FILE)
    res["instance"] = instance_file
    res["method"] = "New"
    res["settings"] = param_set
    print(f"Finished processing {param_set} for {instance_file} with new method.")
    return res

def process_current(args):
    param_set,param_values,instance_file = args
    print(f"Processing {param_set} for {instance_file} with current method.")
    LOG_FILE = f'logs/CPLEX_Settings/CurrentMethod_{param_set}.log'
    currentInstance = ID.InstanceData(f"instances/{instance_file}.json")

    param_values["timelimit"] = 1800  # 30 minutes

    probe = MA.prob_metodo_actual(currentInstance)
    SLV.apply_params(probe, param_values)
    res = SLV.solve(probe, log_file=LOG_FILE)
    res["instance"] = instance_file
    res["method"] = "Current"
    res["settings"] = param_set
    print(f"Finished processing {param_set} for {instance_file} with current method.")
    return res

def top_CPLEX_settings(
    results,
    w_obj=0.70,
    w_time=0.30
):
    df = results.copy()

    df["objective_value"] = df["objective_value"].fillna(df["objective_value"].max())
    df["time_s"] = df["time_s"].fillna(df["time_s"].max())

    df["bad_status"] = ~df["status"].str.contains(
        "optimal|feasible", case=False, na=False
    )

    df["obj_norm"] = (
        (df["objective_value"] - df["objective_value"].min())
        / (df["objective_value"].max() - df["objective_value"].min() + 1e-9)
    )

    df["time_norm"] = (
        (df["time_s"] - df["time_s"].min())
        / (df["time_s"].max() - df["time_s"].min() + 1e-9)
    )

    df["score"] = (
        w_obj * df["obj_norm"]
        + w_time * df["time_norm"]
        + df["bad_status"] * 10 # Penalizo fuertemente los estados malos
    )

    best_params = set(
        df.sort_values("score", ascending=True)["settings"].head(3)
        )

    return best_params
