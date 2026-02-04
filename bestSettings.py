#---- Librerias ----#
import multiprocessing as mp
import pandas as pd
import re
import cplex

#---- Modulos ----#
import instanceData as ID
import Solver as SLV
import Metodo_Nuevo as MN
import results as RS

def search_best_params(instance_file):

    params = {
            
        #   ns0 | ns1 | ns2 → nodeselect
        #   hf0 | hf1 | hf5 | hfA → heuristicfreq (A = automático = -1)
        #   cuts0 | cuts1 | cuts2
        #   ps0 | ps1

            # ======================================================
            # BASELINES
            # ======================================================

            "ns1_hfA_cuts0_ps1_baseline": {
                "threads": 1,
                "mip.strategy.nodeselect": 1,
                "mip.strategy.heuristicfreq": -1,
                "preprocessing.presolve": 1,
            },

            "ns1_hfA_cuts0_ps0_baseline_sin_presolve": {
                "threads": 1,
                "mip.strategy.nodeselect": 1,
                "mip.strategy.heuristicfreq": -1,
                "preprocessing.presolve": 0,
            },

            # ======================================================
            # CONSERVADORAS
            # ======================================================

            "ns0_hf0_cuts0_ps0_conservadora": {
                "threads": 1,
                "mip.strategy.nodeselect": 0,
                "mip.strategy.heuristicfreq": 0,
                "mip.cuts.covers": 0,
                "mip.cuts.flowcovers": 0,
                "mip.cuts.mircut": 0,
                "preprocessing.presolve": 0,
            },
            "ns0_hf0_cuts0_ps1_conservadora_ps": {
                "threads": 1,
                "mip.strategy.nodeselect": 0,
                "mip.strategy.heuristicfreq": 0,
                "mip.cuts.covers": 0,
                "mip.cuts.flowcovers": 0,
                "mip.cuts.mircut": 0,
                "preprocessing.presolve": 1,
            },

            # ======================================================
            # HEURÍSTICAS
            # ======================================================
            "ns0_hf1_cuts0_ps1_heur_agresiva": {
                "threads": 1,
                "mip.strategy.nodeselect": 0,
                "mip.strategy.heuristicfreq": 1,
                "preprocessing.presolve": 1,
            },

            "ns2_hf5_cuts0_ps1_heur_moderada": {
                "threads": 1,
                "mip.strategy.nodeselect": 2,
                "mip.strategy.heuristicfreq": 5,
                "preprocessing.presolve": 1,
            },

            "ns0_hf1_cuts0_ps0_heur_agresiva_sin_ps": {
                "threads": 1,
                "mip.strategy.nodeselect": 0,
                "mip.strategy.heuristicfreq": 1,
                "preprocessing.presolve": 0,
            },

            # ======================================================
            # CORTES
            # ======================================================
            "ns1_hfA_cuts1_ps1_cuts_moderados": {
                "threads": 1,
                "mip.strategy.nodeselect": 1,
                "mip.cuts.covers": 1,
                "mip.cuts.flowcovers": 1,
                "mip.cuts.mircut": 1,
                "preprocessing.presolve": 1,
            },

            "ns1_hfA_cuts2_ps1_cuts_agresivos": {
                "threads": 1,
                "mip.strategy.nodeselect": 1,
                "mip.cuts.covers": 2,
                "mip.cuts.flowcovers": 2,
                "mip.cuts.mircut": 2,
                "preprocessing.presolve": 1,
            },

            "ns1_hfA_cuts2_ps0_cuts_agresivos_sin_ps": {
                "threads": 1,
                "mip.strategy.nodeselect": 1,
                "mip.cuts.covers": 2,
                "mip.cuts.flowcovers": 2,
                "mip.cuts.mircut": 2,
                "preprocessing.presolve": 0,
            },

            # ======================================================
            # MIXTOS
            # ======================================================
            "ns2_hf5_cuts1_ps1_mixto_balanceado": {
                "threads": 1,
                "mip.strategy.nodeselect": 2,
                "mip.strategy.heuristicfreq": 5,
                "mip.cuts.covers": 1,
                "mip.cuts.flowcovers": 1,
                "mip.cuts.mircut": 1,
                "preprocessing.presolve": 1,
            },

            "ns1_hf1_cuts2_ps1_mixto_agresivo": {
                "threads": 1,
                "mip.strategy.nodeselect": 1,
                "mip.strategy.heuristicfreq": 1,
                "mip.cuts.covers": 2,
                "mip.cuts.flowcovers": 2,
                "mip.cuts.mircut": 2,
                "preprocessing.presolve": 1,
            },

            # ======================================================
            # RÁPIDOS
            # ======================================================
            "ns0_hf1_cuts1_ps1_rapido": {
                "threads": 1,
                "mip.strategy.nodeselect": 0,
                "mip.strategy.heuristicfreq": 1,
                "mip.cuts.covers": 1,
                "preprocessing.presolve": 1,
            },

            "ns0_hf1_cuts0_ps0_rapido_minimo": {
                "threads": 1,
                "mip.strategy.nodeselect": 0,
                "mip.strategy.heuristicfreq": 1,
                "preprocessing.presolve": 0,
            },
        }
    
    settings_res = []
    
    tasks = [
    (param_set, param_values, instance_file,settings_res)
    for param_set, param_values in params.items()
    ]

    n_cores = mp.cpu_count()

    with mp.Pool(processes=n_cores) as pool:
        settings_res = pool.map(process, tasks)

    df = pd.DataFrame(settings_res)
    df.to_csv("Results/settings_results.csv", index=False)
    RS.make_tables(df)

    return None

def process(args):
    param_set,param_values,instance_file,settings_res = args
    LOG_FILE = f'logs/CPLEX_Settings/NewMethod_{param_set}.log'
    currentInstance = ID.InstanceData(f"instances/{instance_file}.json")

    probe = MN.prob_metodo_nuevo(currentInstance)
    SLV.apply_params(probe, param_values)
    res = SLV.solve(probe, log_file=LOG_FILE)
    res["instance"] = instance_file
    res["method"] = "Nuevo"
    res["settings"] = param_set
    settings_res.append(res)

    return None