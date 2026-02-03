
#---- Librerias ----#
import cplex

#---- Modulos ----#
import generateInstsance as GI
import instanceData as ID
import Solver as SLV
import Metodo_Actual as MA
import Metodo_Nuevo as MN

# ----------------------------------------------- #
##################### main ########################
# ----------------------------------------------- #

def main():
    # Generar instancias de prueba
    GI.test_instances()
    instances = [
        "small_instance",
        "medium_instance",
        "medium_instance_large_dist",
        "large_instance"
    ]
    
    params_default = {

        # --- General ---
        "threads": 0,

        # --- Presolve ---
        "preprocessing.presolve": 1,

        # --- Cuts ---
        "mip.cuts.mircut": 1,
        "mip.cuts.gomory": 1,
        "mip.cuts.flowcovers": 1,
        "mip.cuts.implied": 1,

        # --- Heurísticas ---
        "mip.strategy.heuristicfreq": 10,

        # --- Estrategia de búsqueda ---
        "mip.strategy.search": (
            cplex.Cplex().parameters.mip.strategy.search.values.traditional
        ),

        # --- Selección de nodos ---
        "mip.strategy.nodeselect": 1,

        # --- Branching ---
        "mip.strategy.variableselect": 3
    }

    res_list = []

    """
    # Resolver instancias con el metodo actual y los parametros por defecto
    for instance_file in instances:
        instance_path = f"Instances/{instance_file}.json"
        currentInstance = ID.InstanceData(instance_path)

        probe = MA.prob_metodo_actual(currentInstance)
        SLV.apply_params(probe, params_default)
        res = SLV.solve(probe,mipgap=0.005,log_file=f"Logs/{instance_file}_actualMethod_noMethodSettings.log")
        res["instance"] = instance_file
        res["method"] = "Actual"
        res["settings"] = "-"
        res_list.append(res)
    """

    # Resolver instancias con el metodo nuevo y los parametros por defecto
    for instance_file in instances:
        instance_path = f"Instances/{instance_file}.json"
        currentInstance = ID.InstanceData(instance_path)

        probe = MN.prob_metodo_nuevo(currentInstance)
        SLV.apply_params(probe, params_default)
        res = SLV.solve(probe,mipgap=0.005,log_file=f"Logs/{instance_file}_newMethod_noMethodSettings.log")
        res["instance"] = instance_file
        res["method"] = "Nuevo"
        res["settings"] = "-"
        res_list.append(res)

    print(res_list)

if __name__ == "__main__":
    main()