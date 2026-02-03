
#---- Librerias ----#
import cplex
import pandas as pd

#---- Modulos ----#
import generateInstsance as GI
import bestSettings as BS
import businessCases as BC
import instanceData as ID
import Solver as SLV
import Metodo_Actual as MA
import Metodo_Nuevo as MN
import results as RS

# ----------------------------------------------- #
##################### main ########################
# ----------------------------------------------- #

def main():
    # Generar instancias de prueba
    
    instances = GI.test_instances()

    params_compare = BS.search_best_params("medium_instance")
    
    """
    params_default = {
        "threads": 0,
        "preprocessing.presolve": 1,
        "mip.cuts.mircut": 1,
        "mip.cuts.gomory": 1,
        "mip.cuts.flowcovers": 1,
        "mip.cuts.implied": 1,
        "mip.strategy.heuristicfreq": 10,
        "mip.strategy.search": (cplex.Cplex().parameters.mip.strategy.search.values.traditional),
        "mip.strategy.nodeselect": 1,
        "mip.strategy.variableselect": 3
    }

    res_list = BC.process_BC(params_default,instances)
    """
if __name__ == "__main__":
    main()