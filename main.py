
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

    print("Starting process...")

    print("Generating instances...")
    instances = GI.test_instances()

    print("Getting best parameters for both methods...")
    bestParams_new, bestParams_current = BS.search_best_params("medium_instance")

    print("Solving instances with both methods and best parameters...")
    res_list = BC.process_BC(bestParams_new, bestParams_current,instances)

if __name__ == "__main__":
    main()