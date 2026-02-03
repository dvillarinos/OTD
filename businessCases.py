
#---- Librerias ----#
import re
import cplex
import pandas as pd

#---- Modulos ----#
import generateInstsance as GI
import bestSettings as BS
import instanceData as ID
import Solver as SLV
import Metodo_Actual as MA
import Metodo_Nuevo as MN
import results as RS

# ----------------------------------------------- #
################## TestCases ######################
# ----------------------------------------------- #

def process_BC(params_default, instances):
    res_list = []

    # Resolver instancias con el metodo actual y los parametros por defecto
    for instance_file in instances:
        instance_path = f"Instances/{instance_file}.json"
        currentInstance = ID.InstanceData(instance_path)

        # Resolver con metodo actual
        probe = MA.prob_metodo_actual(currentInstance)
        SLV.apply_params(probe, params_default)
        res = SLV.solve(probe,mipgap=0.005,log_file=f"Logs/{instance_file}_actualMethod_noMethodSettings.log")
        res["instance"] = instance_file
        res["method"] = "Actual"
        res["settings"] = "-"
        print(f"Solved {instance_file} with Actual Method. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
        res_list.append(res)

        # Resolver con metodo nuevo
        probe = MN.prob_metodo_nuevo(currentInstance)
        SLV.apply_params(probe, params_default)
        res = SLV.solve(probe,mipgap=0.005,log_file=f"Logs/{instance_file}_newMethod_noMethodSettings.log")
        res["instance"] = instance_file
        res["method"] = "Nuevo"
        res["settings"] = "-"
        print(f"Solved {instance_file} with New Method - All restrictions. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
        res_list.append(res)

        # Resolver con metodo nuevo -  Sin restricciones de 1/3 con el camion
        probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=False, Max_Bicis=None, Max_Autos=None, Exclusivos=True, Refrigerados=True,Precedencia=True))
        SLV.apply_params(probe, params_default)
        res = SLV.solve(probe,mipgap=0.005,log_file=f"Logs/{instance_file}_newMethod_Min_No_Camion=False.log")
        res["instance"] = instance_file
        res["method"] = "Nuevo"
        res["settings"] = "Min_No_Camion=False"
        print(f"Solved {instance_file} with New Method - Min_No_Camion=False. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
        res_list.append(res)

        # Resolver con metodo nuevo -  Sin Exclusivos
        probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=True, Max_Bicis=None, Max_Autos=None, Exclusivos=False, Refrigerados=True,Precedencia=True))
        SLV.apply_params(probe, params_default)
        res = SLV.solve(probe,mipgap=0.005,log_file=f"Logs/{instance_file}_newMethod_Exclusivos=False.log")
        res["instance"] = instance_file
        res["method"] = "Nuevo"
        res["settings"] = "Exclusivos=False"
        print(f"Solved {instance_file} with New Method - Exclusivos=False. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
        res_list.append(res)

        # Resolver con metodo nuevo -  Sin Refrigerados
        probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=True, Max_Bicis=None, Max_Autos=None, Exclusivos=True, Refrigerados=False,Precedencia=True))
        SLV.apply_params(probe, params_default)
        res = SLV.solve(probe,mipgap=0.005,log_file=f"Logs/{instance_file}_newMethod_Refrigerados=False.log")
        res["instance"] = instance_file
        res["method"] = "Nuevo"
        res["settings"] = "Refrigerados=False"
        print(f"Solved {instance_file} with New Method - Refrigerados=False. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
        res_list.append(res)

        # Resolver con metodo nuevo -  Sin Precedencia
        probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=True, Max_Bicis=None, Max_Autos=None, Exclusivos=True, Refrigerados=True,Precedencia=False))
        SLV.apply_params(probe, params_default)
        res = SLV.solve(probe,mipgap=0.005,log_file=f"Logs/{instance_file}_newMethod_Precedencia=False.log")
        res["instance"] = instance_file
        res["method"] = "Nuevo"
        res["settings"] = "Precedencia=False"
        print(f"Solved {instance_file} with New Method - Precedencia=False. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
        res_list.append(res)

        # Resolver con metodo nuevo -  Sin Min_No_Camion y tope Bicis/Autos
        Max_Bicis = int(currentInstance.N() * 0.5)
        Max_Autos = Max_Bicis // 2

        probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=False, Max_Bicis=Max_Bicis, Max_Autos=Max_Autos, Exclusivos=True, Refrigerados=True,Precedencia=False))
        SLV.apply_params(probe, params_default)
        res = SLV.solve(probe,mipgap=0.005,log_file=f"Logs/{instance_file}_newMethod_Min_No_Camion=False_MaxBicis_MaxAutos.log")
        res["instance"] = instance_file
        res["method"] = "Nuevo"
        res["settings"] = f"Min_No_Camion=False, Max_Bicis={Max_Bicis}, Max_Autos={Max_Autos}"
        print(f"Solved {instance_file} with New Method - Min_No_Camion=False y tope Bicis/Autos. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
        res_list.append(res)

    df = pd.DataFrame(res_list)
    df.to_csv("Results/results.csv", index=False)
    RS.make_tables(df)

    return res_list