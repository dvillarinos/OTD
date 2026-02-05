
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

def process_BC(bestParams_new, bestParams_current, instances ,gap=1e-4):
    res_list = []

    # Resolver instancias con el metodo actual y los parametros por defecto
    for instance_file in instances:
        instance_path = f"instances/{instance_file}.json"
        currentInstance = ID.InstanceData(instance_path)

        N0 = 25
        T0 = 60
        alpha = 1.7

        
        # Resolver con metodo actual
        for params, params_values in bestParams_current.items():
            params = params.replace(" ","")
            params_values["threads"] = 0
            params_values["timelimit"] = int(T0 * (currentInstance.N() / N0) ** alpha) # type: ignore
            print(f"Solving instance: {instance_file} with Actual Method - Params: {params}.")
            probe = MA.prob_metodo_actual(currentInstance)
            SLV.apply_params(probe, params_values)
            res = SLV.solve(probe,mipgap=gap,log_file=f"logs/BC/{params}_{instance_file}_actualMethod_allRest.log")
            res["instance"] = instance_file
            res["method"] = "Actual"
            res["restrictions"] = "All"
            res["CPLEX_settings"] = {params}
            print(f"Solved {instance_file} with Actual Method - Params: {params}. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
            res_list.append(res)

        # Resolver con metodo nuevo
        for params, params_values in bestParams_new.items():
            params = params.replace(" ","")
            params_values["threads"] = 0
            params_values["timelimit"] = int(T0 * (currentInstance.N() / N0) ** alpha) # type: ignore
            print(f"Solving instance: {instance_file} with New Method - Params: {params}.")
            probe = MN.prob_metodo_nuevo(currentInstance)
            SLV.apply_params(probe, params_values)
            res = SLV.solve(probe,mipgap=gap,log_file=f"logs/BC/{params}_{instance_file}_newMethod_allRest.log")
            res["instance"] = instance_file
            res["method"] = "Nuevo"
            res["restrictions"] = "All"
            res["CPLEX_settings"] = {params}
            print(f"Solved {instance_file} with New Method - All restrictions - Params: {params}. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
            res_list.append(res)

            # Resolver con metodo nuevo -  Sin restricciones de 1/3 con el camion
            print(f"Solving instance: {instance_file} with New Method and Min_No_Camion=False - Params: {params}.")
            probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=False, Max_Bicis=None, Max_Autos=None, Exclusivos=True, Refrigerados=True,Precedencia=True))
            SLV.apply_params(probe, params_values)
            res = SLV.solve(probe,mipgap=gap,log_file=f"logs/BC/{params}_{instance_file}_newMethod_Min_No_Camion=False.log")
            res["instance"] = instance_file
            res["method"] = "Nuevo"
            res["restrictions"]  = "Min_No_Camion=False"
            res["CPLEX_settings"] = {params}
            print(f"Solved {instance_file} with New Method - Min_No_Camion=False - Params: {params}. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
            res_list.append(res)

            # Resolver con metodo nuevo -  Sin Exclusivos
            print(f"Solving instance: {instance_file} with New Method and Exclusivos=False - Params: {params}.")
            probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=True, Max_Bicis=None, Max_Autos=None, Exclusivos=False, Refrigerados=True,Precedencia=True))
            SLV.apply_params(probe, params_values)
            res = SLV.solve(probe,mipgap=gap,log_file=f"logs/BC/{params}_{instance_file}_newMethod_Exclusivos=False.log")
            res["instance"] = instance_file
            res["method"] = "Nuevo"
            res["restrictions"]  = "Exclusivos=Disabled"
            res["CPLEX_settings"] = {params}
            print(f"Solved {instance_file} with New Method - Exclusivos=False - Params: {params}. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
            res_list.append(res)

            # Resolver con metodo nuevo -  Sin Refrigerados
            print(f"Solving instance: {instance_file} with New Method and Refrigerados=False - Params: {params}.")
            probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=True, Max_Bicis=None, Max_Autos=None, Exclusivos=True, Refrigerados=False,Precedencia=True))
            SLV.apply_params(probe, params_values)
            res = SLV.solve(probe,mipgap=gap,log_file=f"logs/BC/{params}_{instance_file}_newMethod_Refrigerados=False.log")
            res["instance"] = instance_file
            res["method"] = "Nuevo"
            res["restrictions"]  = "Refrigerados=Disabled"
            res["CPLEX_settings"] = {params}
            print(f"Solved {instance_file} with New Method - Refrigerados=False - Params: {params}. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
            res_list.append(res)

            # Resolver con metodo nuevo -  Sin Precedencia
            print(f"Solving instance: {instance_file} with New Method and Precedencia=False - Params: {params}.")
            probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=True, Max_Bicis=None, Max_Autos=None, Exclusivos=True, Refrigerados=True,Precedencia=False))
            SLV.apply_params(probe, params_values)
            res = SLV.solve(probe,mipgap=gap,log_file=f"logs/BC/{params}_{instance_file}_newMethod_Precedencia=False.log")
            res["instance"] = instance_file
            res["method"] = "Nuevo"
            res["restrictions"]  = "Precedencia=Disabled"
            res["CPLEX_settings"] = {params}
            print(f"Solved {instance_file} with New Method - Precedencia=False - Params: {params}. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
            res_list.append(res)

            # Resolver con metodo nuevo -  Sin Min_No_Camion y tope Bicis/Autos
            Max_Bicis = int(currentInstance.N() * 0.5)
            Max_Autos = Max_Bicis // 2

            print(f"Solving instance: {instance_file} with New Method and Min_No_Camion=False, Max_Bicis={Max_Bicis}, Max_Autos={Max_Autos} - Params: {params}.")

            probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=False, Max_Bicis=Max_Bicis, Max_Autos=Max_Autos, Exclusivos=True, Refrigerados=True,Precedencia=False))
            SLV.apply_params(probe, params_values)
            res = SLV.solve(probe,mipgap=gap,log_file=f"logs/BC/{params}_{instance_file}_newMethod_Min_No_Camion=False_MaxBicis_MaxAutos.log")
            res["instance"] = instance_file
            res["method"] = "Nuevo"
            res["restrictions"]  = f"Min_No_Camion=Disabled, Max_Bicis={Max_Bicis}, Max_Autos={Max_Autos}"
            res["CPLEX_settings"] = {params}
            print(f"Solved {instance_file} with New Method - Min_No_Camion=False, Max_Bicis={Max_Bicis}, Max_Autos={Max_Autos} - Params: {params}. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
            res_list.append(res)

            # Resolver con metodo nuevo -  Sin Restricciones
            print(f"Solving instance: {instance_file} with New Method and All restrictions disabled - Params: {params}.")
            probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=False, Max_Bicis=None, Max_Autos=None, Exclusivos=False, Refrigerados=False,Precedencia=False))
            SLV.apply_params(probe, params_values)
            res = SLV.solve(probe,mipgap=gap,log_file=f"logs/BC/{params}_{instance_file}_newMethod_AllRestrDisabled.log")
            res["instance"] = instance_file
            res["method"] = "Nuevo"
            res["restrictions"]  = "All restrictions disabled"
            res["CPLEX_settings"] = {params}
            print(f"Solved {instance_file} with New Method - All restrictions disabled - Params: {params}. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
            res_list.append(res)

            # Resolver con metodo nuevo -  Solo Exclusivos
            print(f"Solving instance: {instance_file} with New Method and Exclusivos=Enabled - Params: {params}.")
            probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=False, Max_Bicis=None, Max_Autos=None, Exclusivos=True, Refrigerados=False,Precedencia=False))
            SLV.apply_params(probe, params_values)
            res = SLV.solve(probe,mipgap=gap,log_file=f"logs/BC/{params}_{instance_file}_newMethod_Exclusivos=Enabled.log")
            res["instance"] = instance_file
            res["method"] = "Nuevo"
            res["restrictions"]  = "Exclusivos=Enabled"
            res["CPLEX_settings"] = {params}
            print(f"Solved {instance_file} with New Method - Exclusivos=Enabled - Params: {params}. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
            res_list.append(res)


            # Resolver con metodo nuevo -  Solo Refrigerados
            print(f"Solving instance: {instance_file} with New Method and Refrigerados=Enabled - Params: {params}.")
            probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=False, Max_Bicis=None, Max_Autos=None, Exclusivos=False, Refrigerados=True,Precedencia=False))
            SLV.apply_params(probe, params_values)
            res = SLV.solve(probe,mipgap=gap,log_file=f"logs/BC/{params}_{instance_file}_newMethod_Refrigerados=Enabled.log")
            res["instance"] = instance_file
            res["method"] = "Nuevo"
            res["restrictions"]  = "Refrigerados=Enabled"
            res["CPLEX_settings"] = {params}
            print(f"Solved {instance_file} with New Method - Refrigerados=Enabled - Params: {params}. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
            res_list.append(res)


            # Resolver con metodo nuevo -  Solo Precedencia
            print(f"Solving instance: {instance_file} with New Method and Precedencia=Enabled - Params: {params}.")
            probe = MN.prob_metodo_nuevo(currentInstance,MN.settings(Min_No_Camion=False, Max_Bicis=None, Max_Autos=None, Exclusivos=False, Refrigerados=False,Precedencia=True))
            SLV.apply_params(probe, params_values)
            res = SLV.solve(probe,mipgap=gap,log_file=f"logs/BC/{params}_{instance_file}_newMethod_Precedencia=Enabled.log")
            res["instance"] = instance_file
            res["method"] = "Nuevo"
            res["restrictions"]  = "Precedencia=Enabled"
            res["CPLEX_settings"] = {params}
            print(f"Solved {instance_file} with New Method - Precedencia=Enabled - Params: {params}. Status: {res['status']}, Time: {res['time_s']}, Obj: {res['objective_value']}")
            res_list.append(res)


    df = pd.DataFrame(res_list)
    df.to_csv("Results/results.csv", index=False)

    return res_list