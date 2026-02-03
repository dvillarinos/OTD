#---- Librerias ----#
import time

# ----------------------------------------------- #
############# Parámetros + solve ##################
# ----------------------------------------------- #

def apply_params(prob, params):
    for k, v in params.items():
        eval(f"prob.parameters.{k}.set({repr(v)})")

def solve(prob, mipgap=5e-3, log_file=None):
    # 1. Configuración de CPLEX
    if log_file is not None:
        log = open(log_file, "w")
        prob.set_log_stream(log)
        prob.set_results_stream(log)
        prob.set_warning_stream(log)
        prob.set_error_stream(log)

    prob.parameters.mip.tolerances.mipgap.set(mipgap)

    # 2. Solución
    t0 = time.time()
    prob.solve()
    t1 = time.time()

    if log_file is not None:
        log.close()
    
    # 3. Extracción de resultados
    status = prob.solution.get_status_string(prob.solution.get_status())
    
											 
    obj = None
    active_vars = {}

    # 4. Verificación de factibilidad
    if prob.solution.is_primal_feasible():
												  
        obj = prob.solution.get_objective_value()
		
															  
        vals = dict(zip(prob.variables.get_names(), prob.solution.get_values()))
        
        active_vars = {
            name: round(val, 5)
            for name, val in vals.items()
            if val > 0.5 and prob.variables.get_types(name) == prob.variables.type.binary
        }
        
    # 5. Retorno del diccionario de resultados
    return {
        "status": status,
        "time_s": round(t1 - t0, 3),
        "objective_value": obj,
        "active_vars": active_vars
    }
