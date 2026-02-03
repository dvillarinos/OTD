#---- Librerias ----#
import cplex

#---- Modulos ----#
import instanceData as ID
import findSubtour as FS

# ----------------------------------------------- #
################ Método Actual  ###################
# ----------------------------------------------- #
def prob_metodo_actual(instance_data):
    
    #--- Extraer datos de la instancia ---#
    DESTINOS = instance_data.DESTINOS()
    COSTOS = instance_data.COSTOS()

    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)
    var_names = []
    obj_coeffs = []
    var_map = {}

    idx = 0
    for i in DESTINOS:
        for j in DESTINOS:
            if i != j:
                name = f"x_{i}_{j}"
                var_names.append(name)
                obj_coeffs.append(COSTOS[(i, j)])
                var_map[name] = idx
                idx += 1

    prob.variables.add(
        obj=obj_coeffs,
        types=[prob.variables.type.binary] * len(var_names),
        names=var_names
    )
        
    # --- 2. Restricciones ---
    constraints = []
    rhs = []
    senses = []
    
    # Una sola salida por nodo
    for i in DESTINOS:
        lhs_vars = [f"x_{i}_{j}" for j in DESTINOS if i != j]
        lhs_coeffs = [1.0] * len(lhs_vars)
        constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
        rhs.append(1.0)
        senses.append("E")

    # Una sola entrada por nodo
    for j in DESTINOS:
        lhs_vars = [f"x_{i}_{j}" for i in DESTINOS if i != j]
        lhs_coeffs = [1.0] * len(lhs_vars)
        constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
        rhs.append(1.0)
        senses.append("E")

    # Añadir todas las restricciones al modelo
    prob.linear_constraints.add(
        lin_expr=constraints,
        senses=senses,
        rhs=rhs
    )

    # Restricciones de subtours mediante callback
    tsp_callback = prob.register_callback(FS.TSPLazyConstraintCallback)
    tsp_callback.var_idx = var_map
    tsp_callback.Nodos = set(DESTINOS)

    return prob