#---- Librerias ----#
import cplex
import numpy as np

#---- Modulos ----#
import instanceData as ID
import findSubtour as FS

# ----------------------------------------------- #
################ Método Nuevo  ####################
# ----------------------------------------------- #

class settings:
    def __init__(self, Min_No_Camion=True, Max_Bicis=None, Max_Autos=None, Exclusivos=True, Refrigerados=True,Precedencia=True):
        self.Min_No_Camion = Min_No_Camion
        self.Max_Bicis = Max_Bicis
        self.Max_Autos = Max_Autos
        self.Exclusivos = Exclusivos
        self.Refrigerados = Refrigerados
        self.Precedencia = Precedencia

    def min_no_camion(self):
        return self.Min_No_Camion
    def max_bicis(self):
        return self.Max_Bicis
    def max_autos(self):
        return self.Max_Autos
    def exclusivos(self):
        return self.Exclusivos
    def refrigerados(self):
        return self.Refrigerados
    def precedencia(self):
        return self.Precedencia
    
    
def prob_metodo_nuevo(instance_data,settings=settings()):

    #--- Extraer datos de la instancia ---#
    N = instance_data.N()
    DESTINOS = instance_data.DESTINOS()

    DISTANCIAS = instance_data.DISTANCIAS()
    DISTANCIA_MAX_BICI = instance_data.DISTANCIA_MAX_BICI()
    DISTANCIA_MAX_VEH = instance_data.DISTANCIA_MAX_VEH()

    COSTOS = instance_data.COSTOS()
    COSTO_BICI = instance_data.COSTO_BICI()
    COSTO_VEH = instance_data.COSTO_VEH()

    REFRIGERADOS = instance_data.REFRIGERADOS()
    EXCLUSIVOS = instance_data.EXCLUSIVOS()
    PRECEDENCIA = instance_data.PRECEDENCIA()

    
    #--- Configuracion de las restricciones ---#    													
    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)
    var_names = []
    obj_coeffs = []
    var_map = {}
    
    DEPOSITO = 1
    CLIENTES = [c for c in DESTINOS if c != DEPOSITO]
    BIG_M = N + 10

    # 1. Crear variables
													 
    idx = 0
    
    # 1.1 Variables x_ij (camión)
    for i in DESTINOS:
        for j in DESTINOS:
            if i != j:  
                name = f"x_{i}_{j}"
                var_names.append(name)
                obj_coeffs.append(COSTOS[(i, j)])
                var_map[name] = idx
                idx += 1

    # 1.2 Variables bici_ij
    for i in DESTINOS:
        for j in CLIENTES:
            if i != j and DISTANCIAS.get((i, j), float('inf')) <= DISTANCIA_MAX_BICI:
                name = f"bici_{i}_{j}"
                var_names.append(name)
                obj_coeffs.append(0.0)
                var_map[name] = idx
                idx += 1

    # 1.3 Variables auto_ij
    for i in DESTINOS:
        for j in CLIENTES:
            if i != j and DISTANCIAS.get((i, j), float('inf')) <= DISTANCIA_MAX_VEH:
                name = f"auto_{i}_{j}"
                var_names.append(name)
                obj_coeffs.append(0.0)
                var_map[name] = idx
                idx += 1

    # 1.4 Variables y_bici_i
    for i in DESTINOS:
        name = f"y_bici_{i}"
        var_names.append(name)
        obj_coeffs.append(COSTO_BICI)
        var_map[name] = idx
        idx += 1

    # 1.5 Variables y_auto_i
    for i in DESTINOS:
        name = f"y_auto_{i}"
        var_names.append(name)
        obj_coeffs.append(COSTO_VEH)
        var_map[name] = idx
        idx += 1

    num_vars_binarias = len(var_names)
    lower_bounds = [0.0] * num_vars_binarias 
    upper_bounds = [1.0] * num_vars_binarias
    types_list = [prob.variables.type.binary] * num_vars_binarias

    # 1.6 Variables u_i para MTZ (solo para clientes)
    for i in CLIENTES:
        name = f"u_{i}"
        var_names.append(name)
        obj_coeffs.append(0.0)
        var_map[name] = idx
        idx += 1

    num_vars_continuas = len(CLIENTES)
    lower_bounds += [2.0] * num_vars_continuas
    upper_bounds += [float(N)] * num_vars_continuas
    types_list += [prob.variables.type.continuous] * num_vars_continuas

    prob.variables.add(
        obj=obj_coeffs,
        lb=lower_bounds,
        ub=upper_bounds,
        types=types_list,
        names=var_names
    )
        
    # --- 2. Restricciones ---
    constraints = []
    rhs = []
    senses = []

    # ----- Modelo Base ------

    # R1: Cada cliente es visitado exactamente una vez
    for j in CLIENTES: 
        lhs_vars = []
        lhs_coeffs = []

        for i in DESTINOS:
            if i != j:

                # Camión
                lhs_vars.append(f"x_{i}_{j}")
                lhs_coeffs.append(1.0)
                
                # Bici
																			  
                if f"bici_{i}_{j}" in var_map:
                    lhs_vars.append(f"bici_{i}_{j}")
                    lhs_coeffs.append(1.0)
                
                # Auto
																			 
                if f"auto_{i}_{j}" in var_map:
                    lhs_vars.append(f"auto_{i}_{j}")
                    lhs_coeffs.append(1.0)
        
        constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
        rhs.append(1.0)
        senses.append("E")

    # R2: El camión sale del depósito
    lhs_vars = [f"x_{DEPOSITO}_{j}" for j in CLIENTES]
    constraints.append(cplex.SparsePair(ind=lhs_vars, val=[1.0] * len(lhs_vars)))
    rhs.append(1.0)
    senses.append("E")
    
    # R3: El camión vuelve al depósito
    lhs_vars = [f"x_{i}_{DEPOSITO}" for i in CLIENTES]
    constraints.append(cplex.SparsePair(ind=lhs_vars, val=[1.0] * len(lhs_vars)))
    rhs.append(1.0)
    senses.append("E")

    # R4: Conservación de flujo del camión en nodos intermedios
    for k in CLIENTES:
        lhs_vars = []
        lhs_coeffs = []
        
        for i in DESTINOS:
                        
            if i != k:
                lhs_vars.append(f"x_{i}_{k}")
                lhs_coeffs.append(1.0)
        
                            
        for j in DESTINOS:
            if j != k:
                lhs_vars.append(f"x_{k}_{j}")
                lhs_coeffs.append(-1.0)
        
        constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
        rhs.append(0.0)
        senses.append("E")
        
    # R5: Restricciones MTZ para eliminación de subtours				   
    for i in CLIENTES:
        for j in CLIENTES:
            if i != j:
                lhs_vars = [f"u_{i}", f"u_{j}", f"x_{i}_{j}"]
                lhs_coeffs = [1.0, -1.0, float(N)]                
                constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
                rhs.append(float(N) - 1.0)
                senses.append("L")

    # ----- Adicionales ------

    # R6: A lo sumo un repartidor (bici O auto) por parada
    for i in DESTINOS:
        lhs_vars = [f"y_bici_{i}", f"y_auto_{i}"]				
        lhs_coeffs = [1.0, 1.0]
        constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
        rhs.append(1.0)
        senses.append("L")

    # R7: Una bici puede visitar hasta 2 clientes
    for i in DESTINOS:
        lhs_vars = []
        lhs_coeffs = []
        for j in CLIENTES:
                    
                                                                            
            if i != j and f"bici_{i}_{j}" in var_map:
                lhs_vars.append(f"bici_{i}_{j}")
                lhs_coeffs.append(1.0)
        if lhs_vars:
            lhs_vars.append(f"y_bici_{i}")
            lhs_coeffs.append(-2.0)
            constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
            rhs.append(0.0)
            senses.append("L")
        
    # R8: Un auto puede visitar hasta 5 clientes
    for i in DESTINOS:
        lhs_vars = []
        lhs_coeffs = []
        for j in CLIENTES:                                           
            if i != j and f"auto_{i}_{j}" in var_map:
                lhs_vars.append(f"auto_{i}_{j}")
                lhs_coeffs.append(1.0)
        if lhs_vars:
            lhs_vars.append(f"y_auto_{i}")
            lhs_coeffs.append(-5.0)
            constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
            rhs.append(0.0)
            senses.append("L")

    # R9: Solo se usa bici/auto desde paradas del camión
    for i in CLIENTES:
        lhs_vars = []
        lhs_coeffs = []        
        for j in CLIENTES:
            if i != j:                                                
                if f"bici_{i}_{j}" in var_map:
                    lhs_vars.append(f"bici_{i}_{j}")
                    lhs_coeffs.append(1.0)						 
                if f"auto_{i}_{j}" in var_map:
                    lhs_vars.append(f"auto_{i}_{j}")
                    lhs_coeffs.append(1.0)

        if lhs_vars:
            for k in DESTINOS:
                if k != i:
                    lhs_vars.append(f"x_{k}_{i}")
                    lhs_coeffs.append(-BIG_M)
            constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
            rhs.append(0.0)
            senses.append("L")

    # R10: Máximo 1 producto refrigerado por bici
    for i in DESTINOS:
        lhs_vars = []
        lhs_coeffs = []
        for j in REFRIGERADOS:
            if i != j and f"bici_{i}_{j}" in var_map:
                lhs_vars.append(f"bici_{i}_{j}")
                lhs_coeffs.append(1.0)
        if lhs_vars:
            lhs_vars.append(f"y_bici_{i}")
            lhs_coeffs.append(-1.0)
            constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
            rhs.append(0.0)
            senses.append("L")

    # R11: Máximo 1 producto refrigerado por auto
    for i in DESTINOS:
        lhs_vars = []
        lhs_coeffs = []
        for j in REFRIGERADOS:
            if i != j and f"auto_{i}_{j}" in var_map:
                lhs_vars.append(f"auto_{i}_{j}")
                lhs_coeffs.append(1.0)
        if lhs_vars:
            lhs_vars.append(f"y_auto_{i}")
            lhs_coeffs.append(-1.0)
            constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
            rhs.append(0.0)
            senses.append("L")

    # ---- Configurables -----

    # R12: Al menos 1/3 de clientes NO visitados por camión (restricción deseable)
    if settings.min_no_camion():
        lhs_vars = []
        lhs_coeffs = []
        for j in CLIENTES:
            for i in DESTINOS:
                if i != j:											  
                    if f"bici_{i}_{j}" in var_map:
                        lhs_vars.append(f"bici_{i}_{j}")
                        lhs_coeffs.append(1.0)					 
                    if f"auto_{i}_{j}" in var_map:
                        lhs_vars.append(f"auto_{i}_{j}")
                        lhs_coeffs.append(1.0)

        minimo_no_camion = np.ceil(len(CLIENTES) / 3.0)
        
        constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
        rhs.append(minimo_no_camion)
        senses.append("G")

    # R13: Clientes exclusivos deben ser visitados por camión
    if settings.exclusivos():
        for j in EXCLUSIVOS:
            if j != DEPOSITO:
                lhs_vars = [f"x_{i}_{j}" for i in DESTINOS if i != j]
                constraints.append(cplex.SparsePair(ind=lhs_vars, val=[1.0] * len(lhs_vars)))
                rhs.append(1.0)
                senses.append("E")	
								   
    # R14: Límite de bicis
    if settings.max_bicis() is not None:
        K_BICIS = settings.max_bicis()
        lhs_vars = [f"y_bici_{i}" for i in DESTINOS]
        constraints.append(cplex.SparsePair(ind=lhs_vars, val=[1.0] * len(lhs_vars)))
        rhs.append(K_BICIS)
        senses.append("L")

    # R15: Límite de autos
    if settings.max_autos() is not None:
        K_AUTOS = settings.max_autos()				  
        lhs_vars = [f"y_auto_{i}" for i in DESTINOS]							  
        constraints.append(cplex.SparsePair(ind=lhs_vars, val=[1.0] * len(lhs_vars)))
        rhs.append(K_AUTOS)
        senses.append("L")

    # R16: Restricciones de precedencia
    if settings.precedencia():
        for a, b in PRECEDENCIA:															 
            if a in CLIENTES and b in CLIENTES:
                lhs_vars = [f"u_{a}", f"u_{b}"]
                lhs_coeffs = [1.0, -1.0]            
                constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
                rhs.append(-1.0)
                senses.append("L")

    # Añadir todas las restricciones al modelo
    prob.linear_constraints.add(
        lin_expr=constraints,
        senses=senses,
        rhs=rhs
    )				  

    return prob