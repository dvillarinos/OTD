import os
import time
import json
import random
import numpy as np
import cplex
from cplex.callbacks import LazyConstraintCallback
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import itertools

										 

# ----------------------------------------------- #
### Función para generar la instancia de prueba ###
# ----------------------------------------------- #

def generar_instancia(N=50, rango_distancia=(1, 50), rango_tarifa=(1, 2)):
    random.seed(62)

    # --- Parámetros base ---
    deposito_clientes_ids = list(range(1, N + 1))
    clientes_ids = deposito_clientes_ids.copy()
    clientes_ids.remove(1)

    tarifa_base = round(random.uniform(rango_tarifa[0], rango_tarifa[1]), 2)
    
    # Restricciones
    num_refrigerados = int(np.ceil(N // 10))
    clientes_refrigerados = random.sample(clientes_ids, num_refrigerados)
    clientes_refrigerados.sort()
        
    num_clientes_exclusivos = len(clientes_refrigerados) + 5
    
    poblacion_disponible = [c for c in clientes_ids if c not in clientes_refrigerados]
    
    # Seleccionar clientes adicionales (que no sean depósito ni refrigerados)
    clientes_exclusivos = random.sample(poblacion_disponible, num_clientes_exclusivos)
    clientes_exclusivos.append(1)
        
    clientes_exclusivos.sort()
    num_exclusivos = len(clientes_exclusivos)
        
    # --- Generar Distancias y Costos Asimétricos (Camión) ---
    distancias_costos = []
    
    # Listas temporales para calcular el costo/distancia promedio del camión
    todas_distancias = []
    todos_costos = []
    
    for i in deposito_clientes_ids:
        for j in deposito_clientes_ids:
            if i < j:
                
                # 1. Distancia base para el par {i, j}
                distancia = random.randint(rango_distancia[0], rango_distancia[1])
                
                # 2. C(i, j)
                costo_base = int(round(distancia * tarifa_base))
                factor_ij = random.uniform(0.8, 1.2)
                costo_ij = int(round(costo_base * factor_ij))
                
                # 3. C(j, i)
                factor_ji = random.uniform(0.8, 1.2)
                costo_ji = int(round(costo_base * factor_ji))
                
                # Recolectar datos para calcular métricas
                todas_distancias.append(distancia)
                todos_costos.append(costo_ij)
                todos_costos.append(costo_ji)
                
                # Agregar (i, j)
                distancias_costos.append({
                    "i": i,
                    "j": j,
                    "d_ij": distancia,
                    "c_ij": costo_ij
                })
                
                # Agregar (j, i)
                distancias_costos.append({
                    "i": j,
                    "j": i,
                    "d_ij": distancia,
                    "c_ij": costo_ji
                })
    
    # --- Cálculo de Parámetros Dinámicos ---
    
    referencia_distancia = np.median(todas_distancias) 
    d_max_bici = int(np.ceil(referencia_distancia * .20))
    d_max_veh = int(np.ceil(referencia_distancia * .90))

    referencia_costo = np.median(todos_costos)     
    costo_bici = int(np.ceil(referencia_costo * .30))
    costo_veh  = int(np.ceil(referencia_costo * .70))

    # Generación de Pares de Precedencia ---
    def generar_precedencia(clientes_ids, pares):
        par = random.sample(clientes_ids, 2)
        if par not in pares and [par[1], par[0]] not in pares:
            return par
        else:
            return generar_precedencia(clientes_ids, pares)

    n_pares_precedencia = max(2, N // 20)
    pares_precedencia = []       
    
    for _ in range(n_pares_precedencia):
        par = generar_precedencia(clientes_ids, pares_precedencia)
        pares_precedencia.append(par)

    # --- Ensamblar el JSON ---
    instancia = {
        "cantidad_clientes": N,
        "costo_bici": costo_bici,
        "costo_veh": costo_veh,
        "d_max_bici": d_max_bici,
        "d_max_veh": d_max_veh,
        "cantidad_refrigerados": num_refrigerados,
        "clientes_refrigerados": clientes_refrigerados,
        "cantidad_exclusivos": num_exclusivos,
        "clientes_exclusivos": clientes_exclusivos,
        "distancias_costos": distancias_costos,
        "pares_precedencia": pares_precedencia
    }
    
    nombre_archivo = "instancia_prueba.json"
    with open(nombre_archivo, 'w') as f:
        json.dump(instancia, f, indent=2)

    return None


# ----------------------------------------------- #
############ Funciones auxiliares ################
# ----------------------------------------------- #

def cargar_instancia(nombre_archivo):
    if not os.path.isfile(nombre_archivo):
        generar_instancia(N=50)

    with open(nombre_archivo, 'r') as f:
        instancia = json.load(f)
    return instancia

def leer_instancia(nombre_archivo):
    instancia = cargar_instancia(nombre_archivo)
    N = instancia["cantidad_clientes"]
    DESTINOS = list(range(1, N + 1))

    REFRIGERADOS = instancia["clientes_refrigerados"]
    EXCLUSIVOS = instancia["clientes_exclusivos"]


    DISTANCIA_MAX_BICI = instancia["d_max_bici"]
    DISTANCIA_MAX_VEH = instancia["d_max_veh"]
    
    COSTO_BICI = instancia["costo_bici"]
    COSTO_VEH = instancia["costo_veh"]
    
    DISTANCIAS = {}
    COSTOS = {}
    distancias_costos = instancia.get("distancias_costos", [])
    for entrada in distancias_costos:
        i = entrada["i"]
        j = entrada["j"]
        d_ij = entrada["d_ij"]
        c_ij = entrada["c_ij"]
        DISTANCIAS[(i, j)] = d_ij
        COSTOS[(i, j)] = c_ij
    
    PRECEDENCIA = instancia["pares_precedencia"]

    return N, DESTINOS, REFRIGERADOS, EXCLUSIVOS, DISTANCIAS, COSTOS, DISTANCIA_MAX_BICI, DISTANCIA_MAX_VEH, COSTO_BICI, COSTO_VEH, PRECEDENCIA

# ----------------------------------------------- #
######### Callback - Método DFJ / B&C #############
# ----------------------------------------------- #

def encontrarSubtour(Nodos, x_solution):
    V = list(Nodos)
    num_nodos = len(V)
    
    # Construir el grafo basado en los arcos con x_ij = 1
    grafo = {node: [] for node in Nodos}
    
    for (i, j), val in x_solution.items():
        if val > 0.5:
            grafo[i].append(j)

    # Buscar componentes conexas (subtours)
    visitados = set()
    
    for start_node in Nodos:
        if start_node not in visitados:
            # BFS para encontrar una componente conexa
            componente = []
            queue = deque([start_node])
            visitados.add(start_node)
            
            while queue:
                u = queue.popleft()
                componente.append(u)
                for v in grafo.get(u, []):
                    if v not in visitados:
                        visitados.add(v)
                        queue.append(v)
            
            if len(componente) < num_nodos:
                return componente
                
    return []

class TSPLazyConstraintCallback(LazyConstraintCallback):
    def __init__(self, env):
        super().__init__(env)
        self.var_idx = {}
        self.Nodos = set() 

    def __call__(self):        
        # 1. Obtener los valores de las variables x_ij de la solución entera actual
        x_solucion_cruda = self.get_values()
        
        # 2. Reconstruir la solución x_ij en formato de diccionario { (i, j): valor }
        x_solution = {}
        for var_name, idx in self.var_idx.items():
            if var_name.startswith("x_"):
                # Extraer i y j del nombre: x_i_j
                parts = var_name.split('_')
                i = int(parts[1])
                j = int(parts[2])
                x_solution[(i, j)] = x_solucion_cruda[idx]
        
        # 3. Llamar al algoritmo de separación para encontrar subciclos
        subtour = encontrarSubtour(self.Nodos, x_solution)
        
        # 4. Si se encuentra un subciclo
        if subtour:
            S = set(subtour)
            V_menos_S = self.Nodos - S
            
            # Formular la restricción SEC: Sum(x_ij para i en S, j en V\S) >= 1
            lhs_vars = []
            lhs_coeffs = []
            
            for i in S:
                for j in V_menos_S:			  
                    var_name = f"x_{i}_{j}"								  
                    try:
                        lhs_vars.append(self.var_idx[var_name])
                        lhs_coeffs.append(1.0)
                    except KeyError:										
                        pass
            
            # 5. Añadir la restricción como Lazy Constraint
            if lhs_vars:		  
                self.add(
                    cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs),
                    sense="G", 
                    rhs=1.0
                )

# ----------------------------------------------- #
################ Método Actual  ###################
# ----------------------------------------------- #
def prob_metodo_actual(N, DESTINOS, REFRIGERADOS, EXCLUSIVOS, DISTANCIAS, COSTOS, DISTANCIA_MAX_BICI, DISTANCIA_MAX_VEH, COSTO_BICI, COSTO_VEH, PRECEDENCIA):
    # Crear las variables x_ij y la función objetivo
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
    tsp_callback = prob.register_callback(TSPLazyConstraintCallback)
    tsp_callback.var_idx = var_map
    tsp_callback.Nodos = set(DESTINOS)

    return prob

# ----------------------------------------------- #
################ Método Nuevo  ####################
# ----------------------------------------------- #
def prob_metodo_nuevo(
    N, DESTINOS, REFRIGERADOS, EXCLUSIVOS, DISTANCIAS, COSTOS,
    DISTANCIA_MAX_BICI, DISTANCIA_MAX_VEH,
    COSTO_BICI, COSTO_VEH,
    PRECEDENCIA,
    N_BICIS=None, K_AUTOS=None,
    RESTRICCIONES=None
):

    if RESTRICCIONES is None:
        RESTRICCIONES = {}

    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)

    var_names = []
    obj_coeffs = []
    var_map = {}

    DEPOSITO = 1
    CLIENTES = [c for c in DESTINOS if c != DEPOSITO]
    BIG_M = N + 10

    # =====================================================
    # 1. Variables
    # =====================================================
    idx = 0

    # x_ij (camión)
    for i in DESTINOS:
        for j in DESTINOS:
            if i != j:
                name = f"x_{i}_{j}"
                var_names.append(name)
                obj_coeffs.append(COSTOS[(i, j)])
                var_map[name] = idx
                idx += 1

    # bici_ij
    for i in DESTINOS:
        for j in CLIENTES:
            if i != j and DISTANCIAS.get((i, j), float("inf")) <= DISTANCIA_MAX_BICI:
                name = f"bici_{i}_{j}"
                var_names.append(name)
                obj_coeffs.append(0.0)
                var_map[name] = idx
                idx += 1

    # auto_ij
    for i in DESTINOS:
        for j in CLIENTES:
            if i != j and DISTANCIAS.get((i, j), float("inf")) <= DISTANCIA_MAX_VEH:
                name = f"auto_{i}_{j}"
                var_names.append(name)
                obj_coeffs.append(0.0)
                var_map[name] = idx
                idx += 1

    # y_bici_i
    for i in DESTINOS:
        name = f"y_bici_{i}"
        var_names.append(name)
        obj_coeffs.append(COSTO_BICI)
        var_map[name] = idx
        idx += 1

    # y_auto_i
    for i in DESTINOS:
        name = f"y_auto_{i}"
        var_names.append(name)
        obj_coeffs.append(COSTO_VEH)
        var_map[name] = idx
        idx += 1

    lower_bounds = [0.0] * len(var_names)
    upper_bounds = [1.0] * len(var_names)
    types_list = [prob.variables.type.binary] * len(var_names)

    # u_i (MTZ)
    for i in CLIENTES:
        name = f"u_{i}"
        var_names.append(name)
        obj_coeffs.append(0.0)
        var_map[name] = idx
        idx += 1

    lower_bounds += [2.0] * len(CLIENTES)
    upper_bounds += [float(N)] * len(CLIENTES)
    types_list += [prob.variables.type.continuous] * len(CLIENTES)

    prob.variables.add(
        obj=obj_coeffs,
        lb=lower_bounds,
        ub=upper_bounds,
        types=types_list,
        names=var_names
    )

    # =====================================================
    # 2. Restricciones
    # =====================================================
    constraints, rhs, senses = [], [], []

    # R1: Cada cliente es atendido exactamente una vez
    for j in CLIENTES:
        lhs_vars, lhs_coeffs = [], []
        for i in DESTINOS:
            if i != j:
                lhs_vars.append(f"x_{i}_{j}")
                lhs_coeffs.append(1.0)
                if f"bici_{i}_{j}" in var_map:
                    lhs_vars.append(f"bici_{i}_{j}")
                    lhs_coeffs.append(1.0)
                if f"auto_{i}_{j}" in var_map:
                    lhs_vars.append(f"auto_{i}_{j}")
                    lhs_coeffs.append(1.0)
        constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
        rhs.append(1.0)
        senses.append("E")

    # R2 y R3: salida y regreso al depósito
    constraints.append(cplex.SparsePair(
        ind=[f"x_{DEPOSITO}_{j}" for j in CLIENTES],
        val=[1.0] * len(CLIENTES)
    ))
    rhs.append(1.0)
    senses.append("E")

    constraints.append(cplex.SparsePair(
        ind=[f"x_{i}_{DEPOSITO}" for i in CLIENTES],
        val=[1.0] * len(CLIENTES)
    ))
    rhs.append(1.0)
    senses.append("E")

    # R4: Conservación de flujo
    for k in CLIENTES:
        lhs_vars, lhs_coeffs = [], []
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

    # R5: Exclusividad repartidor
    if RESTRICCIONES.get("exclusividad_repartidor", True):
        for i in DESTINOS:
            constraints.append(cplex.SparsePair(
                ind=[f"y_bici_{i}", f"y_auto_{i}"],
                val=[1.0, 1.0]
            ))
            rhs.append(1.0)
            senses.append("L")

    # R6: Capacidad bici
    if RESTRICCIONES.get("capacidad_bici", True):
        for i in DESTINOS:
            lhs_vars, lhs_coeffs = [], []
            for j in CLIENTES:
                if f"bici_{i}_{j}" in var_map:
                    lhs_vars.append(f"bici_{i}_{j}")
                    lhs_coeffs.append(1.0)
            if lhs_vars:
                lhs_vars.append(f"y_bici_{i}")
                lhs_coeffs.append(-2.0)
                constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
                rhs.append(0.0)
                senses.append("L")

    # R7: Capacidad auto
    if RESTRICCIONES.get("capacidad_auto", True):
        for i in DESTINOS:
            lhs_vars, lhs_coeffs = [], []
            for j in CLIENTES:
                if f"auto_{i}_{j}" in var_map:
                    lhs_vars.append(f"auto_{i}_{j}")
                    lhs_coeffs.append(1.0)
            if lhs_vars:
                lhs_vars.append(f"y_auto_{i}")
                lhs_coeffs.append(-5.0)
                constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
                rhs.append(0.0)
                senses.append("L")

    # R9: Mínimo no atendidos por camión
    if RESTRICCIONES.get("min_no_camion", True):
        lhs_vars, lhs_coeffs = [], []
        for j in CLIENTES:
            for i in DESTINOS:
                if f"bici_{i}_{j}" in var_map:
                    lhs_vars.append(f"bici_{i}_{j}")
                    lhs_coeffs.append(1.0)
                if f"auto_{i}_{j}" in var_map:
                    lhs_vars.append(f"auto_{i}_{j}")
                    lhs_coeffs.append(1.0)
        constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
        rhs.append(np.ceil(len(CLIENTES) / 3.0))
        senses.append("G")

    # R10 y R11: Refrigerados
    if RESTRICCIONES.get("refrigerados", True):
        for i in DESTINOS:
            lhs_vars, lhs_coeffs = [], []
            for j in REFRIGERADOS:
                if f"bici_{i}_{j}" in var_map:
                    lhs_vars.append(f"bici_{i}_{j}")
                    lhs_coeffs.append(1.0)
            if lhs_vars:
                lhs_vars.append(f"y_bici_{i}")
                lhs_coeffs.append(-1.0)
                constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
                rhs.append(0.0)
                senses.append("L")

        for i in DESTINOS:
            lhs_vars, lhs_coeffs = [], []
            for j in REFRIGERADOS:
                if f"auto_{i}_{j}" in var_map:
                    lhs_vars.append(f"auto_{i}_{j}")
                    lhs_coeffs.append(1.0)
            if lhs_vars:
                lhs_vars.append(f"y_auto_{i}")
                lhs_coeffs.append(-1.0)
                constraints.append(cplex.SparsePair(ind=lhs_vars, val=lhs_coeffs))
                rhs.append(0.0)
                senses.append("L")

    # R12: Exclusivos
    if RESTRICCIONES.get("exclusivos", True):
        for j in EXCLUSIVOS:
            if j != DEPOSITO:
                constraints.append(cplex.SparsePair(
                    ind=[f"x_{i}_{j}" for i in DESTINOS if i != j],
                    val=[1.0] * (len(DESTINOS) - 1)
                ))
                rhs.append(1.0)
                senses.append("E")

    # R13 y R14: Límites globales
    if N_BICIS is not None and RESTRICCIONES.get("limite_bicis", True):
        constraints.append(cplex.SparsePair(
            ind=[f"y_bici_{i}" for i in DESTINOS],
            val=[1.0] * len(DESTINOS)
        ))
        rhs.append(N_BICIS)
        senses.append("L")

    if K_AUTOS is not None and RESTRICCIONES.get("limite_autos", True):
        constraints.append(cplex.SparsePair(
            ind=[f"y_auto_{i}" for i in DESTINOS],
            val=[1.0] * len(DESTINOS)
        ))
        rhs.append(K_AUTOS)
        senses.append("L")

    # R15: MTZ
    for i in CLIENTES:
        for j in CLIENTES:
            if i != j:
                constraints.append(cplex.SparsePair(
                    ind=[f"u_{i}", f"u_{j}", f"x_{i}_{j}"],
                    val=[1.0, -1.0, float(N)]
                ))
                rhs.append(float(N) - 1.0)
                senses.append("L")

    # R16: Precedencia
    if RESTRICCIONES.get("precedencia", True):
        for a, b in PRECEDENCIA:
            if a in CLIENTES and b in CLIENTES:
                constraints.append(cplex.SparsePair(
                    ind=[f"u_{a}", f"u_{b}"],
                    val=[1.0, -1.0]
                ))
                rhs.append(-1.0)
                senses.append("L")

    prob.linear_constraints.add(lin_expr=constraints, senses=senses, rhs=rhs)
    return prob

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

# ----------------------------------------------- #
############# ANALISIS RESTRICCIONES ##############
# ----------------------------------------------- #

def explorar_restricciones(N, DESTINOS, REFRIGERADOS, EXCLUSIVOS, DISTANCIAS, COSTOS, DISTANCIA_MAX_BICI, DISTANCIA_MAX_VEH, COSTO_BICI, COSTO_VEH, PRECEDENCIA, N_BICIS=None, K_AUTOS=None):
    params = {
        "threads": 1,
        "mip.strategy.nodeselect": 1,
        "mip.strategy.heuristicfreq": -1,
        "preprocessing.presolve": 1,
    }
    migap = 0.5
    log_file = 'logs/Exploracion_Restricciones.log'
    
    prob = prob_metodo_nuevo(N, DESTINOS, REFRIGERADOS, EXCLUSIVOS, DISTANCIAS, COSTOS, DISTANCIA_MAX_BICI, DISTANCIA_MAX_VEH, COSTO_BICI, COSTO_VEH, PRECEDENCIA, N_BICIS, K_AUTOS)
    apply_params(prob, params)
    res = solve(prob, migap, log_file)

    status = res['status']
    costo = res['objective_value']
    tiempo = res['time_s']
    
    print(f"Status: {status} | Costo: {costo if costo else 'INF/INFACTIBLE'} | Tiempo: {tiempo}s")

    return res

# ----------------------------------------------- #
############# ANALISIS COMPUTACIONAL ##############
# ----------------------------------------------- #
def analizar_configuraciones(N, DESTINOS, REFRIGERADOS, EXCLUSIVOS, DISTANCIAS, COSTOS, DISTANCIA_MAX_BICI, DISTANCIA_MAX_VEH, COSTO_BICI, COSTO_VEH, PRECEDENCIA, N_BICIS, K_AUTOS):
    # --- Parámetros de CPLEX --- #
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

    
    prob = prob_metodo_nuevo(N, DESTINOS, REFRIGERADOS, EXCLUSIVOS, DISTANCIAS, COSTOS, DISTANCIA_MAX_BICI, DISTANCIA_MAX_VEH, COSTO_BICI, COSTO_VEH, PRECEDENCIA, N_BICIS, K_AUTOS)
    
    return prob


# ----------------------------------------------- #
##################### main ########################
# ----------------------------------------------- #

def main():
    FILE = 'instancia_prueba.json'
    GAP = 5e-3

    # --- Leo instancia de prueba --- #
    N, DESTINOS, REFRIGERADOS, EXCLUSIVOS, DISTANCIAS, COSTOS, DISTANCIA_MAX_BICI, DISTANCIA_MAX_VEH, COSTO_BICI, COSTO_VEH, PRECEDENCIA = leer_instancia(FILE)


    # --- Ejecutar Métodos --- #
    print("\n--- Método Actual ---")
    for param_set, param_values in params.items():
        LOG_FILE = f'logs/Metodo_Actual_{param_set}.log'
        print(f"\nParámetros: {param_set}")
        procesar_metodo(
            metodo_nuevo=False,
            params=param_values,
            migap=GAP,
            log_file=LOG_FILE,
            N=N,
            DESTINOS=DESTINOS,
            REFRIGERADOS=REFRIGERADOS,
            EXCLUSIVOS=EXCLUSIVOS,
            DISTANCIAS=DISTANCIAS,
            COSTOS=COSTOS,
            DISTANCIA_MAX_BICI=DISTANCIA_MAX_BICI,
            DISTANCIA_MAX_VEH=DISTANCIA_MAX_VEH,
            COSTO_BICI=COSTO_BICI,
            COSTO_VEH=COSTO_VEH,
            PRECEDENCIA=PRECEDENCIA
        )

    print("\n--- Método Nuevo ---")
    for param_set, param_values in params.items():
        LOG_FILE = f'logs/Metodo_Nuevo_{param_set}.log'
        print(f"\nParámetros: {param_set}")
        procesar_metodo(
            metodo_nuevo=True,
            params=param_values,
            migap=GAP,
            log_file=LOG_FILE,
            N=N,
            DESTINOS=DESTINOS,
            REFRIGERADOS=REFRIGERADOS,
            EXCLUSIVOS=EXCLUSIVOS,
            DISTANCIAS=DISTANCIAS,
            COSTOS=COSTOS,
            DISTANCIA_MAX_BICI=DISTANCIA_MAX_BICI,
            DISTANCIA_MAX_VEH=DISTANCIA_MAX_VEH,
            COSTO_BICI=COSTO_BICI,
            COSTO_VEH=COSTO_VEH,
            PRECEDENCIA=PRECEDENCIA
        )
if __name__ == "__main__":
    main()
