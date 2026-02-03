#---- Librerias ----#
from collections import deque
import cplex
from cplex.callbacks import LazyConstraintCallback

# ----------------------------------------------- #
######### Callback - Método DFJ / B&C #############
# ----------------------------------------------- #

def findSubtour(Nodos, x_solution):
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
        subtour = findSubtour(self.Nodos, x_solution)
        
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