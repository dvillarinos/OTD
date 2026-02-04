# ---- Librerías ----
import random
import json
from re import S
import numpy as np


# ----------------------------------------------- #
############# Instancias de Prueba ################
# ----------------------------------------------- #
def test_instances():
    path = "instances/"

    TestCases = [
                ["small_instance",25,(1, 50),(1, 5)],
                ["medium_instance",50,(1, 50),(1, 5)],
                ["large_instance",75,(1, 75),(1, 7)],
                ["extralarge_instance",100,(1, 75),(1, 7)],
                ["longDistances_instance",60,(1, 200),(1, 10)],
                ]
    
    SEED = 12
    instaces = []
    for instace, N, rango_distancia, rango_tarifa in TestCases:
        instaces.append(instace)
        file_name = path + f"{instace}.json"
        newInstance(file_name,
                    N=N,
                    rango_distancia=rango_distancia,
                    rango_tarifa=rango_tarifa,
                    seed=SEED
                    )
        SEED += 13  # Cambiar semilla para la siguiente instancia
    
    return instaces
   

# ----------------------------------------------- #
### Función para generar la instancia de prueba ###
# ----------------------------------------------- #

def newInstance(
    file_name,
    N=50,
    rango_distancia=(1, 50),
    rango_tarifa=(1, 2),
    seed=62
):
    random.seed(seed)

    # --- Parámetros base ---
    deposito_clientes_ids = list(range(1, N + 1))
    clientes_ids = deposito_clientes_ids.copy()
    clientes_ids.remove(1)  # el depósito no cuenta como cliente normal

    tarifa_base = round(random.uniform(rango_tarifa[0], rango_tarifa[1]), 2)

    # -------------------------------------------------
    # Clientes refrigerados (al menos 1)
    # -------------------------------------------------
    num_refrigerados = max(1, int(np.ceil(N / 10)))
    clientes_refrigerados = random.sample(clientes_ids, num_refrigerados)
    clientes_refrigerados.sort()

    # -------------------------------------------------
    # Clientes exclusivos del camión
    #   - Incluyen SIEMPRE a los refrigerados
    #   - Incluyen al depósito
    # -------------------------------------------------
    poblacion_disponible = [c for c in clientes_ids if c not in clientes_refrigerados]

    num_clientes_exclusivos_extra = min(
        len(poblacion_disponible),
        len(clientes_refrigerados) + 5
    )

    clientes_exclusivos = random.sample(
        poblacion_disponible,
        num_clientes_exclusivos_extra
    )

    clientes_exclusivos = list(
        set(clientes_exclusivos)
        | set(clientes_refrigerados)
        | {1}
    )

    clientes_exclusivos.sort()
    num_exclusivos = len(clientes_exclusivos)

    # -------------------------------------------------
    # Distancias y costos asimétricos del camión
    # -------------------------------------------------
    distancias_costos = []
    todas_distancias = []
    todos_costos = []

    for i in deposito_clientes_ids:
        for j in deposito_clientes_ids:
            if i < j:
                distancia = random.randint(rango_distancia[0], rango_distancia[1])

                costo_base = int(round(distancia * tarifa_base))

                factor_ij = random.uniform(0.8, 1.2)
                factor_ji = random.uniform(0.8, 1.2)

                costo_ij = int(round(costo_base * factor_ij))
                costo_ji = int(round(costo_base * factor_ji))

                todas_distancias.append(distancia)
                todos_costos.extend([costo_ij, costo_ji])

                distancias_costos.append({
                    "i": i,
                    "j": j,
                    "d_ij": distancia,
                    "c_ij": costo_ij
                })

                distancias_costos.append({
                    "i": j,
                    "j": i,
                    "d_ij": distancia,
                    "c_ij": costo_ji
                })

    # -------------------------------------------------
    # Parámetros dinámicos
    # -------------------------------------------------
    referencia_distancia = np.median(todas_distancias)
    d_max_bici = int(np.ceil(referencia_distancia * 0.20))
    d_max_veh = int(np.ceil(referencia_distancia * 0.90))

    referencia_costo = np.median(todos_costos)
    costo_bici = int(np.ceil(referencia_costo * 0.30))
    costo_veh = int(np.ceil(referencia_costo * 0.70))

    # -------------------------------------------------
    # Pares de precedencia (SOLO clientes visitados por el camión)
    # -------------------------------------------------
    def generar_precedencia(clientes, pares):
        while True:
            par = random.sample(clientes, 2)
            if par not in pares and par[::-1] not in pares:
                return par

    candidatos_precedencia = [c for c in clientes_exclusivos if c != 1]

    n_pares_precedencia = max(2, N // 20)
    pares_precedencia = []

    if len(candidatos_precedencia) >= 2:
        for _ in range(min(n_pares_precedencia,
                           len(candidatos_precedencia) * (len(candidatos_precedencia) - 1) // 2)):
            par = generar_precedencia(candidatos_precedencia, pares_precedencia)
            pares_precedencia.append(par)

    # -------------------------------------------------
    # Validaciones finales
    # -------------------------------------------------
    assert set(clientes_refrigerados).issubset(set(clientes_exclusivos))
    assert 1 in clientes_exclusivos
    assert len(distancias_costos) == N * (N - 1)

    # -------------------------------------------------
    # Ensamblar JSON
    # -------------------------------------------------
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

    with open(file_name, "w") as f:
        json.dump(instancia, f, indent=2)

    return None
