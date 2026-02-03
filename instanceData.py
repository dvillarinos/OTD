import json
from typing import Dict, List, Tuple, Set

class InstanceData:
    def __init__(self, instance_path: str):
        with open(instance_path, 'r') as f:
            instancia = json.load(f)

        self._N: int = instancia["cantidad_clientes"]
        self._DESTINOS: List[int] = list(range(1, self._N + 1))

        self._REFRIGERADOS: Set[int] = set(instancia["clientes_refrigerados"])
        self._EXCLUSIVOS: Set[int] = set(instancia["clientes_exclusivos"])

        self._DISTANCIA_MAX_BICI: float = instancia["d_max_bici"]
        self._DISTANCIA_MAX_VEH: float = instancia["d_max_veh"]

        self._COSTO_BICI: float = instancia["costo_bici"]
        self._COSTO_VEH: float = instancia["costo_veh"]

        self._DISTANCIAS: Dict[Tuple[int, int], float] = {}
        self._COSTOS: Dict[Tuple[int, int], float] = {}

        for entrada in instancia.get("distancias_costos", []):
            i = entrada["i"]
            j = entrada["j"]
            self._DISTANCIAS[(i, j)] = entrada["d_ij"]
            self._COSTOS[(i, j)] = entrada["c_ij"]

        self._PRECEDENCIA: List[Tuple[int, int]] = instancia["pares_precedencia"]

    def N(self) -> int:
        return self._N

    def DISTANCIA_MAX_BICI(self) -> float:
        return self._DISTANCIA_MAX_BICI

    def DISTANCIA_MAX_VEH(self) -> float:
        return self._DISTANCIA_MAX_VEH

    def COSTO_BICI(self) -> float:
        return self._COSTO_BICI

    def COSTO_VEH(self) -> float:
        return self._COSTO_VEH

    def DESTINOS(self) -> List[int]:
        return self._DESTINOS

    def REFRIGERADOS(self) -> Set[int]:
        return self._REFRIGERADOS

    def EXCLUSIVOS(self) -> Set[int]:
        return self._EXCLUSIVOS

    def PRECEDENCIA(self) -> List[Tuple[int, int]]:
        return self._PRECEDENCIA

    def DISTANCIAS(self) -> Dict[Tuple[int, int], float]:
        return self._DISTANCIAS

    def COSTOS(self) -> Dict[Tuple[int, int], float]:
        return self._COSTOS

    def distancia(self, i: int, j: int) -> float:
        return self._DISTANCIAS[(i, j)]

    def costo(self, i: int, j: int) -> float:
        return self._COSTOS[(i, j)]
