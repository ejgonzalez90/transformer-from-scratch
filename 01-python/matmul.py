"""Multiplicación de matrices con listas puras (sin NumPy).

Una matriz es una lista de filas: [[1, 0, 2, 3]] es una matriz de 1×4.
"""

# Alias de tipo: una "Matriz" es una lista de listas de float.
# (En C# sería algo como `using Matriz = List<List<double>>;`)
Matriz = list[list[float]]


def dot(u: list[float], v: list[float]) -> float:
    """Producto punto: suma de u[i] * v[i].

    dot([1, 0, 2, 3], [1, 0, 1, 0]) == 2
    """
    # Sin corchetes, lo que va adentro de sum(...) es una expresión generadora:
    # produce los productos de a uno, a medida que sum los pide (como un
    # IEnumerable con yield en C#). Con corchetes, sum([...]) armaría primero
    # una lista entera en memoria y recién después la sumaría. El resultado es
    # el mismo; el generador se ahorra la lista intermedia.
    return sum(i * j for i, j in zip(u, v, strict=True))


def columna(m: Matriz, j: int) -> list[float]:
    """Devuelve la columna j de m, como lista.

    columna([[1, 2], [3, 4]], 1) == [2, 4]
    """
    # List comprehension: [EXPRESIÓN for ELEMENTO in ITERABLE]
    # Equivale a m.Select(r => r[j]) en LINQ, o a este for:
    #     c = []
    #     for r in m:
    #         c.append(r[j])
    # Ej.: m = [[1, 2], [3, 4]], j = 1 → r toma [1, 2] y después [3, 4] → [2, 4]
    return [r[j] for r in m]


def transponer(m: Matriz) -> Matriz:
    """Devuelve mᵀ: la columna j de m pasa a ser la fila j.

    (n×k) → (k×n)
    transponer([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    """
    # range(len(m[0])) recorre los índices de columna: 0, 1, ..., k-1.
    # Por cada uno, columna(m, x) devuelve una lista, así que el resultado es
    # una lista de listas: Enumerable.Range(0, k).Select(x => columna(m, x)).
    # Ej.: m = [[1, 2, 3], [4, 5, 6]] → x = 0, 1, 2 → [[1, 4], [2, 5], [3, 6]]
    # También se puede filtrar, como un Where: [x for x in v if x > 0]
    return [columna(m, x) for x in range(len(m[0]))]


def matmul(a: Matriz, b: Matriz) -> Matriz:
    """(n×k) · (k×m) = (n×m).

    El elemento [i][j] del resultado es dot(fila i de a, columna j de b).
    Si las dimensiones no encajan, lanzá ValueError.
    """
    if len(a[0]) != len(b):
        raise ValueError(
            f"no se puede multiplicar ({len(a)}×{len(a[0])}) · ({len(b)}×{len(b[0])}): "
            f"las columnas de a ({len(a[0])}) tienen que coincidir con las filas de b ({len(b)})"
        )

    # bᵀ se calcula UNA vez, afuera. Si fuera dentro de la comprehension
    # ([... for c in transponer(b)] for r in a), la de adentro se evalúa entera
    # por cada fila de a, y transponer correría n veces.
    # Cada fila de bᵀ es una columna de b: es la misma idea que Kᵀ en Q·Kᵀ.
    bt = transponer(b)

    # Equivale a:
    #     resultado = []
    #     for r in a:                       # por cada fila de a...
    #         fila = []
    #         for c in bt:                  # ...y cada columna de b (fila de bᵀ)
    #             fila.append(dot(r, c))    # la casilla es su producto punto
    #         resultado.append(fila)
    # Ej.: a = [[1, 2], [3, 4]], bt = [[5, 7], [6, 8]]
    #      r = [1, 2] → [dot([1, 2], [5, 7]), dot([1, 2], [6, 8])] = [19, 22]
    #      r = [3, 4] → [43, 50]                    → [[19, 22], [43, 50]]
    return [[dot(r, c) for c in bt] for r in a]
