"""Softmax con NumPy.

softmax(x)_i = e^(x_i) / Σ_j e^(x_j)

Convierte puntajes cualesquiera en probabilidades: todas positivas y que suman 1.
En la atención se aplica a cada FILA de Q·Kᵀ / √d_k (sección 6 de las notas).
"""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax sobre el último eje (por fila, si x es una matriz).

    softmax(np.array([1.0, 2.0]))  ≈ [0.27, 0.73]
    softmax(np.array([[1.0, 2.0],
                      [0.5, 0.5]]))  ≈ [[0.27, 0.73],
                                         [0.5,  0.5 ]]

    Tiene que ser estable: softmax([1000, 1001]) no puede dar nan.
    """
    # No hay for ni if x.ndim: np.exp, - y / se aplican elemento a elemento, y
    # axis=-1 es siempre el último eje. La misma línea sirve para un vector (n,),
    # una matriz (n, n) o un batch (batch, n, n).
    #
    # Estabilidad: softmax(x - c) == softmax(x), porque
    #     e^(x_i - c) / Σ e^(x_j - c) = e^(-c)·e^(x_i) / (e^(-c)·Σ e^(x_j))
    # y e^(-c) se cancela. Con c = máximo de cada fila:
    #   - el exponente más grande es 0 → e^0 = 1, así que no hay overflow
    #     (float64 desborda a partir de e^709,78; e^1000 = inf e inf/inf = nan);
    #   - ese término vale exactamente 1, así que el denominador es ≥ 1 aunque
    #     los demás den 0 por underflow (e^-2001 → 0.0). Nunca hay 0/0.
    # Tiene que ser el máximo POR FILA y no el de toda la matriz: con
    # [[1000, 1001], [-1000, -1000]], el máximo global (1001) deja la fila 1 en
    # e^-2001 = 0 → 0/0 = nan.
    #
    # Reducción con keepdims=True: max y sum colapsan el último eje pero lo
    # conservan con tamaño 1, (2, 2) → (2, 1). Sin keepdims daría (2,), y
    # (2, 2) - (2,) restaría por COLUMNA, no por fila.
    #
    # Broadcasting: (2, 2) - (2, 1) y (2, 2) / (2, 1) estiran la columna a lo
    # ancho, así que cada fila usa su propio máximo y su propia suma. Las shapes
    # se comparan desde la derecha; una dimensión de tamaño 1 se estira.
    #
    # Ojo: np.exp(a, b) no es pow(a, b). El segundo argumento es `out` y
    # pisa b sin avisar. np.exp ya usa e como base.
    exps = np.exp(x - x.max(axis=-1, keepdims=True))

    # e^ se calcula una sola vez y se reutiliza en el numerador y en el
    # denominador (como bt en matmul).
    return exps / exps.sum(axis=-1, keepdims=True)
