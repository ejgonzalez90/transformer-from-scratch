"""Multi-head attention (sección 7 de las notas).

Q = X·WQ, K = X·WK, V = X·WV                     (n×d_model) cada una
se parten en h cabezas                            (h × n × d_head),  d_head = d_model / h
Z_i = softmax(Q_i·K_iᵀ / √d_head + máscara) · V_i  en cada cabeza
Concat(Z_1..Z_h) · WO                            (n×d_model)
"""

import numpy as np
from mascara import atencion_enmascarada


def dividir_cabezas(M: np.ndarray, h: int) -> np.ndarray:
    """(n × d_model) → (h × n × d_head): la cabeza i se queda con el bloque i de columnas.

    M es una matriz ya proyectada: en multihead se aplica a Q, K y V (no a X).

    dividir_cabezas([[1, 2, 3, 4],
                     [5, 6, 7, 8]], h=2) == [[[1, 2], [5, 6]],    ← cabeza 0
                                             [[3, 4], [7, 8]]]    ← cabeza 1

    Si d_model no se divide por h, lanzá ValueError.
    """
    n, d_model = M.shape
    if d_model % h != 0:
        raise ValueError(f"d_model = {d_model} no se divide en {h} cabezas")
    return np.swapaxes(M.reshape(n, h, d_model // h), 0, 1)


def unir_cabezas(Z: np.ndarray) -> np.ndarray:
    """(h × n × d_head) → (n × h·d_head): la inversa de dividir_cabezas (el Concat)."""
    h, n, d_head = Z.shape
    return np.swapaxes(Z, 0, 1).reshape(n, h * d_head)


def multihead(
    X: np.ndarray,
    WQ: np.ndarray,
    WK: np.ndarray,
    WV: np.ndarray,
    WO: np.ndarray,
    h: int,
    mascara: np.ndarray | None = None,
) -> np.ndarray:
    """X (n×d_model), W* (d_model×d_model) → Z (n×d_model).

    Si viene una máscara (n×n), se aplica en todas las cabezas.
    """

    Q = dividir_cabezas(X @ WQ, h)
    K = dividir_cabezas(X @ WK, h)
    V = dividir_cabezas(X @ WV, h)

    Z = atencion_enmascarada(
        Q,
        K,
        V,
        0 if mascara is None else mascara,
    )

    return unir_cabezas(Z) @ WO
