"""Máscara causal para la self-attention del decodificador (sección 9 de las notas).

El token i solo puede mirar a los tokens 0..i: los posteriores son el "futuro".
Se suma -∞ a esos puntajes ANTES del softmax, y como e^-∞ = 0, su peso queda en 0.
"""

import numpy as np

from atencion import puntajes
from softmax import softmax


def mascara_causal(n: int) -> np.ndarray:
    """Matriz (n×n) para SUMAR a los puntajes: 0 donde se puede mirar, -∞ en el futuro.

    mascara_causal(3) == [[0, -∞, -∞],
                          [0,  0, -∞],
                          [0,  0,  0]]
    """
    return np.where(np.triu(np.ones((n, n)), k=1) > 0, -np.inf, 0)


def atencion_enmascarada(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mascara: np.ndarray
) -> np.ndarray:
    """softmax(Q·Kᵀ / √d_k + mascara) · V"""
    return softmax(puntajes(Q, K) + mascara) @ V
