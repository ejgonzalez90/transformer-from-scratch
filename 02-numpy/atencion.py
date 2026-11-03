"""Self-attention de una cabeza con NumPy.

Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V        (sección 6 de las notas)

Shapes (n = tokens de Q, m = tokens de K y V):
    Q (n×d_k)   K (m×d_k)   V (m×d_v)   →   Z (n×d_v)
En self-attention n == m, porque Q, K y V salen de los mismos tokens.
"""

import numpy as np

from softmax import softmax


def puntajes(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Q·Kᵀ / √d_k: el puntaje de cada token de Q contra cada token de K.

    (n×d_k)·(d_k×m) = (n×m). La fila i son los puntajes del token i.
    """
    return (Q @ K.T) / np.sqrt(Q.shape[-1])


def atencion(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """softmax(puntajes) · V: cada z_i es un promedio de los v, ponderado.

    (n×m)·(m×d_v) = (n×d_v)
    """
    return softmax(puntajes(Q, K)) @ V
