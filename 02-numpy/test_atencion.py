import numpy as np

from atencion import atencion, puntajes

# Sección 6 de las notas: 2 tokens, d_k = 4 → √d_k = 2
Q = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
K = np.array([[1.0, 1.0, 1.0, 0.0], [2.0, 0.0, 2.0, 1.0]])
V = np.array([[1.0, 0.0, 0.0, 2.0], [0.0, 2.0, 2.0, 0.0]])


# --- Paso 1: puntajes --------------------------------------------------


def test_puntajes_ejemplo_de_las_notas():
    # Q·Kᵀ = [[q1·k1, q1·k2], [q2·k1, q2·k2]] = [[2, 4], [1, 1]];  / 2
    assert np.allclose(puntajes(Q, K), [[1.0, 2.0], [0.5, 0.5]])


def test_puntajes_shape_n_por_m():
    # 1 token de Q contra 3 de K: (1×4)·(4×3) = (1×3)
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    k = np.ones((3, 4))
    assert puntajes(q, k).shape == (1, 3)


def test_puntajes_divide_por_raiz_de_d_k():
    # d_k = 9 → √d_k = 3;  q·k = 9 → 9 / 3 = 3
    q = np.ones((1, 9))
    k = np.ones((1, 9))
    assert np.allclose(puntajes(q, k), [[3.0]])


# --- Paso 2: atención --------------------------------------------------


def test_atencion_ejemplo_de_las_notas():
    # pesos = softmax por fila = [[0,27 0,73], [0,5 0,5]]
    # z1 = 0,27·v1 + 0,73·v2 = [0,27  1,46  1,46  0,54]
    # z2 = 0,5·v1  + 0,5·v2  = [0,5   1     1     1   ]
    esperado = np.array([[0.2689, 1.4621, 1.4621, 0.5379], [0.5, 1.0, 1.0, 1.0]])
    assert np.allclose(atencion(Q, K, V), esperado, atol=1e-4)


def test_atencion_shape_n_por_d_v():
    # d_v no tiene por qué ser igual a d_k: Q, K (3×4) y V (3×2) → Z (3×2)
    q = np.arange(12.0).reshape(3, 4)
    k = np.arange(12.0).reshape(3, 4) / 10
    v = np.arange(6.0).reshape(3, 2)
    assert atencion(q, k, v).shape == (3, 2)


def test_atencion_con_v_iguales_devuelve_ese_v():
    # Si todos los v son iguales, el promedio ponderado es ese mismo v,
    # sin importar los pesos (porque los pesos de cada fila suman 1).
    v = np.array([[3.0, -1.0], [3.0, -1.0]])
    assert np.allclose(atencion(Q, K, v), [[3.0, -1.0], [3.0, -1.0]])
