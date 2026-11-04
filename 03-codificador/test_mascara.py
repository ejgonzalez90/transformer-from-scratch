import numpy as np

from mascara import atencion_enmascarada, mascara_causal

# Los mismos Q, K y V de la sección 6 de las notas
Q = np.array([[1.0, 0.0, 1.0, 0.0],
              [0.0, 1.0, 0.0, 1.0]])
K = np.array([[1.0, 1.0, 1.0, 0.0],
              [2.0, 0.0, 2.0, 1.0]])
V = np.array([[1.0, 0.0, 0.0, 2.0],
              [0.0, 2.0, 2.0, 0.0]])

INF = np.inf


# --- Paso 1: la máscara ------------------------------------------------

def test_mascara_3x3():
    esperado = np.array([[0.0, -INF, -INF],
                         [0.0,  0.0, -INF],
                         [0.0,  0.0,  0.0]])
    assert np.array_equal(mascara_causal(3), esperado)


def test_mascara_1x1_no_tapa_nada():
    assert np.array_equal(mascara_causal(1), [[0.0]])


def test_mascara_shape():
    assert mascara_causal(5).shape == (5, 5)


# --- Paso 2: atención enmascarada --------------------------------------

def test_atencion_enmascarada_ejemplo_de_las_notas():
    # Sección 9: puntajes / 2 = [[1, 2], [0,5, 0,5]]
    #   + máscara               [[1, -∞], [0,5, 0,5]]
    #   softmax →               [[1, 0],  [0,5, 0,5]]
    # z1 = 1·v1 + 0·v2         = [1, 0, 0, 2]
    # z2 = 0,5·v1 + 0,5·v2     = [0,5, 1, 1, 1]
    esperado = np.array([[1.0, 0.0, 0.0, 2.0],
                         [0.5, 1.0, 1.0, 1.0]])
    assert np.allclose(atencion_enmascarada(Q, K, V, mascara_causal(2)), esperado)


def test_el_primer_token_solo_se_ve_a_si_mismo():
    rng = np.random.default_rng(0)
    q, k, v = rng.normal(size=(3, 4, 4))
    z = atencion_enmascarada(q, k, v, mascara_causal(4))
    assert np.allclose(z[0], v[0])


def test_el_ultimo_token_ve_todo_igual_que_sin_mascara():
    # La última fila de la máscara es toda 0: no cambia nada.
    rng = np.random.default_rng(1)
    q, k, v = rng.normal(size=(3, 4, 4))
    sin_mascara = atencion_enmascarada(q, k, v, np.zeros((4, 4)))
    con_mascara = atencion_enmascarada(q, k, v, mascara_causal(4))
    assert np.allclose(con_mascara[-1], sin_mascara[-1])
    assert not np.allclose(con_mascara[0], sin_mascara[0])
