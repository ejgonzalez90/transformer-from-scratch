import numpy as np

from softmax import softmax

# A mano: e¹ = 2,718 y e² = 7,389; suman 10,107
#   2,718 / 10,107 = 0,2689      7,389 / 10,107 = 0,7311


# --- Paso 1: un vector -------------------------------------------------

def test_vector_ejemplo_de_las_notas():
    # Sección 6, token 1: puntajes / √d_k = [1, 2]
    assert np.allclose(softmax(np.array([1.0, 2.0])), [0.2689, 0.7311], atol=1e-4)


def test_vector_suma_1():
    p = softmax(np.array([3.0, -1.0, 0.0, 2.0]))
    assert np.isclose(p.sum(), 1.0)
    assert (p > 0).all()


def test_vector_puntajes_iguales_dan_probabilidades_iguales():
    assert np.allclose(softmax(np.array([0.5, 0.5])), [0.5, 0.5])


def test_vector_mantiene_la_shape():
    assert softmax(np.array([1.0, 2.0, 3.0])).shape == (3,)


# --- Paso 2: una matriz, por fila --------------------------------------

def test_matriz_por_fila_ejemplo_de_las_notas():
    # Sección 6: Q·Kᵀ / 2 = [[1, 2], [0,5, 0,5]]
    puntajes = np.array([[1.0, 2.0],
                         [0.5, 0.5]])
    esperado = np.array([[0.2689, 0.7311],
                         [0.5,    0.5   ]])
    assert np.allclose(softmax(puntajes), esperado, atol=1e-4)


def test_matriz_cada_fila_suma_1():
    p = softmax(np.array([[1.0, 2.0, 3.0],
                          [0.0, 0.0, 9.0]]))
    assert p.shape == (2, 3)
    assert np.allclose(p.sum(axis=1), [1.0, 1.0])


# --- Paso 3: estabilidad -----------------------------------------------

def test_estable_con_numeros_grandes():
    # e^1000 no entra en un float64: da inf, e inf / inf = nan
    p = softmax(np.array([1000.0, 1001.0]))
    assert not np.isnan(p).any()
    assert np.allclose(p, [0.2689, 0.7311], atol=1e-4)


def test_estable_en_matriz():
    p = softmax(np.array([[1000.0, 1001.0],
                          [-1000.0, -1000.0]]))
    assert np.allclose(p, [[0.2689, 0.7311], [0.5, 0.5]], atol=1e-4)
