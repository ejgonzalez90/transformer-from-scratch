import numpy as np
import pytest

from atencion import puntajes
from mascara import mascara_causal
from multihead import dividir_cabezas, multihead, unir_cabezas

X = np.array([[1.0, 2.0, 3.0, 4.0],
              [5.0, 6.0, 7.0, 8.0]])


# --- Paso 1: partir y unir cabezas --------------------------------------

def test_dividir_en_2_cabezas():
    # (2×4) → (2 cabezas × 2 tokens × 2 dims)
    esperado = np.array([[[1.0, 2.0], [5.0, 6.0]],     # cabeza 0: columnas 0 y 1
                         [[3.0, 4.0], [7.0, 8.0]]])    # cabeza 1: columnas 2 y 3
    assert np.array_equal(dividir_cabezas(X, 2), esperado)


def test_dividir_shape_como_el_paper():
    # 3 tokens, d_model = 512, 8 cabezas → (8, 3, 64)
    assert dividir_cabezas(np.zeros((3, 512)), 8).shape == (8, 3, 64)


def test_dividir_con_d_model_no_divisible():
    with pytest.raises(ValueError):
        dividir_cabezas(X, 3)   # 4 / 3 no da entero


def test_unir_es_la_inversa_de_dividir():
    assert np.array_equal(unir_cabezas(dividir_cabezas(X, 2)), X)


# --- Paso 2: puntajes con varias cabezas a la vez ------------------------
# Esto es en 02-numpy/atencion.py: hoy puntajes usa K.T, que con 3 dimensiones
# da vuelta TODOS los ejes. Hay que transponer solo los dos últimos.

def test_puntajes_con_cabezas_es_cada_cabeza_por_separado():
    Q = np.array([[[1.0, 0.0], [0.0, 1.0]],      # cabeza 0: (2×2)
                  [[1.0, 1.0], [2.0, 0.0]]])     # cabeza 1: (2×2)
    K = np.array([[[1.0, 1.0], [2.0, 0.0]],
                  [[0.0, 1.0], [1.0, 1.0]]])
    esperado = np.stack([puntajes(Q[0], K[0]), puntajes(Q[1], K[1])])
    assert puntajes(Q, K).shape == (2, 2, 2)
    assert np.allclose(puntajes(Q, K), esperado)


# --- Paso 3: multi-head completo -----------------------------------------
# Con W = identidad, Q = K = V = X, y las cuentas salen a mano.
# Es el ejemplo de las notas (§7): la mitad izquierda es la "gramática"
# y la derecha, el "significado".

I4 = np.eye(4)
X_NOTAS = np.array([[1.0, 0.0, 0.0, 1.0],    # token 0
                    [1.0, 0.0, 0.0, 0.0]])   # token 1


def test_multihead_2_cabezas_a_mano():
    # Cabeza 0 (cols 0-1): q0 = q1 = [1 0], k0 = k1 = [1 0]  → pesos [0,5 0,5] en las dos filas
    #                      v0 = v1 = [1 0]                    → z = [1 0] en las dos filas
    # Cabeza 1 (cols 2-3): q0 = [0 1], k0 = [0 1], k1 = [0 0]
    #   fila 0: [1, 0] / √2 → softmax → [0,67 0,33] → z = 0,67·[0 1] + 0,33·[0 0] = [0 0,67]
    #   fila 1: q1 = [0 0] → [0, 0]   → softmax → [0,5  0,5 ] → z = [0 0,5]
    # Concat · WO (identidad):
    esperado = np.array([[1.0, 0.0, 0.0, 0.6698],
                         [1.0, 0.0, 0.0, 0.5   ]])
    assert np.allclose(multihead(X_NOTAS, I4, I4, I4, I4, h=2), esperado, atol=1e-4)


def test_multihead_con_mascara_el_primer_token_solo_se_ve_a_si_mismo():
    # Con máscara causal, el token 0 copia su propio v en cada cabeza → vuelve x0
    z = multihead(X_NOTAS, I4, I4, I4, I4, h=2, mascara=mascara_causal(2))
    assert np.allclose(z[0], X_NOTAS[0])


def test_multihead_aplica_WO():
    # WO que invierte el orden de las columnas: [a b c d] → [d c b a]
    WO = np.fliplr(I4)
    sin_wo = multihead(X_NOTAS, I4, I4, I4, I4, h=2)
    con_wo = multihead(X_NOTAS, I4, I4, I4, WO, h=2)
    assert np.allclose(con_wo, sin_wo[:, ::-1])


def test_multihead_shape_n_por_d_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 8))
    WQ, WK, WV, WO = rng.normal(size=(4, 8, 8))
    assert multihead(X, WQ, WK, WV, WO, h=4).shape == (3, 8)
