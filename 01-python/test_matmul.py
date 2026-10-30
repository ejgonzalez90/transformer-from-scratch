import pytest

from matmul import columna, dot, matmul, transponer

# WQ de las notas, sección 6: (4×3)
WQ = [
    [1, 0, 1],
    [0, 2, 1],
    [1, 1, 0],
    [0, 1, 2],
]


# --- dot ---------------------------------------------------------------


def test_dot_vectores_diferentes():
    with pytest.raises(ValueError):
        dot([1, 0, 1], [1, 1, 1, 0])


def test_dot_ejemplo_de_las_notas():
    # q1 · k1 de la sección 6
    assert dot([1, 0, 1, 0], [1, 1, 1, 0]) == 2


def test_dot_vectores_vacios():
    assert dot([], []) == 0


# --- columna -----------------------------------------------------------


def test_columna_de_2x2():
    assert columna([[1, 2], [3, 4]], 1) == [2, 4]


def test_columna_de_WQ():
    assert columna(WQ, 0) == [1, 0, 1, 0]


# --- transponer --------------------------------------------------------

def test_transponer_2x3():
    # (2×3) → (3×2)
    assert transponer([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


def test_transponer_WQ():
    # (4×3) → (3×4): cada fila de WQᵀ es una columna de WQ
    assert transponer(WQ) == [
        [1, 0, 1, 0],
        [0, 2, 1, 1],
        [1, 1, 0, 2],
    ]


def test_transponer_dos_veces_vuelve_al_original():
    assert transponer(transponer(WQ)) == WQ


# --- matmul ------------------------------------------------------------


def test_matmul_x1_por_WQ():
    # (1×4) · (4×3) = (1×3)
    assert matmul([[1, 0, 2, 3]], WQ) == [[3, 5, 7]]


def test_matmul_identidad():
    i2 = [[1, 0], [0, 1]]
    m = [[2, 3], [4, 5]]
    assert matmul(m, i2) == m


def test_matmul_2x2():
    # [[1,2],[3,4]] · [[5,6],[7,8]]
    # fila 0: [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
    # fila 1: [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
    assert matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]


def test_matmul_shape_2x4_por_4x3():
    x = [[1, 0, 2, 3], [0, 1, 0, 1]]
    z = matmul(x, WQ)
    assert len(z) == 2  # n filas, como x
    assert len(z[0]) == 3  # m columnas, como WQ


def test_matmul_dimensiones_que_no_encajan():
    # (1×3) · (4×3): 3 != 4
    with pytest.raises(ValueError):
        matmul([[1, 2, 3]], WQ)
