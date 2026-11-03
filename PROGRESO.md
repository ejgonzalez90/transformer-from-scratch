# Progreso

## Estado

- Módulo en curso: **3. Bloque del codificador**. El módulo 2 está cerrado: `softmax` y `atencion` en `02-numpy/`, 14 tests.
- Última sesión: 03/11/2026

## Ya entendido (de la lectura)

- Q, K y V salen de `x · WQ`, `x · WK` y `x · WV`. Las matrices W se aprenden en el entrenamiento.
- Atención: `softmax(Q·Kᵀ / √d_k) · V`.
- Multi-head: 8 cabezas de 64 dimensiones, concatenadas y multiplicadas por `WO`.
- La máscara del decodificador pone `-∞` en las posiciones futuras.

## Ya entendido (del código)

- Módulo 0: `.venv` creado con Python 3.14.3 y pytest 9.2.0. Se activa con `source .venv/Scripts/activate`.
- `test_estoy_en_el_venv` verifica que pytest corre con el Python del entorno.
- Recorrer valores en vez de índices (`for fila in m`) y usar `zip(u, v, strict=True)`: sin `strict`, `zip` corta en la lista más corta sin avisar.
- `list.append` es O(1) amortizado, como `List<T>.Add`. La comprehension es más idiomática y algo más rápida, pero tiene la misma complejidad.
- Expresión generadora `sum(x * y for ...)`: no arma la lista intermedia y se consume una sola vez. Hay ejemplos comentados en `01-python/matmul.py`.
- `[[0] * p] * m` repite la misma fila m veces (son referencias); hay que usar una comprehension anidada.
- En Python todo es objeto y las listas guardan referencias, incluso a `0`. Lo que importa es **mutar contra reasignar**: `int`, `float`, `str` y `tuple` son inmutables, así que compartirlos no hace daño (como `string` en C#); `list` y `dict` son mutables. `x[i] = …` reasigna un casillero, y si la lista está compartida, el cambio se ve desde todas las referencias. `*` copia superficialmente; `copy.deepcopy` copia en profundidad. `==` es como `Equals` e `is` es como `ReferenceEquals`.
- `matmul` = `[[dot(r, c) for c in bt] for r in a]` con `bt = transponer(b)` calculada afuera: en una comprehension anidada, la de adentro se evalúa entera por cada vuelta de la de afuera. `matmul` = `dot(fila i de a, fila j de bᵀ)`. Si se transpone `b` una vez antes de los bucles, no se recalcula cada columna por fila: la 200×200 bajó de 1,85 s a 1,05 s. Es la misma idea que `Q·Kᵀ`.
- NumPy: las operaciones son elemento a elemento (`np.exp`, `*`, `/`), así que no hacen falta `for` ni `if x.ndim`. Una reducción (`sum`, `max`, `mean`) colapsa el eje que se le indica; `axis=-1` es siempre el último eje, y `keepdims=True` lo conserva con tamaño 1. No es `flatten`: es como `reduce` en JS.
- Broadcasting: las shapes se comparan desde la derecha, y una dimensión de tamaño 1 se estira. `(2,2) / (2,1)` divide cada fila por su propio valor.
- `np.exp(a, b)` no es `pow`: el segundo argumento es `out`, y pisa `b` sin dar error.
- Softmax estable: `softmax(x − c) = softmax(x)`, porque `e^(−c)` se cancela arriba y abajo. Con `c` = máximo **por fila**, el mayor exponente es 0, así que no hay overflow, y el término máximo vale exactamente 1, así que el denominador es ≥ 1 aunque los demás términos den 0 por underflow. Con el máximo de toda la matriz, una fila lejana da 0/0 = nan. `float64` desborda en `e^709,78`.
- Self-attention: `softmax(Q·Kᵀ / √d_k) · V`, con `d_k = Q.shape[-1]` (el último eje, como `axis=-1`; con un batch, `shape[1]` serían los tokens). Q (n×d_k), K (m×d_k), V (m×d_v) → Z (n×d_v).
- Cada fila de pesos suma 1, así que `z_i` es un promedio ponderado de los `v`. Si todos los `v` son iguales, `z_i = (Σ pesos)·v = v`.
- `X (m×512) · WQ (512×64) = Q (m×64)`: Q tiene una fila por token. Las cabezas se reparten **columnas** (dimensiones), no tokens: 8 × 64 = 512, más o menos el mismo costo que una cabeza de 512.
- En `(m×n)·(n×q)` con `bᵀ` de `(q×n)`, `dot` siempre recibe dos vectores de largo `n`, así que `strict=True` solo verifica `n`. `m` y `q` son la cantidad de vueltas de cada `for`, o sea, la shape del resultado `(m×q)`; nunca se comparan entre sí.

## Dudas abiertas

-

## Registro de sesiones

| Fecha | Módulo | Qué hice | Qué quedó pendiente |
|---|---|---|---|
| 30/10/2026 | 0 y 1 | Armé el `.venv` e instalé pytest; `00-entorno` pasa (2 tests). Vimos módulos, imports y cómo pytest descubre tests. Implementé `dot`, `columna`, `transponer` y `matmul` en `01-python/matmul.py`: pasan los 15 tests. | Módulo 1 cerrado. Próximo: módulo 2 (agregar NumPy a `requirements.txt`, `softmax` estable). |
| 02/11/2026 | 2 | Instalé NumPy 2.5.4. `A @ A` de 200×200: 0,5 ms contra 1,13 s con listas (~2.300 veces más rápido), y da el mismo resultado. Vimos `shape`, operaciones elemento a elemento, `axis`, `keepdims` y broadcasting. | — |
| 03/11/2026 | 2 | Saqué las rutas locales de la historia del repo y copié las notas a `notas/`. Implementé `softmax` estable (8 tests, sin warnings) y `puntajes` + `atencion`, que reproducen `z1 = [0,27 1,46 1,46 0,54]` de la sección 6. Módulo 2 cerrado. | Módulo 3: máscara, multi-head (y `K.T` con batch). |
