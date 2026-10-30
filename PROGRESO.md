# Progreso

## Estado

- Módulo en curso: **2. NumPy** (el módulo 1 está cerrado: `01-python/matmul.py`, 15 tests).
- Última sesión: 30/10/2026

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

## Dudas abiertas

-

## Registro de sesiones

| Fecha | Módulo | Qué hice | Qué quedó pendiente |
|---|---|---|---|
| 30/10/2026 | 0 y 1 | Armé el `.venv` e instalé pytest; `00-entorno` pasa (2 tests). Vimos módulos, imports y cómo pytest descubre tests. Implementé `dot`, `columna`, `transponer` y `matmul` en `01-python/matmul.py`: pasan los 15 tests. | Módulo 1 cerrado. Próximo: módulo 2 (agregar NumPy a `requirements.txt`, `softmax` estable). |
