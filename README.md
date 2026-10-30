# transformer-from-scratch

¿Sabés qué? Los Transformers son un bardo loco. La única forma que tengo de entender este delirio es armar uno y verlo correr, matriz por matriz. Y ya que estoy, aprendo Python.

Base teórica: mis notas de *The Illustrated Transformer* en [`notas/illustrated-transformer.md`](notas/illustrated-transformer.md).

## Ruta

Cada módulo tiene su carpeta (`01-python/`, `02-numpy/`, …), con los ejercicios y sus tests.

| # | Módulo | Qué construyo | Cómo sé que funciona |
|---|---|---|---|
| 0 | Entorno | `venv`, `pip`, `pytest`, correr un script | `pytest` pasa con un test vacío |
| 1 | Python para alguien que viene de C#/TS | Multiplicación de matrices con listas puras (`x1 · WQ`) | `matmul([[1,0,2,3]], WQ) == [[3,5,7]]` |
| 2 | NumPy | `softmax` estable y self-attention de una cabeza | Reproduce la sección 6 de las notas: `[0,27 0,73]` |
| 3 | El bloque del codificador en NumPy | Máscara, multi-head, codificación posicional, layer norm, feed-forward | Shapes `(n, d_model)` de punta a punta y tests con números a mano |
| 4 | PyTorch | Tensores, autograd, `nn.Module`; portar el módulo 3 | Mismos resultados que la versión NumPy |
| 5 | Mini GPT (*decoder-only*) | Modelo a nivel de caracteres entrenado en CPU con un texto chico | La pérdida baja y genera texto que "se parece" al corpus |
| 6 | (Opcional) Codificador-decodificador completo | Una tarea de juguete, como invertir una secuencia | Exactitud alta en el conjunto de prueba |

## Reglas del repo

- El código lo escribo yo. La IA explica, propone ejercicios y revisa.
- Cada ejercicio tiene un test con números calculados a mano.
- El avance se anota en `PROGRESO.md`.

## Entorno

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash en Windows
pip install -r requirements.txt
pytest
```
