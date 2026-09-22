# NumPy for Machine Learning

Five marimo notebooks covering the NumPy functions that actually turn up in the machine learning demos in this repository, as a bridge between `Lecture5/IntroductionToNumpy.py`

| Notebook                     | Covers                                                                                               |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| `NumPyForMLPart1Creation.py` | `array`, `arange`, `linspace`, `zeros`, `full`, `zeros_like`, `fromfile`, dtype and shape            |
| `NumPyForMLPart2Shapes.py`   | `reshape`, `ravel`, `flatten`, `newaxis`, `expand_dims`, `transpose`, `meshgrid`, `c_`, broadcasting |
| `NumPyForMLPart3Maths.py`    | ufuncs, `clip`, `@`/`matmul`, `sum`, `mean`, `argmax`, the `axis` parameter, `where`                 |
| `NumPyForMLPart4Random.py`   | `default_rng`, the legacy `random` functions, `isclose`, `allclose`, gradient checking               |
| `NumPyForMLPart5.py`         | `random`, `seed`, `shuffle`, `permutation`                                                           |

Run them with uv:

```
uv run marimo edit NumPyForML/NumPyForMLPart1Creation.py
```
