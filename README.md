# Hands-on ML Study

A focused, reproducible workspace for studying machine learning by building small experiments, recording conclusions, and turning reusable ideas into tested Python code.

## Repository layout

- `notebooks/`: numbered study notebooks and experiments.
- `src/hands_on_ml/`: reusable code extracted from notebooks.
- `tests/`: fast checks for reusable code.
- `data/`: local datasets; raw and generated data stay out of Git.
- `reports/figures/`: exported plots used in study notes.

## Working rhythm

1. Start an experiment in a numbered notebook such as `01_end_to_end_ml.ipynb`.
2. Keep notebooks readable and record the conclusion next to the result.
3. Move reusable logic into `src/hands_on_ml`.
4. Add a small test before relying on that logic in another notebook.
5. Commit code and conclusions, not large datasets or generated outputs.

## Useful commands

```bash
jupyter lab
pytest
ruff check .
```
