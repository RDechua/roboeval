# Dashboard

Pure data + figure logic for the Phase 5 Plotly/Dash app. The Dash skeleton itself lives at
`analysis/dashboard/app.py`; this package contains only the loaders, dataclasses, and figure
builders so it can be unit-tested under `mypy --strict` without importing `dash`.

::: roboeval.dashboard
