# Hugging Face Spaces Docker SDK target.
#
# Build context is the REPO ROOT, not analysis/dashboard/. HF Spaces
# clones the repo and runs `docker build` from the root with the path
# to this Dockerfile, so the COPY paths below are repo-relative.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Project metadata, library code, and dashboard skeleton.
COPY pyproject.toml README.md /app/
COPY roboeval /app/roboeval
COPY analysis/dashboard /app/analysis/dashboard

# The only data file the runtime dashboard reads — schema v2 bundles
# cells + ablation + Welch's t-tests inline (built by
# scripts/build_headline_json.py from the gitignored auto_labels at
# build time so the runtime image stays self-contained).
COPY data/headline.json /app/data/headline.json

RUN pip install --upgrade pip \
 && pip install --no-deps -e . \
 && pip install -r analysis/dashboard/requirements.txt
# `--no-deps` skips the full pyproject ML stack (lerobot, mujoco, torch,
# sb3, etc.) — none of which the dashboard imports at runtime. The
# `roboeval.dashboard` subpackage only needs plotly + dash + dbc, which
# requirements.txt installs explicitly.

ENV PYTHONPATH=/app

EXPOSE 7860
CMD ["gunicorn", "--chdir", "analysis/dashboard", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "2", \
     "--timeout", "60", \
     "app:server"]
