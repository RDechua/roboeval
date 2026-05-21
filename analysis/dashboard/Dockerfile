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

# Project metadata and library code.
COPY pyproject.toml README.md /app/
COPY roboeval /app/roboeval

# Dashboard runtime + deploy artifacts.
COPY analysis/dashboard /app/analysis/dashboard

# Pre-baked data the dashboard reads at boot.
COPY data/headline.json /app/data/headline.json
COPY docs/figures/phase4_ablation.json /app/docs/figures/phase4_ablation.json
COPY outputs/eval/act_spatial_y+5cm/eval_results_w6k2wole.json \
     /app/outputs/eval/act_spatial_y+5cm/eval_results_w6k2wole.json
COPY outputs/residual/y+5cm_sparse/eval_results_o6ukyo53.json \
     /app/outputs/residual/y+5cm_sparse/eval_results_o6ukyo53.json
COPY outputs/residual/y+5cm_shaped/eval_results_43czuigy.json \
     /app/outputs/residual/y+5cm_shaped/eval_results_43czuigy.json

# auto_labels sidecars for the three Phase 4 conditions
# (data.load_phase4_eval_results does a sibling lookup by run_id).
COPY data/taxonomy/auto_labels_w6k2wole.json /app/data/taxonomy/auto_labels_w6k2wole.json
COPY data/taxonomy/auto_labels_o6ukyo53.json /app/data/taxonomy/auto_labels_o6ukyo53.json
COPY data/taxonomy/auto_labels_43czuigy.json /app/data/taxonomy/auto_labels_43czuigy.json

RUN pip install --upgrade pip \
 && pip install -e . \
 && pip install -r analysis/dashboard/requirements.txt

ENV PYTHONPATH=/app

EXPOSE 7860
CMD ["gunicorn", "--chdir", "analysis/dashboard", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "2", \
     "--timeout", "60", \
     "app:server"]
