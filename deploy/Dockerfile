# syntax=docker/dockerfile:1
# =====================================================================
# SPECTRA inference image (CPU). Serves the sequence-based ablation model
# via FastAPI. Graph backends (dgl / torch_geometric) are NOT needed here.
# Build:  docker build -f deploy/Dockerfile -t spectra:latest .
# Run:    docker run -p 8000:8000 -e SPECTRA_CHECKPOINT=/models/model.pt \
#                 -v $PWD/models:/models:ro spectra:latest
# =====================================================================
ARG PYTHON_VERSION=3.10-slim
ARG ESM_MODEL=facebook/esm2_t6_8M_UR50D

# ----------------------- Stage 1: builder -----------------------
FROM python:${PYTHON_VERSION} AS builder
ARG ESM_MODEL
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf-cache
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# isolated venv, copied wholesale into the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU torch from the CPU wheel index, then pinned serving deps
RUN pip install --upgrade pip \
    && pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
COPY deploy/requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# install the package WITHOUT its heavy research deps (lightning, dgl, etc.)
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY spectra ./spectra
RUN pip install --no-deps .

# bake ESM-2 weights into the image so runtime needs no network
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('${ESM_MODEL}'); AutoModel.from_pretrained('${ESM_MODEL}')"

# ----------------------- Stage 2: runtime -----------------------
FROM python:${PYTHON_VERSION} AS runtime
ARG ESM_MODEL
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    SPECTRA_MODE=E SPECTRA_DEVICE=cpu \
    SPECTRA_ESM_CKPT=${ESM_MODEL}

# bring over the venv (with package + deps) and the baked HF cache
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf-cache /opt/hf-cache

# run as an unprivileged user
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /opt/hf-cache
USER appuser
WORKDIR /home/appuser

# PORT-aware entrypoint so the same image runs on Cloud Run / Fly / Render
# (which inject $PORT) and locally (defaults to 8000).
COPY --chown=appuser:appuser deploy/entrypoint.sh /home/appuser/entrypoint.sh
RUN chmod +x /home/appuser/entrypoint.sh

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://localhost:{p}/health').getcode()==200 else 1)"

# Provide the trained weights at runtime (mount) OR bake them in (Dockerfile.cloudrun):
#   -e SPECTRA_CHECKPOINT=/models/model.pt  -v $PWD/models:/models:ro
CMD ["/home/appuser/entrypoint.sh"]
