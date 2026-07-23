# Hugging Face Space (Docker SDK) for the SPECTRA inference API.
# Self-contained CPU image: installs the package from GitHub, bakes ESM-2, and
# mints an (untrained) smoke checkpoint so the Space goes live with no manual
# model upload. Swap in trained weights later (see README swap note).
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as uid 1000; write only under the user's home.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface \
    SPECTRA_MODE=E SPECTRA_DEVICE=cpu \
    SPECTRA_ESM_CKPT=facebook/esm2_t6_8M_UR50D \
    SPECTRA_CHECKPOINT=/home/user/app/model.pt \
    PORT=7860
USER user
WORKDIR /home/user/app

# CPU torch + pinned serving stack + the SPECTRA package (no heavy research deps)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers==4.40.2 huggingface-hub==0.23.0 \
      fastapi==0.111.0 "uvicorn[standard]==0.29.0" pydantic==2.7.1 \
      numpy==1.26.4 pandas==2.2.2 \
 && pip install --no-cache-dir --no-deps "git+https://github.com/92kunheekim/SPECTRA.git@main"

# Bake ESM-2 weights, then mint the untrained smoke checkpoint (self-contained).
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'); AutoModel.from_pretrained('facebook/esm2_t6_8M_UR50D')" \
 && python -m spectra.inference.dummy_checkpoint --mode E --out /home/user/app/model.pt

EXPOSE 7860
CMD ["sh","-c","uvicorn spectra.inference.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
