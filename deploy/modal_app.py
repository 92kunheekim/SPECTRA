"""Deploy the SPECTRA inference API on Modal (serverless, scale-to-zero, no card).

    pip install modal
    modal token new                      # one-time browser OAuth (GitHub/Google)
    modal deploy deploy/modal_app.py

Modal prints a public URL, e.g.
    https://<workspace>--spectra-tcr-pmhc-fastapi-app.modal.run
Test it:
    curl  <url>/health
    open  <url>/docs
    curl -X POST <url>/predict -H 'content-type: application/json' \\
         -d @deploy/sample_request.json

NOTE: this serves the SPECTRA mode-E architecture with an UNTRAINED smoke
checkpoint (minted at image-build time), so probabilities are placeholders --
it proves the serving path end to end. To serve the trained 15-fold-CV weights,
add your real model.pt to the image (image.add_local_file / a Modal Volume) and
drop the dummy_checkpoint step below.
"""
import os
import modal

app = modal.App("spectra-tcr-pmhc")

ESM = "facebook/esm2_t6_8M_UR50D"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential")
    # CPU torch from the CPU wheel index, then the pinned serving stack
    .pip_install("torch==2.2.2", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "transformers==4.40.2", "huggingface-hub==0.23.0", "fastapi==0.111.0",
        "uvicorn[standard]==0.29.0", "pydantic==2.7.1", "numpy==1.26.4", "pandas==2.2.2",
    )
    # the SPECTRA package WITHOUT its heavy research deps (lightning/dgl/pyg)
    .run_commands(
        "pip install --no-deps 'git+https://github.com/92kunheekim/SPECTRA.git@main'"
    )
    .env({
        "SPECTRA_MODE": "E",
        "SPECTRA_DEVICE": "cpu",
        "SPECTRA_ESM_CKPT": ESM,
        "SPECTRA_CHECKPOINT": "/root/model.pt",
        "HF_HOME": "/root/.cache/huggingface",
    })
    # bake ESM-2 weights + mint the untrained smoke checkpoint (self-contained)
    .run_commands(
        f"python -c \"from transformers import AutoTokenizer, AutoModel; "
        f"AutoTokenizer.from_pretrained('{ESM}'); AutoModel.from_pretrained('{ESM}')\"",
        "python -m spectra.inference.dummy_checkpoint --mode E --out /root/model.pt",
    )
)


@app.function(image=image, memory=2048, timeout=120)
@modal.asgi_app()
def fastapi_app():
    # Load the checkpoint into the FastAPI app state once per container start,
    # explicitly (so it does not depend on ASGI-lifespan support).
    from spectra.inference import api, serving
    if api._STATE["model"] is None:
        api._STATE["model"], api._STATE["tok"] = serving.load_model(
            os.environ["SPECTRA_CHECKPOINT"],
            mode=api._STATE["mode"],
            device="cpu",
        )
    return api.app
