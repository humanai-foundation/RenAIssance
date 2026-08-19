# ============================================
# RenAIssance OCR Backend
# Multi-stage build — one Dockerfile, two variants (GPU / CPU)
#
#   Build the GPU image (default):
#     docker build -t renaissance-backend:gpu ./backend
#   Build the CPU image (Mac / no-NVIDIA hosts):
#     docker build --build-arg VARIANT=cpu \
#                  --build-arg RUNTIME_BASE=ubuntu:24.04 \
#                  -t renaissance-backend:cpu ./backend
#
# The two variants differ ONLY in the torch + paddle wheels and the runtime
# base image; all application code and other pins are identical. The app
# already selects cuda/mps/cpu at runtime, so the same code runs on either.
#
# Stage 1 — ubuntu:24.04  (+Python 3.11 deadsnakes)
#            Installs all Python deps into /opt/venv (variant-specific wheels)
#
# Stage 2 — ${RUNTIME_BASE}  (+Python 3.11 deadsnakes)
#            ubuntu:24.04 for BOTH variants. The GPU variant does NOT need
#            the multi-GB nvidia/cuda runtime base: torch ships its CUDA
#            user-space libs as nvidia-*-cu12 pip wheels and paddle RPATHs
#            into the same site-packages/nvidia directory, while the host
#            driver (libcuda.so.1) is injected at `docker run --gpus all`
#            time by the NVIDIA Container Toolkit. Dropping the CUDA base
#            saves ~3.5 GB of system CUDA libraries nothing ever loads.
#
# WHY the same base OS for both stages:
#   Native extensions (.so wheels) are linked against the host distro's
#   glibc, OpenSSL and libstdc++.  Copying a venv from Debian 12 into
#   Ubuntu 24.04 (or vice-versa) causes "GLIBC_2.38 not found" /
#   "OPENSSL_3.3.0 not found" errors.  Both stages are Ubuntu 24.04,
#   so they share the builder's ABI.
#
# GPU variant requirements on the host:
#   - NVIDIA driver >= 560 (supports CUDA 12.6)
#   - NVIDIA Container Toolkit installed
#   - Run:  docker run --gpus all -p 8000:8000 <image>
# ============================================

# Global build args (consumed in the stages below; re-declared per stage).
# RUNTIME_BASE stays overridable but ubuntu:24.04 is correct for both
# variants — see the Stage 2 note above.
ARG VARIANT=gpu
ARG RUNTIME_BASE=ubuntu:24.04

############################
# Stage 1 — dependency builder
############################
FROM ubuntu:24.04 AS builder

ARG VARIANT

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python 3.11 from deadsnakes PPA + build tools.
# Ubuntu 24.04 ships Python 3.12 by default; using 3.11 here to match
# paddlepaddle-gpu ABI and keep numpy/other pinned versions stable.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a Python 3.11 virtual environment — all packages go here.
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .

# Install Python dependencies in two steps to sidestep a hard conflict
# between torch and paddle over their nvidia-*-cu12 pins (torch pins one set
# of CUDA user-space libs, paddle pins another; pip's resolver cannot satisfy
# both simultaneously). Both wheels bundle their own CUDA libs internally, so
# the duplicate site-packages nvidia-* entries are not needed at runtime —
# installing paddle with --no-deps avoids the conflict.
#
# Variant split:
#   gpu: torch 2.5.1 CUDA build (PyPI default) + paddlepaddle-gpu 3.3.0 (cu126)
#   cpu: torch 2.5.1 CPU build (pytorch cpu index) + paddlepaddle 3.3.0 (cpu).
#        torch CPU is installed FIRST so `-r requirements.txt` finds it already
#        satisfied and does not pull the multi-GB CUDA wheel.
#
# BuildKit cache mount keeps ~/.cache/pip warm across builds so rebuilds
# after a requirements.txt edit don't re-download multi-GB PyTorch/Paddle
# wheels. Enable with:  DOCKER_BUILDKIT=1 docker build ...  (default in
# modern Docker).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    if [ "$VARIANT" = "cpu" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && \
        pip install -r requirements.txt && \
        pip install --no-deps \
            --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
            paddlepaddle==3.3.0 ; \
    else \
        pip install -r requirements.txt && \
        pip install --no-deps \
            --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
            paddlepaddle-gpu==3.3.0 ; \
    fi

# NOTE: no font pre-download step. paddlex 3.0 downloaded fonts into its own
# root-owned install dir (failing at runtime for non-root appuser), which is
# why older builds baked them in. paddlex 3.4 changed this: fonts are lazy and
# cached under PADDLE_PDX_CACHE_HOME (=/paddle_models, a writable named volume),
# so they download on first use to a writable location — same pattern as models.

# Slim the venv before copying it into the runtime stage.
#
# 1. Deduplicate OpenCV: requirements pin opencv-python-headless but
#    paddleocr → paddlex ALSO pulls opencv-contrib-python, so the image
#    ends up with two full OpenCV builds sharing one cv2/ directory.
#    paddlex verifies its [ocr] extra by installed-package METADATA, so
#    contrib must be the one that stays (removing it breaks PPStructureV3
#    creation with a "requires additional dependencies" error even though
#    the cv2 module itself would import fine). Force-reinstall contrib
#    LAST so the shared cv2/ directory is complete. Saves ~170 MB.
RUN pip uninstall -y opencv-python-headless opencv-python 2>/dev/null; \
    pip install --no-deps --force-reinstall --no-cache-dir opencv-contrib-python==4.10.0.84
#
# 1b. paddleocr hard-depends on the LangChain/OpenAI client stack for its
#     PP-ChatOCR document-chat pipelines, which this app never uses (we
#     only run PPStructureV3 + PaddleOCR detection). The paddlex [ocr]
#     dependency check passes without them — verified by running layout
#     detection after removal. Saves ~150 MB and a large dep surface.
RUN pip uninstall -y \
        langchain langchain-community langchain-core langchain-openai \
        langchain-text-splitters langsmith openai tiktoken \
        dataclasses-json marshmallow typing-inspect jsonpatch \
        requests-toolbelt 2>/dev/null || true
#
# 2. Inference-only cuts (GPU variant; the CPU wheels never install these):
#    - triton: JIT compiler backend for torch.compile — this app only runs
#      eager inference, torch imports it lazily and works without it.
#      (NCCL must stay: libtorch_cuda.so declares DT_NEEDED libnccl.so.2,
#      so `import torch` fails without it even for single-GPU inference.)
#    - C/C++ headers (torch/include, nvidia/**/include, triton headers are
#      gone with triton): only needed to compile extensions, never shipped.
#    - *.a static libs: link-time only.
# 3. Strip all __pycache__ dirs and *.pyc/*.pyo files. Python regenerates
#    them lazily on first import — the builder-time bytecode for torch +
#    paddle + transformers alone is several hundred MB of dead weight.
#    We intentionally keep pip and setuptools installed; paddlex
#    introspects pkg_resources at runtime.
RUN SP=/opt/venv/lib/python3.11/site-packages && \
    rm -rf "$SP/triton" "$SP"/triton-*.dist-info \
           "$SP/torch/include" "$SP/torch/test" && \
    find "$SP/nvidia" -maxdepth 2 -type d -name include -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type f \( -name '*.a' -o -name '*.pyc' -o -name '*.pyo' \) -delete && \
    find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} +


############################
# Stage 2 — runtime (CUDA for gpu, plain Ubuntu for cpu)
############################
FROM ${RUNTIME_BASE}

ARG VARIANT

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOME=/home/appuser \
    # Surfaced to the entrypoint preflight + the app so it can tailor messages.
    RENAISSANCE_VARIANT=${VARIANT} \
    # venv from builder; takes precedence over any system Python
    PATH="/opt/venv/bin:$PATH" \
    # PaddleX / PaddleOCR model cache — mapped to a named Docker volume.
    # PADDLE_PDX_CACHE_HOME is the variable paddlex actually reads
    # (paddlex/utils/cache.py); the previous PADDLE_PDX_MODEL_CACHE_HOME
    # was only read by our own code, so models silently re-downloaded to
    # the ephemeral ~/.paddlex on EVERY container recreation instead of
    # persisting in the volume.
    PADDLE_PDX_CACHE_HOME=/paddle_models \
    # HuggingFace cache for the fine-tuned corrector's base model
    # (google/gemma-3-4b-it, ~8 GB, GATED — needs HF_TOKEN at runtime). Point it
    # at the SAME persisted named volume as the paddle models so first-use
    # downloads survive container recreation instead of re-downloading — same
    # fix the paddle weights got. The LoRA adapter itself is baked into the
    # image (models/weights/postprocess), only the base weights download.
    HF_HOME=/paddle_models/hf \
    # Disable experimental Paddle PIR to avoid ConvertPirAttribute errors
    FLAGS_enable_pir_api=0 \
    FLAGS_enable_pir_in_executor=0 \
    # Plain-Ubuntu base: tell the NVIDIA Container Toolkit what to inject
    # when the container is started with --gpus / a device reservation.
    # (The nvidia/cuda images set these for you; ubuntu:24.04 does not.
    # Harmless for the cpu variant — without a device request the toolkit
    # never engages.)
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Central user-tracking endpoint, baked into the published image so every
# self-hosted instance reports signups to the one shared Supabase table.
# These are PUBLIC-by-design values: the publishable (anon) key can only add a
# signup row (INSERT-only Row Level Security) — it cannot read/edit/delete any
# data, and the Postgres password / service_role key are never included here.
# Override at runtime with -e SUPABASE_URL / -e SUPABASE_PUBLISHABLE_KEY.
ENV SUPABASE_URL=https://hljxjbcarzbsbvwffmzu.supabase.co \
    SUPABASE_PUBLISHABLE_KEY=sb_publishable_wKkx48mKpDZBk9h3gwDfOw_c6MxHIiz

# Install ONLY the Python 3.11 runtime (no -dev, no pip) plus the shared
# libraries that native extensions need at runtime.
# IMPORTANT: same deadsnakes PPA + same Python 3.11 as the builder so every
# .so in /opt/venv resolves its symbols correctly. One combined RUN keeps the
# apt cache clean-up in the same layer as the install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && apt-get purge -y --auto-remove software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully-built venv from the builder.
# Both stages are Ubuntu 24.04, so glibc / OpenSSL / libstdc++ are identical
# — no ABI mismatch possible.
COPY --from=builder /opt/venv /opt/venv

# Create non-root user and the model-cache directory up front so that the
# subsequent COPY --chown writes into correct ownership in a single layer
# (avoids a recursive chown over /app afterwards).
#
# At runtime /paddle_models is shadowed by a named Docker volume so
# downloaded models persist across container restarts (lazy-loaded on
# first use). We used to pre-download PaddleOCR weights here, but the
# named volume mounted at /paddle_models shadows this path at runtime,
# so the baked-in models were discarded anyway — bloating the image by
# ~1.5 GB for nothing. Lazy first-run download is the correct pattern.
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --home /home/appuser appuser && \
    mkdir -p /home/appuser /paddle_models /app/storage/transcripts /app/storage/datasets && \
    chown -R appuser:appgroup /home/appuser /paddle_models /app/storage

############################
# App setup
############################
WORKDIR /app
COPY --chown=appuser:appgroup . .

# Normalize the TrOCR checkpoint's processor assets. transformers 4.53 reads
# the shipped processor_config.json/tokenizer.json natively, so this is now a
# belt-and-suspenders step: it re-derives the processor from the base model
# (microsoft/trocr-base-printed, shared vocabulary) so the directory is
# guaranteed loadable by the pinned versions regardless of how the checkpoint
# was serialized. No-op if the trocr dir is absent. Ownership is reset so
# appuser can read the regenerated files at runtime.
#
# HF_HOME is overridden for this ONE step: it normally points into
# /paddle_models (the named volume), and this RUN executes as root — a root
# -owned /paddle_models/hf baked into the image seeds every fresh volume
# root-owned, so appuser could never download the corrector's base model at
# runtime (PermissionError in huggingface_hub). Keep build-time HF writes out
# of the volume path.
RUN HF_HOME=/tmp/hf-build python /app/scripts/normalize_trocr_config.py && \
    rm -rf /tmp/hf-build && \
    chown -R appuser:appgroup /app/models/weights/trocr /home/appuser /paddle_models && \
    chmod +x /app/scripts/entrypoint.sh

USER appuser

EXPOSE 8000
VOLUME ["/app/storage"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Entrypoint runs a non-blocking preflight (logs a clear message if a GPU
# image is started without GPU access, or if RAM is low) then exec's the CMD.
# It NEVER aborts boot: Gemini OCR, preprocessing and export work without a
# GPU; only PaddleOCR layout detection requires one. The hard "specs do not
# meet requirements" gate lives in the host launcher (run.sh / run.ps1),
# which checks before pulling the multi-GB image.
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# Single worker on purpose: PaddleOCR server-class models (~1.5 GB each,
# loaded fresh per detection call and freed after) would otherwise be
# duplicated per-worker — doubling RAM / VRAM footprint and OOM-killing
# the container on larger pages. Detection is already serialized at the
# app level (asyncio lock in layout_detection router), so extra workers
# add no throughput for GPU-bound work.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
