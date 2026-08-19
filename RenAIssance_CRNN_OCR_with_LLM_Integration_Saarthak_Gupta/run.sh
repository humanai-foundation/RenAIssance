#!/usr/bin/env bash
# ============================================================================
# RenAIssance — one-command launcher (Linux / macOS / WSL)
#
#   ./run.sh              Detect GPU, pre-check specs, pull & run the right image
#   ./run.sh --cpu        Force the CPU image (no NVIDIA GPU needed)
#   ./run.sh --build      Build locally from source instead of pulling images
#   ./run.sh --native     macOS only: run natively to use the Apple GPU (MPS)
#   ./run.sh --down       Stop the stack
#   ./run.sh --help       Show this help
#
# The hard "your machine does not meet the requirements" gate runs HERE,
# before any multi-GB image is pulled. The container also runs a lighter
# preflight as a safety net.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")"

# ---- tunables -------------------------------------------------------------
MIN_DRIVER_MAJOR=560     # CUDA 12.6 needs NVIDIA driver >= 560
MIN_VRAM_GB=4            # below this the GPU image is not worth it
MIN_RAM_GB=4             # hard floor; app recommends >= 8
REC_RAM_GB=8
CUDA_BASE_IMAGE="nvidia/cuda:12.6.3-base-ubuntu24.04"
URL="http://localhost:5173"

# ---- flags ----------------------------------------------------------------
FORCE_CPU=0; BUILD=0; DOWN=0; NATIVE=0
for arg in "$@"; do
  case "$arg" in
    --cpu)    FORCE_CPU=1 ;;
    --build)  BUILD=1 ;;
    --down)   DOWN=1 ;;
    --native) NATIVE=1 ;;
    -h|--help)
      sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
die()   { red "ERROR: $*"; exit 1; }

OS="$(uname -s)"

# ---- docker availability --------------------------------------------------
command -v docker >/dev/null 2>&1 || die "Docker is not installed or not on PATH."
if docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DC=(docker-compose)
else
  die "Docker Compose v2 (\"docker compose\") is required."
fi
docker info >/dev/null 2>&1 || die "Docker daemon is not running. Start Docker Desktop / the docker service and retry."

# ---- helpers --------------------------------------------------------------
total_ram_gb() {
  if [ "$OS" = "Darwin" ]; then
    echo $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
  else
    echo $(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))
  fi
}

free_disk_gb() {
  # GB free on the filesystem holding this repo (proxy for docker storage).
  df -Pk . 2>/dev/null | awk 'NR==2{print int($4/1024/1024)}'
}

have_nvidia_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

driver_major() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null \
    | head -n1 | cut -d. -f1
}

vram_gb() {
  local mib
  mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)"
  [ -n "$mib" ] && echo $(( mib / 1024 )) || echo 0
}

gpu_works_in_docker() {
  docker run --rm --gpus all "$CUDA_BASE_IMAGE" nvidia-smi -L >/dev/null 2>&1
}

# ---- down -----------------------------------------------------------------
if [ "$DOWN" = "1" ]; then
  bold "Stopping all RenAIssance compose stacks…"
  for f in docker-compose.yml docker-compose.cpu.yml \
           docker-compose.images.yml docker-compose.images.cpu.yml; do
    [ -f "$f" ] && "${DC[@]}" -f "$f" down 2>/dev/null || true
  done
  green "Stopped."
  exit 0
fi

# ---- native (macOS MPS) ---------------------------------------------------
if [ "$NATIVE" = "1" ]; then
  [ "$OS" = "Darwin" ] || die "--native is for macOS (Apple GPU via MPS). On Linux/Windows use the GPU image."
  exec ./run-native.sh
fi

bold "RenAIssance launcher — OS: $OS"

# ---- common spec checks ---------------------------------------------------
RAM_GB="$(total_ram_gb)"
DISK_GB="$(free_disk_gb)"
echo "  RAM: ${RAM_GB} GB   Free disk: ${DISK_GB:-?} GB"
if [ "$RAM_GB" -lt "$MIN_RAM_GB" ]; then
  red "Your machine does not meet the requirements: only ${RAM_GB} GB RAM (minimum ${MIN_RAM_GB} GB)."
  exit 1
fi
[ "$RAM_GB" -lt "$REC_RAM_GB" ] && \
  red "  WARNING: ${RAM_GB} GB RAM detected; ${REC_RAM_GB} GB+ recommended. Large pages may be slow / OOM."

# ---- decide variant -------------------------------------------------------
VARIANT="gpu"
if [ "$FORCE_CPU" = "1" ]; then
  VARIANT="cpu"; echo "  --cpu given → CPU image."
elif [ "$OS" = "Darwin" ]; then
  VARIANT="cpu"
  echo "  macOS detected → CPU image (Docker on macOS has no GPU passthrough)."
  echo "  To use the Apple GPU, run natively:  ./run.sh --native"
elif ! have_nvidia_gpu; then
  VARIANT="cpu"; echo "  No NVIDIA GPU detected → CPU image."
fi

# ---- GPU gate (only when we intend to use the GPU image) ------------------
if [ "$VARIANT" = "gpu" ]; then
  DMAJ="$(driver_major)"; VRAM="$(vram_gb)"
  echo "  NVIDIA driver: ${DMAJ:-?}   VRAM: ${VRAM} GB"

  fail=""
  [ -z "$DMAJ" ] && fail="could not read NVIDIA driver version"
  [ -n "$DMAJ" ] && [ "$DMAJ" -lt "$MIN_DRIVER_MAJOR" ] && \
    fail="NVIDIA driver $DMAJ is too old (need >= $MIN_DRIVER_MAJOR for CUDA 12.6)"
  [ "$VRAM" -lt "$MIN_VRAM_GB" ] && \
    fail="only ${VRAM} GB VRAM (minimum ${MIN_VRAM_GB} GB)"

  if [ -z "$fail" ]; then
    echo "  Verifying NVIDIA Container Toolkit (GPU access inside Docker)…"
    gpu_works_in_docker || fail="Docker cannot access the GPU — install the NVIDIA Container Toolkit"
  fi

  if [ -n "$fail" ]; then
    red "Your machine does not meet the GPU requirements: $fail."
    printf "Fall back to the CPU image instead? [y/N] "
    read -r ans || ans=""
    case "$ans" in
      y|Y) VARIANT="cpu" ;;
      *) echo "Aborting. Fix the above or run ./run.sh --cpu."; exit 1 ;;
    esac
  fi
fi

green "Selected variant: ${VARIANT}  (mode: $([ "$BUILD" = 1 ] && echo build || echo published-images))"

# ---- pick compose file ----------------------------------------------------
if [ "$BUILD" = "1" ]; then
  FILE="$([ "$VARIANT" = cpu ] && echo docker-compose.cpu.yml || echo docker-compose.yml)"
else
  FILE="$([ "$VARIANT" = cpu ] && echo docker-compose.images.cpu.yml || echo docker-compose.images.yml)"
fi
echo "  Compose file: $FILE"

# ---- launch ---------------------------------------------------------------
if [ "$BUILD" = "1" ]; then
  "${DC[@]}" -f "$FILE" up -d --build
else
  bold "Pulling images (this can take a while the first time)…"
  "${DC[@]}" -f "$FILE" pull
  "${DC[@]}" -f "$FILE" up -d
fi

green "RenAIssance is starting. Open: $URL"
echo "  Logs:  ${DC[*]} -f $FILE logs -f"
echo "  Stop:  ./run.sh --down"
