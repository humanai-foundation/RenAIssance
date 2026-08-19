# RenAIssance

RenAIssance is a full-stack web app for reading historical documents. You upload a PDF or image, clean up the scan, detect the text regions, and transcribe the text with your choice of engine. It was built for early modern Spanish books, but the pipeline works on any printed page.

The workflow is a five step wizard: Upload, Select pages, Preprocess, Text detection, and OCR & export.

## Features

- **OCR engines:** Gemini, ChatGPT, a local CRNN, and a local TrOCR (fine-tuned from `microsoft/trocr-base-printed`).
- **Optional AI cleanup:** correct raw OCR with Gemini, OpenAI, DeepSeek, Qwen, or a Spanish fine-tuned corrector (a `gemma-3-4b-it` LoRA adapter). The fine-tuned corrector needs an NVIDIA GPU.
- **Export:** download the transcription as TXT, DOCX, or PDF.

## What you need

- Docker version 24 or newer, with Compose v2.
- About 15 GB of free disk for the images and model weights.
- A Gemini or OpenAI API key if you want to use those engines. You enter the key in the app, not in a file.
- An NVIDIA GPU with driver 560 or newer and the NVIDIA Container Toolkit if you want GPU acceleration. This is optional. Without a GPU the app runs on the CPU image, which works everywhere and is only slower.

## Run it

Clone the repository and run the launcher for your system. It detects your GPU, checks your machine before downloading anything, and starts the correct image automatically.

### Linux and Windows (WSL)

```bash
git clone https://github.com/<your-org>/RenAIssance.git
cd RenAIssance
./run.sh
```

Then open http://localhost:5173 in your browser. The first run downloads the images and can take several minutes. Some models download the first time you use them, and the app shows a progress bar while that happens.

Useful flags:

```bash
./run.sh --cpu     # force the CPU image
./run.sh --build   # build from source instead of pulling published images
./run.sh --down    # stop everything (your saved data is kept)
```

### Windows (PowerShell)

```powershell
.\run.ps1          # auto-detect GPU or CPU
.\run.ps1 -Cpu     # force the CPU image
.\run.ps1 -Down    # stop
```

## Running on a Mac

Docker on macOS cannot reach the GPU, so on a Mac the app always runs on the **CPU image**. This is expected and everything works, it is just slower than on an NVIDIA machine. The local Spanish fine-tuned corrector is the only feature that requires a GPU, so on a Mac use Gemini or OpenAI for the AI cleanup step instead.

Follow these steps.

1. Install Docker Desktop for Mac from https://www.docker.com/products/docker-desktop and start it. Wait until the whale icon in the menu bar stops animating, which means Docker is ready.
2. Install Git if you do not have it. The simplest way is to run `git --version` in the Terminal, which offers to install the developer tools if Git is missing.
3. Clone the repository and enter the folder:

   ```bash
   git clone https://github.com/<your-org>/RenAIssance.git
   cd RenAIssance
   ```

4. Start the app. The launcher sees there is no NVIDIA GPU and picks the CPU image for you:

   ```bash
   ./run.sh
   ```

   If you prefer to be explicit, run `./run.sh --cpu`.

5. Wait for the download to finish. The first run pulls a few gigabytes, so give it a few minutes on a normal connection.
6. Open http://localhost:5173 in your browser. The app is ready.
7. To stop the app later, run `./run.sh --down`. Your transcripts and datasets are saved and will still be there next time.

### Optional: use the Apple GPU (Apple Silicon)

Docker cannot use the Apple GPU, but you can run the app directly on your Mac to get GPU acceleration for the local TrOCR and CRNN engines through Apple's Metal (MPS). Text detection still runs on the CPU because PaddleOCR has no Metal backend.

You need Homebrew, Python 3.11, and Node.js:

```bash
brew install python@3.11 node
./run-native.sh
```

The script creates a local Python environment, installs everything, and starts the app on http://localhost:5173. The first run is slow because it installs the dependencies.

## Using the app

1. **Upload** a PDF or an image (PNG, JPG, TIFF, or BMP).
2. **Select** the pages you want from the thumbnail grid.
3. **Preprocess** the pages. Toggle and tune the cleanup operations and watch the before and after preview. You can also load a pre-tested pipeline for a known book type.
4. **Detect** the text regions on each page.
5. **Read and export.** Pick an engine, transcribe, edit anything that needs fixing, optionally run the AI cleanup, and download the result.

The app opens on a simple sign up screen that asks for a name, an email, and an institution. It exists only for lightweight usage tracking and never asks for an API key. You enter provider keys later, in the reading step, and only if you use a cloud engine.

## Compose files

The launcher just selects the right Compose file. You can run them directly if you prefer.

| Host | File |
|------|------|
| NVIDIA GPU, published images | `docker-compose.images.yml` |
| No GPU or Mac, published images | `docker-compose.images.cpu.yml` |
| NVIDIA GPU, build from source | `docker-compose.yml` |
| No GPU or Mac, build from source | `docker-compose.cpu.yml` |

```bash
docker compose -f docker-compose.images.cpu.yml up -d    # run (Mac / no GPU)
docker compose -f docker-compose.images.cpu.yml down      # stop, data is kept
```

## Manual Docker commands

If you would rather not use the launcher or Compose, you can pull and run the published images by hand. The commands below use the GPU backend. On a Mac or any machine without an NVIDIA GPU, remove the `--gpus all` line from the backend command and everything still works on the CPU.

### Install (run once)

Pull the images and create the network and the volumes that hold your data. The volumes persist forever, so you only create them once.

```bash
docker pull saarthakg004/renaissance-backend:latest
docker pull saarthakg004/renaissance-frontend:latest

docker network create renaissance 2>/dev/null || true
docker volume create paddle_models 2>/dev/null || true
docker volume create renaissance_storage 2>/dev/null || true
```

### Run

Start the backend, then the frontend. The `--network-alias backend` on the backend is required, because the frontend proxies API calls to `http://backend:8000` and must be able to find it by that name.

```bash
docker run -d --name renaissance-backend \
  --gpus all \
  --network renaissance --network-alias backend \
  -p 8000:8000 \
  -v paddle_models:/paddle_models \
  -v renaissance_storage:/app/storage \
  --restart unless-stopped \
  saarthakg004/renaissance-backend:latest

docker run -d --name renaissance-frontend \
  --network renaissance \
  -p 5173:8080 \
  --restart unless-stopped \
  saarthakg004/renaissance-frontend:latest
```

Wait about 30 seconds for the backend to start, then open http://localhost:5173.

### Stop and delete the containers (your data is kept)

This removes only the containers. The volumes, and therefore your datasets and transcripts, stay on disk. Run the two `docker run` commands above again to start fresh containers with the same data.

```bash
docker rm -f renaissance-backend renaissance-frontend
```

### Delete everything, including your data

This is the full cleanup. It removes the containers, the network, and the volumes. Everything you saved is deleted.

```bash
docker rm -f renaissance-backend renaissance-frontend
docker network rm renaissance
docker volume rm paddle_models renaissance_storage
```

## Development and tests

```bash
# Backend tests (CPU-safe, no GPU needed). CI runs these on every push.
cd backend
pip install -r requirements-dev.txt
pytest tests -q

# Frontend dev server
cd frontend
npm install
npm run dev
```

Model weights are not stored in Git. Continuous integration downloads them from Google Drive when it builds the images, so a normal clone stays small.

## Troubleshooting

| Symptom | Fix |
|---|---|
| The launcher says your machine does not meet the GPU requirements | Install or repair the NVIDIA driver (560 or newer) and the Container Toolkit, or run `./run.sh --cpu` to use the CPU image. |
| The GPU image logs that it started without a usable CUDA device | The container was launched without GPU access. Use the launcher, or run the CPU image. The server still starts, only PaddleOCR detection is unavailable. |
| The local Spanish corrector fails on a Mac | It needs an NVIDIA GPU. Use Gemini or OpenAI for the cleanup step instead. |
| Port 8000 or 5173 is already in use | Stop the other program using it, or remap the port in the Compose file. |
| The frontend says it cannot reach the backend | The backend is still starting. The first health check can take about 30 seconds. |

View logs with `docker logs -f renaissance-backend`.
