# ============================================================================
# RenAIssance — one-command launcher (Windows / PowerShell)
#
#   .\run.ps1            Detect GPU, pre-check specs, pull & run the right image
#   .\run.ps1 -Cpu       Force the CPU image (no NVIDIA GPU needed)
#   .\run.ps1 -Build     Build locally from source instead of pulling images
#   .\run.ps1 -Down      Stop the stack
#
# The hard "your machine does not meet the requirements" gate runs HERE,
# before any multi-GB image is pulled. GPU on Windows requires WSL2 + an
# NVIDIA driver with WSL support + the NVIDIA Container Toolkit in the WSL
# Docker engine (Docker Desktop with the WSL2 backend handles this).
# ============================================================================
[CmdletBinding()]
param(
  [switch]$Cpu,
  [switch]$Build,
  [switch]$Down
)
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$MinDriverMajor = 560
$MinVramGb      = 4
$MinRamGb       = 4
$RecRamGb       = 8
$CudaBaseImage  = 'nvidia/cuda:12.6.3-base-ubuntu24.04'
$Url            = 'http://localhost:5173'

function Die($m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
function Info($m) { Write-Host $m }
function Ok($m)  { Write-Host $m -ForegroundColor Green }

# ---- docker availability --------------------------------------------------
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Die 'Docker is not installed or not on PATH.' }
try { docker compose version *> $null } catch { Die 'Docker Compose v2 ("docker compose") is required.' }
try { docker info *> $null } catch { Die 'Docker daemon is not running. Start Docker Desktop and retry.' }

function Get-RamGb {
  [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
}
function Test-NvidiaGpu {
  if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $false }
  try { nvidia-smi -L *> $null; return $true } catch { return $false }
}
function Get-DriverMajor {
  try { (nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | Select-Object -First 1).Split('.')[0] }
  catch { '' }
}
function Get-VramGb {
  try { [int]((nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | Select-Object -First 1)) / 1024 -as [int] }
  catch { 0 }
}
function Test-GpuInDocker {
  try { docker run --rm --gpus all $CudaBaseImage nvidia-smi -L *> $null; return $true } catch { return $false }
}

# ---- down -----------------------------------------------------------------
if ($Down) {
  Write-Host 'Stopping all RenAIssance compose stacks…'
  foreach ($f in 'docker-compose.yml','docker-compose.cpu.yml','docker-compose.images.yml','docker-compose.images.cpu.yml') {
    if (Test-Path $f) { try { docker compose -f $f down *> $null } catch {} }
  }
  Ok 'Stopped.'; exit 0
}

Write-Host 'RenAIssance launcher — OS: Windows' -ForegroundColor White

# ---- common spec checks ---------------------------------------------------
$RamGb = Get-RamGb
Info "  RAM: $RamGb GB"
if ($RamGb -lt $MinRamGb) {
  Write-Host "Your machine does not meet the requirements: only $RamGb GB RAM (minimum $MinRamGb GB)." -ForegroundColor Red
  exit 1
}
if ($RamGb -lt $RecRamGb) { Write-Host "  WARNING: $RamGb GB RAM; ${RecRamGb}GB+ recommended." -ForegroundColor Red }

# ---- decide variant -------------------------------------------------------
$Variant = 'gpu'
if ($Cpu) { $Variant = 'cpu'; Info '  -Cpu given → CPU image.' }
elseif (-not (Test-NvidiaGpu)) { $Variant = 'cpu'; Info '  No NVIDIA GPU detected → CPU image.' }

# ---- GPU gate -------------------------------------------------------------
if ($Variant -eq 'gpu') {
  $dmaj = Get-DriverMajor; $vram = Get-VramGb
  Info "  NVIDIA driver: $dmaj   VRAM: $vram GB"
  $fail = ''
  if (-not $dmaj) { $fail = 'could not read NVIDIA driver version' }
  elseif ([int]$dmaj -lt $MinDriverMajor) { $fail = "NVIDIA driver $dmaj too old (need >= $MinDriverMajor for CUDA 12.6)" }
  elseif ($vram -lt $MinVramGb) { $fail = "only $vram GB VRAM (minimum $MinVramGb GB)" }
  if (-not $fail) {
    Info '  Verifying NVIDIA Container Toolkit (GPU access inside Docker)…'
    if (-not (Test-GpuInDocker)) { $fail = 'Docker cannot access the GPU — ensure WSL2 + NVIDIA Container Toolkit' }
  }
  if ($fail) {
    Write-Host "Your machine does not meet the GPU requirements: $fail." -ForegroundColor Red
    $ans = Read-Host 'Fall back to the CPU image instead? [y/N]'
    if ($ans -match '^[Yy]') { $Variant = 'cpu' } else { Info 'Aborting. Fix the above or run .\run.ps1 -Cpu.'; exit 1 }
  }
}

$mode = if ($Build) { 'build' } else { 'published-images' }
Ok "Selected variant: $Variant  (mode: $mode)"

# ---- pick compose file ----------------------------------------------------
if ($Build) { $File = if ($Variant -eq 'cpu') { 'docker-compose.cpu.yml' } else { 'docker-compose.yml' } }
else        { $File = if ($Variant -eq 'cpu') { 'docker-compose.images.cpu.yml' } else { 'docker-compose.images.yml' } }
Info "  Compose file: $File"

# ---- launch ---------------------------------------------------------------
if ($Build) {
  docker compose -f $File up -d --build
} else {
  Write-Host 'Pulling images (this can take a while the first time)…'
  docker compose -f $File pull
  docker compose -f $File up -d
}

Ok "RenAIssance is starting. Open: $Url"
Info "  Logs:  docker compose -f $File logs -f"
Info "  Stop:  .\run.ps1 -Down"
