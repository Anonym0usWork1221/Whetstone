<#
.SYNOPSIS
    Builds a release package for Windows.

.DESCRIPTION
    Produces dist\whetstone-<version>-windows-x86_64-sm<arch>.zip and a SHA256
    checksum.

    The CUDA architecture is part of the artifact name on purpose. Whetstone
    compiles for exactly one GPU family -- the kernels use capabilities that
    differ by architecture -- so a package built for sm_75 will not run on an
    older card, and shipping it under a generic name would be misleading.

    NOTE: this script has not been executed on Windows by its author, who
    develops on Linux. The build itself is exercised on Windows by the release
    workflow in .github/workflows/release.yml. If something here is wrong,
    please open an issue rather than working around it silently.

.PARAMETER Arch
    CUDA compute capability without the dot. Default 75 (Turing).

.PARAMETER Out
    Output directory. Default .\dist

.PARAMETER SkipTests
    Skip the test run (CI usually runs them separately).

.PARAMETER KeepDir
    Keep the staging directory after zipping.

.EXAMPLE
    .\scripts\deploy.ps1
    .\scripts\deploy.ps1 -Arch 86 -Out C:\tmp\dist
#>

[CmdletBinding()]
param(
    [string]$Arch = $(if ($env:WHETSTONE_CUDA_ARCH) { $env:WHETSTONE_CUDA_ARCH } else { "75" }),
    [string]$Out,
    [switch]$SkipTests,
    [switch]$KeepDir
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
if (-not $Out) { $Out = Join-Path $Root "dist" }

function Info($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  ok $m"  -ForegroundColor Green }
function Warn($m) { Write-Host "warn: $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "error: $m" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- preflight

Info "preflight"

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Die "cargo not found. Install Rust: https://rustup.rs"
}
Ok "cargo $((cargo --version).Split(' ')[1])"

if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Die @"
nvcc not found. Whetstone needs the CUDA Toolkit.
  1. Install from https://developer.nvidia.com/cuda-downloads
  2. Build from a 'x64 Native Tools Command Prompt for VS' so cl.exe is on PATH
     (nvcc requires the MSVC host compiler)
  3. Or set CUDA_PATH to your toolkit root
"@
}
$nvccVersion = (nvcc --version | Select-String -Pattern "release ([0-9.]+)").Matches.Groups[1].Value
Ok "nvcc $nvccVersion"

# nvcc needs the MSVC host compiler. Failing here with a clear message beats
# failing later inside a compile with a confusing one.
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Die @"
cl.exe not found. nvcc uses MSVC as its host compiler on Windows.
Open 'x64 Native Tools Command Prompt for VS 2022' and run this script there,
or run vcvars64.bat first.
"@
}
Ok "cl.exe on PATH"

$archList = (nvcc --list-gpu-arch) -join " "
if ($archList -notmatch "compute_$Arch") {
    Die "this nvcc cannot target sm_$Arch. Supported: $archList"
}
Ok "nvcc supports sm_$Arch"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | Select-Object -First 1)
    if ($gpu) { Ok "gpu $gpu" }
} else {
    Warn "no nvidia-smi; building without running GPU tests"
    $SkipTests = $true
}

$Version = (Select-String -Path (Join-Path $Root "Cargo.toml") -Pattern '^version\s*=\s*"(.*)"' |
            Select-Object -First 1).Matches.Groups[1].Value
if (-not $Version) { Die "could not read version from Cargo.toml" }

# Provenance. Absent in a source zip, which is fine -- the binary reports
# "unknown" rather than claiming a commit it was not built from.
$GitSha = "unknown"
if (Get-Command git -ErrorAction SilentlyContinue) {
    try {
        $null = git rev-parse --git-dir 2>$null
        if ($LASTEXITCODE -eq 0) {
            $GitSha = (git rev-parse --short=12 HEAD).Trim()
            $dirty = git status --porcelain --untracked-files=no
            if ($dirty) {
                $GitSha = "$GitSha-dirty"
                Warn "working tree has uncommitted changes; marking build as dirty"
            }
        }
    } catch { $GitSha = "unknown" }
}

$Name  = "whetstone-$Version-windows-x86_64-sm$Arch"
$Stage = Join-Path $Out $Name

Info "building whetstone $Version ($GitSha) for sm_$Arch"

# ------------------------------------------------------------------- build

$env:WHETSTONE_CUDA_ARCH = $Arch
$env:WHETSTONE_GIT_SHA   = $GitSha
if (-not $env:SOURCE_DATE_EPOCH) {
    $env:SOURCE_DATE_EPOCH = [string][int][double]::Parse(
        (Get-Date -UFormat %s))
}

cargo build --release --locked
if ($LASTEXITCODE -ne 0) { Die "build failed" }

$Exe = Join-Path $Root "target\release\whetstone.exe"
if (-not (Test-Path $Exe)) { Die "build produced no binary at $Exe" }
Ok "built target\release\whetstone.exe"

if (-not $SkipTests) {
    Info "running correctness tests"
    cargo test --release --locked
    if ($LASTEXITCODE -ne 0) { Die "tests failed" }
    Ok "correctness tests passed"

    # Performance checks are reported, not enforced: they are timing-based, so
    # running them right after a build measures machine contention as much as
    # the kernels. A flaky release gate is worse than no gate.
    Info "performance checks (informational)"
    cargo test --release --locked -- --ignored --nocapture
    if ($LASTEXITCODE -ne 0) { Warn "performance checks did not pass; not blocking the release" }
} else {
    Warn "tests skipped"
}

# ----------------------------------------------------------------- package

Info "packaging"

if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force -Path $Stage, "$Stage\bin", "$Stage\bench", "$Stage\docs" | Out-Null

Copy-Item $Exe "$Stage\bin\whetstone.exe"
Copy-Item "$Root\bench\chat.py","$Root\bench\baseline_hf.py",
          "$Root\bench\reference_numpy.py","$Root\bench\tokenizer.py" "$Stage\bench\"
if (Test-Path "$Root\scripts\download_model.py") {
    Copy-Item "$Root\scripts\download_model.py" "$Stage\bench\"
}
Copy-Item "$Root\README.md","$Root\LICENSE" "$Stage\"
Copy-Item "$Root\docs\FORMAT.md","$Root\docs\ROADMAP.md" "$Stage\docs\"
if (Test-Path "$Root\CHANGELOG.md") { Copy-Item "$Root\CHANGELOG.md" "$Stage\" }
Copy-Item "$Root\scripts\run.bat" "$Stage\run.bat"

$builtAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$rustcVersion = (rustc --version).Split(' ')[1]

@"
version:    $Version
commit:     $GitSha
built:      $builtAt
target:     x86_64-pc-windows-msvc
cuda arch:  sm_$Arch
nvcc:       $nvccVersion
rustc:      $rustcVersion
"@ | Set-Content -Encoding UTF8 "$Stage\VERSION"

$ccMajor = $Arch.Substring(0,1)
$ccMinor = $Arch.Substring(1,1)

@"
Whetstone $Version - Windows x86_64, CUDA sm_$Arch

Required:
  * NVIDIA GPU with compute capability $ccMajor.$ccMinor
    (sm_75 = Turing: RTX 2060/2070/2080, GTX 1650 Super/1660, T4, Quadro RTX)
  * NVIDIA driver new enough for the CUDA runtime below
  * cudart64_*.dll - from the CUDA Toolkit or the runtime redistributable.
    If whetstone.exe fails to start with a missing-DLL error, that is why.

Optional, for the Python benchmark and chat harness:
  * Python 3.10+
  * pip install torch transformers safetensors regex

Check your card:
    nvidia-smi --query-gpu=name,compute_cap --format=csv
    bin\whetstone.exe probe
"@ | Set-Content -Encoding UTF8 "$Stage\REQUIREMENTS.txt"

$Zip = Join-Path $Out "$Name.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
Compress-Archive -Path $Stage -DestinationPath $Zip -CompressionLevel Optimal

$hash = (Get-FileHash -Algorithm SHA256 $Zip).Hash.ToLower()
"$hash  $Name.zip" | Set-Content -Encoding ASCII "$Zip.sha256"

if (-not $KeepDir) { Remove-Item -Recurse -Force $Stage }

$sizeMb = [math]::Round((Get-Item $Zip).Length / 1MB, 2)

Write-Host ""
Info "done"
Write-Host "  $Zip ($sizeMb MB)"
Write-Host "  $Zip.sha256"
Write-Host ""
Write-Host "  verify:  (Get-FileHash -Algorithm SHA256 '$Name.zip').Hash"
Write-Host "  unpack:  Expand-Archive '$Name.zip' -DestinationPath . ; cd '$Name' ; .\run.bat probe"
Write-Host ""
