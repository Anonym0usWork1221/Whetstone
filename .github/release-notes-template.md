---

## Which file do I want?

| file | for |
|---|---|
| `whetstone-*-linux-x86_64-sm75.tar.gz` | Linux, NVIDIA Turing (RTX 20-series, GTX 16-series, T4) |
| `whetstone-*-windows-x86_64-sm75.zip` | Windows, same GPUs |

Whetstone compiles for **one** GPU architecture. An `sm75` build will not run on
an older card. Check yours:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

`compute_cap 7.5` → take the `sm75` artifact. For anything else, build from
source with the matching architecture:

```bash
WHETSTONE_CUDA_ARCH=86 ./scripts/deploy.sh --arch 86     # Ampere
```

## Install

**Linux**

```bash
tar xzf whetstone-*-linux-x86_64-sm75.tar.gz
cd whetstone-*-linux-x86_64-sm75
./run.sh doctor      # GPU, driver, binary, Python, model — all at once
./run.sh probe       # what your GPU can actually do
```

**Windows**

```powershell
Expand-Archive whetstone-*-windows-x86_64-sm75.zip -DestinationPath .
cd whetstone-*-windows-x86_64-sm75
.\run.bat doctor
.\run.bat probe
```

## Using it

```bash
./run.sh setup                                   # one-time: Python env for chat/bench
./run.sh download                                # fetch Qwen2.5-0.5B-Instruct
./run.sh inspect  models/Qwen2.5-0.5B-Instruct   # architecture + roofline
./run.sh convert  models/Qwen2.5-0.5B-Instruct model.wstone
./run.sh verify   model.wstone models/Qwen2.5-0.5B-Instruct
./run.sh chat                                    # live chat with a tok/s readout
./run.sh bench                                   # throughput run
```

## Requirements

- NVIDIA GPU matching the artifact's architecture
- NVIDIA driver new enough for the bundled CUDA runtime version
- `libcudart` — from the CUDA toolkit or the runtime redistributable
- Python 3.10+ **only** for the chat and benchmark harness; the CLI itself needs
  nothing but the binary

## Verify your download

```bash
sha256sum -c SHA256SUMS
```

```powershell
(Get-FileHash -Algorithm SHA256 whetstone-*-windows-x86_64-sm75.zip).Hash
```

## Documentation

- [`docs/FORMAT.md`](https://github.com/Anonym0usWork1221/Whetstone/blob/main/docs/FORMAT.md) — the `.wstone` container specification
- [`docs/ROADMAP.md`](https://github.com/Anonym0usWork1221/Whetstone/blob/main/docs/ROADMAP.md) — what is built, what is next, and what this project will not do
- [`docs/RELEASES.md`](https://github.com/Anonym0usWork1221/Whetstone/blob/main/docs/RELEASES.md) — release contents and versioning policy
