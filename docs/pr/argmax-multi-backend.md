### PR Category

Operator

### Type of Change

Performance Optimization

### Description

Adds backend-specific `argmax` implementations for MetaX, Hygon, MThreads, and PPU. The implementations use backend-tuned Triton kernels for common reduction layouts and retain the generic implementation as a fallback for unsupported layouts, dtypes, or shapes.

- MetaX uses stride-aware row, column, and global reductions with SM-aware splitting, partial/final reduction, stable NaN/tie handling, and hardware-tuned thresholds for the C550. Size-1 reductions return an all-zero index output through a dedicated fast path.
- Hygon streams contiguous long last-dimension reductions in bounded blocks with one CTA per output row. FP16/BF16 comparisons use FP32, and the implementation preserves first-NaN and lowest-index tie semantics.
- MThreads uses an output-block kernel for non-inner reductions, processing multiple output rows per CTA and looping over the reduction axis to coalesce contiguous inner elements. Fixed heuristics replace runtime autotune.
- PPU uses row, output-block, and strided-global kernels with explicit first-NaN and lowest-index tie handling, while falling back to the generic implementation for unsupported layouts.
- All backends keep the existing public API and fall back to the generic implementation when a specialized path is not applicable.

### Issue

N/A

### Progress

- [ ] Change is properly reviewed (1 reviewer required, 2 recommended).
- [x] Correctness checks pass on MetaX C550, Hygon BW, MTT S5000, and PPU-ZW810E.
- [x] Performance is measured against PyTorch on all four backends; detailed metrics are included where the source validation recorded them.

### Performance

Devices and environments:

- MetaX C550 (104 SM), PyTorch 2.10.0+metax3.8.1.0, Triton 3.6.0
- Hygon BW (80 SM), PyTorch 2.4.1+das.opt1.dtk25041, Triton 3.6.0+dtk2604
- MThreads MTT S5000 (80 GiB), MUSA driver 3.3.5-server, PyTorch 2.9.1, TorchMUSA 2.9.1+a18d871, FlagTree/Triton 3.6.0
- PPU-ZW810E (64 SM), PyTorch 2.10.0, Triton 3.6.0

```bash
python -m pytest tests/test_argmax.py -q
python -m pytest -s -q benchmark/test_argmax.py
python -m pytest -s -q benchmark/test_argmax.py --mode cudagraph
```

Speedup is **PyTorch latency / FlagGems latency**.

#### Cross-platform validation summary

| Platform | Device | Correctness validation | Reported benchmark result |
| --- | --- | --- | --- |
| MetaX | C550 (104 SM) | 126/126 argmax tests | CUDA Graph 51/51 and Kernel 51/51 at or above 0.95x PyTorch; details below |
| Hygon | BW (80 SM) | 126/126 argmax tests | Kernel 42/51 and CUDA Graph 47/51 at or above 0.95x PyTorch; details below |
| MThreads | MTT S5000 | 126 argmax tests; FP16/FP32/BF16 and ATen registration | MUSA Graph 42/51 and Kernel 51/51 at or above 0.95x PyTorch; details below |
| PPU | ZW810E (64 SM) | 135 existing argmax tests | 51/51 SUCCESS in both Kernel and CUDA Graph modes; details below |

The MThreads metrics below are from the validation run on commit `de033a4b4`; missing metrics are not inferred.

#### MetaX performance

| Mode | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA Graph | 51 | 1.008x | 4.792x | 1.738x | 51 / 51 |
| Kernel (non-Graph) | 51 | 1.065x | 2.595x | 1.557x | 51 / 51 |

The size-1 reduction fast path raises FP32 `(64, 1, 64)` CUDA Graph speedup from 0.942x to 1.031x.

<details>
<summary>MetaX CUDA Graph - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 1.852x | 1.876x | 1.887x |
| (64, 64) | 1 | 2.177x | 2.176x | 2.110x |
| (4096, 4096) | 1 | 1.307x | 1.242x | 1.259x |
| (64, 512, 512) | 1 | 1.456x | 1.548x | 1.455x |
| (1024, 1024, 1024) | 1 | 1.653x | 1.059x | 1.664x |
| (1049600,) | None (global) | 1.845x | 1.774x | 1.838x |
| (1073741824,) | None (global) | 1.498x | 1.059x | 1.500x |
| (1024, 1) | 1 | 1.042x | 1.008x | 1.045x |
| (1024, 16) | 1 | 1.945x | 1.950x | 1.748x |
| (1024, 256) | 1 | 2.103x | 2.168x | 2.056x |
| (1024, 4096) | 1 | 2.073x | 1.508x | 2.012x |
| (1024, 65536) | 1 | 1.503x | 1.068x | 1.505x |
| (1024, 1048576) | 1 | 1.515x | 1.157x | 1.515x |
| (64, 1, 64) | 1 | 1.070x | 1.031x | 1.071x |
| (64, 16, 64) | 1 | 4.758x | 4.659x | 4.792x |
| (64, 256, 64) | 1 | 4.192x | 4.693x | 4.397x |
| (64, 4096, 64) | 1 | 1.783x | 1.226x | 1.459x |

</details>

<details>
<summary>MetaX Kernel (non-Graph) - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 1.767x | 1.773x | 1.831x |
| (64, 64) | 1 | 1.368x | 1.368x | 1.351x |
| (4096, 4096) | 1 | 1.539x | 1.400x | 1.503x |
| (64, 512, 512) | 1 | 1.506x | 1.589x | 1.502x |
| (1024, 1024, 1024) | 1 | 1.668x | 1.065x | 1.677x |
| (1049600,) | None (global) | 1.770x | 1.758x | 1.791x |
| (1073741824,) | None (global) | 1.504x | 1.067x | 1.504x |
| (1024, 1) | 1 | 1.173x | 1.173x | 1.151x |
| (1024, 16) | 1 | 1.535x | 1.571x | 1.571x |
| (1024, 256) | 1 | 1.617x | 1.639x | 1.590x |
| (1024, 4096) | 1 | 1.882x | 1.658x | 1.842x |
| (1024, 65536) | 1 | 1.545x | 1.124x | 1.553x |
| (1024, 1048576) | 1 | 1.508x | 1.166x | 1.511x |
| (64, 1, 64) | 1 | 1.154x | 1.154x | 1.154x |
| (64, 16, 64) | 1 | 2.065x | 2.065x | 2.115x |
| (64, 256, 64) | 1 | 2.531x | 2.595x | 2.562x |
| (64, 4096, 64) | 1 | 1.847x | 1.353x | 1.582x |

</details>

#### Hygon performance

Environment: Hygon BW (80 SM), PyTorch 2.4.1+das.opt1.dtk25041, Triton 3.6.0+dtk2604. The available container did not provide the DTK 26.04/PyTorch 2.9 combination pinned by `pyproject.toml`. Because the current tree imports `torch.float8_e8m0fnu` from unrelated `dstack` code, validation used a process-local alias to an existing FP8 dtype; the argmax implementation and tests were unchanged.

| Mode | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA Graph | 51 | 0.870x | 52.672x | 2.726x | 47 / 51 |
| Kernel (non-Graph) | 51 | 0.525x | 12.403x | 1.661x | 42 / 51 |

All 51 benchmark cases executed successfully in both modes. CUDA Graph cases below 0.95x are `(1024, 1)` and `(64, 1, 64)` for FP16/BF16. Kernel cases below 0.95x are concentrated in global reductions and small or narrow rows; the individual values are retained below.

<details>
<summary>Hygon CUDA Graph - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 4.531x | 3.202x | 4.467x |
| (64, 64) | 1 | 4.396x | 3.776x | 4.417x |
| (4096, 4096) | 1 | 1.684x | 1.353x | 1.597x |
| (64, 512, 512) | 1 | 10.796x | 3.083x | 9.707x |
| (1024, 1024, 1024) | 1 | 3.698x | 1.926x | 2.424x |
| (1049600,) | None (global) | 3.223x | 2.777x | 3.106x |
| (1073741824,) | None (global) | 1.983x | 1.123x | 1.775x |
| (1024, 1) | 1 | 0.870x | 1.049x | 0.877x |
| (1024, 16) | 1 | 2.195x | 2.098x | 2.140x |
| (1024, 256) | 1 | 2.457x | 2.398x | 2.204x |
| (1024, 4096) | 1 | 1.287x | 1.350x | 1.559x |
| (1024, 65536) | 1 | 1.164x | 1.562x | 1.124x |
| (1024, 1048576) | 1 | 0.984x | 1.311x | 1.003x |
| (64, 1, 64) | 1 | 0.901x | 1.057x | 0.898x |
| (64, 16, 64) | 1 | 7.528x | 9.503x | 5.975x |
| (64, 256, 64) | 1 | 26.148x | 29.151x | 52.672x |
| (64, 4096, 64) | 1 | 5.489x | 5.602x | 4.658x |

</details>

<details>
<summary>Hygon Kernel (non-Graph) - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 0.552x | 1.547x | 2.201x |
| (64, 64) | 1 | 1.239x | 2.525x | 1.331x |
| (4096, 4096) | 1 | 1.587x | 1.537x | 1.578x |
| (64, 512, 512) | 1 | 12.403x | 2.693x | 5.101x |
| (1024, 1024, 1024) | 1 | 5.750x | 1.911x | 2.292x |
| (1049600,) | None (global) | 0.547x | 0.525x | 3.209x |
| (1073741824,) | None (global) | 1.621x | 0.914x | 2.195x |
| (1024, 1) | 1 | 1.310x | 1.000x | 1.195x |
| (1024, 16) | 1 | 2.013x | 0.802x | 0.815x |
| (1024, 256) | 1 | 0.869x | 1.241x | 1.199x |
| (1024, 4096) | 1 | 1.388x | 1.850x | 1.959x |
| (1024, 65536) | 1 | 1.102x | 1.467x | 0.918x |
| (1024, 1048576) | 1 | 0.985x | 1.371x | 0.934x |
| (64, 1, 64) | 1 | 1.143x | 1.167x | 1.167x |
| (64, 16, 64) | 1 | 1.121x | 1.214x | 0.984x |
| (64, 256, 64) | 1 | 5.903x | 4.202x | 3.927x |
| (64, 4096, 64) | 1 | 5.397x | 3.619x | 4.554x |

</details>

#### MThreads performance

Validated on MTT S5000 (80 GiB) with MUSA driver 3.3.5-server, PyTorch 2.9.1, TorchMUSA 2.9.1+a18d871 and FlagTree/Triton 3.6.0. Commands:

```bash
MTHREADS_VISIBLE_DEVICES=0 python -m pytest tests/test_argmax.py -q
MTHREADS_VISIBLE_DEVICES=0 python -m pytest -s -q benchmark/test_argmax.py
MTHREADS_VISIBLE_DEVICES=0 python -m pytest -s -q benchmark/test_argmax.py --mode cudagraph
```

- Correctness: 126 passed in 4.57 s.
- ATen dispatch: 9/9 checks passed across FP16/FP32/BF16 and row/inner-reduction shapes.
- Kernel benchmark: 51 cases passed in 61.73 s.
- Graph benchmark: 51 cases passed in 159.90 s.

FlagTree/Triton 3.6.0's `do_bench_cudagraph` hardcodes `torch.cuda` stream, event and graph APIs, which this TorchMUSA build does not expose. The Graph run used a temporary, non-committed timing-layer shim that maps those APIs to the native `torch.musa` `Stream`, `Event`, `stream`, `graph` and `MUSAGraph` implementations. The argmax implementation, test inputs and measurements were unchanged.

| Mode | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| MUSA Graph | 51 | 0.819x | 4.807x | 1.398x | 42 / 51 |
| Kernel (non-Graph) | 51 | 0.967x | 5.284x | 1.416x | 51 / 51 |

The Graph measurements below are from one stable benchmark run. Nine cases fall below 0.95x, concentrated in memory-bound long-reduction shapes plus one small 3-D case; they are retained unchanged below.

| Shape | Dim | Dtype | Torch latency (ms) | FlagGems latency (ms) | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| (4096, 4096) | 1 | FP16 | 0.045190 | 0.051383 | 0.879x |
| (1024, 65536) | 1 | FP16 | 0.177078 | 0.214061 | 0.827x |
| (1024, 1048576) | 1 | FP16 | 2.716997 | 3.179672 | 0.854x |
| (4096, 4096) | 1 | FP32 | 0.058263 | 0.070080 | 0.831x |
| (1049600,) | None (global) | FP32 | 0.016711 | 0.018447 | 0.906x |
| (4096, 4096) | 1 | BF16 | 0.044183 | 0.050849 | 0.869x |
| (1024, 65536) | 1 | BF16 | 0.177464 | 0.216785 | 0.819x |
| (1024, 1048576) | 1 | BF16 | 2.748009 | 3.171237 | 0.867x |
| (64, 1, 64) | 1 | BF16 | 0.002745 | 0.003082 | 0.891x |

<details>
<summary>MThreads MUSA Graph - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 1.160x | 1.050x | 1.324x |
| (64, 64) | 1 | 1.573x | 1.457x | 1.721x |
| (4096, 4096) | 1 | 0.879x | 0.831x | 0.869x |
| (64, 512, 512) | 1 | 1.351x | 1.320x | 1.377x |
| (1024, 1024, 1024) | 1 | 1.490x | 1.196x | 0.973x |
| (1049600,) | None (global) | 1.375x | 0.906x | 1.425x |
| (1073741824,) | None (global) | 1.142x | 1.125x | 1.004x |
| (1024, 1) | 1 | 1.183x | 1.155x | 1.267x |
| (1024, 16) | 1 | 2.322x | 2.113x | 2.649x |
| (1024, 256) | 1 | 1.503x | 1.286x | 1.468x |
| (1024, 4096) | 1 | 1.064x | 1.002x | 1.123x |
| (1024, 65536) | 1 | 0.827x | 1.290x | 0.819x |
| (1024, 1048576) | 1 | 0.854x | 1.158x | 0.867x |
| (64, 1, 64) | 1 | 1.093x | 1.183x | 0.891x |
| (64, 16, 64) | 1 | 4.807x | 2.694x | 4.368x |
| (64, 256, 64) | 1 | 2.229x | 1.660x | 2.807x |
| (64, 4096, 64) | 1 | 2.995x | 1.920x | 3.026x |

</details>

<details>
<summary>MThreads Kernel (non-Graph) - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 1.369x | 1.099x | 1.361x |
| (64, 64) | 1 | 1.527x | 1.319x | 1.521x |
| (4096, 4096) | 1 | 0.967x | 1.074x | 0.968x |
| (64, 512, 512) | 1 | 1.233x | 1.254x | 1.238x |
| (1024, 1024, 1024) | 1 | 1.509x | 1.124x | 1.005x |
| (1049600,) | None (global) | 1.330x | 1.128x | 1.329x |
| (1073741824,) | None (global) | 1.368x | 1.034x | 1.198x |
| (1024, 1) | 1 | 1.160x | 1.191x | 1.160x |
| (1024, 16) | 1 | 2.356x | 1.885x | 2.340x |
| (1024, 256) | 1 | 1.356x | 1.207x | 1.366x |
| (1024, 4096) | 1 | 1.127x | 1.163x | 1.134x |
| (1024, 65536) | 1 | 1.018x | 1.257x | 1.025x |
| (1024, 1048576) | 1 | 0.977x | 1.002x | 0.994x |
| (64, 1, 64) | 1 | 1.082x | 1.041x | 1.060x |
| (64, 16, 64) | 1 | 5.284x | 4.256x | 5.236x |
| (64, 256, 64) | 1 | 2.336x | 1.299x | 2.338x |
| (64, 4096, 64) | 1 | 2.737x | 1.383x | 2.741x |

</details>

#### PPU performance

| Mode | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA Graph | 51 | 1.003x | 5.854x | 1.972x | 51 / 51 |
| Kernel (non-Graph) | 51 | 0.978x | 5.693x | 1.737x | 51 / 51 |

The PPU measurements below are from one stable standard benchmark run. Repeated runs showed transient variance on a few global and small-shape cases.

<details>
<summary>PPU CUDA Graph - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 2.437x | 2.245x | 2.633x |
| (64, 64) | 1 | 3.044x | 2.905x | 3.039x |
| (4096, 4096) | 1 | 1.342x | 1.236x | 1.362x |
| (64, 512, 512) | 1 | 1.535x | 1.567x | 1.556x |
| (1024, 1024, 1024) | 1 | 1.278x | 1.011x | 1.245x |
| (1049600,) | None (global) | 2.656x | 2.374x | 2.798x |
| (1073741824,) | None (global) | 2.227x | 1.242x | 2.243x |
| (1024, 1) | 1 | 1.033x | 1.035x | 1.026x |
| (1024, 16) | 1 | 3.468x | 3.209x | 3.281x |
| (1024, 256) | 1 | 2.233x | 2.147x | 2.230x |
| (1024, 4096) | 1 | 2.561x | 2.216x | 2.566x |
| (1024, 65536) | 1 | 1.807x | 1.114x | 1.814x |
| (1024, 1048576) | 1 | 1.517x | 1.397x | 1.526x |
| (64, 1, 64) | 1 | 1.003x | 1.055x | 1.020x |
| (64, 16, 64) | 1 | 4.206x | 4.769x | 4.221x |
| (64, 256, 64) | 1 | 4.510x | 5.854x | 4.481x |
| (64, 4096, 64) | 1 | 1.204x | 1.676x | 1.195x |

</details>

<details>
<summary>PPU Kernel (non-Graph) - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 1.845x | 1.586x | 1.824x |
| (64, 64) | 1 | 3.217x | 2.884x | 3.217x |
| (4096, 4096) | 1 | 1.077x | 1.010x | 1.084x |
| (64, 512, 512) | 1 | 1.218x | 0.978x | 1.139x |
| (1024, 1024, 1024) | 1 | 1.269x | 1.007x | 1.246x |
| (1049600,) | None (global) | 1.742x | 1.655x | 1.788x |
| (1073741824,) | None (global) | 1.919x | 1.183x | 2.010x |
| (1024, 1) | 1 | 1.100x | 1.100x | 1.100x |
| (1024, 16) | 1 | 3.349x | 3.222x | 3.655x |
| (1024, 256) | 1 | 2.326x | 2.019x | 2.398x |
| (1024, 4096) | 1 | 2.065x | 1.525x | 2.068x |
| (1024, 65536) | 1 | 1.301x | 1.048x | 1.308x |
| (1024, 1048576) | 1 | 1.371x | 1.357x | 1.466x |
| (64, 1, 64) | 1 | 1.065x | 1.065x | 1.065x |
| (64, 16, 64) | 1 | 4.953x | 5.693x | 4.939x |
| (64, 256, 64) | 1 | 3.602x | 3.592x | 3.477x |
| (64, 4096, 64) | 1 | 1.055x | 1.142x | 1.040x |

</details>
