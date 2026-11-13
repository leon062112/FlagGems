### PR Category

Operator

### Type of Change

Performance Optimization

### Description

Adds backend-specific `argmax` implementations for MetaX, Hygon, MThreads, and PPU. The implementations use backend-tuned Triton kernels for common reduction layouts and retain the generic implementation as a fallback for unsupported layouts, dtypes, or shapes.

- MetaX uses stride-aware row, column, and global reductions with SM-aware splitting, partial/final reduction, stable NaN/tie handling, and hardware-tuned thresholds for the C550.
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

- MetaX C550
- Hygon BW (80 SM)
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
| MetaX | C550 | Original argmax tests, NaN/tie/strided views, ATen dispatch, CUDA Graph replay | Benchmark validation completed; quantitative latency data was not recorded in the source commit |
| Hygon | BW (80 SM) | 126 argmax tests | CUDA Graph 51/51 and Kernel 51/51 at or above 0.95x PyTorch |
| MThreads | MTT S5000 | 126 argmax tests; FP16/FP32/BF16 and ATen registration | MUSA Graph 42/51 and Kernel 51/51 at or above 0.95x PyTorch; details below |
| PPU | ZW810E (64 SM) | 135 existing argmax tests | 51/51 SUCCESS in both Kernel and CUDA Graph modes; details below |

Hygon reports the 51/51 threshold result but does not include min/max/geometric-mean values. MetaX reports benchmark validation without raw latency values. The MThreads metrics below are from the validation run on commit `de033a4b4`; missing metrics are not inferred.

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
