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
- MThreads MTT S5000
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
| MThreads | MTT S5000 | Original argmax tests with FP16/FP32/BF16, ATen registration | 51 benchmark cases completed; quantitative latency data was not recorded in the source commit |
| PPU | ZW810E (64 SM) | 135 existing argmax tests | 51/51 SUCCESS in both Kernel and CUDA Graph modes; details below |

Hygon reports the 51/51 threshold result but does not include min/max/geometric-mean values. MetaX and MThreads report benchmark validation without raw latency values. Missing metrics are not inferred.

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
