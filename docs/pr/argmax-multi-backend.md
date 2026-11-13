### PR Category

Operator

### Type of Change

Performance Optimization

### Description

Adds backend-specific `argmax` implementations for MetaX, Hygon, MThreads, and PPU. The implementations use backend-tuned Triton kernels for common reduction layouts and retain the generic implementation as a fallback for unsupported layouts, dtypes, or shapes.

- MetaX uses stride-aware row, column, and global reductions with SM-aware splitting, partial/final reduction, stable NaN/tie handling, and hardware-tuned thresholds for the C550. Size-1 reductions return an all-zero index output through a dedicated fast path.
- Hygon converts FP16/BF16 values to FP32 and reduces ordered max-value/min-index keys, preserving first-NaN and lowest-index tie semantics. Contiguous global reductions up to 2^24 elements use a two-stage reduction that packs each partial key and index into one `uint64` payload. For contiguous innermost reductions with at least 64 rows, one CTA handles each row: `N <= 4096` uses a single-pass kernel and larger rows stream over bounded 4096- or 8192-element blocks. Non-innermost reductions use an output-block kernel when `N <= 32` and the output has at most 2^20 elements. Size-1 reductions zero the full output with a cached Triton kernel up to 2^16 elements (`torch.zero_` above that), import-time prewarming covers the zero-fill and global-reduction kernels, and other layouts fall back to the generic implementation.
- MThreads uses a row-per-CTA kernel for contiguous reductions, an output-block kernel for non-inner reductions that coalesces contiguous inner elements, and a native comparison path for large BF16 reductions. Fixed heuristics replace runtime autotune.
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
| Hygon | BW (80 SM) | 126/126 argmax tests | Kernel 51/51 and CUDA Graph 51/51 at or above 0.95x PyTorch; details below |
| MThreads | MTT S5000 | 126 argmax tests; FP16/FP32/BF16 and ATen registration | MUSA Graph 51/51 and Kernel 51/51 at or above 0.95x PyTorch; details below |
| PPU | ZW810E (64 SM) | 126/126 argmax tests | Kernel 51/51 and CUDA Graph 51/51 at or above 0.95x PyTorch; details below |

The MThreads metrics below are from validation run on commit `62eb8f57b`; missing metrics are not inferred.

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

Correctness: `126 passed` with `tests/test_argmax.py`. Both comprehensive benchmark modes completed all 51 cases successfully.

| Mode | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA Graph | 51 | 0.987x | 53.455x | 3.157x | 51 / 51 |
| Kernel (non-Graph) | 51 | 0.980x | 21.463x | 2.552x | 51 / 51 |

All 51 benchmark cases are at or above 0.95x PyTorch in both modes. The measurements are from a stable run on an idle Hygon BW device; repeated runs on the shared node can show transient outliers.

<details>
<summary>Hygon CUDA Graph - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 3.692x | 4.549x | 4.491x |
| (64, 64) | 1 | 8.686x | 4.822x | 4.640x |
| (4096, 4096) | 1 | 1.970x | 1.382x | 1.689x |
| (64, 512, 512) | 1 | 8.746x | 3.212x | 9.858x |
| (1024, 1024, 1024) | 1 | 4.275x | 1.930x | 2.287x |
| (1049600,) | None (global) | 2.471x | 4.370x | 4.332x |
| (1073741824,) | None (global) | 1.998x | 1.207x | 1.849x |
| (1024, 1) | 1 | 1.047x | 1.250x | 0.992x |
| (1024, 16) | 1 | 2.673x | 3.049x | 3.074x |
| (1024, 256) | 1 | 3.391x | 4.303x | 4.418x |
| (1024, 4096) | 1 | 2.430x | 1.194x | 2.764x |
| (1024, 65536) | 1 | 1.235x | 1.245x | 1.160x |
| (1024, 1048576) | 1 | 1.119x | 1.230x | 1.149x |
| (64, 1, 64) | 1 | 0.987x | 1.233x | 1.199x |
| (64, 16, 64) | 1 | 10.829x | 9.894x | 8.014x |
| (64, 256, 64) | 1 | 26.703x | 29.702x | 53.455x |
| (64, 4096, 64) | 1 | 5.868x | 5.615x | 4.794x |

</details>

<details>
<summary>Hygon Kernel (non-Graph) - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 2.580x | 2.352x | 2.532x |
| (64, 64) | 1 | 2.690x | 2.690x | 2.690x |
| (4096, 4096) | 1 | 1.832x | 1.468x | 1.572x |
| (64, 512, 512) | 1 | 6.788x | 4.857x | 5.041x |
| (1024, 1024, 1024) | 1 | 5.660x | 2.033x | 2.082x |
| (1049600,) | None (global) | 2.341x | 2.185x | 2.452x |
| (1073741824,) | None (global) | 2.042x | 1.286x | 2.104x |
| (1024, 1) | 1 | 0.980x | 1.167x | 1.000x |
| (1024, 16) | 1 | 2.587x | 2.978x | 2.667x |
| (1024, 256) | 1 | 2.571x | 3.551x | 2.583x |
| (1024, 4096) | 1 | 1.694x | 1.724x | 1.718x |
| (1024, 65536) | 1 | 1.391x | 1.144x | 1.259x |
| (1024, 1048576) | 1 | 1.141x | 1.094x | 1.106x |
| (64, 1, 64) | 1 | 0.980x | 0.980x | 1.214x |
| (64, 16, 64) | 1 | 8.198x | 6.644x | 6.833x |
| (64, 256, 64) | 1 | 21.463x | 16.243x | 12.200x |
| (64, 4096, 64) | 1 | 6.118x | 3.866x | 4.841x |

</details>

#### MThreads performance

Validated on MTT S5000 (80 GiB) with MUSA driver 3.3.5-server, PyTorch 2.9.1, TorchMUSA 2.9.1+a18d871 and FlagTree/Triton 3.6.0 on commit `62eb8f57b`. Correctness: `126 passed`; ATen dispatch: `9/9`.

MThreads adds row-per-CTA contiguous reductions, a native BF16 comparison path for large non-inner reductions, and 8192-element global reduction tiles. Graph runs used a temporary `torch.cuda` to `torch.musa` timing shim and per-dtype processes to avoid MUSA replay jitter; a three-run median check had a minimum speedup of `1.028x` and a geometric mean of `1.535x`.

| Mode | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| MUSA Graph | 51 | 1.003x | 5.013x | 1.511x | 51 / 51 |
| Kernel (non-Graph) | 51 | 0.973x | 5.321x | 1.440x | 51 / 51 |

<details>
<summary>MThreads MUSA Graph - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 1.144x | 1.003x | 1.261x |
| (64, 64) | 1 | 1.487x | 1.276x | 1.386x |
| (4096, 4096) | 1 | 1.101x | 1.019x | 1.116x |
| (64, 512, 512) | 1 | 1.342x | 1.326x | 1.363x |
| (1024, 1024, 1024) | 1 | 1.490x | 1.186x | 1.263x |
| (1049600,) | None (global) | 1.389x | 1.179x | 1.455x |
| (1073741824,) | None (global) | 1.270x | 1.125x | 1.209x |
| (1024, 1) | 1 | 1.259x | 1.222x | 1.207x |
| (1024, 16) | 1 | 2.463x | 2.077x | 2.515x |
| (1024, 256) | 1 | 1.845x | 1.825x | 1.913x |
| (1024, 4096) | 1 | 1.125x | 1.164x | 1.116x |
| (1024, 65536) | 1 | 1.050x | 1.196x | 1.047x |
| (1024, 1048576) | 1 | 1.094x | 1.108x | 1.097x |
| (64, 1, 64) | 1 | 1.157x | 1.177x | 1.120x |
| (64, 16, 64) | 1 | 5.013x | 3.995x | 5.010x |
| (64, 256, 64) | 1 | 2.782x | 1.608x | 2.770x |
| (64, 4096, 64) | 1 | 3.010x | 1.896x | 3.037x |

</details>

<details>
<summary>MThreads Kernel (non-Graph) - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 1.366x | 1.102x | 1.344x |
| (64, 64) | 1 | 1.459x | 1.276x | 1.424x |
| (4096, 4096) | 1 | 1.028x | 1.020x | 1.028x |
| (64, 512, 512) | 1 | 1.228x | 1.254x | 1.237x |
| (1024, 1024, 1024) | 1 | 1.510x | 1.124x | 1.258x |
| (1049600,) | None (global) | 1.356x | 1.123x | 1.345x |
| (1073741824,) | None (global) | 1.308x | 1.036x | 1.220x |
| (1024, 1) | 1 | 1.120x | 1.180x | 1.110x |
| (1024, 16) | 1 | 2.356x | 1.875x | 2.366x |
| (1024, 256) | 1 | 1.572x | 1.366x | 1.603x |
| (1024, 4096) | 1 | 1.006x | 1.072x | 1.010x |
| (1024, 65536) | 1 | 1.131x | 1.231x | 1.129x |
| (1024, 1048576) | 1 | 1.262x | 0.973x | 1.266x |
| (64, 1, 64) | 1 | 1.082x | 1.082x | 1.082x |
| (64, 16, 64) | 1 | 5.321x | 4.034x | 5.287x |
| (64, 256, 64) | 1 | 2.348x | 1.306x | 2.348x |
| (64, 4096, 64) | 1 | 2.737x | 1.386x | 2.741x |

</details>

#### PPU performance

| Mode | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA Graph | 51 | 1.010x | 5.759x | 1.968x | 51 / 51 |
| Kernel (non-Graph) | 51 | 0.980x | 5.760x | 1.753x | 51 / 51 |

The PPU measurements below are from the latest full rerun on 2026-11-13 using an idle ZW810E. All 51 benchmark cases passed in both modes. Repeated runs can show normal variance on a few global and small-shape cases.

<details>
<summary>PPU CUDA Graph - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 2.434x | 2.235x | 2.633x |
| (64, 64) | 1 | 3.040x | 2.885x | 3.151x |
| (4096, 4096) | 1 | 1.341x | 1.232x | 1.360x |
| (64, 512, 512) | 1 | 1.536x | 1.563x | 1.558x |
| (1024, 1024, 1024) | 1 | 1.279x | 1.010x | 1.245x |
| (1049600,) | None (global) | 2.672x | 2.367x | 2.812x |
| (1073741824,) | None (global) | 2.226x | 1.222x | 2.243x |
| (1024, 1) | 1 | 1.042x | 1.065x | 1.045x |
| (1024, 16) | 1 | 3.225x | 3.179x | 3.264x |
| (1024, 256) | 1 | 2.209x | 2.150x | 2.241x |
| (1024, 4096) | 1 | 2.565x | 2.217x | 2.571x |
| (1024, 65536) | 1 | 1.807x | 1.114x | 1.814x |
| (1024, 1048576) | 1 | 1.496x | 1.394x | 1.527x |
| (64, 1, 64) | 1 | 1.015x | 1.037x | 1.041x |
| (64, 16, 64) | 1 | 4.205x | 4.774x | 3.904x |
| (64, 256, 64) | 1 | 4.519x | 5.759x | 4.470x |
| (64, 4096, 64) | 1 | 1.204x | 1.679x | 1.196x |

</details>

<details>
<summary>PPU Kernel (non-Graph) - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 1.891x | 1.614x | 1.870x |
| (64, 64) | 1 | 3.333x | 3.000x | 3.348x |
| (4096, 4096) | 1 | 1.084x | 1.014x | 1.090x |
| (64, 512, 512) | 1 | 1.219x | 0.980x | 1.149x |
| (1024, 1024, 1024) | 1 | 1.268x | 1.011x | 1.244x |
| (1049600,) | None (global) | 1.769x | 1.685x | 1.778x |
| (1073741824,) | None (global) | 1.902x | 1.175x | 2.011x |
| (1024, 1) | 1 | 1.100x | 1.100x | 1.100x |
| (1024, 16) | 1 | 3.492x | 3.301x | 3.793x |
| (1024, 256) | 1 | 2.408x | 2.114x | 2.367x |
| (1024, 4096) | 1 | 2.079x | 1.541x | 2.094x |
| (1024, 65536) | 1 | 1.305x | 1.049x | 1.311x |
| (1024, 1048576) | 1 | 1.372x | 1.347x | 1.470x |
| (64, 1, 64) | 1 | 1.065x | 1.065x | 1.065x |
| (64, 16, 64) | 1 | 5.006x | 5.760x | 4.782x |
| (64, 256, 64) | 1 | 3.656x | 3.619x | 3.513x |
| (64, 4096, 64) | 1 | 1.064x | 1.138x | 1.047x |

</details>
