### PR Category

Operator

### Type of Change

Performance Optimization

### Description

Adds backend-specific `argmax` implementations for MetaX, Hygon, MThreads, and PPU. The implementations use backend-tuned Triton kernels for common reduction layouts and retain the generic implementation as a fallback for unsupported layouts, dtypes, or shapes.

- MetaX validates reduction dimensions and empty reduction axes, supports 0D input, and uses stride-aware row, column, small non-inner, and global reductions with SM-aware splitting, partial/final reduction, first-NaN and lowest-index tie handling, and hardware-tuned thresholds for the C550. Size-1 reductions return an all-zero index output through a dedicated fast path; uncollapsible strided views are materialized by a Triton copy kernel.
- Hygon converts FP16/BF16 values to FP32 and reduces ordered max-value/min-index keys. Its specialized paths preserve first-NaN and lowest-index tie semantics, while generic fallback paths retain the existing generic behavior. Contiguous global reductions up to 2^24 elements use a two-stage reduction that packs each partial key and index into one `uint64` payload. For contiguous innermost reductions with at least 64 rows, one CTA handles each row: `N <= 4096` uses a single-pass kernel and larger rows stream over bounded 4096- or 8192-element blocks. Non-innermost reductions use an output-block kernel when `N <= 32` and the output has at most 2^20 elements. Size-1 reductions use a Triton zero-fill kernel launched through the public `libentry(cache_binding=True)` path, scalar and empty-dimension errors follow PyTorch, and other layouts fall back to the generic implementation.
- MThreads uses a row-per-CTA kernel for contiguous reductions, an output-block kernel for non-inner reductions that coalesces contiguous inner elements, and a native comparison path for large BF16 reductions. Specialized paths normalize NaN to `+inf` and repair the lowest NaN index, preserve lowest-index ties, and follow PyTorch behavior for scalar input, invalid or empty reduction dimensions, and size-1 reductions. Long contiguous rows stream full tiles without masks, while global reductions use a tree reduction and repair NaN partials before the final index selection. Fixed heuristics replace runtime autotune.
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
| Hygon | BW (80 SM) | 126/126 argmax tests | Kernel 51/51 and CUDA Graph 51/51 completed; stable run at or above 0.95x, with occasional transient outliers noted below |
| MThreads | MTT S5000 | 126 argmax tests; FP16/FP32/BF16 and ATen registration | Kernel 51/51 at or above 0.95x PyTorch; MUSA Graph 50/51 at or above 0.95x (minimum 0.944x); details below |
| PPU | ZW810E (64 SM) | 126/126 argmax tests | Kernel 51/51 and CUDA Graph 51/51 at or above 0.95x PyTorch; details below |

#### MetaX performance

| Mode | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA Graph | 51 | 1.000x | 4.613x | 1.710x | 51 / 51 |
| Kernel (non-Graph) | 51 | 1.065x | 2.384x | 1.536x | 51 / 51 |

The size-1 reduction fast path raises FP32 `(64, 1, 64)` CUDA Graph speedup from 0.942x to 1.027x.

<details>
<summary>MetaX CUDA Graph - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 1.847x | 1.876x | 1.887x |
| (64, 64) | 1 | 2.152x | 2.182x | 2.135x |
| (4096, 4096) | 1 | 1.309x | 1.243x | 1.259x |
| (64, 512, 512) | 1 | 1.457x | 1.547x | 1.451x |
| (1024, 1024, 1024) | 1 | 1.653x | 1.058x | 1.664x |
| (1049600,) | None (global) | 1.841x | 1.798x | 1.841x |
| (1073741824,) | None (global) | 1.498x | 1.059x | 1.500x |
| (1024, 1) | 1 | 1.033x | 1.000x | 1.010x |
| (1024, 16) | 1 | 1.947x | 1.946x | 1.762x |
| (1024, 256) | 1 | 2.094x | 2.164x | 2.063x |
| (1024, 4096) | 1 | 2.053x | 1.498x | 2.007x |
| (1024, 65536) | 1 | 1.503x | 1.069x | 1.504x |
| (1024, 1048576) | 1 | 1.516x | 1.169x | 1.513x |
| (64, 1, 64) | 1 | 1.065x | 1.027x | 1.037x |
| (64, 16, 64) | 1 | 4.613x | 4.492x | 4.523x |
| (64, 256, 64) | 1 | 3.443x | 3.892x | 3.547x |
| (64, 4096, 64) | 1 | 1.783x | 1.225x | 1.455x |

</details>

<details>
<summary>MetaX Kernel (non-Graph) - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 1.743x | 1.782x | 1.786x |
| (64, 64) | 1 | 1.368x | 1.386x | 1.345x |
| (4096, 4096) | 1 | 1.514x | 1.379x | 1.478x |
| (64, 512, 512) | 1 | 1.491x | 1.595x | 1.485x |
| (1024, 1024, 1024) | 1 | 1.667x | 1.065x | 1.677x |
| (1049600,) | None (global) | 1.756x | 1.707x | 1.800x |
| (1073741824,) | None (global) | 1.502x | 1.067x | 1.504x |
| (1024, 1) | 1 | 1.176x | 1.154x | 1.151x |
| (1024, 16) | 1 | 1.554x | 1.467x | 1.554x |
| (1024, 256) | 1 | 1.600x | 1.607x | 1.565x |
| (1024, 4096) | 1 | 1.839x | 1.631x | 1.800x |
| (1024, 65536) | 1 | 1.531x | 1.121x | 1.554x |
| (1024, 1048576) | 1 | 1.509x | 1.153x | 1.508x |
| (64, 1, 64) | 1 | 1.115x | 1.151x | 1.132x |
| (64, 16, 64) | 1 | 2.082x | 2.032x | 2.065x |
| (64, 256, 64) | 1 | 2.345x | 2.384x | 2.345x |
| (64, 4096, 64) | 1 | 1.838x | 1.345x | 1.578x |

</details>

#### Hygon performance

Environment: Hygon BW (80 SM), PyTorch 2.4.1+das.opt1.dtk25041, Triton 3.6.0+dtk2604. The primary reduction kernels and the stable 51-case reference measurements are from code commit `9e4dbb74`. The current branch tip `489037e6` adds the Apache license header, PyTorch-compatible edge-case handling, and a public `libentry(cache_binding=True)` launch path for the singleton zero-fill kernel without changing the reduction kernels. The available container did not provide the DTK 26.04/PyTorch 2.9 combination pinned by `pyproject.toml`. Because the current tree imports `torch.float8_e8m0fnu` from unrelated `dstack` code, validation used a process-local alias to an existing FP8 dtype; the argmax implementation and tests were unchanged.

Correctness: `126 passed` with `tests/test_argmax.py`. Additional manual checks covered scalar `dim=0/-1`, invalid and empty reduction dimensions, size-1 outputs, and first-NaN behavior on specialized paths. Both comprehensive benchmark modes completed all 51 cases successfully.

| Mode | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA Graph | 51 | 0.987x | 53.455x | 3.157x | 51 / 51 |
| Kernel (non-Graph) | 51 | 0.980x | 21.463x | 2.552x | 51 / 51 |

In the stable reference run, all 51 benchmark cases are at or above 0.95x PyTorch in both modes. Isolated reruns on the shared Hygon node can occasionally fall below 0.95x, especially for the roughly 2 microsecond singleton-output cases and a few global reductions; these are accepted as transient timing outliers rather than implementation regressions, and repeat measurements return near or above PyTorch.

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

Validated on MTT S5000 (80 GiB) with MUSA driver 3.3.5-server, PyTorch 2.9.1, TorchMUSA 2.9.1+a18d871 and FlagTree/Triton 3.6.0 on commit `53457e16`. Correctness: `126 passed`; the ATen registration surface is unchanged from the earlier `9/9` dispatch check.

MThreads adds row-per-CTA contiguous reductions, a native BF16 comparison path for large non-inner reductions, and tree-based global reductions. Long contiguous rows stream unmasked full tiles and encode the `+inf` candidate in the temporary index, so NaN repair uses a second lightweight kernel without allocating an extra buffer or regressing graph replay. Global reductions repair NaN partials before the final index selection. Graph runs used a temporary `torch.cuda` to `torch.musa` timing shim and per-dtype processes to avoid cross-dtype MUSA replay jitter. The shim maps `torch.cuda` stream, event, and graph-capture APIs to their `torch.musa` equivalents so the standard CUDA Graph benchmark can run on MUSA; it is benchmark-only and is not part of the operator implementation.

| Mode | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| MUSA Graph | 51 | 0.944x | 3.101x | 1.388x | 50 / 51 |
| Kernel (non-Graph) | 51 | 0.989x | 3.509x | 1.361x | 51 / 51 |

<details>
<summary>MThreads MUSA Graph - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 1.329x | 1.198x | 1.351x |
| (64, 64) | 1 | 1.395x | 1.185x | 1.411x |
| (4096, 4096) | 1 | 0.972x | 0.980x | 0.974x |
| (64, 512, 512) | 1 | 1.584x | 1.195x | 1.199x |
| (1024, 1024, 1024) | 1 | 1.091x | 1.163x | 1.083x |
| (1049600,) | None (global) | 1.302x | 1.210x | 1.412x |
| (1073741824,) | None (global) | 1.141x | 1.190x | 1.087x |
| (1024, 1) | 1 | 1.151x | 1.179x | 1.203x |
| (1024, 16) | 1 | 2.602x | 2.218x | 2.627x |
| (1024, 256) | 1 | 1.843x | 1.523x | 1.741x |
| (1024, 4096) | 1 | 0.952x | 1.152x | 0.944x |
| (1024, 65536) | 1 | 1.085x | 1.166x | 1.057x |
| (1024, 1048576) | 1 | 1.171x | 1.123x | 1.191x |
| (64, 1, 64) | 1 | 1.127x | 1.110x | 1.185x |
| (64, 16, 64) | 1 | 2.819x | 2.011x | 3.101x |
| (64, 256, 64) | 1 | 1.867x | 1.161x | 2.047x |
| (64, 4096, 64) | 1 | 2.649x | 1.738x | 2.797x |

</details>

<details>
<summary>MThreads Kernel (non-Graph) - speedup vs. PyTorch</summary>

| Shape | Dim | FP16 | FP32 | BF16 |
| --- | --- | ---: | ---: | ---: |
| (1048576,) | None (global) | 1.283x | 1.209x | 1.299x |
| (64, 64) | 1 | 1.455x | 1.238x | 1.416x |
| (4096, 4096) | 1 | 0.999x | 1.010x | 0.996x |
| (64, 512, 512) | 1 | 1.371x | 1.208x | 1.131x |
| (1024, 1024, 1024) | 1 | 1.099x | 1.107x | 1.098x |
| (1049600,) | None (global) | 1.263x | 1.082x | 1.279x |
| (1073741824,) | None (global) | 1.281x | 1.109x | 1.266x |
| (1024, 1) | 1 | 1.163x | 1.120x | 1.122x |
| (1024, 16) | 1 | 2.469x | 1.950x | 2.537x |
| (1024, 256) | 1 | 1.521x | 1.313x | 1.542x |
| (1024, 4096) | 1 | 0.989x | 1.057x | 0.989x |
| (1024, 65536) | 1 | 1.140x | 1.224x | 1.140x |
| (1024, 1048576) | 1 | 1.347x | 0.997x | 1.359x |
| (64, 1, 64) | 1 | 1.082x | 1.061x | 1.082x |
| (64, 16, 64) | 1 | 3.438x | 2.142x | 3.509x |
| (64, 256, 64) | 1 | 1.808x | 1.075x | 1.861x |
| (64, 4096, 64) | 1 | 2.472x | 1.321x | 2.618x |

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
