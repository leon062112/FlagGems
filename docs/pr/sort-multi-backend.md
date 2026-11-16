### PR Category

Operator

### Type of Change

Performance Optimization

### Description

Adds backend-specific implementations of `sort` and `sort_stable` for Hygon, MetaX, MThreads and PPU. The implementations follow the Hopper dispatch strategy from #8070 and retain the existing public API and generic fallback behavior.

- Dispatch on row length, row count, packed-key width and the input device's SM count.
- Use row sorting networks for short rows, packed-key tiled merge for larger rows, and radix paths for selected long-row workloads. FP16/BF16/INT16 share the two-pass 16-bit radix criteria.
- Use single-warp initial runs for dense grids, stage-specific merge tiles and register limits, and skip merge-path partition searches when a program handles an entire pair of runs.
- Recover final values directly from packed keys where possible, preserving original signed-zero bits through a masked input load. Stable tie-breaking keeps original indices.
- Support strided and non-last sort dimensions with tiled copies, and use the existing `libentry`, Triton heuristics and JIT launch mechanisms.
- Hygon adds a dedicated packed-row kernel for contiguous rows up to 2048 elements, bypassing `libentry`, heuristics and generic strided parameters. A singleton sort dimension copies values and emits zero indices in one launch. Long 32-bit rows with at most 16 rows use a stable four-pass 8-bit radix path with NaN keys ordered after finite values, while long 64-bit-key rows use a 2048-element initial run, a dedicated second scratch buffer, final-index-buffer reuse, and first-complete-run-pair merging per CTA. The generic path remains for unpackable 64-bit values and unsupported layouts.
- MetaX adapts the Hopper dispatch strategy and adds a size-1-row fast path that returns the input with an all-zero `int64` index tensor.
- MThreads rewrites the backend implementation, extends radix dispatch to selected long INT32/FP32 few-row workloads, keeps radix indices in `int32` scratch, writes `int64` only for the final output, and stores radix prefixes in place to reduce scratch traffic.
- PPU applies the shared dispatch strategy on ZW810E and is validated across the full PR-aligned workload set.

### Issue

N/A

### Progress

- [ ] Change is properly reviewed (1 reviewer required, 2 recommended).
- [x] Correctness checks pass across the reported backend coverage: Hygon BW 90/90 PR-aligned cases; MetaX C550 640/640; MTT S5000 640/640 with device reference and 640/640 with `--ref cpu`; PPU-ZW810E 640/640 with `--ref cpu`.
- [x] Performance is measured against PyTorch on all four backends; detailed results are included where the source validation recorded them.

### Performance

Devices and environments:

- Hygon BW (80 SM), PyTorch 2.4.1+das.opt1.dtk25041, Triton 3.6.0+dtk2604
- MetaX C550 (104 SM), PyTorch 2.10.0+metax3.8.1.0, FlagTree/Triton 3.6.0
- MThreads MTT S5000 (60 SM, 80 GiB), PyTorch 2.9.1, TorchMUSA 2.9.1+a18d871, FlagTree/Triton 3.6.0
- PPU-ZW810E (64 SM), PyTorch 2.10.0, FlagTree/Triton 3.6.0

Speedup is **PyTorch latency / FlagGems latency**. Reported row values are retained unchanged; the geometric means below use the reported three-decimal speedups, with equal weight for each measurement. All measurements use the PR-aligned workload set from #8070: 9 shapes, 5 dtypes and both `sort` and `sort_stable`.

```bash
python -m pytest -s -q benchmark/test_sort.py --mode cudagraph
python -m pytest -s -q benchmark/test_sort.py --mode kernel
```

#### CUDA Graph (`mode=cudagraph`)

The following table summarizes the accepted CUDA Graph runs. Hygon results are from one complete 90-case run: 88/90 measurements meet `0.95×`; `sort` INT32 `(8, 262144)` is 0.905× and `sort_stable` FP32 `(8, 262144)` is 0.926×. The four-pass radix path lowers kernel-mode latency but adds graph nodes, making the packed merge path more competitive under Graph replay.

MThreads supplies supplemental MUSA Graph results through a benchmark-only `torch.cuda` to `torch.musa` timing shim. Its 86/90 result is retained for reference but is not used for `0.95×` acceptance because replay overhead of approximately 0.5-1 ms makes small-shape comparisons unstable. The shim does not change the operator implementation.

| Platform | Measurements | Min | Max | Geometric mean | Speedup ≥ 0.95× |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hygon | 90 | 0.905× | 48.237× | 4.077× | 88 / 90 |
| MetaX | 90 | 1.697× | 21.535× | 5.757× | 90 / 90 |
| PPU | 90 | 1.140× | 7.439× | 2.513× | 90 / 90 |
| All accepted Graph platforms | 270 | 0.905× | 48.237× | 3.893× | 268 / 270 |
| MThreads (supplemental, not acceptance) | 90 | 0.143× | 108.779× | 2.313× | 86 / 90 |

<details>
<summary>Hygon — speedup vs. PyTorch</summary>

**sort — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 9.605× | 12.615× | 8.468× | 12.356× | 9.112× |
| (256, 256) | 11.164× | 10.901× | 9.261× | 16.053× | 45.232× |
| (1024, 1024) | 2.386× | 2.109× | 5.309× | 4.203× | 2.413× |
| (4096, 4096) | 3.576× | 3.304× | 3.747× | 3.885× | 3.715× |
| (1024, 65536) | 1.878× | 2.281× | 1.960× | 2.363× | 2.018× |
| (1024, 1) | 7.514× | 4.906× | 4.185× | 6.015× | 5.884× |
| (1024, 512) | 4.469× | 4.277× | 11.189× | 5.023× | 4.359× |
| (16, 131072) | 1.467× | 0.967× | 1.335× | 0.977× | 1.305× |
| (8, 262144) | 1.408× | 0.905× | 1.280× | 0.959× | 1.251× |

**sort_stable — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 12.422× | 9.821× | 9.878× | 13.094× | 9.986× |
| (256, 256) | 48.237× | 38.463× | 44.477× | 39.644× | 46.278× |
| (1024, 1024) | 2.370× | 4.250× | 2.326× | 2.336× | 2.389× |
| (4096, 4096) | 3.867× | 3.459× | 3.751× | 3.946× | 3.713× |
| (1024, 65536) | 1.878× | 2.280× | 1.961× | 2.374× | 2.021× |
| (1024, 1) | 6.073× | 5.427× | 8.009× | 5.852× | 4.428× |
| (1024, 512) | 4.504× | 7.104× | 11.096× | 9.816× | 11.007× |
| (16, 131072) | 1.421× | 0.997× | 1.320× | 0.993× | 1.313× |
| (8, 262144) | 1.308× | 0.952× | 1.246× | 0.926× | 1.259× |

</details>

<details>
<summary>MetaX — speedup vs. PyTorch</summary>

**sort — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 10.436× | 13.245× | 10.661× | 13.339× | 10.356× |
| (256, 256) | 16.022× | 21.535× | 15.579× | 21.400× | 15.733× |
| (1024, 1024) | 4.540× | 3.188× | 4.390× | 3.149× | 4.341× |
| (4096, 4096) | 5.149× | 4.090× | 4.842× | 4.098× | 4.845× |
| (1024, 65536) | 3.839× | 4.460× | 3.148× | 3.845× | 2.750× |
| (1024, 1) | 17.117× | 15.750× | 17.139× | 15.735× | 17.164× |
| (1024, 512) | 8.196× | 7.511× | 8.050× | 7.456× | 7.970× |
| (16, 131072) | 2.374× | 2.434× | 2.070× | 2.060× | 1.760× |
| (8, 262144) | 2.274× | 2.213× | 1.993× | 1.875× | 1.697× |

**sort_stable — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 10.498× | 13.362× | 10.802× | 13.486× | 10.480× |
| (256, 256) | 16.054× | 21.518× | 15.570× | 21.255× | 15.734× |
| (1024, 1024) | 4.552× | 3.188× | 4.418× | 3.144× | 4.364× |
| (4096, 4096) | 5.174× | 4.089× | 4.854× | 4.100× | 4.850× |
| (1024, 65536) | 3.834× | 4.457× | 3.144× | 3.843× | 2.752× |
| (1024, 1) | 16.537× | 16.312× | 16.558× | 16.341× | 16.580× |
| (1024, 512) | 8.189× | 7.548× | 8.005× | 7.499× | 7.954× |
| (16, 131072) | 2.235× | 2.224× | 2.019× | 1.991× | 1.837× |
| (8, 262144) | 2.143× | 2.021× | 1.950× | 1.814× | 1.765× |

</details>

<details>
<summary>PPU — speedup vs. PyTorch</summary>

**sort — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 5.269× | 6.565× | 5.179× | 6.207× | 5.152× |
| (256, 256) | 6.788× | 7.414× | 6.826× | 6.967× | 6.829× |
| (1024, 1024) | 2.201× | 1.645× | 2.142× | 1.609× | 2.129× |
| (4096, 4096) | 1.424× | 1.174× | 1.382× | 1.140× | 1.368× |
| (1024, 65536) | 1.254× | 1.373× | 1.213× | 1.349× | 1.200× |
| (1024, 1) | 4.042× | 3.908× | 4.093× | 4.070× | 3.978× |
| (1024, 512) | 4.134× | 3.419× | 4.046× | 3.279× | 4.023× |
| (16, 131072) | 1.548× | 1.222× | 1.641× | 1.255× | 1.726× |
| (8, 262144) | 1.537× | 1.194× | 1.653× | 1.147× | 1.699× |

**sort_stable — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 5.372× | 6.287× | 5.164× | 6.220× | 5.167× |
| (256, 256) | 6.957× | 7.439× | 6.803× | 6.946× | 6.823× |
| (1024, 1024) | 2.205× | 1.645× | 2.145× | 1.609× | 2.128× |
| (4096, 4096) | 1.426× | 1.174× | 1.380× | 1.140× | 1.368× |
| (1024, 65536) | 1.251× | 1.374× | 1.213× | 1.348× | 1.200× |
| (1024, 1) | 4.135× | 4.328× | 4.039× | 4.094× | 4.011× |
| (1024, 512) | 4.137× | 3.411× | 4.060× | 3.267× | 4.024× |
| (16, 131072) | 1.558× | 1.232× | 1.652× | 1.254× | 1.724× |
| (8, 262144) | 1.542× | 1.190× | 1.653× | 1.144× | 1.699× |

</details>

<details>
<summary>MThreads — speedup vs. PyTorch (supplemental, not acceptance)</summary>

**sort — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 2.797× | 1.686× | 2.586× | 1.959× | 1.470× |
| (256, 256) | 1.045× | 0.193× | 2.099× | 2.353× | 2.344× |
| (1024, 1024) | 1.368× | 3.169× | 1.833× | 1.575× | 2.328× |
| (4096, 4096) | 2.566× | 1.746× | 1.596× | 1.725× | 1.450× |
| (1024, 65536) | 1.284× | 1.550× | 1.030× | 1.450× | 1.077× |
| (1024, 1) | 1.731× | 1.264× | 1.767× | 1.491× | 1.768× |
| (1024, 512) | 2.803× | 2.136× | 2.711× | 2.256× | 3.042× |
| (16, 131072) | 1.063× | 1.116× | 1.025× | 1.091× | 1.040× |
| (8, 262144) | 1.085× | 1.072× | 1.025× | 1.079× | 0.992× |

**sort_stable — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 6.745× | 4.553× | 11.038× | 11.281× | 6.918× |
| (256, 256) | 4.843× | 6.803× | 12.041× | 6.237× | 2.218× |
| (1024, 1024) | 1.948× | 2.536× | 6.142× | 2.606× | 0.622× |
| (4096, 4096) | 2.466× | 3.577× | 1.848× | 0.350× | 0.143× |
| (1024, 65536) | 1.860× | 1.666× | 1.511× | 1.522× | 1.134× |
| (1024, 1) | 38.309× | 108.779× | 64.004× | 17.990× | 66.286× |
| (1024, 512) | 5.222× | 7.167× | 5.676× | 37.791× | 5.500× |
| (16, 131072) | 1.054× | 1.043× | 1.115× | 1.998× | 1.116× |
| (8, 262144) | 1.042× | 0.998× | 1.093× | 0.977× | 1.074× |

</details>

#### Without CUDA Graph (`mode=kernel`)

The following table summarizes the kernel-mode runs. Hygon kernel results include host-side parameter preparation and launch overhead, so short rows are dominated by fixed launch cost; 78/90 measurements meet `0.95×`. MThreads results are from the second of two independent runs; both runs passed 90/90 measurements.

| Platform | Measurements | Min | Max | Geometric mean | Speedup ≥ 0.95× |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hygon | 90 | 0.662× | 7.166× | 2.071× | 78 / 90 |
| MThreads | 90 | 0.951× | 153.819× | 2.391× | 90 / 90 |
| MetaX | 90 | 1.876× | 5.437× | 3.292× | 90 / 90 |
| PPU | 90 | 1.114× | 5.235× | 2.243× | 90 / 90 |
| All kernel platforms | 360 | 0.662× | 153.819× | 2.459× | 348 / 360 |

<details>
<summary>Hygon — speedup vs. PyTorch</summary>

**sort — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 2.581× | 1.143× | 1.000× | 1.285× | 2.510× |
| (256, 256) | 6.600× | 2.876× | 1.911× | 3.281× | 4.398× |
| (1024, 1024) | 3.752× | 3.159× | 3.700× | 3.432× | 3.673× |
| (4096, 4096) | 3.562× | 3.282× | 3.501× | 3.776× | 3.480× |
| (1024, 65536) | 1.878× | 2.289× | 1.969× | 2.377× | 2.024× |
| (1024, 1) | 1.342× | 1.478× | 1.486× | 1.256× | 1.427× |
| (1024, 512) | 4.565× | 6.131× | 4.789× | 7.166× | 4.626× |
| (16, 131072) | 1.148× | 0.843× | 0.988× | 0.932× | 1.012× |
| (8, 262144) | 1.087× | 0.894× | 1.018× | 0.837× | 1.014× |

**sort_stable — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 2.176× | 1.708× | 0.911× | 1.062× | 0.942× |
| (256, 256) | 3.679× | 3.729× | 1.899× | 2.967× | 2.615× |
| (1024, 1024) | 3.803× | 3.151× | 3.722× | 3.477× | 3.673× |
| (4096, 4096) | 3.629× | 3.299× | 3.530× | 3.761× | 3.594× |
| (1024, 65536) | 1.885× | 2.284× | 1.965× | 2.375× | 2.022× |
| (1024, 1) | 1.507× | 1.121× | 1.030× | 0.662× | 1.265× |
| (1024, 512) | 4.879× | 5.896× | 4.016× | 6.371× | 4.807× |
| (16, 131072) | 1.229× | 0.778× | 0.962× | 0.823× | 1.079× |
| (8, 262144) | 1.142× | 0.773× | 0.935× | 0.806× | 1.054× |

</details>

<details>
<summary>MThreads — speedup vs. PyTorch</summary>

**sort — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 3.486× | 1.893× | 2.556× | 2.078× | 2.254× |
| (256, 256) | 2.201× | 2.214× | 2.161× | 2.451× | 2.418× |
| (1024, 1024) | 1.611× | 1.359× | 1.571× | 1.452× | 1.709× |
| (4096, 4096) | 1.807× | 1.662× | 1.473× | 1.652× | 1.464× |
| (1024, 65536) | 1.249× | 1.605× | 1.039× | 1.492× | 1.058× |
| (1024, 1) | 1.677× | 1.524× | 1.651× | 1.508× | 1.733× |
| (1024, 512) | 2.330× | 2.166× | 2.250× | 2.185× | 2.477× |
| (16, 131072) | 0.992× | 1.008× | 0.998× | 1.009× | 1.012× |
| (8, 262144) | 0.951× | 0.961× | 0.965× | 0.957× | 0.979× |

**sort_stable — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 5.117× | 9.336× | 6.839× | 9.329× | 6.669× |
| (256, 256) | 5.960× | 11.720× | 5.903× | 11.379× | 5.948× |
| (1024, 1024) | 1.477× | 2.309× | 1.475× | 2.330× | 1.467× |
| (4096, 4096) | 1.807× | 1.664× | 1.475× | 1.648× | 1.463× |
| (1024, 65536) | 1.251× | 1.620× | 1.040× | 1.494× | 1.056× |
| (1024, 1) | 78.432× | 153.819× | 78.900× | 152.502× | 77.457× |
| (1024, 512) | 4.386× | 7.505× | 4.319× | 7.043× | 4.299× |
| (16, 131072) | 0.991× | 1.011× | 0.995× | 1.010× | 1.008× |
| (8, 262144) | 0.951× | 0.962× | 0.970× | 0.959× | 0.976× |

</details>

<details>
<summary>MetaX — speedup vs. PyTorch</summary>

**sort — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 3.492× | 3.934× | 3.532× | 3.951× | 3.532× |
| (256, 256) | 4.543× | 5.230× | 4.380× | 5.243× | 4.493× |
| (1024, 1024) | 3.505× | 2.705× | 3.421× | 2.685× | 3.401× |
| (4096, 4096) | 5.073× | 4.056× | 4.776× | 4.065× | 4.770× |
| (1024, 65536) | 3.835× | 4.462× | 3.138× | 3.841× | 2.748× |
| (1024, 1) | 2.192× | 2.216× | 2.212× | 2.173× | 2.212× |
| (1024, 512) | 4.672× | 5.412× | 4.600× | 5.385× | 4.600× |
| (16, 131072) | 2.500× | 2.371× | 2.171× | 2.183× | 1.944× |
| (8, 262144) | 2.412× | 2.181× | 2.108× | 2.001× | 1.876× |

**sort_stable — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 3.516× | 3.951× | 3.590× | 3.967× | 3.548× |
| (256, 256) | 4.603× | 5.230× | 4.443× | 5.257× | 4.551× |
| (1024, 1024) | 3.485× | 2.709× | 3.421× | 2.694× | 3.394× |
| (4096, 4096) | 5.083× | 4.055× | 4.777× | 4.060× | 4.771× |
| (1024, 65536) | 3.827× | 4.466× | 3.138× | 3.834× | 2.748× |
| (1024, 1) | 2.235× | 2.231× | 2.173× | 2.255× | 2.216× |
| (1024, 512) | 4.720× | 5.437× | 4.639× | 5.379× | 4.622× |
| (16, 131072) | 2.483× | 2.371× | 2.176× | 2.190× | 1.944× |
| (8, 262144) | 2.406× | 2.176× | 2.102× | 2.002× | 1.882× |

</details>

<details>
<summary>PPU — speedup vs. PyTorch</summary>

**sort — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 4.160× | 4.645× | 4.284× | 4.710× | 4.259× |
| (256, 256) | 5.099× | 5.235× | 5.083× | 4.990× | 5.051× |
| (1024, 1024) | 1.607× | 1.550× | 1.575× | 1.553× | 1.536× |
| (4096, 4096) | 1.411× | 1.165× | 1.378× | 1.131× | 1.358× |
| (1024, 65536) | 1.256× | 1.374× | 1.217× | 1.349× | 1.207× |
| (1024, 1) | 4.225× | 4.250× | 4.225× | 4.250× | 4.225× |
| (1024, 512) | 3.054× | 3.120× | 2.987× | 2.991× | 2.907× |
| (16, 131072) | 1.485× | 1.217× | 1.603× | 1.240× | 1.659× |
| (8, 262144) | 1.451× | 1.114× | 1.584× | 1.137× | 1.616× |

**sort_stable — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 4.197× | 4.645× | 4.284× | 4.710× | 4.259× |
| (256, 256) | 5.099× | 5.230× | 5.070× | 5.088× | 5.089× |
| (1024, 1024) | 1.618× | 1.549× | 1.575× | 1.564× | 1.537× |
| (4096, 4096) | 1.413× | 1.164× | 1.380× | 1.130× | 1.360× |
| (1024, 65536) | 1.258× | 1.372× | 1.217× | 1.346× | 1.208× |
| (1024, 1) | 4.225× | 4.250× | 4.250× | 4.250× | 4.225× |
| (1024, 512) | 3.054× | 3.116× | 2.995× | 2.986× | 2.905× |
| (16, 131072) | 1.485× | 1.219× | 1.602× | 1.241× | 1.659× |
| (8, 262144) | 1.451× | 1.115× | 1.584× | 1.138× | 1.615× |

</details>
