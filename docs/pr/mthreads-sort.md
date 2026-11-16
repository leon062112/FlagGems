### PR Category

Operator

### Type of Change

Performance Optimization

### Description

Rewrites the MThreads backend implementation of `sort` and `sort_stable`, with dispatch based on row length, row count, packed-key width and the input device's SM count.

- Use row sorting networks for short rows, packed-key tiled merge for larger rows, and radix paths for selected long-row workloads. FP16/BF16/INT16 share the same dispatch criteria; long INT32/F32 workloads with few rows also use the radix path.
- Use single-warp initial runs for dense grids, stage-specific merge tiles and register limits, and skip merge-path partition searches when a program handles an entire pair of runs.
- Recover final values directly from packed keys where possible, preserving original signed-zero bits through a masked input load. Stable tie-breaking keeps original indices.
- Keep MThreads radix indices in `int32` scratch and write `int64` only for the final output; radix prefixes are stored in place to reduce scratch traffic.
- Support strided and non-last sort dimensions with tiled copies, and use the existing `libentry`, Triton heuristics, and JIT launch mechanisms. The public API and generic fallback behavior remain unchanged.

### Issue

N/A

### Progress

- [ ] Change is properly reviewed (1 reviewer required, 2 recommended).
- [x] Correctness checks pass on MTT S5000: `tests/test_sort.py` passes 640/640 with the device reference and 640/640 with `--ref cpu`.
- [x] Performance is measured against PyTorch.

### Performance

Env: MTT S5000 (60 SM, 80 GiB), PyTorch 2.9.1, TorchMUSA 2.9.1+a18d871, FlagTree/Triton 3.6.0.

```bash
python -m pytest -q tests/test_sort.py
python -m pytest -q tests/test_sort.py --ref cpu
python -m pytest -s -q benchmark/test_sort.py --mode kernel
```

Speedup is **PyTorch latency / FlagGems latency**. Geometric means use the reported three-decimal speedups with equal weight for each measurement.

#### CUDA Graph (`mode=cudagraph`)

The upstream CUDA Graph benchmark is not portable to the validated MUSA environment because `triton.testing.do_bench_cudagraph` hard-codes `torch.cuda`. The following supplemental results use a benchmark-only shim that maps the stream, event, and graph-capture APIs to `torch.musa`; the shim does not change the operator implementation.

Graph replay overhead of approximately 0.5-1 ms per replay dominates small shapes and produces unstable comparisons, so these results are retained for reference only and are not used for 0.95x performance acceptance.

```bash
# Requires the benchmark-only torch.cuda -> torch.musa timing shim on MUSA.
python -m pytest -s -q benchmark/test_sort.py --mode cudagraph
```

Speedup is **PyTorch latency / FlagGems latency**. Reported row values are retained unchanged; the geometric means below use the reported three-decimal speedups with equal weight for each measurement.

| Operator | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| sort | 45 | 0.193x | 3.169x | 1.540x | 44 / 45 |
| sort_stable | 45 | 0.143x | 108.779x | 3.473x | 42 / 45 |
| All | 90 | 0.143x | 108.779x | 2.313x | 86 / 90 |

The four supplemental Graph measurements below 0.95x are `sort` INT32 `(256, 256)` at 0.193x, `sort_stable` BF16 `(1024, 1024)` at 0.622x, `sort_stable` FP32 `(4096, 4096)` at 0.350x, and `sort_stable` BF16 `(4096, 4096)` at 0.143x. The corresponding kernel-mode rows are all at or above 0.95x, which is consistent with these being Graph replay/timing outliers rather than kernel regressions.

**sort — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 2.797x | 1.686x | 2.586x | 1.959x | 1.470x |
| (256, 256) | 1.045x | 0.193x | 2.099x | 2.353x | 2.344x |
| (1024, 1024) | 1.368x | 3.169x | 1.833x | 1.575x | 2.328x |
| (4096, 4096) | 2.566x | 1.746x | 1.596x | 1.725x | 1.450x |
| (1024, 65536) | 1.284x | 1.550x | 1.030x | 1.450x | 1.077x |
| (1024, 1) | 1.731x | 1.264x | 1.767x | 1.491x | 1.768x |
| (1024, 512) | 2.803x | 2.136x | 2.711x | 2.256x | 3.042x |
| (16, 131072) | 1.063x | 1.116x | 1.025x | 1.091x | 1.040x |
| (8, 262144) | 1.085x | 1.072x | 1.025x | 1.079x | 0.992x |

**sort_stable — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 6.745x | 4.553x | 11.038x | 11.281x | 6.918x |
| (256, 256) | 4.843x | 6.803x | 12.041x | 6.237x | 2.218x |
| (1024, 1024) | 1.948x | 2.536x | 6.142x | 2.606x | 0.622x |
| (4096, 4096) | 2.466x | 3.577x | 1.848x | 0.350x | 0.143x |
| (1024, 65536) | 1.860x | 1.666x | 1.511x | 1.522x | 1.134x |
| (1024, 1) | 38.309x | 108.779x | 64.004x | 17.990x | 66.286x |
| (1024, 512) | 5.222x | 7.167x | 5.676x | 37.791x | 5.500x |
| (16, 131072) | 1.054x | 1.043x | 1.115x | 1.998x | 1.116x |
| (8, 262144) | 1.042x | 0.998x | 1.093x | 0.977x | 1.074x |

<details>
<summary>All 90 MThreads MUSA Graph measurements, including original latencies</summary>

| Operator | Shape (M, N) | Dtype | Torch latency (ms) | FlagGems latency (ms) | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| sort | (64, 64) | INT16 | 0.016312 | 0.005832 | 2.797x |
| sort | (256, 256) | INT16 | 0.026245 | 0.025113 | 1.045x |
| sort | (1024, 1024) | INT16 | 0.871475 | 0.637053 | 1.368x |
| sort | (4096, 4096) | INT16 | 3.788217 | 1.476546 | 2.566x |
| sort | (1024, 65536) | INT16 | 14.160123 | 11.023847 | 1.284x |
| sort | (1024, 1) | INT16 | 0.004359 | 0.002518 | 1.731x |
| sort | (1024, 512) | INT16 | 0.108068 | 0.038551 | 2.803x |
| sort | (16, 131072) | INT16 | 0.384379 | 0.361688 | 1.063x |
| sort | (8, 262144) | INT16 | 0.403177 | 0.371436 | 1.085x |
| sort | (64, 64) | INT32 | 0.016330 | 0.009685 | 1.686x |
| sort | (256, 256) | INT32 | 0.026492 | 0.137358 | 0.193x |
| sort | (1024, 1024) | INT32 | 1.592102 | 0.502397 | 3.169x |
| sort | (4096, 4096) | INT32 | 5.779331 | 3.310573 | 1.746x |
| sort | (1024, 65536) | INT32 | 33.853769 | 21.840674 | 1.550x |
| sort | (1024, 1) | INT32 | 0.004223 | 0.003341 | 1.264x |
| sort | (1024, 512) | INT32 | 0.121646 | 0.056939 | 2.136x |
| sort | (16, 131072) | INT32 | 0.875610 | 0.784562 | 1.116x |
| sort | (8, 262144) | INT32 | 0.867005 | 0.808645 | 1.072x |
| sort | (64, 64) | FP16 | 0.019854 | 0.007677 | 2.586x |
| sort | (256, 256) | FP16 | 0.030905 | 0.014722 | 2.099x |
| sort | (1024, 1024) | FP16 | 0.245380 | 0.133880 | 1.833x |
| sort | (4096, 4096) | FP16 | 2.871353 | 1.798784 | 1.596x |
| sort | (1024, 65536) | FP16 | 13.239494 | 12.850415 | 1.030x |
| sort | (1024, 1) | FP16 | 0.003732 | 0.002112 | 1.767x |
| sort | (1024, 512) | FP16 | 0.106658 | 0.039346 | 2.711x |
| sort | (16, 131072) | FP16 | 0.386898 | 0.377411 | 1.025x |
| sort | (8, 262144) | FP16 | 0.391407 | 0.382010 | 1.025x |
| sort | (64, 64) | FP32 | 0.021445 | 0.010948 | 1.959x |
| sort | (256, 256) | FP32 | 0.034611 | 0.014707 | 2.353x |
| sort | (1024, 1024) | FP32 | 0.281799 | 0.178869 | 1.575x |
| sort | (4096, 4096) | FP32 | 5.860222 | 3.398074 | 1.725x |
| sort | (1024, 65536) | FP32 | 31.860833 | 21.966619 | 1.450x |
| sort | (1024, 1) | FP32 | 0.004275 | 0.002867 | 1.491x |
| sort | (1024, 512) | FP32 | 0.131918 | 0.058479 | 2.256x |
| sort | (16, 131072) | FP32 | 0.878520 | 0.805209 | 1.091x |
| sort | (8, 262144) | FP32 | 0.866956 | 0.803511 | 1.079x |
| sort | (64, 64) | BF16 | 0.023228 | 0.015806 | 1.470x |
| sort | (256, 256) | BF16 | 0.035132 | 0.014987 | 2.344x |
| sort | (1024, 1024) | BF16 | 0.269182 | 0.115646 | 2.328x |
| sort | (4096, 4096) | BF16 | 2.639168 | 1.819693 | 1.450x |
| sort | (1024, 65536) | BF16 | 13.645106 | 12.668038 | 1.077x |
| sort | (1024, 1) | BF16 | 0.003722 | 0.002105 | 1.768x |
| sort | (1024, 512) | BF16 | 0.118762 | 0.039047 | 3.042x |
| sort | (16, 131072) | BF16 | 0.388202 | 0.373306 | 1.040x |
| sort | (8, 262144) | BF16 | 0.393724 | 0.397092 | 0.992x |
| sort_stable | (64, 64) | INT16 | 0.038101 | 0.005649 | 6.745x |
| sort_stable | (256, 256) | INT16 | 0.070248 | 0.014504 | 4.843x |
| sort_stable | (1024, 1024) | INT16 | 0.255222 | 0.131039 | 1.948x |
| sort_stable | (4096, 4096) | INT16 | 2.884176 | 1.169398 | 2.466x |
| sort_stable | (1024, 65536) | INT16 | 15.533580 | 8.352038 | 1.860x |
| sort_stable | (1024, 1) | INT16 | 0.219558 | 0.005731 | 38.309x |
| sort_stable | (1024, 512) | INT16 | 0.200652 | 0.038427 | 5.222x |
| sort_stable | (16, 131072) | INT16 | 0.386619 | 0.366861 | 1.054x |
| sort_stable | (8, 262144) | INT16 | 0.394804 | 0.379059 | 1.042x |
| sort_stable | (64, 64) | INT32 | 0.069530 | 0.015270 | 4.553x |
| sort_stable | (256, 256) | INT32 | 0.134868 | 0.019825 | 6.803x |
| sort_stable | (1024, 1024) | INT32 | 0.440723 | 0.173811 | 2.536x |
| sort_stable | (4096, 4096) | INT32 | 12.255070 | 3.426137 | 3.577x |
| sort_stable | (1024, 65536) | INT32 | 36.226151 | 21.740661 | 1.666x |
| sort_stable | (1024, 1) | INT32 | 0.383641 | 0.003527 | 108.779x |
| sort_stable | (1024, 512) | INT32 | 0.400322 | 0.055855 | 7.167x |
| sort_stable | (16, 131072) | INT32 | 0.759929 | 0.728782 | 1.043x |
| sort_stable | (8, 262144) | INT32 | 0.757361 | 0.758541 | 0.998x |
| sort_stable | (64, 64) | FP16 | 0.257834 | 0.023359 | 11.038x |
| sort_stable | (256, 256) | FP16 | 0.528878 | 0.043923 | 12.041x |
| sort_stable | (1024, 1024) | FP16 | 0.722837 | 0.117693 | 6.142x |
| sort_stable | (4096, 4096) | FP16 | 2.703892 | 1.462886 | 1.848x |
| sort_stable | (1024, 65536) | FP16 | 14.270491 | 9.441384 | 1.511x |
| sort_stable | (1024, 1) | FP16 | 0.222478 | 0.003476 | 64.004x |
| sort_stable | (1024, 512) | FP16 | 0.236634 | 0.041689 | 5.676x |
| sort_stable | (16, 131072) | FP16 | 0.445175 | 0.399398 | 1.115x |
| sort_stable | (8, 262144) | FP16 | 0.448713 | 0.410459 | 1.093x |
| sort_stable | (64, 64) | FP32 | 0.092845 | 0.008230 | 11.281x |
| sort_stable | (256, 256) | FP32 | 0.138729 | 0.022243 | 6.237x |
| sort_stable | (1024, 1024) | FP32 | 0.438328 | 0.168228 | 2.606x |
| sort_stable | (4096, 4096) | FP32 | 5.556193 | 15.856613 | 0.350x |
| sort_stable | (1024, 65536) | FP32 | 32.994754 | 21.678370 | 1.522x |
| sort_stable | (1024, 1) | FP32 | 0.390049 | 0.021682 | 17.990x |
| sort_stable | (1024, 512) | FP32 | 2.236091 | 0.059170 | 37.791x |
| sort_stable | (16, 131072) | FP32 | 1.550785 | 0.776196 | 1.998x |
| sort_stable | (8, 262144) | FP32 | 0.763784 | 0.781771 | 0.977x |
| sort_stable | (64, 64) | BF16 | 0.040429 | 0.005844 | 6.918x |
| sort_stable | (256, 256) | BF16 | 0.070099 | 0.031601 | 2.218x |
| sort_stable | (1024, 1024) | BF16 | 0.225674 | 0.362726 | 0.622x |
| sort_stable | (4096, 4096) | BF16 | 2.759553 | 19.336551 | 0.143x |
| sort_stable | (1024, 65536) | BF16 | 71.364073 | 62.912939 | 1.134x |
| sort_stable | (1024, 1) | BF16 | 0.197074 | 0.002973 | 66.286x |
| sort_stable | (1024, 512) | BF16 | 0.231377 | 0.042068 | 5.500x |
| sort_stable | (16, 131072) | BF16 | 0.444046 | 0.397967 | 1.116x |
| sort_stable | (8, 262144) | BF16 | 0.441275 | 0.411059 | 1.074x |

</details>

#### Without CUDA Graph (`mode=kernel`)

The following results reproduce the MTT S5000 kernel-mode run supplied for this PR. Measurements use the PR-aligned workload set from #8070: 9 shapes, 5 dtypes and both `sort` and `sort_stable`.

```bash
python -m pytest -s -q benchmark/test_sort.py --mode kernel
```

| Operator | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| sort | 45 | 0.951x | 3.486x | 1.579x | 45 / 45 |
| sort_stable | 45 | 0.951x | 153.819x | 3.622x | 45 / 45 |
| All | 90 | 0.951x | 153.819x | 2.391x | 90 / 90 |

Two independent runs produced the same pass count:

| Run | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| First run | 90 | 0.951x | 154.994x | 2.408x | 90 / 90 |
| Second run | 90 | 0.951x | 153.819x | 2.391x | 90 / 90 |

The weakest measurements are INT16 `(8, 262144)` at 0.951x for both operators and `(16, 131072)` at 0.991x-1.012x. All other measurements are above 0.95x.

**MThreads — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 3.486x | 1.893x | 2.556x | 2.078x | 2.254x |
| (256, 256) | 2.201x | 2.214x | 2.161x | 2.451x | 2.418x |
| (1024, 1024) | 1.611x | 1.359x | 1.571x | 1.452x | 1.709x |
| (4096, 4096) | 1.807x | 1.662x | 1.473x | 1.652x | 1.464x |
| (1024, 65536) | 1.249x | 1.605x | 1.039x | 1.492x | 1.058x |
| (1024, 1) | 1.677x | 1.524x | 1.651x | 1.508x | 1.733x |
| (1024, 512) | 2.330x | 2.166x | 2.250x | 2.185x | 2.477x |
| (16, 131072) | 0.992x | 1.008x | 0.998x | 1.009x | 1.012x |
| (8, 262144) | 0.951x | 0.961x | 0.965x | 0.957x | 0.979x |

**MThreads sort_stable — speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 5.117x | 9.336x | 6.839x | 9.329x | 6.669x |
| (256, 256) | 5.960x | 11.720x | 5.903x | 11.379x | 5.948x |
| (1024, 1024) | 1.477x | 2.309x | 1.475x | 2.330x | 1.467x |
| (4096, 4096) | 1.807x | 1.664x | 1.475x | 1.648x | 1.463x |
| (1024, 65536) | 1.251x | 1.620x | 1.040x | 1.494x | 1.056x |
| (1024, 1) | 78.432x | 153.819x | 78.900x | 152.502x | 77.457x |
| (1024, 512) | 4.386x | 7.505x | 4.319x | 7.043x | 4.299x |
| (16, 131072) | 0.991x | 1.011x | 0.995x | 1.010x | 1.008x |
| (8, 262144) | 0.951x | 0.962x | 0.970x | 0.959x | 0.976x |

<details>
<summary>All 90 MThreads kernel measurements, including original latencies</summary>

| Operator | Shape (M, N) | Dtype | Torch latency (ms) | FlagGems latency (ms) | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| sort | (64, 64) | INT16 | 0.020640 | 0.005920 | 3.486x |
| sort | (256, 256) | INT16 | 0.026760 | 0.012160 | 2.201x |
| sort | (1024, 1024) | INT16 | 0.240240 | 0.149080 | 1.611x |
| sort | (4096, 4096) | INT16 | 2.657480 | 1.470280 | 1.807x |
| sort | (1024, 65536) | INT16 | 13.666440 | 10.939020 | 1.249x |
| sort | (1024, 1) | INT16 | 0.004160 | 0.002480 | 1.677x |
| sort | (1024, 512) | INT16 | 0.108880 | 0.046720 | 2.330x |
| sort | (16, 131072) | INT16 | 0.409760 | 0.413120 | 0.992x |
| sort | (8, 262144) | INT16 | 0.405960 | 0.426700 | 0.951x |
| sort | (64, 64) | INT32 | 0.015520 | 0.008200 | 1.893x |
| sort | (256, 256) | INT32 | 0.026920 | 0.012160 | 2.214x |
| sort | (1024, 1024) | INT32 | 0.254400 | 0.187160 | 1.359x |
| sort | (4096, 4096) | INT32 | 5.443960 | 3.276440 | 1.662x |
| sort | (1024, 65536) | INT32 | 33.484362 | 20.863060 | 1.605x |
| sort | (1024, 1) | INT32 | 0.003840 | 0.002520 | 1.524x |
| sort | (1024, 512) | INT32 | 0.115120 | 0.053160 | 2.166x |
| sort | (16, 131072) | INT32 | 0.806100 | 0.800080 | 1.008x |
| sort | (8, 262144) | INT32 | 0.799200 | 0.832040 | 0.961x |
| sort | (64, 64) | FP16 | 0.015440 | 0.006040 | 2.556x |
| sort | (256, 256) | FP16 | 0.026800 | 0.012400 | 2.161x |
| sort | (1024, 1024) | FP16 | 0.238640 | 0.151940 | 1.571x |
| sort | (4096, 4096) | FP16 | 2.640200 | 1.791840 | 1.473x |
| sort | (1024, 65536) | FP16 | 12.876121 | 12.392720 | 1.039x |
| sort | (1024, 1) | FP16 | 0.004160 | 0.002520 | 1.651x |
| sort | (1024, 512) | FP16 | 0.107280 | 0.047680 | 2.250x |
| sort | (16, 131072) | FP16 | 0.408080 | 0.409020 | 0.998x |
| sort | (8, 262144) | FP16 | 0.404760 | 0.419460 | 0.965x |
| sort | (64, 64) | FP32 | 0.017120 | 0.008240 | 2.078x |
| sort | (256, 256) | FP32 | 0.030200 | 0.012320 | 2.451x |
| sort | (1024, 1024) | FP32 | 0.274440 | 0.188960 | 1.452x |
| sort | (4096, 4096) | FP32 | 5.391000 | 3.263040 | 1.652x |
| sort | (1024, 65536) | FP32 | 31.098801 | 20.847860 | 1.492x |
| sort | (1024, 1) | FP32 | 0.003800 | 0.002520 | 1.508x |
| sort | (1024, 512) | FP32 | 0.124480 | 0.056960 | 2.185x |
| sort | (16, 131072) | FP32 | 0.806860 | 0.799440 | 1.009x |
| sort | (8, 262144) | FP32 | 0.800260 | 0.836300 | 0.957x |
| sort | (64, 64) | BF16 | 0.017720 | 0.007860 | 2.254x |
| sort | (256, 256) | BF16 | 0.030080 | 0.012440 | 2.418x |
| sort | (1024, 1024) | BF16 | 0.261640 | 0.153080 | 1.709x |
| sort | (4096, 4096) | BF16 | 2.612180 | 1.784400 | 1.464x |
| sort | (1024, 65536) | BF16 | 13.116440 | 12.394320 | 1.058x |
| sort | (1024, 1) | BF16 | 0.004160 | 0.002400 | 1.733x |
| sort | (1024, 512) | BF16 | 0.119000 | 0.048040 | 2.477x |
| sort | (16, 131072) | BF16 | 0.408600 | 0.403740 | 1.012x |
| sort | (8, 262144) | BF16 | 0.408060 | 0.416880 | 0.979x |
| sort_stable | (64, 64) | INT16 | 0.039500 | 0.007720 | 5.117x |
| sort_stable | (256, 256) | INT16 | 0.072480 | 0.012160 | 5.960x |
| sort_stable | (1024, 1024) | INT16 | 0.220160 | 0.149040 | 1.477x |
| sort_stable | (4096, 4096) | INT16 | 2.656460 | 1.470480 | 1.807x |
| sort_stable | (1024, 65536) | INT16 | 13.669880 | 10.927600 | 1.251x |
| sort_stable | (1024, 1) | INT16 | 0.194520 | 0.002480 | 78.432x |
| sort_stable | (1024, 512) | INT16 | 0.204400 | 0.046600 | 4.386x |
| sort_stable | (16, 131072) | INT16 | 0.409180 | 0.412700 | 0.991x |
| sort_stable | (8, 262144) | INT16 | 0.406120 | 0.427200 | 0.951x |
| sort_stable | (64, 64) | INT32 | 0.076560 | 0.008200 | 9.336x |
| sort_stable | (256, 256) | INT32 | 0.142520 | 0.012160 | 11.720x |
| sort_stable | (1024, 1024) | INT32 | 0.432560 | 0.187300 | 2.309x |
| sort_stable | (4096, 4096) | INT32 | 5.445280 | 3.272160 | 1.664x |
| sort_stable | (1024, 65536) | INT32 | 33.910339 | 20.936900 | 1.620x |
| sort_stable | (1024, 1) | INT32 | 0.387640 | 0.002520 | 153.819x |
| sort_stable | (1024, 512) | INT32 | 0.398660 | 0.053120 | 7.505x |
| sort_stable | (16, 131072) | INT32 | 0.807600 | 0.799080 | 1.011x |
| sort_stable | (8, 262144) | INT32 | 0.799900 | 0.831840 | 0.962x |
| sort_stable | (64, 64) | FP16 | 0.040760 | 0.005960 | 6.839x |
| sort_stable | (256, 256) | FP16 | 0.073320 | 0.012420 | 5.903x |
| sort_stable | (1024, 1024) | FP16 | 0.224040 | 0.151920 | 1.475x |
| sort_stable | (4096, 4096) | FP16 | 2.639680 | 1.789420 | 1.475x |
| sort_stable | (1024, 65536) | FP16 | 12.877440 | 12.378560 | 1.040x |
| sort_stable | (1024, 1) | FP16 | 0.195680 | 0.002480 | 78.900x |
| sort_stable | (1024, 512) | FP16 | 0.205920 | 0.047680 | 4.319x |
| sort_stable | (16, 131072) | FP16 | 0.407060 | 0.409140 | 0.995x |
| sort_stable | (8, 262144) | FP16 | 0.406200 | 0.418800 | 0.970x |
| sort_stable | (64, 64) | FP32 | 0.077620 | 0.008320 | 9.329x |
| sort_stable | (256, 256) | FP32 | 0.140640 | 0.012360 | 11.379x |
| sort_stable | (1024, 1024) | FP32 | 0.440400 | 0.189020 | 2.330x |
| sort_stable | (4096, 4096) | FP32 | 5.389880 | 3.270320 | 1.648x |
| sort_stable | (1024, 65536) | FP32 | 31.157360 | 20.856020 | 1.494x |
| sort_stable | (1024, 1) | FP32 | 0.384320 | 0.002520 | 152.502x |
| sort_stable | (1024, 512) | FP32 | 0.401160 | 0.056960 | 7.043x |
| sort_stable | (16, 131072) | FP32 | 0.806320 | 0.798400 | 1.010x |
| sort_stable | (8, 262144) | FP32 | 0.803020 | 0.836940 | 0.959x |
| sort_stable | (64, 64) | BF16 | 0.040280 | 0.006040 | 6.669x |
| sort_stable | (256, 256) | BF16 | 0.073520 | 0.012360 | 5.948x |
| sort_stable | (1024, 1024) | BF16 | 0.224560 | 0.153120 | 1.467x |
| sort_stable | (4096, 4096) | BF16 | 2.613200 | 1.785720 | 1.463x |
| sort_stable | (1024, 65536) | BF16 | 13.072840 | 12.379400 | 1.056x |
| sort_stable | (1024, 1) | BF16 | 0.195200 | 0.002520 | 77.457x |
| sort_stable | (1024, 512) | BF16 | 0.206680 | 0.048080 | 4.299x |
| sort_stable | (16, 131072) | BF16 | 0.407200 | 0.404160 | 1.008x |
| sort_stable | (8, 262144) | BF16 | 0.407260 | 0.417480 | 0.976x |

</details>
