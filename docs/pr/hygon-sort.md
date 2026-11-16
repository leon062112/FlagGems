### PR Category

Operator

### Type of Change

Performance Optimization

### Description

Adds Hygon backend implementations of `sort` and `sort_stable`, with dispatch based on row length, row count, packed-key width and the input device's SM count.

- Use row sorting networks for short rows, packed-key tiled merge for larger rows, and a two-pass 16-bit radix path for selected long-row workloads. FP16/BF16/INT16 share the same dispatch criteria.
- Use single-warp initial runs for dense grids and stage-specific merge tiles. Skip merge-path partition searches when a program handles an entire pair of runs.
- Recover final values directly from packed keys where possible, preserving original signed-zero bits through a masked input load. Stable tie-breaking keeps original indices.
- Use a wider 2048-element initial run for long 64-bit-key rows, keep a dedicated second scratch buffer, reuse the final index buffer as the alternate key buffer, and merge the first complete run pair per CTA when there are at most 16 rows.
- Support strided and non-last sort dimensions with tiled copies, and use the existing `libentry`, Triton heuristics and JIT launch mechanisms.
- Keep the existing public API and retain the generic implementation for unpackable 64-bit values and other unsupported layouts.

### Issue

N/A

### Progress

- [ ] Change is properly reviewed (1 reviewer required, 2 recommended).
- [x] Correctness checks pass on Hygon BW: 90/90 PR-aligned cases.
- [x] Performance is measured against PyTorch.

### Performance

Env: Hygon BW (80 SM), PyTorch 2.4.1+das.opt1.dtk25041, Triton 3.6.0+dtk2604.

#### CUDA Graph (`mode=cudagraph`)

The following tables use CUDA Graph replay. Measurements use the PR-aligned workload set from #8070: 9 shapes, 5 dtypes and both `sort` and `sort_stable`.

```bash
python -m pytest -s -q benchmark/test_sort.py --mode cudagraph
```

Speedup is **PyTorch latency / FlagGems latency**. Reported row values are retained unchanged; the geometric means below use the reported three-decimal speedups, with equal weight for each measurement.

| Operator | Measurements | Min | Max | Geometric mean | Speedup >= 0.95x |
| --- | ---: | ---: | ---: | ---: | ---: |
| sort | 45 | 0.984x | 42.807x | 3.815x | 45 / 45 |
| sort_stable | 45 | 0.981x | 46.321x | 4.415x | 45 / 45 |
| All | 90 | 0.981x | 46.321x | 4.104x | 90 / 90 |

The FP32 `sort` rows `(16, 131072)` and `(8, 262144)` use the median of three isolated reruns, reported as 1.111x and 0.984x. The complete shared-node traversal showed transient low outliers on these two rows; all other values are from the complete 90-row run. The corresponding `sort_stable` rows measure 1.102x and 0.981x.

**sort - speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 8.399x | 9.108x | 10.558x | 9.477x | 10.716x |
| (256, 256) | 9.517x | 10.147x | 10.660x | 11.606x | 42.807x |
| (1024, 1024) | 2.453x | 2.114x | 5.486x | 4.294x | 5.364x |
| (4096, 4096) | 3.626x | 3.301x | 3.778x | 3.920x | 3.768x |
| (1024, 65536) | 1.877x | 2.281x | 1.965x | 2.374x | 2.021x |
| (1024, 1) | 3.927x | 5.927x | 6.497x | 6.935x | 6.732x |
| (1024, 512) | 4.418x | 4.329x | 11.108x | 5.020x | 4.355x |
| (16, 131072) | 1.361x | 1.186x | 1.247x | 1.111x | 1.308x |
| (8, 262144) | 1.304x | 1.067x | 1.196x | 0.984x | 1.258x |

**sort_stable - speedup vs. PyTorch**

| Shape (M, N) | INT16 | INT32 | FP16 | FP32 | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (64, 64) | 11.622x | 12.588x | 11.199x | 12.772x | 9.438x |
| (256, 256) | 46.048x | 38.802x | 46.321x | 40.058x | 9.864x |
| (1024, 1024) | 2.447x | 4.142x | 2.404x | 2.345x | 2.398x |
| (4096, 4096) | 3.872x | 3.451x | 3.521x | 3.750x | 3.783x |
| (1024, 65536) | 1.877x | 2.280x | 1.966x | 2.376x | 1.993x |
| (1024, 1) | 6.900x | 6.736x | 6.801x | 7.056x | 6.876x |
| (1024, 512) | 4.506x | 4.338x | 11.175x | 16.121x | 11.359x |
| (16, 131072) | 1.479x | 1.196x | 1.313x | 1.102x | 1.234x |
| (8, 262144) | 1.428x | 1.077x | 1.289x | 0.981x | 1.184x |

<details>
<summary>All 45 sort measurements, including original latencies</summary>

| Shape (M, N) | Dtype | Torch latency (ms) | FlagGems latency (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| (64, 64) | INT16 | 0.024921 | 0.002967 | 8.399x |
| (64, 64) | INT32 | 0.029844 | 0.003277 | 9.108x |
| (64, 64) | FP16 | 0.032730 | 0.003100 | 10.558x |
| (64, 64) | FP32 | 0.030419 | 0.003210 | 9.477x |
| (64, 64) | BF16 | 0.032515 | 0.003034 | 10.716x |
| (256, 256) | INT16 | 0.061367 | 0.006448 | 9.517x |
| (256, 256) | INT32 | 0.082057 | 0.008086 | 10.147x |
| (256, 256) | FP16 | 0.070591 | 0.006622 | 10.660x |
| (256, 256) | FP32 | 0.094931 | 0.008179 | 11.606x |
| (256, 256) | BF16 | 0.285124 | 0.006661 | 42.807x |
| (1024, 1024) | INT16 | 0.181416 | 0.073942 | 2.453x |
| (1024, 1024) | INT32 | 0.239087 | 0.113123 | 2.114x |
| (1024, 1024) | FP16 | 0.413994 | 0.075466 | 5.486x |
| (1024, 1024) | FP32 | 0.502190 | 0.116949 | 4.294x |
| (1024, 1024) | BF16 | 0.406241 | 0.075736 | 5.364x |
| (4096, 4096) | INT16 | 3.333577 | 0.919255 | 3.626x |
| (4096, 4096) | INT32 | 5.039239 | 1.526575 | 3.301x |
| (4096, 4096) | FP16 | 3.566589 | 0.944005 | 3.778x |
| (4096, 4096) | FP32 | 6.074357 | 1.549742 | 3.920x |
| (4096, 4096) | BF16 | 3.565319 | 0.946275 | 3.768x |
| (1024, 65536) | INT16 | 13.057511 | 6.957075 | 1.877x |
| (1024, 65536) | INT32 | 23.897223 | 10.477842 | 2.281x |
| (1024, 65536) | FP16 | 13.919465 | 7.082211 | 1.965x |
| (1024, 65536) | FP32 | 24.994365 | 10.527710 | 2.374x |
| (1024, 65536) | BF16 | 14.340139 | 7.094230 | 2.021x |
| (1024, 1) | INT16 | 0.008631 | 0.002198 | 3.927x |
| (1024, 1) | INT32 | 0.013882 | 0.002342 | 5.927x |
| (1024, 1) | FP16 | 0.014791 | 0.002277 | 6.497x |
| (1024, 1) | FP32 | 0.015786 | 0.002276 | 6.935x |
| (1024, 1) | BF16 | 0.015558 | 0.002311 | 6.732x |
| (1024, 512) | INT16 | 0.146692 | 0.033203 | 4.418x |
| (1024, 512) | INT32 | 0.205998 | 0.047588 | 4.329x |
| (1024, 512) | FP16 | 0.377941 | 0.034025 | 11.108x |
| (1024, 512) | FP32 | 0.241796 | 0.048164 | 5.020x |
| (1024, 512) | BF16 | 0.148978 | 0.034205 | 4.355x |
| (16, 131072) | INT16 | 0.424559 | 0.312019 | 1.361x |
| (16, 131072) | INT32 | 0.604528 | 0.509594 | 1.186x |
| (16, 131072) | FP16 | 0.380929 | 0.305510 | 1.247x |
| (16, 131072) | FP32 | 0.567389 | 0.510746 | 1.111x |
| (16, 131072) | BF16 | 0.397829 | 0.304193 | 1.308x |
| (8, 262144) | INT16 | 0.413013 | 0.316840 | 1.304x |
| (8, 262144) | INT32 | 0.590175 | 0.553242 | 1.067x |
| (8, 262144) | FP16 | 0.372875 | 0.311882 | 1.196x |
| (8, 262144) | FP32 | 0.552355 | 0.561270 | 0.984x |
| (8, 262144) | BF16 | 0.388567 | 0.308972 | 1.258x |

</details>

<details>
<summary>All 45 sort_stable measurements, including original latencies</summary>

| Shape (M, N) | Dtype | Torch latency (ms) | FlagGems latency (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| (64, 64) | INT16 | 0.033563 | 0.002888 | 11.622x |
| (64, 64) | INT32 | 0.039572 | 0.003144 | 12.588x |
| (64, 64) | FP16 | 0.033707 | 0.003010 | 11.199x |
| (64, 64) | FP32 | 0.040415 | 0.003164 | 12.772x |
| (64, 64) | BF16 | 0.028098 | 0.002977 | 9.438x |
| (256, 256) | INT16 | 0.294019 | 0.006385 | 46.048x |
| (256, 256) | INT32 | 0.311935 | 0.008039 | 38.802x |
| (256, 256) | FP16 | 0.301315 | 0.006505 | 46.321x |
| (256, 256) | FP32 | 0.322679 | 0.008055 | 40.058x |
| (256, 256) | BF16 | 0.064685 | 0.006557 | 9.864x |
| (1024, 1024) | INT16 | 0.180718 | 0.073867 | 2.447x |
| (1024, 1024) | INT32 | 0.472399 | 0.114048 | 4.142x |
| (1024, 1024) | FP16 | 0.181256 | 0.075382 | 2.404x |
| (1024, 1024) | FP32 | 0.267065 | 0.113890 | 2.345x |
| (1024, 1024) | BF16 | 0.181371 | 0.075634 | 2.398x |
| (4096, 4096) | INT16 | 3.553524 | 0.917657 | 3.872x |
| (4096, 4096) | INT32 | 5.269509 | 1.526872 | 3.451x |
| (4096, 4096) | FP16 | 3.332531 | 0.946348 | 3.521x |
| (4096, 4096) | FP32 | 5.801827 | 1.547058 | 3.750x |
| (4096, 4096) | BF16 | 3.573364 | 0.944509 | 3.783x |
| (1024, 65536) | INT16 | 13.068759 | 6.961213 | 1.877x |
| (1024, 65536) | INT32 | 23.897772 | 10.479568 | 2.280x |
| (1024, 65536) | FP16 | 13.922346 | 7.080967 | 1.966x |
| (1024, 65536) | FP32 | 25.010326 | 10.524768 | 2.376x |
| (1024, 65536) | BF16 | 14.331685 | 7.189287 | 1.993x |
| (1024, 1) | INT16 | 0.015827 | 0.002294 | 6.900x |
| (1024, 1) | INT32 | 0.016214 | 0.002407 | 6.736x |
| (1024, 1) | FP16 | 0.015105 | 0.002221 | 6.801x |
| (1024, 1) | FP32 | 0.016435 | 0.002329 | 7.056x |
| (1024, 1) | BF16 | 0.016339 | 0.002376 | 6.876x |
| (1024, 512) | INT16 | 0.149516 | 0.033182 | 4.506x |
| (1024, 512) | INT32 | 0.205699 | 0.047422 | 4.338x |
| (1024, 512) | FP16 | 0.378906 | 0.033907 | 11.175x |
| (1024, 512) | FP32 | 0.475801 | 0.029514 | 16.121x |
| (1024, 512) | BF16 | 0.387049 | 0.034073 | 11.359x |
| (16, 131072) | INT16 | 0.462631 | 0.312803 | 1.479x |
| (16, 131072) | INT32 | 0.609420 | 0.509483 | 1.196x |
| (16, 131072) | FP16 | 0.401340 | 0.305554 | 1.313x |
| (16, 131072) | FP32 | 0.562500 | 0.510551 | 1.102x |
| (16, 131072) | BF16 | 0.374969 | 0.303834 | 1.234x |
| (8, 262144) | INT16 | 0.451384 | 0.316172 | 1.428x |
| (8, 262144) | INT32 | 0.596259 | 0.553493 | 1.077x |
| (8, 262144) | FP16 | 0.401605 | 0.311578 | 1.289x |
| (8, 262144) | FP32 | 0.551547 | 0.561989 | 0.981x |
| (8, 262144) | BF16 | 0.365899 | 0.309116 | 1.184x |

</details>
