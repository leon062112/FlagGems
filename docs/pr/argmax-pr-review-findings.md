# Argmax PR 静态审查结论

## 审查范围

- 代码分支：`origin/argmax`，最新提交 `62eb8f57b` (`perf(mthreads): optimize contiguous argmax reductions`)。
- 文档分支：`origin/docs/argmax-pr-description`，最新提交 `e1c5bf64f` (`docs: align Hygon argmax implementation description`)。
- 审查方式：仅做静态代码、文档和文件结构检查；本次未运行 kernel 或 pytest。
- 说明：当前工作区检出的文档分支不包含 `origin/argmax` 的 MetaX `713cc06d5` 和 MThreads `62eb8f57b` 最新改动，因此代码结论均以 `origin/argmax` 为准。

## 结论

当前状态不建议直接开 PR。文件结构和注册关系基本正确，但存在以下需要先处理的问题：

1. PR 文档中的 Hygon 实现描述与实际代码明显不一致。
2. MetaX 缺少 `dim` 边界校验，并在空 reduction 维度上可能触发内部除零。
3. MThreads、MetaX 部分路径以及 Hygon 全局路径的 NaN 语义与 PyTorch 不一致，需要在文档中限定范围或补齐实现。
4. 新增路径缺少针对 NaN、非连续、非法 `dim`、0 维和空 reduction 维度的回归测试。

## 必须修

### 1. 修正 PR 文档中的 Hygon 描述

文件：`docs/pr/argmax-multi-backend.md:14`

最新描述（`e1c5bf64f`）仍然包含以下与 `origin/argmax` 代码不符的内容：

- “Contiguous global reductions up to 2^24 elements use a two-stage reduction that packs each partial key and index into one `uint64` payload”：Hygon 全局路径实际调用通用 `argmax_kernel_1` / `argmax_kernel_2`，没有 `uint64` 打包。
- “`N <= 4096` uses a single-pass kernel and larger rows stream over bounded 4096- or 8192-element blocks”：Hygon 只为连续最后一维、`k == 1`、`n >= 4096`、`m >= 64` 的输入使用 `argmax_lastdim_kernel`；`n < 4096` 会回退通用实现。代码中没有 `N <= 4096` 单遍 kernel。
- “Non-innermost reductions use an output-block kernel when `N <= 32` and the output has at most 2^20 elements”：Hygon 没有 output-block kernel；非最后一维直接回退通用实现。
- “Size-1 reductions zero the full output with a cached Triton kernel up to 2^16 elements (`torch.zero_` above that)”：代码中的 `argmax_zero_kernel` 对所有 size-1 情况都直接启动普通 `@triton.jit` kernel，没有 2^16 阈值，也没有 `torch.zero_` 分支。
- “import-time prewarming covers the zero-fill and global-reduction kernels”：Hygon 模块没有 import 时预热逻辑。
- “preserving first-NaN and lowest-index tie semantics”：该语义只由 `argmax_lastdim_kernel` 实现；全局路径和通用回退路径不保证 first-NaN。

实际实现可参考 `src/flag_gems/runtime/backend/_hygon/ops/argmax.py:96`、`:138`、`:42`、`:59`。建议将 Hygon 描述改为类似：

> Hygon specializes contiguous last-dimension reductions with `k == 1`, `n >= 4096`, and `m >= 64`. FP16/BF16 values are promoted to FP32, `N` is streamed in bounded blocks, and the kernel tracks the first NaN and uses leftmost tie-breaking. Other layouts and dtypes fall back to the generic implementation. Size-1 reductions use a dedicated zero-fill kernel. Contiguous global reductions up to `2^24` elements use the generic two-stage kernels.

### 2. 补齐 MetaX 的 `dim` 校验和 0 维输入处理

文件：`src/flag_gems/runtime/backend/_metax/ops/argmax.py`（`origin/argmax` 版本）

- `:512`、`:528`、`:540` 在 `dim %= inp.ndim` 前没有做范围检查。`argmax(x, dim=2)` 在 2D 输入上会静默变成 `dim=0`，而 PyTorch 应抛出 `IndexError`。
- 0 维输入且 `dim=0` 时会执行 `dim %= 0`，在 `:540` 触发 `ZeroDivisionError`。PyTorch 接受 0 维输入的 `dim=0`，Thead/PPU 实现也已经显式支持该情况。
- 建议在进入任何取模或 `shape[dim]` 访问前统一校验：0 维只允许 `dim=0/-1`，其余维度要求 `-ndim <= dim < ndim`，非法值抛 `IndexError`。

### 3. 处理空 reduction 维度

文件：`src/flag_gems/runtime/backend/_metax/ops/argmax.py`（`origin/argmax` 版本）

例如 shape `(3, 0, 4)`、`dim=1` 时，输出非空但 `n=0`，代码会进入 column 路径。`tile_n` 可能为 0，随后 `chunk`/`split` 计算会出现 `triton.cdiv(0, 0)`（`:631`）。PyTorch 对这种输入应给出明确错误，建议在入口对 `n == 0` 直接报错，不要进入 kernel 选择逻辑。

### 4. 明确或修复 NaN 语义

PyTorch 的 `argmax` 将 NaN 视为最大值，并返回第一个 NaN 的索引。当前实现情况：

- MThreads：`_argmax_contiguous_kernel`、`_argmax_block_kernel`、`_argmax_block_native_kernel` 直接使用 `tl.max(..., return_indices=True)`，没有 NaN 分支。对应 `origin/argmax` 的 `src/flag_gems/runtime/backend/_mthreads/ops/argmax.py:95`、`:140`、`:189`。
- MetaX：专用 `argmax_pair` / `argmax_reduce` 是 NaN-aware 的，但小 non-inner fast path 在 `:532` 回退通用实现，通用路径不保证 first-NaN。
- Hygon：`argmax_lastdim_kernel` 显式处理 first-NaN，但全局路径和通用回退路径不保证。

如果 NaN 属于本 PR 的验收范围，这是 blocker。若本轮只要求普通数值精度，建议在文档中明确写出“只有专用 last-dim/pair kernel 保证 first-NaN，通用回退路径继承 generic 行为”，不要笼统声明所有路径都保持该语义。

### 5. 确认宿主函数调用约束

按仓库 `AGENTS.md` 的要求，宿主函数不应调用 PyTorch compute op，只能做分配/初始化等操作。以下代码直接调用了 `.contiguous()`：

- MetaX：`origin/argmax` 的 `src/flag_gems/runtime/backend/_metax/ops/argmax.py:577`。
- MThreads：`origin/argmax` 的 `src/flag_gems/runtime/backend/_mthreads/ops/argmax.py:332`、`:337`、`:366`。

这不是数值正确性问题，但属于明确的代码规范风险。建议改为专用 copy kernel，或把非连续输入限制为明确的 fallback，并在 PR 中说明原因。

## 非阻塞建议

- Hygon 新文件缺少 Apache license header：`src/flag_gems/runtime/backend/_hygon/ops/argmax.py:1`。其他三个新增后端文件和相邻 Hygon 算子都有该 header。
- Hygon 的 `argmax_zero_kernel` 没有 `@libentry`。如果保留“cached kernel”描述，应补装饰器；否则文档应改为普通 JIT zero-fill kernel。
- `tests/test_argmax.py` 和 `benchmark/test_argmax.py` 均未修改。现有 126 个 case 只覆盖 shape/dtype/keepdim 组合，缺少 NaN、非连续 view、非法 `dim`、0 维、空 reduction 维度、整数/布尔 dtype 的回归测试。
- MThreads graph 结果依赖临时 `torch.cuda` 到 `torch.musa` 的 timing shim，但仓库中没有该 shim。建议在文档中给出 shim 路径，或说明这是临时环境方案，避免性能结果无法复现。
- 文档目前只有 MThreads 标注了代码提交 `62eb8f57b`；建议 MetaX、Hygon、PPU 的指标也标注对应代码提交，便于追溯。

## 已确认通过

- 四个后端均已在各自 `ops/__init__.py` 中注册 `argmax`，并加入 `__all__`。
- `conf/operators.yaml` 中已有 `argmax` 元数据，无需新增。
- `git diff --check` 在代码分支和文档分支均通过。
- Thead/PPU 的 `_argmax_pair` 明确处理 first-NaN 和最低索引，0 维、非法 `dim`、空输入也有显式分支，是四个实现中最完整的。
- MThreads 文档中的实现描述基本准确：contiguous row-per-CTA、non-inner output-block、大 BF16 native comparison、8192 全局 tile 都能在代码中找到对应路径。
- 文档中的 benchmark 表与汇总指标内部一致，51 个 case 的最小值均不低于 0.95x（按文档记录）。

## 建议合并顺序

1. 修正 Hygon 文档描述，确保 PR 描述与代码一致。
2. 修复 MetaX `dim` 校验、0 维输入和空 reduction 维度处理。
3. 决定 NaN 语义范围：要么补齐 MThreads/MetaX/Hygon 回退路径，要么在文档中明确限定。
4. 增加边界和特殊值回归测试。
5. 处理 Hygon license header、`.contiguous()` 规范和 MThreads shim 可复现性。
6. 更新文档中的代码提交哈希并重新记录对应性能结果。
