# Argmax PR 静态审查结论

## 审查范围

- 代码分支：`origin/argmax`，最新提交 `d8ec47f6` (`fix(hygon): align argmax edge-case behavior with PyTorch`)。
- 文档分支：`origin/docs/argmax-pr-description`，本次更新后包含最新的 Hygon 实现说明和审查结论。
- 审查方式：静态代码、文档和文件结构检查，加上 Hygon 边界场景手工验证；按要求未修改 `tests/test_argmax.py` 或 `benchmark/test_argmax.py`。
- 说明：Hygon 文档、头文件和边界行为已更新；下列仍待处理项主要针对其他后端和未修改的回归测试覆盖。

## 结论

Hygon 的代码、Apache header 和 PR 描述已经对齐，主要快路径与 51 项性能结果保持不变。整个 PR 仍有以下待处理项：

1. MetaX 缺少 `dim` 边界校验，并在空 reduction 维度上可能触发内部除零。
2. MThreads 和 MetaX 的部分 NaN 语义与 PyTorch 不一致。
3. 新增路径缺少针对 NaN、非连续、非法 `dim`、0 维和空 reduction 维度的回归测试；本轮按要求未修改测试文件。

## 必须修

### 1. Hygon（已处理）

- PR 描述与当前实现一致：packed `uint64` global reduction、short/long innermost dispatch、small non-inner kernel、size-1 zero-fill、import-time prewarm 均能在代码中找到对应路径。
- Apache license header 已补齐。
- 标量 `dim=0/-1`、非法 `dim`、空 reduction 维和 size-1 大输出已按 PyTorch 行为修复；size-1 不再调用 `out.zero_()`。
- `126` 项原有精度测试通过；额外手工验证覆盖了标量、非法/空 reduction、size-1 和专用路径 first-NaN 行为。
- 非连续输入的 NaN 仍由 generic fallback 处理，first-NaN 保证仅限 Hygon 专用路径；PR 描述已限定该范围。

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
- Hygon：short/long inner、global packed 和 small non-inner 专用路径均使用 ordered key 保证 first-NaN；非连续输入的 generic fallback 不保证，PR 描述已明确限定。

如果 NaN 属于本 PR 的验收范围，这是 blocker。若本轮只要求普通数值精度，建议在文档中明确写出“只有专用 last-dim/pair kernel 保证 first-NaN，通用回退路径继承 generic 行为”，不要笼统声明所有路径都保持该语义。

### 5. 确认宿主函数调用约束

按仓库 `AGENTS.md` 的要求，宿主函数不应调用 PyTorch compute op，只能做分配/初始化等操作。以下代码直接调用了 `.contiguous()`：

- MetaX：`origin/argmax` 的 `src/flag_gems/runtime/backend/_metax/ops/argmax.py:577`。
- MThreads：`origin/argmax` 的 `src/flag_gems/runtime/backend/_mthreads/ops/argmax.py:332`、`:337`、`:366`。

这不是数值正确性问题，但属于明确的代码规范风险。建议改为专用 copy kernel，或把非连续输入限制为明确的 fallback，并在 PR 中说明原因。

## 非阻塞建议

- Hygon 的 zero-fill runner 依赖 Triton 私有 run/warmup 接口，并在 import 时预热；需要 reviewer 接受该方案，或后续改为公共接口。
- `tests/test_argmax.py` 和 `benchmark/test_argmax.py` 按要求未修改。现有 126 个 case 只覆盖 shape/dtype/keepdim 组合，边界场景目前依赖手工验证。
- MThreads graph 结果依赖临时 `torch.cuda` 到 `torch.musa` 的 timing shim，但仓库中没有该 shim。建议在文档中给出 shim 路径，或说明这是临时环境方案，避免性能结果无法复现。
- 文档目前只有 MThreads 和 Hygon 标注了代码提交；建议 MetaX、PPU 的指标也标注对应代码提交，便于追溯。

## 已确认通过

- 四个后端均已在各自 `ops/__init__.py` 中注册 `argmax`，并加入 `__all__`。
- `conf/operators.yaml` 中已有 `argmax` 元数据，无需新增。
- `git diff --check` 在代码分支和文档分支均通过。
- Thead/PPU 的 `_argmax_pair` 明确处理 first-NaN 和最低索引，0 维、非法 `dim`、空输入也有显式分支，是四个实现中最完整的。
- MThreads 文档中的实现描述基本准确：contiguous row-per-CTA、non-inner output-block、大 BF16 native comparison、8192 全局 tile 都能在代码中找到对应路径。
- 文档中的 benchmark 表与汇总指标内部一致，51 个 case 的最小值均不低于 0.95x（按文档记录）。

## 建议合并顺序

1. 修复 MetaX `dim` 校验、0 维输入和空 reduction 维度处理。
2. 决定并补齐 MThreads/MetaX 的 NaN 语义范围。
3. 在后续允许修改测试时增加边界和特殊值回归测试。
4. 处理 `.contiguous()` 规范和 MThreads shim 可复现性。
5. 补齐 MetaX、PPU 的代码提交哈希。
