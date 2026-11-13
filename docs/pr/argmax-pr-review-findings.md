# Argmax PR 静态审查结论

## 审查范围

- 代码分支：`origin/argmax`，最新提交 `489037e6` (`perf(hygon): use public launch path for singleton argmax`)。
- 文档分支：`origin/docs/argmax-pr-description`，本次更新后包含最新的 Hygon 实现说明和审查结论。
- 审查方式：静态代码、文档和文件结构检查，加上 Hygon/MetaX 边界场景手工验证和完整性能测试；按要求未修改 `tests/test_argmax.py` 或 `benchmark/test_argmax.py`。
- 说明：Hygon 和 MetaX 的文档、边界行为及性能结果已更新；下列仍待处理项主要针对 MThreads 和未修改的回归测试覆盖。

## 结论

Hygon 和 MetaX 的代码、PR 描述及边界行为已经对齐，两者的 51 项性能测试均完成，稳定参考结果保持在 0.95x 以上。共享节点上的极短 case 偶发低于 0.95x，已作为瞬时测量波动记录在 PR 描述中。整个 PR 仍有以下待处理项：

1. MThreads 的部分 NaN 语义与 PyTorch 不一致。
2. 新增路径缺少针对 NaN、非连续、非法 `dim`、0 维和空 reduction 维度的回归测试；本轮按要求未修改测试文件。

## 必须修

### 1. Hygon（已处理）

- PR 描述与当前实现一致：packed `uint64` global reduction、short/long innermost dispatch、small non-inner kernel 和 size-1 zero-fill 均能在代码中找到对应路径。
- Apache license header 已补齐。
- 标量 `dim=0/-1`、非法 `dim`、空 reduction 维和 size-1 大输出已按 PyTorch 行为修复；size-1 不再调用 `out.zero_()`。
- size-1 zero-fill 改用公开的 `libentry(cache_binding=True)` 和普通 `kernel[grid]` 启动，不再使用 Triton 私有编译对象字段，也不再在 import 时预热。
- `126` 项原有精度测试通过；额外手工验证覆盖了标量、非法/空 reduction、size-1 和专用路径 first-NaN 行为。
- 非连续输入的 NaN 仍由 generic fallback 处理，first-NaN 保证仅限 Hygon 专用路径；PR 描述已限定该范围。

### 2. MetaX 边界、NaN 和宿主约束（已处理）

文件：`src/flag_gems/runtime/backend/_metax/ops/argmax.py`（`origin/argmax` 版本）

- 在取模和访问 `shape[dim]` 前统一校验 `dim`，非法值抛出 `IndexError`；0 维输入的 `dim=0/-1` 返回标量索引 0。
- 空 reduction 维度、空输入的 `dim=None` 均按 PyTorch 行为抛出 `IndexError`，不再进入除零或 kernel 编译路径。
- 小 non-inner 回退改为 NaN-aware `argmax_pair` kernel，专用与回退路径均保证 first-NaN 和最低索引语义。
- 无法折叠的 strided view 改由 Triton copy kernel 物化，不再在宿主函数中调用 `.contiguous()`。
- 修正小 non-inner tile 的 `TILE_N` 为 2 的幂，避免非 2 次幂 reduction 长度编译失败。
- `126` 项原有精度测试通过；额外手工检查覆盖标量、非法/空 reduction、NaN、非连续输入和整数 dtype。

### 3. 明确或修复 NaN 语义

PyTorch 的 `argmax` 将 NaN 视为最大值，并返回第一个 NaN 的索引。当前实现情况：

- MThreads：`_argmax_contiguous_kernel`、`_argmax_block_kernel`、`_argmax_block_native_kernel` 直接使用 `tl.max(..., return_indices=True)`，没有 NaN 分支。对应 `origin/argmax` 的 `src/flag_gems/runtime/backend/_mthreads/ops/argmax.py:95`、`:140`、`:189`。
- Hygon：short/long inner、global packed 和 small non-inner 专用路径均使用 ordered key 保证 first-NaN；非连续输入的 generic fallback 不保证，PR 描述已明确限定。

如果 NaN 属于本 PR 的验收范围，这是 blocker。若本轮只要求普通数值精度，建议在文档中明确写出“只有专用 last-dim/pair kernel 保证 first-NaN，通用回退路径继承 generic 行为”，不要笼统声明所有路径都保持该语义。

### 4. 确认宿主函数调用约束

按仓库 `AGENTS.md` 的要求，宿主函数不应调用 PyTorch compute op，只能做分配/初始化等操作。以下代码直接调用了 `.contiguous()`：

- MThreads：`origin/argmax` 的 `src/flag_gems/runtime/backend/_mthreads/ops/argmax.py:332`、`:337`、`:366`。

这不是数值正确性问题，但属于明确的代码规范风险。建议改为专用 copy kernel，或把非连续输入限制为明确的 fallback，并在 PR 中说明原因。

## 非阻塞建议

- Hygon 的 zero-fill runner 已改为公开 LibEntry 启动路径，不依赖 Triton 私有 `run`/`warmup` 字段，也没有 import 时预热。
- `tests/test_argmax.py` 和 `benchmark/test_argmax.py` 按要求未修改。现有 126 个 case 只覆盖 shape/dtype/keepdim 组合，边界场景目前依赖手工验证。
- MThreads graph 结果依赖临时 `torch.cuda` 到 `torch.musa` 的 timing shim，但仓库中没有该 shim。建议在文档中给出 shim 路径，或说明这是临时环境方案，避免性能结果无法复现。
- 建议 PPU 的指标也标注对应代码提交，便于追溯。

## 已确认通过

- 四个后端均已在各自 `ops/__init__.py` 中注册 `argmax`，并加入 `__all__`。
- `conf/operators.yaml` 中已有 `argmax` 元数据，无需新增。
- `git diff --check` 在代码分支和文档分支均通过。
- Thead/PPU 的 `_argmax_pair` 明确处理 first-NaN 和最低索引，0 维、非法 `dim`、空输入也有显式分支，是四个实现中最完整的。
- MThreads 文档中的实现描述基本准确：contiguous row-per-CTA、non-inner output-block、大 BF16 native comparison、8192 全局 tile 都能在代码中找到对应路径。
- MetaX CUDA Graph 和非 Graph 均为 51/51、不低于 0.95x；边界和 NaN 手工检查通过，指标已标注代码提交 `e55d65da8`。
- 文档中的 benchmark 表与稳定参考汇总内部一致；共享节点偶发低于 0.95x 的场景已明确标注为瞬时测量波动。

## 建议合并顺序

1. 决定并补齐 MThreads 的 NaN 语义范围。
2. 在后续允许修改测试时增加边界和特殊值回归测试。
3. 处理 MThreads `.contiguous()` 规范和 graph timing shim 可复现性。
4. 补齐 PPU 的代码提交哈希。
