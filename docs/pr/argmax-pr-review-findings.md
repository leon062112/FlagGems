# Argmax PR 静态审查结论

## 审查范围

- 代码分支：`origin/argmax`，最新提交 `53457e16` (`perf(mthreads): keep argmax NaN handling graph-friendly`)。
- 文档分支：`origin/docs/argmax-pr-description`，本次更新后包含最新的 Hygon、MetaX 和 MThreads 实现说明及审查结论。
- 审查方式：静态代码、文档和文件结构检查，加上 Hygon/MetaX 边界场景手工验证和完整性能测试；按要求未修改 `tests/test_argmax.py` 或 `benchmark/test_argmax.py`。
- 说明：Hygon、MetaX 和 MThreads 的文档、边界行为及性能结果已更新；下列仍待处理项主要针对未修改的回归测试覆盖和 MThreads 宿主函数的 `.contiguous()` 规范风险。

## 结论

Hygon、MetaX 和 MThreads 的代码、PR 描述及边界行为已经对齐。Hygon 和 MetaX 的 51 项性能测试均完成，稳定参考结果保持在 0.95x 以上；MThreads Kernel 模式 51/51 不低于 0.95x，MUSA Graph 模式 50/51 不低于 0.95x，BF16 `(1024, 4096)` 为 0.944x。共享节点上的极短 case 偶发低于 0.95x，已作为瞬时测量波动记录在 PR 描述中。整个 PR 仍有以下待处理项：

1. 新增路径缺少针对 NaN、非连续、非法 `dim`、0 维和空 reduction 维度的回归测试；本轮按要求未修改测试文件。
2. MThreads 仍通过宿主函数中的 `.contiguous()` 物化非连续输入，属于代码规范风险而非数值正确性问题。

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

### 3. MThreads NaN 和边界语义（已处理）

PyTorch 的 `argmax` 将 NaN 视为最大值，并返回第一个 NaN 的索引。当前实现情况：

- MThreads 的 FP16/FP32/BF16 专用路径先将 NaN 规范化为 `+inf`，再通过独立的轻量修复阶段定位最低 NaN 索引；连续长行路径把 `+inf` 标记编码进临时 index，避免额外 buffer 和 graph replay 回退。
- 标量 `dim=0/-1`、非法 `dim`、空 reduction 维、空输入 `dim=None` 和 size-1 reduction 均已按 PyTorch 行为处理。
- 手工验证覆盖标量、非法/空 reduction、FP16/FP32/BF16 first-NaN、多 NaN、全局 NaN、`+inf` 与 NaN 混合、非连续和转置输入；原有 `126` 项精度测试全部通过。
- Hygon：short/long inner、global packed 和 small non-inner 专用路径均使用 ordered key 保证 first-NaN；非连续输入的 generic fallback 不保证，PR 描述已明确限定。

### 4. 确认宿主函数调用约束

按仓库 `AGENTS.md` 的要求，宿主函数不应调用 PyTorch compute op，只能做分配/初始化等操作。以下代码直接调用了 `.contiguous()`：

- MThreads：`origin/argmax` 的 `src/flag_gems/runtime/backend/_mthreads/ops/argmax.py:761`、`:764`、`:783`、`:803`。

这不是数值正确性问题，但属于明确的代码规范风险。建议改为专用 copy kernel，或把非连续输入限制为明确的 fallback，并在 PR 中说明原因。

## 非阻塞建议

- Hygon 的 zero-fill runner 已改为公开 LibEntry 启动路径，不依赖 Triton 私有 `run`/`warmup` 字段，也没有 import 时预热。
- `tests/test_argmax.py` 和 `benchmark/test_argmax.py` 按要求未修改。现有 126 个 case 只覆盖 shape/dtype/keepdim 组合，边界场景目前依赖手工验证。
- MThreads graph 结果依赖临时 `torch.cuda` 到 `torch.musa` 的 timing shim；PR 描述已明确说明该 shim 仅用于 benchmark、不属于算子实现。
- 建议 PPU 的指标也标注对应代码提交，便于追溯。

## 已确认通过

- 四个后端均已在各自 `ops/__init__.py` 中注册 `argmax`，并加入 `__all__`。
- `conf/operators.yaml` 中已有 `argmax` 元数据，无需新增。
- `git diff --check` 在代码分支和文档分支均通过。
- Thead/PPU 的 `_argmax_pair` 明确处理 first-NaN 和最低索引，0 维、非法 `dim`、空输入也有显式分支，是四个实现中最完整的。
- MThreads 文档与实现一致：contiguous row-per-CTA、non-inner output-block、大 BF16 native comparison、unmasked full-tile streaming、encoded-index NaN repair 和全局 tree reduction 都能在代码中找到对应路径。
- MetaX CUDA Graph 和非 Graph 均为 51/51、不低于 0.95x；边界和 NaN 手工检查通过，指标已标注代码提交 `e55d65da8`。
- 文档中的 benchmark 表与稳定参考汇总内部一致；共享节点偶发低于 0.95x 的场景已明确标注为瞬时测量波动。

## 建议合并顺序

1. 在后续允许修改测试时增加边界和特殊值回归测试。
2. 处理 MThreads `.contiguous()` 宿主函数规范风险。
3. 补齐 PPU 的代码提交哈希。
