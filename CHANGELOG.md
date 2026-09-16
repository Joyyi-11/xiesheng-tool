# 变更日志

本文件记录 `xiesheng-tool` 公开镜像的关键同步与变更。完整开发历史见私有仓 `lianyi-xiesheng`，公开仓为人工策展子集、非盲拷。

## 2026-09-16 — 规范化体系对齐（v24 规则 + v23 渲染器）

本次同步私有仓的两类滞后，使公开仓与私有仓在规范化体系上对齐。两笔提交均已在 `main` 落地。

### 1. 校对规则收敛到确定性层（关联 issue #1 / #2，已关闭）

- 提交 `4dcd5f2`（来自私有仓 `429ac69` / `1d9610e`）
- `src/processor/normalize.py` 新增 `normalize_en_punct`（外文词后半角标点转全角）与 `merge_repeat_chars`（≥3 连同字合并，拟声 / 应答白名单除外），并提供 CLI：
  `python -m src.processor.normalize <md> [--apply]`
- `src/processor/rules.py`：`RULES_SPEC_VERSION` 23 → 24；`EDIT_FIDELITY` 开出「删口语噪音」例外口并加信息量守恒判据；`EDIT_REDO` 由「可以合并」改为强制、限定 ≥3 连、拟声与应答叠用除外；`EDIT_EN_PUNCT` 补确定性脚本兜底说明
- `src/processor/session_edit.py`：校验器增加两条硬闸门（转录区残留半角标点 / ≥3 连重复字即报错）
- 测试 300 passed

### 2. 渲染器与文档对齐（v23 行为）

- 提交 `ba5f503`
- `src/renderer/markdown.py` 同步私有版：加粗幂等保护（`bold_anchors --apply` 二次套 `**` 时不再翻倍）、核心观点三段式段间不留空格、术语表改为可选小节
- 配对 `tests/test_renderer.py`（含术语表缺失断言）
- `README.md`「输出格式」段对齐：术语表移至核心观点之后并标记「可选」，核心观点三段式格式说明与渲染器行为一致
- 测试 301 passed

### 刻意保留的差异

公开仓 README 标题与「单集校订流程 SOP」整段仍与私有仓不同：公开仓是人工策展子集，且未 ship `bold_anchors.py` 等模块，文档不应指向不存在的文件。
