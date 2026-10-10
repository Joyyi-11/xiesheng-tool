# 变更日志

本文件记录 `xiesheng-tool` 公开镜像的关键同步与变更。完整开发历史见私有仓 `lianyi-xiesheng`，公开仓为人工策展子集、非盲拷。

## 2026-10-10 — 分离前人数预检（probe）+ 中文读数日期规范化

来源：私有仓提交 `b0ddaa8`。两处都是「ASR 出来的结果不好读、不好分」的老问题，本次给出过程内的解法而非事后补丁。

### 1. 说话人分离：把判断从「塌缩之后」提前到「分离之前」

实际问题不是「圆桌人多才会并成一簇」——**双人访谈反而更容易塌**，因为只有 2 簇时没有缓冲。根因在 `diarize.estimate_speakers()` 的一条短路：

```python
_single_speaker_sim_p10: float = 0.16
if sim_p10 >= 0.16 and min_k <= 1:
    return 1, ...   # 直接判单人，连 GMM-BIC 都不跑
```

远程访谈播客普遍越过这条线（实测 #757 = 0.2393、#706 = 0.1702），于是**无论后端是 cam++ 还是 wespeaker，走自动路径都会把它当单人**。这也是此前只能等 `collapse_check` 报坍缩、再人工 `--speakers 2` 重跑的原因。

- 新增 `src/diarization/probe.py`：分离前跑 VAD + speaker embedding + 多 K 轮廓系数扫描，输出建议后端与人数。绕开短路的办法是 `estimate_speakers(min_k=2)`——短路条件是 `min_k <= 1`。
- 判据是**多个候选 K 取轮廓系数最优，不是取最大 K**。轮廓系数天然随 K 增大而下降，取最大 K 会被「刚好过线的细碎 K」骗到。四份真实样本标定：

  | 样本 | 实际情况 | sim_p10 | K=2 轮廓系数 | probe 结论 |
  |---|---|---|---|---|
  | #757 | 双人访谈 | 0.2362（短路） | 0.3098 | wespeaker K=2 ✅ |
  | #706 | 双人访谈 | 0.1702（短路） | 0.4122 | wespeaker K=2 ✅ |
  | #753 | 主讲 + 插入乔布斯原声 | 0.0696 | 0.4014 | wespeaker K=2 ✅ |
  | #28 | 确凿单口 | 0.3127（短路） | **0.0712** | campplus ✅ 不误报 |

  #28 是关键反例：它同样命中短路，但轮廓系数够不到阈值。这条门槛区分的就是「真可分」与「被误判」。
- `src/main.py`：默认后端 `campplus` → `auto`，由 probe 接管；顺带修掉一个真 bug——原先 auto 回退 wespeaker 时把 `chosen_k`（可能为 1）当人数传进去，等于白换后端。
- 预检失败（如未装 diarize extra）保守回退 cam++，后置的塌缩检测仍会兜住。

### 2. 成稿规范化：中文读数日期 → 阿拉伯数字

ASR 的逆文本正则化常把日期落成中文读数（`二零二六年，九月二十八日`），难扫读也不利于检索。`src/processor/normalize.py` 新增 `normalize_cn_date` / `count_cn_date`，已接进 `python -m src.processor.normalize <md> --apply`，Phase B 收尾跑一次即可。

覆盖两种形态，并对异常值保守：

- 完整年月日（含口语造成的「年，月」间断句）→ `2026年9月28日`
- 只提年份 → `2023年那篇`
- 越界的（听成「十三月三日」）保持原样，交由校订环节处理

### 3. 测试

- 新增 `tests/test_probe.py`（15 例）：用上述真实标定数据作 fixture，**同时覆盖正例与反例**——只测「双人能切出来」不够，必须保证确凿单口不被误报。
- `tests/test_normalize.py` 追加 `TestCnDate`（11 例）：含列举（「一是……二是……」）与越界值两类不得误伤的负例，以及只有「年+月」无「日」的已知边界（保持原样）。
- 套件 301 → 327 passed。

### 刻意保留的差异

- `scripts/oneoff/build_uncertainty_pack.py`（检索式校订候选包）未 ship：它依赖同目录的 `fix_transcript_surface.py` / `scan_typo_candidates.py`，这两个都不在公开仓，文档一旦指向就是断链。等校订工具链整体开源时一并评估。
- 私有仓 `src/processor/rules.py` 已升到 v25，公开仓仍是 v24（相差约 95 行）；本次不同步这条线，保持每次同步只做一件事。
- 同步时源码统一按 LF 落盘（公开仓 `.gitattributes` 为 `* text=auto`）。

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
