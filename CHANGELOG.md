# 变更日志

本文件记录 `xiesheng-tool` 公开镜像的关键同步与变更。完整开发历史见私有仓 `lianyi-xiesheng`，公开仓为人工策展子集、非盲拷。

## 2026-10-11（三）— 新增四类确定性标点清理（ASR 标点缺陷）+ 失效 anchor 修复，rules v27

来源：私有仓提交 `c572b33` / `7ce307b`。起因是一句质疑：**「标点符号有问题的，校订脚本能不能解决？而不只是改那一集」**。全量 63 份成稿审计后，把可机械判定的四类做成规则——此前这类问题只能靠人工在某集撞见再手改，必然漏。

### 规则覆盖（由 `normalize_punct` 统一驱动，CLI `--apply` 生效）

| 类别 | 形态 | 处置 | 量级 |
|---|---|---|---|
| ① 停顿空格 | `要约件， 他由此`、`困惑 。` | 删空格 | 7858 |
| ② 堆叠/冗余 | `协作中、，然后`、`所述完毕，。另外`、`nba。。` | 保留首个/去重 | 12 |
| ③ 汉字后半角 | `编码，并且运行`、`注解:塞利` | 转全角 | 12 |
| ④ 行首错位 | `\n\n，包括其实线下`、`困境\n。相比之下` | 删 / 与前一行合并 | 6 |

`normalize_en_punct` 原本只管「字母后」的半角标点，汉字后紧跟的半角一直漏网——③ 补的正是这一半。

### 三条实测踩到的坑（代码注释与用例均已锁定，勿改回）

1. **「标点后一律删空格」会毁西文语境**：来源行的 ` | ` 分隔符、`R。 F。 Kuang` 人名缩写、`kid， you always` 英文从句都会被挤坏。故拆成两条——**后接汉字才一律删**；后接西文/数字**仅当标点前也是汉字**才删。
2. **校验器闸门不能照抄「先剥 `**`」**：半角/重复字闸门靠剥 `**` 防漏检，但本闸门剥了会把 `**主持人Nina：** 嗨` 误判成「冒号后多余空格」，实测两集 **85 / 109 处假阳性**。本闸门刻意不剥。
3. **空串 `in` 陷阱再现**：`"" in "。！？"` 恒为 True，会让文末空行被当成「以句末标点开头」而与上一行合并、吃掉文末换行。改用显式长度判空。

### 失效 anchor 修复（`repair_stale_anchors`）

anchor 是回标锚点，须与成稿逐字一致；成稿被确定性清洗后，停留在旧形态的 anchor 会定位不到、回标静默失败。**只对定位不到的 anchor 施加同一套清洗**（能定位的一律不动，逐字保真优先）。333 条 anchor 中失效数 9 → 1。

⚠️ **批量应用的教训**：md 与 handoff 必须分开判定——**md 无需改动的文件不能整体跳过**，否则残留的失效 anchor 永远修不到（本轮首版因此漏了 2 集）。

### 规则与闸门

- `rules.py` → `RULES_SPEC_VERSION` 27，新增 `EDIT_PUNCT_ASR`：明确**哪些交给脚本、哪些必须人工回原文**（ASR 按停顿在句中插入多余逗号、切裂「不是……而是……」这类关联结构，属语义判断，脚本不代做）。
- `validate_session_output` 新增标点硬闸门防回潮。全量 63 集对比：无任一集报错变多（401 → 362，差额即被识别并消除的旧缺陷）。

### 验证

幂等复核 0 改动 · 汉字字符数变化 0（措辞一字未动） · 私有仓 406 passed（2 例因缺 `funasr`，环境性）。

---

## 2026-10-11（二）补丁 — 中文数量转换硬化：堵住空后缀误判与黏连/概数腐蚀

来源：私有仓提交 `749454d`。上一轮把中文数量转阿拉伯数字引入后，全量干跑扫描 71 个成稿发现该规则会**腐蚀读数**，故紧急硬化。同一份转换逻辑同时驱动 `--apply` 与 handoff 同步，必须零误转才敢落到其它已转录集。

### 1. 空串 `in` 陷阱（根因，腐蚀最隐蔽）

`normalize_cn_number` 的「万/亿后裸数字」守卫原写成 `if _after not in "千百十": return run`。当 `_after == ""`（一万五 结尾的五、一亿两 收尾的两）时，`"" not in "千百十"` 在 Python 里恒为 **False**（空串是任何串的子串），守卫整段失效，导致：

- `一万五` → `10005`（应为省略千位的口语，保留中文）
- `一亿两周` → `100000002周`（应为 1亿 + 两周，保留中文）

改为显式判等：`if _after == "" or _after not in "千百十": return run`。

### 2. 两个相邻数量被贪心正则黏连

`_CN_RUN_RE` 是贪婪匹配，会把紧邻的两个数量合成一个 run 错转：

- `五十五十` → `100`、`四十五十` → `90`、`十一二十` → `30`、`十八十九` → `99`（重复进位字「十」）
- `三十万二十万` → `300020万`（重复进位字「万」）

新增读数合法性守卫：**含重复进位字（十/百/千/万/亿 任一出现两次）即整串跳过**。

### 3. 「X 或 Y」概数范围被错转成单值

`七八十`、`两三百`、`四五十`、`五六百`、`一两千`、`三四万`、`二三十万`、`三六万`、`一二一百` 等是两个连续单字读数构成的概数范围，转成单值会丢掉区间。新增守卫：**含连续单字读数（两个数字字紧挨、中间无进位字）即整串跳过**。`七十八`、`八十九`、`五十五` 等真正规范的读数因「读数—进位字—读数」结构不触发该守卫，照常转换。

### 4. 测试

`tests/test_normalize.py` 的 `TestCnNumber` 新增 `test_skips_ambiguous_and_merged_runs`（22 个应保留中文的读数，含上述全部坑）与 `test_still_converts_well_formed`（确保 `五百→500`、`十一万两千→112000` 等规范读数不被误伤）。硬化后全量扫描 71 成稿的 729 处转换零腐蚀。

---

## 2026-10-11（二）— 两条新确定性规则：话轮末逗号转句号 + 中文数量转阿拉伯数字 + rules v26

来源：私有仓提交 `abd92f9`（规则硬化）、`d6ef3be`（#757 应用）。上一轮把句末引语句号内化后，本轮补齐两类同样"纯机械可判定、却长期靠人工校订"的问题，并把术语表规则升级为有机器闸门的硬规则。

### 1. 话轮末逗号转句号（原文转录的组装层缺陷）

`## 原文转录` 里出现大量"一个人讲完却以逗号收尾"的行：根因不是 ASR 故意打逗号，而是说话人分离的边界正好落在逗号上、下一行又换了人。此前只能靠人工校订逐处改，本轮从两层堵死：

- `src/processor/normalize.py` 新增 `normalize_turn_terminal`：`normalize_markdown` 对所有段落生效，把 `【说话人】…，` 的话轮末逗号改为句号。
- `src/utils.py` 的 `merge_same_speaker_blocks`（组装环节）兜底：未来新集从源头干净，不再产生话轮末逗号。
- `src/processor/session_edit.py`：`validate_session_output` 加冗余闸门，漏改会被卡住。

### 2. 中文数量转阿拉伯数字

ASR 把数量落成中文读数（`十六家`、`十二亿用户`、`一百二十位`），难扫读、不利于一眼锁定数字。`src/processor/normalize.py` 新增 `normalize_cn_number`：转阿拉伯数字并保留万/亿单位（`十六家→16家`、`十二亿→12亿`、`三百六十五→365`），刻意避开日期（月/日/号/年）与成语误伤。`--apply` 顺带 `_sync_handoff_numbers` 同步 handoff 的描述字段（`anchor`/`quote` 逐字字段不动，防下次重渲染回潮）。

### 3. 术语表规则升级为硬规则（rules v25 → v26）

此前规则只说"能不写就不写"、无机器闸门，模型默认会列术语表（#757 即漏）。本次：

- `src/processor/rules.py`：`STRUCT_GLOSSARY` 重写——**已单列成核心观点/问题与思考的概念必须就近内联（缩进子条目），绝不许再单列成术语表；无则整节不写**；`RULES_SPEC_VERSION` 25 → 26。
- `src/processor/session_edit.py`：术语表条目若已在核心观点/问题与思考中展开（术语长度 ≥ 3，防短常用词误报）即报错要求内联。
- `src/processor/normalize.py`：GOLDEN 夹具去掉冗余术语表（原把模型塌缩等同时写进核心观点与术语表，正是新规则禁止的形态），相应改写/新增测试。

### 4. 测试

- `tests/test_normalize.py` 追加 `TestCnNumber`（约 13 例：十六家/十二亿/三百六十五/末尾数量/万单位保留/日期与成语负例）与 `TestTurnTerminal`（约 12 例：话轮末逗号转句号/非话轮逗号不动/句末句号不动/幂等），连同上一轮引语句号与日期用例，套件 344 → **372 passed**。
- `tests/test_session_edit.py` 增补术语表冗余闸门用例。

### 刻意保留的差异

- `scripts/oneoff/build_uncertainty_pack.py`（检索式校订候选包）仍未 ship：它依赖同目录的 `fix_transcript_surface.py` / `scan_typo_candidates.py`，这两个都不在公开仓，文档一旦指向就是断链。等校订工具链整体开源时一并评估。
- 同步时源码统一按 LF 落盘（公开仓 `.gitattributes` 为 `* text=auto`）。

## 2026-10-11 — 句末引语句号规则内化 + rules v25 对齐

来源：私有仓提交 `b1337ca`（规则内化）、`2706189`（成稿应用）。把「核心观点末尾的引用原话必须自带句末标点」从一次性脚本提升为确定性层，并补齐上一轮漏同步的 rules v25。

### 1. 句末引语句号：从 oneoff 脚本到确定性层

此前这条规则只活在 `scripts/oneoff/fix_upper_quote_period.py`（一次性脚本），换集就得记得跑、且规则散落两处易漂移。本次把它**只实现一次**在 `src/processor/normalize.py`：

- 新增 `normalize_quote_period` / `scan_quote_period` / `sync_handoff_quotes`；`normalize_markdown` 对 `## 核心观点` 块自动补句末句号（疑问句补问号，写在引号**之内**），`python -m src.processor.normalize <md> --apply` 顺带同步同名 `_handoff.json` 的 `key_points[].quote`（`anchor` 不动——它是转录回标锚点，必须与转录逐字一致，否则下次重渲染会被覆盖回去）。
- 判据刻意窄：**只看位于句末（引号后即换行/条目结尾）且自身缺句末标点**的引语，嵌入自己句子的短语引述（工作的基本单位从「职位」变为「任务」）不误伤。
- 以承接连词（而/但/却/以及…）开头的引语是**残句**（原话本身从半句起），脚本不自动补、只列出来交人工回原文补全整句。
- `scripts/oneoff/fix_upper_quote_period.py` 降级为**薄封装**：规则只在 normalize 实现一次，脚本只做「一次扫完 `output/` 下全部成稿」的批量能力（`--all`）。
- `src/processor/session_edit.py`：`validate_session_output` 加硬闸门，成稿若句末「…」缺标点直接报错（Show Notes 透传豁免）。

### 2. rules v25 对齐（上一轮漏同步的线）

私有仓 `rules.py` 已升到 v25（摘要改为底层逻辑提炼而非观点罗列、核心观点/问题与思考/术语表条款重写、术语表明确排除专名）。本轮把 `src/processor/rules.py` 与 `src/processor/session_edit.py` 一并前向移植，使公开仓与私有仓在规则版本上重新对齐（此前公开仓停在 v24，相差约 95 行）。

### 3. 测试

- `tests/test_normalize.py` 追加 `TestTrailingQuotePeriod`（7 例：缺句号告警 / 句号在内通过 / 问号通过 / 嵌入短语不报 / 句中不报 / 句号在外告警 / Show Notes 豁免）与 `TestCnDate`（11 例）。
- `tests/test_session_edit.py` 追加 `TestTrailingQuotePeriod`（7 例）。
- `tests/test_rules_single_source.py` 同步 v25 措辞断言（摘要/核心观点/术语表新口径）。
- 套件 327 → 344 passed。

### 刻意保留的差异

- `scripts/oneoff/build_uncertainty_pack.py`（检索式校订候选包）仍未 ship：它依赖同目录的 `fix_transcript_surface.py` / `scan_typo_candidates.py`，这两个都不在公开仓，文档一旦指向就是断链。等校订工具链整体开源时一并评估。
- `scripts/oneoff/fix_upper_quote_period.py` 本次**已**随规则内化一并 ship（它现在是只依赖 `src.processor.normalize` 的薄封装，无外部依赖）。
- 同步时源码统一按 LF 落盘（公开仓 `.gitattributes` 为 `* text=auto`）。

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
- 私有仓 `src/processor/rules.py` 当时已升到 v25、公开仓停在 v24（相差约 95 行）；该差异已在次日（2026-10-11）的同步中补齐，见上方条目。
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
