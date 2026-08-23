# 撷声（Xiesheng）

> 撷声，意为「采撷声音里的精华」，把播客中值得留存的内容，转写、校订、提炼成结构化笔记。

面向小宇宙播客的低成本结构化文稿生成工具。输入单集链接，自动完成节目抓取、音频下载、本地转写、说话人区分和 LLM 内容整理，最终输出适合阅读、编辑和引用的 Markdown 文稿。

## 为什么做

现有播客转录服务通常采用按月订阅或有限免费额度。对于只需要偶尔处理单期节目的人，这意味着持续付费；只调用本地 ASR 又只能得到原始逐字稿，仍要人工处理错字、口语、专名、说话人和内容结构。

撷声把计算量最大的语音转写放在本地完成，只把文本校订和内容提炼交给 LLM。这样既避免持续订阅，也把单期 API 支出控制在较低水平，同时保留可回查的原始转录。

## 功能

- 爬取小宇宙播客节目信息、Show Notes
- 下载音频并转写为文字（本地 FunASR SenseVoice-Small，免费；整段音频串行转录，约 9x 实时）
- 千问或 DeepSeek 分块校订：错字修正、口语清理、语义分段、专名纠错（分块并行校订）
- LLM 模型自动回退：按 `--llm-model` 候选列表（逗号分隔）或网关 `/models` 自动选可用模型，404 `model_not_found` 不再白白重试，过载（429/502/503）指数退避
- 会话内校订固化：`python -m src.processor.session_edit <episode>_diarized.txt` 输出固定规范提示词，附带结构校验器保证输出格式稳定
- 独立生成内容提要、人物简介、闪光语句、Mermaid 思维导图大纲和关键词，避免长文输出截断
- 永久保留原始转录，整段音频缓存支持失败后续跑
- 按模型、来源 hash 和提示词版本隔离缓存，避免失败重跑或改提示词后重复消耗 LLM token
- 本地 ASR 纯 CPU 运行，无需 GPU
- 结果缓存：同链接 + 同日期的重复运行直接复用已完成文稿，`--refresh` 强制重跑
- 支持两种后处理方式：有 API Key 走自动化校订，无 Key 走 `--no-llm` 会话内处理
- 记录转写耗时、实时系数和 LLM token 用量
- 输出结构化 Markdown：Show Notes -> 内容提要 -> 闪光语句 -> 大纲 -> 人物简介 -> 全文转录

## 完整链路

从链接到文稿共经过六个阶段：

```text
输入：小宇宙播客链接

处理：
   -> 抓取标题 / 来源 / Show Notes / 音频地址
   -> 下载音频
   -> FunASR SenseVoice-Small 本地转写，保留原始逐字稿
   -> 说话人识别，为不同声音添加标签
   -> LLM 校订
        ├─ 免费会话链路：生成自包含会话输入包
        └─ API 自动校订：分块校订 -> 重试/回退
   -> 内容提炼与 Markdown 组装
   
输出：结构化 Markdown 转录文稿
     标题、来源与时间／Show Notes／内容提要／闪光语句/大纲/思维导图/关键词／人物简介／全文转录       
```

## 两种后处理链路

校订与结构化可以走自动化 API，也可以交给 AI 会话——后者在 API Key 无余额或想保持完全本地时非常实用。两条链路共用同一份转写与说话人区分结果，只在"后处理"这一步分叉。

- **会话内（默认，免费）**：工具只完成转写与说话人区分，产出"会话输入包"——一份自包含文件，包含节目标题、来源、Show Notes 与带 `[SPEAKER_XX]` 标签的全文。内部处理方式已固化为固定规范（`src/processor/session_edit.py`，v4），规范提示词要求产出与自动化链路相同的完整结构——内容提要、闪光语句、Mermaid 思维导图大纲（一级主题带播放时间）、关键词、人物简介与全文转录——并用内置校验器复核会话产出。未配置 API Key 时自动走此链路。
- **自动化（可选）**：同时配置 `LLM_API_KEY` 与 `LLM_BASE_URL` 后直接运行，工具按块校订全文，再生成内容提要、人物简介、闪光台词、Mermaid 思维导图大纲和关键词（规则见 `src/processor/prompt.py`）。模型不可用时自动降级为会话内链路。

## 模型选型

转写后端默认 **FunASR SenseVoice-Small**（CPU 约 9x 实时、长音频精度与 Paraformer 相当、原生多语种、+LLM 后处理即可补足英文专名大小写）。Paraformer-Large 作为可选 `--model paraformer-large`，适合需要**热词定制**（常驻嘉宾名/产品名）或最稳**字级时间戳**的场景。原 faster-whisper 路线已移除——它的提速完全依赖 ffmpeg 切片 + 多进程并行，该机制退役后 Whisper 只剩 ~1x 串行的裸速度，留作兜底只会慢到超时/失败。

## 模型缓存

FunASR 模型（SenseVoice-Small、Paraformer-Large、ct-punc 标点、fsmn-vad）下载后统一缓存在用户主目录 `~/.cache/modelscope/`（约 3 GB），**跨运行、跨项目复用，不会每次重新下载**——只有首次运行某模型时才下载一次。删除该目录后下次运行会重新下载（约 3 GB，视网速需数十分钟），**请勿删除**。

注意：运行日志会打印 `模型缓存已命中` 表示本次直接用本地模型；若看到 `将下载缺失模型` 才表示需要联网拉取。

## 各部分分工

| 部分 | 实际职责 | 不负责什么 |
|------|----------|------------|
| 小宇宙页面抓取 | 获取标题、节目名、发布日期、Show Notes 和音频地址 | Show Notes 来自节目原页面，不由 LLM 生成 |
| 音频处理 | 下载音频，通过 ffmpeg 转换为本地转写所需格式 | 不改变节目内容 |
| FunASR（SenseVoice-Small / Paraformer-Large） | 在本地把音频转成带时间信息的原始文字 | 不负责内容提要、人物判断和文稿结构 |
| 说话人识别 | 根据声音特征区分说话人，生成 `SPEAKER_00` 等标签；VAD 段上限 4s 从源头降低快速接话的混段概率 | 只区分声音，不直接确认真实姓名和身份；混段自动保留 `[SPEAKER_XX]` 而非猜测归属 |
| LLM 全文校订 | 分块修正错字、口语、专名和分段，保留原意 | 不重新创作或扩写播客观点 |
| LLM 内容提炼 | 结合 Show Notes 和校订全文，生成内容提要、人物简介、闪光语句、Mermaid 思维导图大纲和关键词，并在证据充分时映射说话人身份 | 信息不足时不猜测人物身份 |
| Markdown 组装 | 按固定结构写入 Show Notes、提要、闪光语句、大纲、关键词、人物简介和全文 | 不参与语义判断 |

## 快速开始

```powershell
# 安装依赖（依赖以 pyproject.toml 为准，requirements.txt 为兼容转发）
pip install -r requirements.txt

# 转录并产出会话输入包（默认免费链路，无需任何 API Key）
xiesheng https://www.xiaoyuzhoufm.com/episode/xxxxx
# 把生成的 output/<节目名>_diarized.txt 粘贴到 AI 会话做校订与结构化

# 需启用自动化后处理，先配置 OpenAI 兼容接口：
[Environment]::SetEnvironmentVariable("LLM_API_KEY", "your_key", "User")
[Environment]::SetEnvironmentVariable("LLM_BASE_URL", "https://your-provider/v1", "User")
xiesheng https://www.xiaoyuzhoufm.com/episode/xxxxx

# 切换到 DeepSeek 系模型（需已配置 LLM API）
xiesheng https://www.xiaoyuzhoufm.com/episode/xxxxx --llm-provider deepseek

# 指定多个候选模型，按顺序自动回退（需已配置 LLM API）
xiesheng https://www.xiaoyuzhoufm.com/episode/xxxxx --llm-model qwen3.7-plus,gpt-5.2

# 会话内校订：生成固定规范提示词（免费链路）
python -m src.processor.session_edit output/<节目名>_diarized.txt --prompt-only

# 会话内校订：用已配置 LLM 自动完成并校验
python -m src.processor.session_edit output/<节目名>_diarized.txt
```

## 前置依赖

- Python 3.10+
- ffmpeg（用于音频格式转换）

## 配置

默认免费链路无需任何配置。可选自动化 LLM 后处理需要 OpenAI 兼容接口，从操作系统环境变量读取：

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-provider/v1
```

两个变量须同时设置才会启用自动化；只设置其中一个时仍走会话内链路。默认模型为 `qwen3.7-plus`，`--llm-provider deepseek` 使用 `deepseek-v4-flash`，可用 `--llm-model` 覆盖——支持逗号分隔多个候选模型自动回退。模型不可随便可用时自动降级为会话内链路。项目 `.env` 仅保留本地配置。

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `url` | 必填 | 小宇宙播客单集链接 |
| `-o` | `output/` | 输出目录 |
| `--model` | `sensevoice-small` | ASR 模型（sensevoice-small / paraformer-large） |
| `--llm-provider` | `qwen` | 后处理模型提供方（`qwen`/`deepseek`） |
| `--llm-model` | 提供方默认值 | 覆盖默认模型名称；支持逗号分隔多个候选按顺序回退（如 `qwen3.7-plus,gpt-5.2`） |
| `--speakers` | 自动检测 | 明确指定说话人数 |
| `--no-llm` | 否 | 仅转写与说话人区分，产出会话输入包，不需要 LLM API Key |
| `--no-diarization` | 否 | 跳过说话人识别 |
| `--spk-max-seg-ms` | `4000` | 说话人分离粒度：VAD 段上限（毫秒），段越短越不易把两人快速接话并成一段；可回退 `8000` |
| `--audio` | 无 | 复用已有的 16k mono WAV（需为本期音频），跳过下载与转换 |
| `--jobs` | `1` | 预留参数（FunASR 整段串行转录，暂不使用） |
| `--refresh` | 否 | 忽略结果缓存，强制重新转录与整理 |

## 输出格式

```
# 节目标题
> 来源：播客名称 | 节目标题 | 播出日期

# Show Notes
（完整保留的节目介绍）

## 内容提要
- **要点名称**：支撑证据
（不限条数，提炼全部重要观点）

## 闪光语句
- **关键词**：精彩原话
（不限条数，关键短语加粗）

## 大纲
​```mermaid
mindmap
  root((节目标题))
    "00:00 一级主题"
      二级要点
```
（Mermaid 思维导图；一级主题带真实播放时间）

## 关键词
- **关键词**：1-2 句说明

## 人物简介
**身份姓名**：简介
（正文格式）

## 全文转录
（校订后的分角色完整文稿）
```

## 成本

- 转写与说话人区分：本地免费
- 会话内处理：免费（在 AI 会话中完成）
- 可选自动化 LLM 后处理费用取决于模型、节目长度和所选 API 提供方账单
- 个人使用 DeepSeek 校订的历史实测中，典型单期 API 支出低于 0.05 元
- 程序记录输入／输出 token，不再使用过期的固定单价估算

`0.05 元/期` 是特定模型、节目长度和实际账单下的历史结果，不是固定报价或程序保证；更换模型、API 服务或处理更长节目时应以实际账单为准。

## 中间文件

- `output/<节目名>_raw.txt`：未经 LLM 修改的原始转录
- `output/<节目名>_diarized.txt`：自包含会话输入包（标题、来源、Show Notes + 带 `[SPEAKER_XX]` 标签的全文），`--no-llm` 时的直接交付物
- `output/<节目名>.wav`：转写用的 16k mono WAV（按节目命名，避免多期互相覆盖）
- `output/.work/<节目名>/transcribe/`：按音频指纹与模型名持久化的整段转写缓存，支持失败后续跑
- `output/.work/<节目名>/`：按模型与提示词版本隔离的校订分块缓存、`struct.json` 结构化结果缓存
- `output/.work/result_cache.json`：结果缓存索引（链接 + 发布日期），命中时跳过整条链路
- `output/<节目名>.md`：最终文稿
- 会话内校订：`python -m src.processor.session_edit <节目名>_diarized.txt [--prompt-only]`，输出可粘贴的固定规范提示词或直接生成并校验最终文稿

模型请求失败或返回被截断时，该分块不会写入缓存，再次运行相同节目和模型时，来源一致且提示词版本一致的已完成分块会直接复用。

## 限制

- 当前主要支持小宇宙单集页面；网页结构变化可能影响抓取。
- LLM 后处理需要兼容 OpenAI Chat Completions 的 API。
- 说话人识别是辅助能力：VAD 段上限 4s 从源头降低快速接话被并成一段的混段概率；仍无法可靠拆分的混段自动保留 `[SPEAKER_XX]` 标签而非猜测归属，最终由人工复核确认真实身份。

## 项目结构

```
src/
├── main.py                 # CLI 入口
├── config.py               # LLM 配置加载
├── audio.py                # 音频下载与格式转换
├── utils.py                # 计时、计费等工具
├── result_cache.py         # 幂等结果缓存（链接 + 发布日期）
├── models/
│   ├── schemas.py          # 数据模型
│   └── llm_struct.py       # LLM 结构化输出的 Pydantic 校验模型
├── scraper/xiaoyuzhou.py   # 小宇宙页面爬取
├── transcriber/
│   ├── base.py                     # 转写器抽象接口
│   └── funasr_transcriber.py       # FunASR 本地转写（SenseVoice / Paraformer，整段串行 + 缓存）
├── diarization/            # 说话人识别
├── processor/
│   ├── prompt.py           # 忠实校订与提要生成提示词
│   ├── normalize.py        # 文本归一化
│   ├── llm_processor.py    # 并行分块校订、模型探测回退、校验、缓存与文稿整理
│   └── session_edit.py     # 会话内校订：固定规范提示词 + 结构校验器 + CLI
└── renderer/markdown.py    # Markdown 渲染
```

## 作者

连漪（Lianyi）

## 许可证

MIT

```