"""Deterministic in-session transcript editing (the no-LLM path).

The ``--no-llm`` path hands the user a self-contained session package
(``<episode>_diarized.txt``) and asks an AI chat session to do the cleaning and
structuring. Without a fixed spec that work has been ad-hoc: quality varies run
to run and there is no way to check the result.

This module pins the in-session procedure down:

* ``build_session_prompt`` — a versioned, fixed prompt template that wraps the
  session package. Bump ``SESSION_SPEC_VERSION`` whenever the rules change so
  old outputs can be told apart.
* ``validate_session_output`` — structural checks a finished .md must pass
  (headings present, speaker labels resolved, sections non-empty, quote format).
* ``main`` — CLI: from a session package, either emit the prompt for manual
  pasting into a chat, or (when an LLM API is configured) run it end-to-end and
  validate the result before writing the .md.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

from src.config import DEFAULT_LLM_PROVIDER, LLM_MODELS, get_llm_config

logger = logging.getLogger(__name__)

# 改动 SESSION_RULES 或输出结构时必须递增，让旧产出可与新规则区分。
# v11：「Show Notes」纳入「输出结构」必需小节并统一为二级标题 ## Show Notes（与自动渲染路径 markdown.py 对齐）；REQUIRED_HEADINGS 增加 ## Show Notes。
# v10：「内容提要」+「闪光语句」合并为「核心观点」——每条核心观点可附一句可选原话（> 引用块）；移除独立「闪光语句」小节。
# v9：校订规则新增 GB/T 15834-2011 标点规范——引号/书名号并列不用顿号；并新增校验器拦截。
# v8：闪光语句改回正文格式（独立成段、不标注说话人、不用引用块）；v7 的 > 块+——说话人 作废。
# v7：问题与思考改 1. 编号且问答分行；关键词补句号；全文转录说话人改【身份姓名】；
#     新增第 10 条禁止「大纲」板块与 mermaid 思维导图。
SESSION_SPEC_VERSION = 11

REQUIRED_HEADINGS = (
    "## Show Notes",
    "## 摘要",
    "## 核心观点",
    "## 问题与思考",
    "## 关键词",
    "## 人物简介",
    "## 全文转录",
)

# 会话内校订的固定规则：与 src/processor/prompt.py 的自动化规则保持同一套编辑口径。
SESSION_RULES = """你是一名专业的中文播客文稿编辑。下面是一份自包含的会话输入包：节目标题、来源、Show Notes 与带 [SPEAKER_XX] 说话人标签的全文转录。请把全文校订并结构化为最终 Markdown 文稿。

## 输出结构（必须严格包含以下小节，顺序一致）

# 节目标题

> 来源：播客名称 | 节目标题 | 播出日期
（三要素用 | 分隔：播客名、本期节目标题、节目播出日期——日期来自输入包元信息，不是处理时间）

## Show Notes

（完整保留输入包中的 Show Notes 原文，不增删不改写）

## 摘要

（2-3 句话总结本期播客的主题与核心内容；不列要点、不分段，聚焦「这期讲了什么、核心结论是什么」，基于全文但不照搬原文）

## 核心观点

- **主题句。** 支撑证据或展开说明（1-2 句）
（提炼节目中所有重要观点，不限制条数，以覆盖完整为准；每条以完整主题句开头、句号结尾，加粗部分必须是完整句子，只用一次加粗；若某观点有特别值得单独收录、能体现它的原话，在主题句下一行用 > 引用块附上该原话——忠实原文、不加引号、不改写，关键短语可加粗；**无则不加，不要为每条强行配原话**）

## 问题与思考

1. 核心问题（整理式，非原文照抄）？

   对应的思考或回答（1-3 句，经提炼整理，不与原句逐字重复）

（筛选 3-5 个最值得关注、最能引发思考的问题与回答；须为节目真正讨论过的核心议题，不是随机摘取的全文提问；用 1. 2. 3. 编号，问题单独占一行、以问号结尾，换行后另起一行写答案；可与核心观点互补但不重复——若某点已在核心观点中作为论点展开，这里转而呈现其「追问与张力」，不要复述同一结论）

## 关键词

- **关键词**：1-2 句解释。
（提取 4-8 个贯穿节目主题的核心概念、专名或术语，避免一次性细碎名词；每条以「**关键词**：解释」呈现，解释须以句号结尾）

## 人物简介

**身份姓名**：简介
（每人一行，正文格式，不用引用；身份依据 Show Notes 与对话内容，不确定时不编造）

## 全文转录

【身份姓名】说话内容
（说话人标签已替换为真实身份+姓名，用中文方括号【】标注，如【主播湫湫】；按话题分段，段落间空一行）

## 校订规则

1. 忠实保留原文：不改写、不缩写、不省略、不添加原文没有的信息；只修正语音识别的同音错字、专名错误、繁简混用、标点和明显语病；**修正被错误断句的连贯短语——不要在连贯短语中间插入句号造成切断（例如「好不好找工作」被拆成「好。不好找工作」），应按语义连读还原为完整短语**。
2. 说话人映射：根据 Show Notes 与对话内容把 [SPEAKER_XX] 替换为【身份+姓名】（如【主播XX】【嘉宾XX】）。**注意：说话人标签按时间段聚类，一段内可能混入多位说话人（快速问答、无缝接话时尤甚）**：
   - 若一段内出现两个不同人物的人称自述（如“我是A”与“我叫B”同时出现）或明显的问答交替、接话，判定为**多人混段**，必须**按语义拆分为独立段落**并分别标注真实说话人（如【主播XX】与【嘉宾XX】各自成段）；不要机械照搬整段标签，也不要按段首内容给整段贴【】。
   - 语义无法可靠拆分时，保留 [SPEAKER_XX] 不猜测。
   - **校订完成后回查自称一致性**：每段【X】内的第一人称自称（「我是Y」「我叫Y」「我们团队」等）必须与标签 X 一致；若出现指向他人 Y 的自称（如标签【主播湫湫】却写「我是 ACE」），说明该段说话人判错，须按真实自称重标，不要将就原标签。
   - **话轮边界特别警惕**：一段的结尾句若明显开启下一个话题、且下一段以「这个/那/其实/对」等接话词延续同一思路，边界可能被放错——确认尾句归属后再定标签，不要把后一位说话人的开场句误贴给前一位（例如前一段尾句「我最近在做一系列招聘项目」若与下一段「这个同感主要…」连成同一话题，应归给同一说话人）。
3. 删除“嗯、啊、那个”等无意义填充词；“然后、那”只有确属填充词时才删除。
4. 保留“但是、所以、其实、不过”等逻辑转折词。
5. 段落按语义自然分段，每段 2-4 句为宜；同一说话人连续多段只在第一段标注说话人。
6. 双引号统一使用中文直角引号「」（英文术语、代码、URL 内的可保留英文引号）。标有引号或书名号的并列成分之间通常不用顿号（GB/T 15834-2011）：「甲」「乙」或《甲》《乙》直接并列即可，顿号多余；例外是并列项之间有括注等插入成分时宜用顿号。
7. 摘要、核心观点、问题与思考、关键词、人物简介都只基于会话包内容生成，不编造原文没有的信息。
8. 只校订，不创作：不扩写观点、不补充背景、不总结替代原文。
9. 全文转录信息量硬线：保留原文全部事例、数字与对话原貌，口语长叙述不得压缩成概要；校验器以「全文转录 ≥ 原始转录 50%」为硬线（低于 50% 判为过度删减）。若已触发，用 `python -m src.processor.rebuild_transcript <会话包> -o <md>` 从会话包保真重建全文转录后再校订，不要手工重写压缩版。
10. 禁止额外板块：不得生成「大纲」小节，也不得用 mermaid 画思维导图（之前约定已删除该板块）。输出结构严格以上方「输出结构」所列小节为准，不得增删小节。
"""


def build_session_prompt(package_text: str) -> str:
    """Wrap a session package into the fixed, versioned editing prompt.

    ``package_text`` is the whole ``<episode>_diarized.txt`` content (title,
    source, Show Notes, labeled transcript). The returned string is meant to be
    pasted verbatim into an AI chat session.
    """
    return (
        f"[会话内校订规范 v{SESSION_SPEC_VERSION}]\n\n"
        + SESSION_RULES
        + "\n"
        + "---\n\n"
        "以下是待处理的内容。\n\n"
        + package_text
        + "\n"
        + "---\n\n"
        "请直接输出校订后的最终 Markdown 文稿，不要输出其他解释。"
    )


def _split_sections(md_text: str) -> dict[str, str]:
    """Split markdown into {heading: body} by '## ' level headings."""
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in md_text.splitlines():
        m = re.match(r"^(##\s+.+)$", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf)
            current = m.group(1).strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf)
    return sections


def validate_session_output(md_text: str, source_text: str = "") -> list[str]:
    """Return a list of problems with ``md_text`` (empty list means OK).

    Structural checks only — content fidelity is up to the editor. ``source_text``
    is the labeled transcript for a light length sanity check.
    """
    problems: list[str] = []
    sections = _split_sections(md_text)

    for heading in REQUIRED_HEADINGS:
        if heading not in sections:
            problems.append(f"缺少必需小节：{heading}")
            continue
        body = sections[heading].strip()
        if not body:
            problems.append(f"小节为空：{heading}")

    # 来源行：须含播客名 | 节目标题 | 日期 三要素（在标题后、首个 ## 前）
    head_part = md_text.split("## ", 1)[0] if "## " in md_text else md_text
    if "> 来源：" in head_part:
        src_line = next((l for l in head_part.splitlines() if l.strip().startswith("> 来源：")), "")
        if src_line.count("|") < 2:
            problems.append("来源行应含三要素（播客名 | 节目标题 | 播出日期），当前仅 " + src_line.strip())
    else:
        problems.append("缺少来源行（> 来源：播客名 | 节目标题 | 播出日期）")

    # Show Notes：须保留原文小节
    if "## Show Notes" not in md_text:
        problems.append("缺少 Show Notes 小节（应完整保留输入包中的 Show Notes）")

    # 核心观点：条数不限，但每项须以加粗主题句开头
    kp = sections.get("## 核心观点", "")
    kp_items = [line for line in kp.splitlines() if line.strip().startswith("- ")]
    if kp_items and not all(re.search(r"\*\*.+\*\*", item) for item in kp_items):
        problems.append("核心观点中存在未加粗的条目（每条要点须以 **加粗主题句** 开头）")

    # 问题与思考：1. 2. 编号、问答分行，禁用带圈数字 ①②③ 与同行「**问题？** 答案」
    qt_body = sections.get("## 问题与思考", "")
    if qt_body.strip():
        if re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]", qt_body):
            problems.append("问题与思考不应使用带圈数字 ①②③，改用 1. 2. 编号")
        if not re.search(r"^\s*\d+\.", qt_body, re.MULTILINE):
            problems.append("问题与思考应使用 1. 2. 编号（问题单独一行，答案换行另写）")
        if re.search(r"^\s*-\s+\*\*.+?\?\*\*\s+\S", qt_body, re.MULTILINE):
            problems.append("问题与思考不应把问题与答案写在同行（旧格式 **问题？** 答案），应 1. 编号、问题单独一行、答案换行另写")

    # 关键词：4-8 条，每项为 **关键词**：解释，且解释以句号结尾
    keywords = [line for line in sections.get("## 关键词", "").splitlines() if line.strip().startswith("- ")]
    if keywords:
        if not 4 <= len(keywords) <= 8:
            problems.append(f"关键词应提取 4-8 个（实际 {len(keywords)}）")
        if not all(re.search(r"\*\*.+?\*\*[:：]", item) for item in keywords):
            problems.append("关键词中存在格式不符的条目（每条须为 **关键词**：解释）")
        if not all(item.rstrip().endswith(("。", ".")) for item in keywords):
            problems.append("关键词每条解释须以句号结尾")

    # 人物简介：每人一行正文（**身份**：简介），不用引用
    intro = sections.get("## 人物简介", "")
    intro_lines = [line for line in intro.splitlines() if line.strip()]
    if intro:
        if not intro_lines:
            problems.append("人物简介为空")
        elif any(l.strip().startswith(">") for l in intro_lines):
            problems.append("人物简介不应使用引用格式（> 开头），请用正文格式 **身份**：简介")
        elif not all(re.match(r"^\*\*.+?\*\*[:：]", l.strip()) for l in intro_lines):
            problems.append("人物简介条目应为 **身份姓名**：简介 格式")

    # 全文转录：说话人须用中文方括号【身份姓名】，不得用 **加粗**：样式
    transcript = sections.get("## 全文转录", "")
    if re.search(r"^\s*\*\*.+?\*\*[:：]", transcript, re.MULTILINE):
        problems.append("全文转录说话人应使用中文方括号【身份姓名】，不要用 **加粗**：样式")
    # 全文转录：不得残留未映射的 SPEAKER 标签（混段无法可靠拆分时可保留，但需人工复核）
    leftover = re.findall(r"\[SPEAKER_\d+\]", transcript)
    if leftover:
        problems.append(
            f"全文转录残留未映射的说话人标签（混段无法可靠拆分时可保留，但需复核）：{sorted(set(leftover))}"
        )

    # 全文转录：说话人标签与自称一致性（启发式标红疑似错位，不自动改）
    for m in re.finditer(r"^【([^】]+)】", transcript, re.MULTILINE):
        label = m.group(1)
        label_core = re.sub(r"^(主播|嘉宾|主持人|老师|同学|教授|博士|先生|女士)", "", label).strip()
        start = m.end()
        nxt = re.search(r"\n【[^】]+】", transcript[start:])
        seg = transcript[start: start + (nxt.start() if nxt else len(transcript) - start)]
        for sm in re.finditer(
            r"我(?:是|叫|们是|就是)\s*([\u4e00-\u9fa5A-Za-z]{1,8})(?=[，。！？；、\s])", seg
        ):
            y = sm.group(1).strip()
            if y and y != label_core and y not in ("做", "在", "一个", "这个", "那个", "他们", "她们"):
                problems.append(
                    f"全文转录中【{label}】段落出现自称「我是{y}」等指向他人「{y}」的表述，"
                    f"疑似说话人标签错位，请人工复核该段归属"
                )
                break

    # 全文转录：说话人占比失衡（启发式，仅标红不自动改）
    # 两人对话里某人独占绝大多数段落，通常是 diarization 过度切分或标签归并错误
    # （如 vol.231：14 个伪说话人被错归并，嘉宾占满全文）。
    spk_openings = re.findall(r"^\s*【([^】]+)】", transcript, re.MULTILINE)
    if len(spk_openings) >= 4:
        from collections import Counter

        _cnt = Counter(spk_openings)
        _top, _top_n = _cnt.most_common(1)[0]
        _total = sum(_cnt.values())
        if _top_n / _total > 0.85:
            problems.append(
                f"说话人占比失衡：全文转录中【{_top}】独占 {_top_n}/{_total} 段"
                f"（{_top_n / _total:.0%}），疑似 diarization 失败或说话人标签归并错误，请人工复核"
            )

    if source_text and transcript:
        ratio = len(transcript) / max(len(source_text), 1)
        if ratio < 0.5:
            problems.append(
                f"全文转录长度仅为原始转录的 {ratio:.0%}（疑似过度删减）。"
                "校订须保留原文全部信息（事例、数字、对话），只做填充词删除/错字修正/分段，"
                "不得把口语叙述压缩成概要；若已过度压缩，可用 "
            "`python -m src.processor.rebuild_transcript <包> -o <md>` 从会话包保真重建全文转录。"
        )

    # 禁止额外小节（大纲）与 mermaid 思维导图
    if re.search(r"^##\s+大纲", md_text, re.MULTILINE):
        problems.append("不得生成「大纲」板块（之前约定已删除），请移除该小节")
    if "```mermaid" in md_text or re.search(r"^\s*mindmap\s*$", md_text, re.MULTILINE):
        problems.append("不得生成 mermaid 思维导图，请移除相关代码块")

    # 标点：标有引号或书名号的并列成分之间不用顿号（GB/T 15834-2011）
    if re.search(r'([”」』》"])、([「『《“"])', md_text):
        problems.append(
            "标有引号或书名号的并列成分之间不应使用顿号（GB/T 15834-2011），"
            "如「甲」「乙」或《甲》《乙》直接并列即可，顿号多余"
        )

    return problems


def _strip_unwanted_sections(md_text: str) -> str:
    """剥离规范禁止的板块（如「大纲」）与 mermaid 思维导图。

    作为安全网：即使 LLM 仍产出这些多余内容，写盘前也会被移除，
    不让它们进入最终 .md。第 10 条校订规则已从提示词层面禁止，此处兜底。
    """
    lines = md_text.splitlines()
    out: list[str] = []
    in_block: str | None = None  # None / "heading"（大纲）/ "mermaid"
    for line in lines:
        if in_block == "heading":
            # 处于「大纲」板块内：跳过所有行，直到遇到下一个二级标题才结束
            if re.match(r"^##\s+\S", line):
                in_block = None
                out.append(line)
            continue
        if in_block == "mermaid":
            if line.strip() == "```":
                in_block = None
            continue
        if line.strip().startswith("```mermaid"):
            in_block = "mermaid"
            continue
        if re.match(r"^##\s+大纲", line):
            in_block = "heading"
            continue
        out.append(line)
    return "\n".join(out)


def _run_with_llm(package_text: str, llm_config) -> str:
    """Run the fixed session prompt through the LLM once and return the markdown."""
    from src.processor.llm_processor import resolve_llm_config

    config = resolve_llm_config(llm_config)
    from openai import OpenAI

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": SESSION_RULES},
            {"role": "user", "content": package_text},
        ],
        temperature=0.1,
        max_tokens=32_000,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def _extract_source(line: str) -> str:
    """Parse '> 来源：播客  |  日期' into '播客 | 日期'."""
    m = re.match(r">\s*来源[:：]\s*(.+)$", line.strip())
    return m.group(1).strip() if m else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="撷声会话内校订：生成固定校订提示词，或用 LLM 自动完成并校验"
    )
    parser.add_argument("package", type=Path, help="会话输入包（<节目名>_diarized.txt）")
    parser.add_argument("-o", "--output", type=Path, help="输出 .md 路径（默认与包同名）")
    parser.add_argument("--prompt-only", action="store_true",
                        help="只生成可粘贴的会话提示词，不调用 LLM")
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=sorted(LLM_MODELS),
                        help="LLM 提供方（默认: qwen）")
    parser.add_argument("--llm-model",
                        help="覆盖提供方默认模型；可用逗号分隔指定多个候选（如 qwen3.7-plus,gpt-5.2）")
    args = parser.parse_args()

    if not args.package.exists():
        print(f"错误：找不到会话输入包 {args.package}", file=sys.stderr)
        sys.exit(1)
    package_text = args.package.read_text(encoding="utf-8")

    if args.prompt_only:
        print(build_session_prompt(package_text))
        return

    llm_config = get_llm_config(args.llm_provider, args.llm_model)
    if not llm_config.api_key:
        print(
            "未检测到 LLM API Key（LLM_API_KEY / LLM_BASE_URL 未设置）。\n"
            "可用 --prompt-only 生成固定提示词，粘贴到任一 AI 会话完成校订。",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("Running session edit with %s/%s ...", llm_config.provider, llm_config.model)
    try:
        md = _run_with_llm(package_text, llm_config)
    except Exception as exc:
        logger.exception("LLM 校订失败")
        print(f"\n错误: {exc}", file=sys.stderr)
        sys.exit(1)

    # 写盘前先剥离规范禁止的板块（大纲 / mermaid），作为安全网
    md = _strip_unwanted_sections(md)
    source_text = package_text
    problems = validate_session_output(md, source_text=source_text)
    if problems:
        print("校验未通过：", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        # 仍写盘，但提示用户复核
        print("\n（已输出，请按上述问题复核后使用）")

    out = args.output or args.package.with_name(args.package.stem.replace("_diarized", "") + ".md")
    out.write_text(md, encoding="utf-8")
    print(f"\n[OK] 校订文稿已保存: {out}")
    if not problems:
        print("校验通过。")
    else:
        print(f"校验未通过（{len(problems)} 项问题），见上方提示。")
        sys.exit(2)


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
