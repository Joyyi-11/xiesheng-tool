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
SESSION_SPEC_VERSION = 4

REQUIRED_HEADINGS = (
    "## 内容提要",
    "## 闪光语句",
    "## 大纲",
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

# Show Notes

（完整保留输入包中的 Show Notes 原文，不增删不改写）

## 内容提要

- **主题句。** 支撑证据或展开说明（1-2 句）
（提炼节目中所有重要观点，不限制条数，以覆盖完整为准；每条以完整主题句开头、句号结尾，加粗部分必须是完整句子，只用一次加粗）

## 闪光语句

- **关键词或短语**：原话中的精彩表述
（抓取所有值得保留的原话，不限制条数；必须忠实原文，不加引号；每条把最能代表该句的**关键词或关键短语加粗**——加粗对象是原文中出现的词或短语，不是整句）

## 大纲

```mermaid
mindmap
  root((节目标题))
    01:24 一级主题
      二级要点
```
（用 Mermaid mindmap 代码块呈现节目结构；一级主题带真实播放时间（mm:ss，来自输入包的时间段信息，无法确定时用序号），每个一级主题 6-15 字，其下可有 1-5 个二级要点；大纲是结构梳理，与内容提要的论点提炼分工，不要简单重复）

## 关键词

- **关键词**：1-2 句解释
（提取 4-8 个贯穿节目主题的核心概念、专名或术语，避免一次性细碎名词）

## 人物简介

**身份姓名**：简介
（每人一行，正文格式，不用引用；身份依据 Show Notes 与对话内容，不确定时不编造）

## 全文转录

**身份姓名**：说话内容
（说话人标签已替换为真实身份+姓名；按话题分段，段落间空一行）

## 校订规则

1. 忠实保留原文：不改写、不缩写、不省略、不添加原文没有的信息；只修正语音识别的同音错字、专名错误、繁简混用、标点和明显语病。
2. 说话人映射：根据 Show Notes 与对话内容把 [SPEAKER_XX] 替换为“身份+姓名”（如“主播XX”“嘉宾XX”）。**注意：说话人标签按时间段聚类，一段内可能混入多位说话人（快速问答、无缝接话时尤甚）**：
   - 若一段内出现两个不同人物的人称自述（如“我是A”与“我叫B”同时出现）或明显的问答交替、接话，判定为**多人混段**，必须**按语义拆分为独立段落**并分别标注真实说话人（如“主播XX：”与“嘉宾XX：”各自成段）；不要机械照搬整段标签，也不要按段首内容给整段贴名。
   - 语义无法可靠拆分时，保留 [SPEAKER_XX] 不猜测。
3. 删除“嗯、啊、那个”等无意义填充词；“然后、那”只有确属填充词时才删除。
4. 保留“但是、所以、其实、不过”等逻辑转折词。
5. 段落按语义自然分段，每段 2-4 句为宜；同一说话人连续多段只在第一段标注说话人。
6. 双引号统一使用中文直角引号「」（英文术语、代码、URL 内的可保留英文引号）。
7. 内容提要、闪光语句、大纲、关键词、人物简介都只基于会话包内容生成，不编造原文没有的信息。
8. 只校订，不创作：不扩写观点、不补充背景、不总结替代原文。
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
    if "# Show Notes" not in md_text:
        problems.append("缺少 Show Notes 小节（应完整保留输入包中的 Show Notes）")

    # 内容提要：条数不限，但每项须以加粗主题句开头
    kp = sections.get("## 内容提要", "")
    kp_items = [line for line in kp.splitlines() if line.strip().startswith("- ")]
    if kp_items and not all(re.search(r"\*\*.+\*\*", item) for item in kp_items):
        problems.append("内容提要中存在未加粗的条目（每条要点须以 **加粗主题句** 开头）")

    # 闪光语句：条数不限，但每项须含加粗关键词（原文中的词或短语）
    quotes = [line for line in sections.get("## 闪光语句", "").splitlines() if line.strip().startswith("- ")]
    if quotes and not all(re.search(r"\*\*.+\*\*", item) for item in quotes):
        problems.append("闪光语句中存在未加粗关键词的条目（每条须把原话中的 **关键词/短语** 加粗）")

    # 大纲：Mermaid mindmap 代码块，含 root 与一级主题（带时间或序号）
    outline = sections.get("## 大纲", "")
    mermaid_block = re.search(r"```mermaid\s*\n(mindmap[\s\S]*?)```", outline)
    if outline and not mermaid_block:
        problems.append("大纲未使用 ```mermaid mindmap 代码块（应包含 mindmap 与 root((标题))）")
    if mermaid_block:
        body = mermaid_block.group(1)
        top_topics = [l for l in body.splitlines() if re.match(r"^\s{4}\S", l) and "root" not in l]
        if len(top_topics) < 3:
            problems.append(f"大纲一级主题应至少 3 个（实际 {len(top_topics)}）")
        if not all(re.search(r'^\s{4}"?\d{1,2}:\d{2}"?\s', l) or re.search(r"^\s{4}\d{2}\s", l) for l in top_topics):
            problems.append("大纲一级主题应带真实时间（mm:ss）或序号开头")

    # 关键词：4-8 条，且每项为 **关键词**：解释
    keywords = [line for line in sections.get("## 关键词", "").splitlines() if line.strip().startswith("- ")]
    if keywords:
        if not 4 <= len(keywords) <= 8:
            problems.append(f"关键词应提取 4-8 个（实际 {len(keywords)}）")
        if not all(re.search(r"\*\*.+?\*\*[:：]", item) for item in keywords):
            problems.append("关键词中存在格式不符的条目（每条须为 **关键词**：解释）")

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

    # 全文转录：不得残留未映射的 SPEAKER 标签（混段无法可靠拆分时可保留，但需人工复核）
    transcript = sections.get("## 全文转录", "")
    leftover = re.findall(r"\[SPEAKER_\d+\]", transcript)
    if leftover:
        problems.append(
            f"全文转录残留未映射的说话人标签（混段无法可靠拆分时可保留，但需复核）：{sorted(set(leftover))}"
        )

    if source_text and transcript:
        ratio = len(transcript) / max(len(source_text), 1)
        if ratio < 0.5:
            problems.append(f"全文转录长度仅为原始转录的 {ratio:.0%}（疑似过度删减）")

    return problems


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
