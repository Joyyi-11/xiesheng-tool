"""Deterministic in-session transcript editing (the session-link path).

The session-link (``--llm-mode session``) hands the user a self-contained session package
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

from src.config import DEFAULT_LLM_PROVIDER, LLM_MODELS, get_llm_config, llm_configured
from src.diarization.speaker_resolver import COMMON_ALIASES
from src.utils import (
    LEGACY_TRANSCRIPT_HEADING,
    SPEAKER_LABEL_RE,
    TRANSCRIPT_HEADING,
    TRANSCRIPT_HEADING_RE,
    get_transcript_body,
)

logger = logging.getLogger(__name__)

# ============================================================
# 编辑口径单一权威源：session 与 api 两链路的校订/结构化条款
# 全部来自 src.processor.rules（含版本号），本模块不再自维护规则文字。
# 会话链路特有的排版骨架（Markdown 小节/示例行）保留在本模块。
# ============================================================

from src.processor.rules import (  # noqa: E402
    EDIT_RULES,
    RULES_SPEC_VERSION,
    STRUCT_ANCHORS,
    STRUCT_GLOSSARY,
    STRUCT_INTRO,
    STRUCT_KEY_POINT,
    STRUCT_QUESTIONS,
    STRUCT_SUMMARY,
)

# 版本 = rules.RULES_SPEC_VERSION（单一权威源）：改编辑口径只需改 src/processor/rules.py，
# 三处版本随之同步、旧缓存自动作废。禁止在本模块另写一份口径文本。
# 转录小节标题一律引用 src.utils.TRANSCRIPT_HEADING，禁止字面量硬编码。
SESSION_SPEC_VERSION = RULES_SPEC_VERSION

REQUIRED_HEADINGS = (
    "## Show Notes",
    "## 摘要",
    "## 核心观点",
    "## 问题与思考",
    "## 术语表",
    "## 人物简介",
    TRANSCRIPT_HEADING,
)

# 小节别名：更名后旧稿仍可被校验，不因标题改名被判「缺少必需小节」。
_HEADING_ALIASES: dict[str, tuple[str, ...]] = {
    TRANSCRIPT_HEADING: (LEGACY_TRANSCRIPT_HEADING,),
}

# 会话内校订的固定规则：内容口径全部来自 src.processor.rules（与 api 链路同一权威源），
# 本模块只负责「会话特有的排版骨架」（Markdown 小节/示例行）与流程层说明。
def _build_session_rules() -> str:
    edits = "\n".join(f"{i}. {t}" for i, t in enumerate(EDIT_RULES, 1))
    # 会话链路特有条款：api 链路把这些职责交给渲染器/护栏，session 需 LLM 自守。
    session_only = (
        "## 会话链路补充（仅本链路需自守，api 链路由代码护栏等价保证）\n\n"
        f"{len(EDIT_RULES) + 1}. 说话人映射：根据 Show Notes 与对话内容把 [SPEAKER_XX] 替换为【身份+姓名】"
        "（如【主播XX】【嘉宾XX】）。**注意：说话人标签按时间段聚类，一段内可能混入多位说话人"
        "（快速问答、无缝接话时尤甚）**：若一段内出现两个不同人物的人称自述（如「我是A」与「我叫B」"
        "同时出现）或明显的问答交替、接话，判定为**多人混段**，必须**按语义拆分为独立段落**并分别标注"
        "真实说话人；语义无法可靠拆分时，保留 [SPEAKER_XX] 不猜测。\n"
        f"{len(EDIT_RULES) + 2}. **校订完成后回查自称一致性**：每段【X】内的第一人称自称"
        "（「我是Y」「我叫Y」等）必须与标签 X 一致；若出现指向他人 Y 的自称（如标签【主播湫湫】"
        "却写「我是 ACE」），说明该段说话人判错，须按真实自称重标。**话轮边界特别警惕**：一段的"
        "结尾句若明显开启下一个话题、且下一段以「这个/那/其实/对」等接话词延续同一思路，边界可能"
        "被放错——确认尾句归属后再定标签。\n"
        f"{len(EDIT_RULES) + 3}. 摘要、核心观点、问题与思考、术语表、人物简介都只基于会话包内容生成，"
        "不编造原文没有的信息；只校订，不创作：不扩写观点、不补充背景、不总结替代原文。\n"
        f"{len(EDIT_RULES) + 4}. 原文转录信息量硬线：保留原文全部事例、数字与对话原貌，口语长叙述"
        "不得压缩成概要；校验器以「原文转录 ≥ 原始转录 50%」为硬线（低于 50% 判为过度删减）。若已触发，"
        "用 `python -m src.processor.rebuild_transcript <会话包> -o <md>` 从会话包保真重建原文转录后再校订，"
        "不要手工重写压缩版。\n"
        f"{len(EDIT_RULES) + 5}. 禁止额外板块：不得生成「大纲」小节，也不得用 mermaid 画思维导图。"
        "输出结构严格以上方「输出结构」所列小节为准，不得增删小节。"
    )
    return f"""你是一名专业的中文播客文稿编辑。下面是一份自包含的会话输入包：节目标题、来源、Show Notes 与带 [SPEAKER_XX] 说话人标签的原文转录。请把全文校订并结构化为最终 Markdown 文稿。

## 输出结构（必须严格包含以下小节，顺序一致）

# 节目标题

> 来源：播客名称 | 节目标题 | 播出日期
（三要素用 | 分隔：播客名、本期节目标题、节目播出日期——日期来自输入包元信息，不是处理时间）

## Show Notes

（完整保留输入包中的 Show Notes 原文，不增删不改写）

## 摘要

（{STRUCT_SUMMARY}）

## 核心观点

- **观点句。** 展开论述句（1-2 句）「值得单独收录的原话」
  - **观点内特有术语**：1-2 句就近解释。
（{STRUCT_KEY_POINT}）

## 问题与思考

1. **核心问题（整理式，非原文照抄）？**

   对应的思考或回答（直接解答该问题，1-3 句，引用人物须点明身份，不与原句逐字重复）

（用 1. 2. 3. 编号，**序号写在加粗外、问题整句加粗**（`1. **问题？**`），问题单独占一行、以问号结尾，换行后另起一行写答案且答案不加粗。{STRUCT_QUESTIONS}）

## 术语表

- **残留术语**：1-2 句解释。
（{STRUCT_GLOSSARY}）

## 人物简介

**身份姓名**：简介
（{STRUCT_INTRO}）

{TRANSCRIPT_HEADING}

【身份姓名】说话内容
（说话人标签已替换为真实身份+姓名，用中文方括号【】标注，如【主播湫湫】；按话题分段，段落间空一行；同一说话人的相邻段落必须合并为一段——源转录按语音停顿切碎，同一人常被切成连续多段，只在该说话人发言开头贴一次标签，只有说话人真正切换时才另起一段。说话人标签本身不得加粗。{STRUCT_ANCHORS}）

## 校订规则

{edits}

{session_only}"""


SESSION_RULES = _build_session_rules()


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


# 自称校验去噪：这些语素出现在「我是 Y」之后时，Y 多为自述性描述而非他人姓名。
_NON_NAME_MORPH = set(
    "人生产友者的是个一这那做在就否很太不啊呢吧吗嘛跟和与会与告诉"
    "你们他们她们我们之乎也者给被把让叫为及或都更最觉想说看听批拌紧绷"
    # 动词/助词类语素：中文人名几乎不用，出现即说明 Y 是动作陈述而非姓名。
    # 实测误报来源（黄金时代集）：「我是通过了 N1 那个考试」「我就是要先发」
    # 被当成自称他人姓名，人工复核确认是本人口语自述。
    "了过着要还又没挺别"
)

# 职业/头衔词：出现在「我是 Y」里时，Y 是身份/职业自述而非他人姓名。
# 例：Lenny 说「我是设计师」——他本就是主持人，讲职业背景，非串标（连漪确认 OK）。
_PROFESSION_WORDS = {
    "设计师", "工程师", "程序员", "架构师", "产品经理", "项目经理", "研究员",
    "科学家", "教授", "博士", "导师", "老师", "学生", "医生", "护士", "律师",
    "法官", "记者", "编辑", "作者", "作家", "诗人", "艺术家", "画家", "音乐家",
    "歌手", "演员", "导演", "主持人", "主播", "投资人", "创始人", "联创",
    "合伙人", "老板", "经理", "总裁", "总监", "行长", "局长", "校长", "院长",
    "公务员", "军人", "警察", "消防员", "厨师", "司机", "工人", "农民", "商人",
    "企业家", "分析师", "顾问", "教练", "裁判", "运动员", "模特", "网红", "博主",
    "运营", "销售", "会计", "翻译", "码农", "白领",
}

# MBTI 十六型：「我是 INFP」是性格自述，不是姓名（实测误报：AI越厉害集【废废】段）。
_MBTI_TYPES = {
    "intj", "intp", "entj", "entp", "infj", "infp", "enfj", "enfp",
    "istj", "isfj", "estj", "esfj", "istp", "isfp", "estp", "esfp",
}


def _is_plausible_self_name(y: str) -> bool:
    """判断「我是 Y」里的 Y 是否像一个人名（值得作为说话人错位信号）。

    仅 2-3 字纯中文（且不含非姓名语素）或 2-4 字母的拉丁名才算；
    「我是拌面 / 我是真人 / 我是本科学历 / 我是设计师」等自述性短语返回 False，
    避免把本人真实自述或职业背景误判为说话人错位。
    """
    if not y:
        return False
    if y in _PROFESSION_WORDS:
        return False
    if re.fullmatch(r"[一-龥]{2,3}", y):
        return not any(ch in _NON_NAME_MORPH for ch in y)
    if re.fullmatch(r"[A-Za-z]{2,4}", y):
        return y.lower() not in _MBTI_TYPES
    return False


def validate_session_output(md_text: str, source_text: str = "") -> list[str]:
    """Return a list of problems with ``md_text`` (empty list means OK).

    Structural checks only — content fidelity is up to the editor. ``source_text``
    is the labeled transcript for a light length sanity check.
    """
    problems: list[str] = []
    sections = _split_sections(md_text)

    for heading in REQUIRED_HEADINGS:
        actual = heading
        if heading not in sections:
            # 兼容历史别名（如旧稿的「## 全文转录」）：命中别名即视为该小节存在
            alias = next((h for h in _HEADING_ALIASES.get(heading, ()) if h in sections), None)
            if alias is None:
                problems.append(f"缺少必需小节：{heading}")
                continue
            actual = alias
        body = sections[actual].strip()
        if not body and heading != "## 术语表":
            # 术语表为「残留术语」区：允许 0 条（标题仍需存在，见上）
            problems.append(f"小节为空：{heading}")

    # 来源行：须含播客名 | 节目标题 | 日期 三要素（在标题后、首个 ## 前）
    head_part = md_text.split("## ", 1)[0] if "## " in md_text else md_text
    if "> 来源：" in head_part:
        src_line = next(
            (line for line in head_part.splitlines() if line.strip().startswith("> 来源：")), ""
        )
        if src_line.count("|") < 2:
            problems.append("来源行应含三要素（播客名 | 节目标题 | 播出日期），当前仅 " + src_line.strip())
    else:
        problems.append("缺少来源行（> 来源：播客名 | 节目标题 | 播出日期）")

    # Show Notes：须保留原文小节
    if "## Show Notes" not in md_text:
        problems.append("缺少 Show Notes 小节（应完整保留输入包中的 Show Notes）")

    # 核心观点：条数不限，但每项须以加粗主题句开头
    kp = sections.get("## 核心观点", "")
    kp_lines = kp.splitlines()
    kp_items = [line for line in kp_lines if line.strip().startswith("- ")]
    if kp_items and not all(re.search(r"\*\*.+\*\*", item) for item in kp_items):
        problems.append("核心观点中存在未加粗的条目（每条要点须以 **加粗主题句** 开头）")

    # 核心观点：禁止「**观点句**：展开」的框架式冒号（v17）——本节是「观点句、
    # 展开论述句、引用句」三段顺次相接，主题句以句号收尾后直接空格接展开句。
    # 缩进的术语子条目（  - **术语**：解释）是定义式标签，不受本条限制。
    for _kp_line in kp_lines:
        if re.match(r"^-\s+\*\*.+?\*\*\s*[:：]", _kp_line):
            problems.append(
                "核心观点主题句后不应使用冒号引出展开（如「**观点句**：展开」），"
                "应改为「**观点句。** 展开论述句「引用句」」三段顺次相接、空格分隔"
                "（主题句内部必要冒号可保留，但整句须以句号收尾）"
            )
            break

    # 问题与思考：1. 2. 编号、问答分行，禁用带圈数字 ①②③ 与同行「**问题？** 答案」
    qt_body = sections.get("## 问题与思考", "")
    if qt_body.strip():
        if re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]", qt_body):
            problems.append("问题与思考不应使用带圈数字 ①②③，改用 1. 2. 编号")
        if not re.search(r"^\s*\d+\.", qt_body, re.MULTILINE):
            problems.append("问题与思考应使用 1. 2. 编号（问题单独一行，答案换行另写）")
        if re.search(r"^\s*-\s+\*\*.+?\?\*\*\s+\S", qt_body, re.MULTILINE):
            problems.append("问题与思考不应把问题与答案写在同行（旧格式 **问题？** 答案），应 1. 编号、问题单独一行、答案换行另写")
        # v18：问题须整句加粗、序号留在加粗外——问题比答案更该先跳出来
        if re.search(r"^\s*\d+\.\s+(?!\*\*)", qt_body, re.MULTILINE):
            problems.append("问题与思考的问题须整句加粗、序号留在加粗外（写成 1. **问题？**），答案不加粗")
        # 答案不得结尾反抛新问号：直接解答，而非把问题丢回读者
        for _blk in re.split(r"(?m)^\s*\d+\.", qt_body):
            _lines = [ln.rstrip() for ln in _blk.splitlines() if ln.strip()]
            if not _lines:
                continue
            _ans = _lines
            for _i, _ln in enumerate(_lines):
                if _ln.endswith("？") or _ln.endswith("?"):
                    _ans = _lines[_i + 1:]
                    break
            if _ans and (_ans[-1].endswith("？") or _ans[-1].endswith("?")):
                problems.append("问题与思考的答案不应以新问号结尾（须直接解答该问题，勿把问题抛回读者）")
                break

    # 术语表：残留术语 0 起（不强制条数），有条目时每项为 **术语**：解释，且解释以句号结尾
    terms = [line for line in sections.get("## 术语表", "").splitlines() if line.strip().startswith("- ")]
    if terms:
        if not all(re.search(r"\*\*.+?\*\*[:：]", item) for item in terms):
            problems.append("术语表中存在格式不符的条目（每条须为 **术语**：解释）")
        if not all(item.rstrip().endswith(("。", ".")) for item in terms):
            problems.append("术语表每条解释须以句号结尾")

    # 人物简介：每人一行正文（**身份**：简介），不用引用
    intro = sections.get("## 人物简介", "")
    intro_lines = [line for line in intro.splitlines() if line.strip()]
    if intro:
        if intro.strip() == "（无）":
            pass
        elif not intro_lines:
            problems.append("人物简介为空；无法确认任何人物信息时应输出「（无）」")
        elif any(line.strip().startswith(">") for line in intro_lines):
            problems.append("人物简介不应使用引用格式（> 开头），请用正文格式 **身份**：简介")
        elif not all(re.match(r"^\*\*.+?\*\*[:：]", line.strip()) for line in intro_lines):
            problems.append("人物简介条目应为 **身份姓名**：简介 格式")

    # 原文转录：说话人须用中文方括号【身份姓名】，不得用 **加粗**：样式
    transcript = get_transcript_body(sections)
    if re.search(r"^\s*\*\*.+?\*\*[:：]", transcript, re.MULTILINE):
        problems.append("原文转录说话人应使用中文方括号【身份姓名】，不要用 **加粗**：样式")
    # 原文转录：不得残留未映射的 SPEAKER 标签（混段无法可靠拆分时可保留，但需人工复核）
    leftover = SPEAKER_LABEL_RE.findall(transcript)
    if leftover:
        problems.append(
            f"原文转录残留未映射的说话人标签（混段无法可靠拆分时可保留，但需复核）：{sorted(set(leftover))}"
        )

    # 原文转录：说话人标签与自称一致性（启发式标红疑似错位，不自动改）
    # 去噪：①谐音别名（小猪→小朱、秋秋→湫湫）按 COMMON_ALIASES 归一后再比对；
    #       ②只在 Y 像「人名」时才判错位——自述性短语（我是拌面/我是真人/我是本科学历）
    #         含人/生/友/者/的/做/在…语素，或非 2-3 字中文、非 2-4 字母，一律视为本人真实
    #         自述，跳过，避免把口语自述误判为说话人错位。
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
            if not y:
                continue
            # ① 谐音别名归一：归一后即等于本段标签，属正确自述，跳过。
            #    标签可能带未收录的身份前缀（如【课代表立正】自称「我是立正」），
            #    故用「一方包含另一方」判定一致，不要求严格相等。
            y_norm = COMMON_ALIASES.get(y, y)
            #    注意 label_core 可能为空（标签只有身份词如【主播】），此时不做包含判定，
            #    否则空串恒被包含、所有错位都被跳过。
            if y_norm == label_core or y_norm in label or (label_core and label_core in y_norm):
                continue
            # ② 仅当 Y 像人名才判错位；自述性短语（属性/描述）跳过去噪
            if not _is_plausible_self_name(y):
                continue
            if y != label_core:
                problems.append(
                    f"原文转录中【{label}】段落出现自称「我是{y}」等指向他人「{y}」的表述，"
                    f"疑似说话人标签错位，请人工复核该段归属"
                )
                break

    # 原文转录：说话人占比失衡（启发式，仅标红不自动改）
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
                f"说话人占比失衡：原文转录中【{_top}】独占 {_top_n}/{_total} 段"
                f"（{_top_n / _total:.0%}），疑似 diarization 失败或说话人标签归并错误，请人工复核"
            )

    # 原文转录：同一说话人相邻多段未合并（v16）
    # 源转录按语音停顿切碎、同一人还常被判成多个簇，校订后若仍连续多段同标签，
    # 观感极碎且违反校订规则第 5 条。此处只标红提醒，不自动改（避免误并真实轮次）；
    # 确定性合并用 src.utils.merge_same_speaker_blocks。
    _spk_seq = [m.group(1) for m in re.finditer(r"^【([^】]+)】", transcript, re.MULTILINE)]
    if _spk_seq:
        _cur_label, _cur_len = _spk_seq[0], 1
        _max_label, _max_len = _spk_seq[0], 1
        for _prev, _cur in zip(_spk_seq, _spk_seq[1:]):
            if _cur == _prev:
                _cur_len += 1
                if _cur_len > _max_len:
                    _max_label, _max_len = _prev, _cur_len
            else:
                _cur_label, _cur_len = _cur, 1
        if _max_len >= 3:
            problems.append(
                f"原文转录中【{_max_label}】连续 {_max_len} 段未合并：同一说话人的相邻段落必须合并为一段"
                "（源转录按语音停顿切碎，同一人还常被 diarization 判成多个簇），"
                "只有说话人真正切换时才另起一段（见校订规则第 5 条）；"
                "可用 src.utils.merge_same_speaker_blocks 确定性合并"
            )

    if source_text and transcript:
        # 比率只比「正文内容」，先把说话人标签（[SPEAKER_XX] / 【姓名】）与空白剥掉，
        # 否则 diarized 源里 1229 个冗长 [SPEAKER_XX] 标签会把分母虚高近一倍，造成假阳性。
        _strip = lambda s: re.sub(r"【[^】]*】|\[SPEAKER_\d+\]|\s", "", s)
        # 部分会话包头部嵌入了 Show Notes / 简介等非语音 markdown 文本（如「專家型通才」
        # 集 4977 字符头部），首个 [SPEAKER_XX] 标签之后才是语音转录。分母若含头部，
        # 会把保留率虚低（实测 30% vs 真实 103%），此处从首个语音标签起算。
        _src = source_text
        _first_spk = re.search(r"\[SPEAKER_\d+\]", _src)
        if _first_spk:
            _src = _src[_first_spk.start():]
        _src_body = _strip(_src)
        _tr_body = _strip(transcript)
        ratio = len(_tr_body) / max(len(_src_body), 1)
        if ratio < 0.5:
            problems.append(
                f"原文转录长度仅为原始转录的 {ratio:.0%}（疑似过度删减）。"
                "校订须保留原文全部信息（事例、数字、对话），只做填充词删除/错字修正/分段，"
                "不得把口语叙述压缩成概要；若已过度压缩，可用 "
            "`python -m src.processor.rebuild_transcript <包> -o <md>` 从会话包保真重建原文转录。"
        )

    # 禁止额外小节（大纲）与 mermaid 思维导图
    if re.search(r"^##\s+大纲", md_text, re.MULTILINE):
        problems.append("不得生成「大纲」板块（之前约定已删除），请移除该小节")
    if "```mermaid" in md_text or re.search(r"^\s*mindmap\s*$", md_text, re.MULTILINE):
        problems.append("不得生成 mermaid 思维导图，请移除相关代码块")

    # 代码块围栏完整性：v12 七段结构（Show Notes/摘要/核心观点/问题与思考/
    # 术语表/人物简介/原文转录）均为标题+列表+段落，无代码块需求；
    # 任何 ``` 围栏（含未闭合）都属异常，需移除（引用原文用「」或缩进，勿用 ```）。
    # 这是「内容被整段塞进未闭合代码块」类问题的机器可验硬约束（api 链路由
    # build_output_markdown 纯函数保证无围栏，此规则主要兜 session 链路/手动稿）。
    fence_count = md_text.count("```")
    if fence_count:
        if fence_count % 2 != 0:
            problems.append(
                f"检测到 {fence_count} 个代码块围栏（奇数，未闭合）：v12 结构不应含代码块，"
                "请移除多余围栏（引用原文用「」或缩进，勿用 ```）"
            )
        else:
            problems.append(
                f"检测到 {fence_count} 个代码块围栏：v12 结构不应含代码块，请移除"
                "（引用原文用「」或缩进，勿用 ```）"
            )

    # 标点：标有引号或书名号的并列成分之间不用顿号（GB/T 15834-2011）。
    # 只查「校订者撰写的章节」（摘要→人物简介：即 ## 摘要 与原文转录之间）。
    # Show Notes 是节目源简介透传（可能为长文，含源文案引号+顿号，如「"人工智能+哲学"、
    # "计算社会科学"」）、原文转录是源语音，均不归校订管，排除在检查范围外。
    _editorial = md_text
    _ab = re.search(r"^##\s*摘要\s*$", md_text, re.MULTILINE)
    _tr_m = TRANSCRIPT_HEADING_RE.search(md_text)
    if _ab and _tr_m:
        _editorial = md_text[_ab.end() : _tr_m.start()]
    elif _ab and not _tr_m:
        _editorial = md_text[_ab.end() :]
    elif _tr_m and not _ab:
        _editorial = md_text[:_tr_m.start()]
    if re.search(r'([”」』》"])、([「『《“"])', _editorial):
        problems.append(
            "标有引号或书名号的并列成分之间不应使用顿号（GB/T 15834-2011），"
            "如「甲」「乙」或《甲》《乙》直接并列即可，顿号多余"
        )

    # 原文转录标点过度切分检测（2026-09-08 连漪拍板调整判据）：
    # 旧判据「单话轮内短句总数 ≥5」会把正常的疑问短句（如「为什么会这样呢？」）误判为切分错误。
    # 真正的过度切分特征是 ASR 按自然停顿逐句补句号、出现「连续多个以句号收尾的极短句」
    # （如「好。」「对。」「是的。」），故改为「连续 ≥3 个以句号收尾的短句（≤12 字）」才报，
    # 既兜住真过度切分，又放过正常的疑问/感叹短句。后续校订中持续观察阈值是否合适。
    _trans_m = TRANSCRIPT_HEADING_RE.search(md_text)
    if _trans_m:
        for _tline in md_text[_trans_m.end() :].splitlines():
            _lm = re.match(r"^【[^】]*】\s*(.*)$", _tline)
            if not _lm:
                continue
            _tbody = _lm.group(1).strip()
            if not _tbody:
                continue
            # 切出「句子 + 句末标点」成对序列，逐对扫描连续以句号收尾的短句
            _pairs = re.findall(r"([^。？！]*)([。？！])", _tbody)
            _run = 0
            _first = None
            for _s, _mk in _pairs:
                _clean = re.sub(r"\s", "", _s)
                if 0 < len(_clean) <= 12 and _mk == "。":
                    _run += 1
                    if _first is None:
                        _first = _s[:12]
                    if _run >= 3:
                        problems.append(
                            "原文转录疑似标点过度切分：同一话轮内出现连续多个以句号收尾的极短句"
                            f"（如「{_first}。」），应按语义连读为逗号连接的长句"
                            "（见校订规则第 1 条 (b)）"
                        )
                        break
                else:
                    _run = 0
                    _first = None

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
    """Run the fixed session prompt through the LLM and validate completion."""
    from src.processor.llm_processor import _request_text, resolve_llm_config

    config = resolve_llm_config(llm_config)
    from openai import OpenAI

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    content, _, _ = _request_text(
        client,
        config.model,
        SESSION_RULES,
        package_text,
        max_tokens=32_000,
    )
    return content.strip()


def _extract_source(line: str) -> str:
    """Parse '> 来源：播客  |  日期' into '播客 | 日期'."""
    m = re.match(r">\s*来源[:：]\s*(.+)$", line.strip())
    return m.group(1).strip() if m else ""


def _run_check(md_path: Path) -> None:
    """校验已有 .md（运行 validate_session_output），打印问题后退出。

    供会话链路交付前自检：任何 .md 落盘后跑一遍，0 问题才交付，
    避免代码块未闭合等结构问题被遗漏（api 链路由 build_output_markdown
    纯函数保证无围栏，此检查主要兜 session 链路/手动稿）。
    """
    if not md_path.exists():
        print(f"错误：找不到文件 {md_path}", file=sys.stderr)
        sys.exit(1)
    md = md_path.read_text(encoding="utf-8")
    problems = validate_session_output(md)
    if problems:
        print(f"校验未通过（{len(problems)} 项）：", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(2)
    print("校验通过。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="撷声会话内校订：生成固定校订提示词，或用 LLM 自动完成并校验"
    )
    parser.add_argument("package", type=Path, nargs="?",
                        help="会话输入包（<节目名>_diarized.txt）；与 --check 互斥")
    parser.add_argument("--check", type=Path,
                        help="校验已有 .md 文件（运行 validate_session_output 并打印问题），不调用 LLM")
    parser.add_argument("-o", "--output", type=Path, help="输出 .md 路径（默认与包同名）")
    parser.add_argument("--prompt-only", action="store_true",
                        help="只生成可粘贴的会话提示词，不调用 LLM")
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=sorted(LLM_MODELS),
                        help="LLM 提供方（默认: qwen）")
    parser.add_argument("--llm-model",
                        help="覆盖提供方默认模型；可用逗号分隔指定多个候选（如 qwen3.7-plus,gpt-5.2）")
    args = parser.parse_args()

    if args.check is not None:
        _run_check(args.check)
        return
    if args.package is None:
        parser.error("必须提供 package（会话输入包）或 --check（校验已有 .md）")

    if not args.package.exists():
        print(f"错误：找不到会话输入包 {args.package}", file=sys.stderr)
        sys.exit(1)
    package_text = args.package.read_text(encoding="utf-8")

    if args.prompt_only:
        print(build_session_prompt(package_text))
        return

    llm_config = get_llm_config(args.llm_provider, args.llm_model)
    if not llm_configured():
        print(
            "未同时检测到 LLM_API_KEY 与 LLM_BASE_URL。\n"
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
