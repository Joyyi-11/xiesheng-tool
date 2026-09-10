"""Prompt templates for DeepSeek LLM processing.

api 链路（--llm-mode api）：CLEAN（分块校订）→ STRUCT（结构化 JSON）。
两条链路的**编辑口径**（校订规则 + 结构化产物规则）由
``src.processor.rules`` 单一权威源提供——session 链路（SESSION_RULES）与
本模块引用同一批条款常量，杜绝两份文本人工同步的漂移。

本模块保留 api 链路特有的**流程条款**（分块/JSON 契约/保留 [SPEAKER_XX]）：
这些是 api 的调用形态差异，不是编辑口径，不进 rules。
"""

from src.processor import rules
from src.processor.rules import (
    RULES_SPEC_VERSION,
    STRUCT_ANCHORS,
    STRUCT_GLOSSARY,
    STRUCT_INTRO,
    STRUCT_KEY_POINT,
    STRUCT_MAPPING,
    STRUCT_QUESTIONS,
    STRUCT_SUMMARY,
)

# ============================================================
# CLEAN：逐分块校订（api 特有流程 + 共享校订口径）
# ============================================================

# api 链路特有：续编在共享口径（rules.EDIT_RULES，1-16）之后
_CLEAN_API_ONLY = (
    "## api 分块契约（仅本链路）\n\n"
    "17. 原样保留 [SPEAKER_XX] 标签；不要猜测或替换说话人姓名（说话人映射由后置 STRUCT 阶段完成）。\n"
    "18. 只校订目标分块。上下文（上一分块末尾/下一分块开头）仅用于理解，不能出现在输出中。\n"
    "19. 返回合法 JSON：{\"chunk_id\": 整数, \"cleaned_text\": \"校订后的完整目标分块\"}。"
    "不要返回其他文字。"
)

# 共享校订口径编号化（与 session 链路同一套，见 src.processor.rules）
_EDIT_NUMBERED = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(rules.EDIT_RULES))

# 断句是两套规则里最易混的：第 3 条防「连贯短语被句号切断」，第 4 条防「两句粘连不补句号」。
# 编号随 rules.EDIT_RULES 顺序（1=忠实保留 …），此处只作提示不重复条款，避免文字二次漂移。
_EDIT_HINT = (
    "注意区分两类标点问题：连贯短语/复合词被句号切断（应连读，见上第 3 条）与"
    "确为两个独立句子却粘连缺句号（应补句号，见上第 4 条）——先判断归属再动手，"
    "不要在每个自然停顿处都补句号。"
)

CLEAN_SYSTEM_PROMPT = f"""你是中文播客逐字稿校订员。目标是在不丢失信息、不改变原意的前提下，让转录文稿准确、流畅、易读。以下校订口径与另一条链路完全一致（规则源版本 v{RULES_SPEC_VERSION}），先逐条执行：

{_EDIT_NUMBERED}

## 校订提示
{_EDIT_HINT}

{_CLEAN_API_ONLY}"""

CLEAN_USER_PROMPT = """## Show Notes（用于核对专名）
{show_notes}

## 上一分块末尾（只作上下文）
{previous_context}

## 目标分块 {chunk_id}
{chunk_text}

## 下一分块开头（只作上下文）
{next_context}

请完整校订目标分块，并返回 chunk_id={chunk_id} 的 JSON。"""


# ============================================================
# STRUCT：一次性结构化（api JSON 契约 + 共享结构化口径）
# ============================================================

STRUCT_SYSTEM_PROMPT = f"""你是中文播客文稿编辑。基于已经校订的完整逐字稿，一次生成该节目的全部阅读辅助信息。可以概括，但不能编造逐字稿和 Show Notes 中没有的信息，也不能改写逐字稿本身。

返回合法 JSON，格式为：
{{
  "key_points": [{{"point": "完整主题句", "evidence": "来自原文的简短支撑", "quote": "可选，逐字稿中直接出现的原句，忠实原文、不改写，无则留空字符串", "anchor": "可选，该观点在原文转录区逐字出现的支撑句/短语（用于转录回标加粗），无可靠原文子串则留空", "terms": [{{"term": "观点中出现、读者可能不懂的特有术语", "desc": "1-2 句客观解释"}}]}}],
  "speaker_intro": "**身份姓名**：简介",
  "speaker_mapping": {{"SPEAKER_00": "身份姓名"}},
  "keywords": [{{"key": "残留术语", "desc": "1-2 句说明"}}],
  "summary": "2-3 句话的节目主题与核心结论总结",
  "questions": [{{"question": "核心问题（整理式，非原文照抄）", "answer": "直接解答该问题的回答，1-3 句，引用人物须点明身份", "anchor": "可选，该问题在原文转录区逐字出现的支撑句/短语（用于转录回标加粗），无可靠原文子串则留空"}}]
}}

各字段的撰写口径与另一条链路完全一致（规则源版本 v{RULES_SPEC_VERSION}）：

1. {STRUCT_SUMMARY}
2. {STRUCT_KEY_POINT}
3. {STRUCT_QUESTIONS}
4. {STRUCT_GLOSSARY}
5. {STRUCT_INTRO}
6. {STRUCT_MAPPING}
7. {STRUCT_ANCHORS}
8. 所有双引号统一用直角引号「」，不用英文直引号 "。
9. 只返回 JSON，不返回正文或 Markdown 代码围栏。"""

STRUCT_USER_PROMPT = """## 节目信息
标题：{title}
播客：{podcast_name}

## Show Notes
{show_notes}

## 已校订逐字稿
{transcript_text}

请基于以上内容，一次生成 key_points、speaker_intro、speaker_mapping、keywords、summary、questions（含可选的 anchor 回标锚点），并返回上述格式的 JSON。"""
