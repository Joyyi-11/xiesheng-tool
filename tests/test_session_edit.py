"""Tests for deterministic in-session transcript editing."""

from src.processor.session_edit import (
    REQUIRED_HEADINGS,
    SESSION_RULES,
    SESSION_SPEC_VERSION,
    build_session_prompt,
    validate_session_output,
)

GOLDEN = """# 174. 我们还能给算法当多久的品味老师？｜对谈亚马逊AGI查晟

> 来源：起朱楼宴宾客 | 174. 我们还能给算法当多久的品味老师？｜对谈亚马逊AGI查晟 | 2026-07-21

## Show Notes

这是一期对谈节目，嘉宾是亚马逊 AGI 优化与研究团队负责人查晟。

## 摘要

本期对谈亚马逊 AGI 查晟，围绕开源大模型、模型塌缩与人类品味在 AI 时代的角色展开。

## 核心观点

- **中国团队坚持开源大模型。** 但缺乏数据分发。
- **模型塌缩不可避免。** 关键在于对 AI 数据的筛选。
- **AI 无法替代人类品味。** 目前 AI 更需要的是 harness。

## 问题与思考

1. **开源模型最大的结构性缺陷是什么？**

   无法从用户使用回流训练数据，数据飞轮转不起来。

2. **模型塌缩可控吗？**

   关键在于对生成数据做质量过滤与多样性控制，维持多样性。

## 术语表

- **数据飞轮**：开源模型无法从用户回流数据的结构性缺陷。
- **模型塌缩**：AI 用生成数据训练导致能力退化的现象。
- **认知负债**：过度依赖 AI 导致思考能力下降。
- **主权 AI**：各国训练符合自身价值观的大模型。
- **品味（Taste）**：人类定义问题与指引优化的能力。

## 人物简介

**主播**：大卫翁，播客主理人
**嘉宾**：查晟，亚马逊 AGI 优化与研究团队负责人

## 全文转录

【主播】大家好，欢迎来到我的博客节目。

【嘉宾】谢谢，我最近在波士顿参加学术会议。
"""


def test_session_rules_are_versioned():
    assert SESSION_SPEC_VERSION == 18


def test_build_session_prompt_wraps_package():
    package = "# 测试标题\n\n> 来源：测试播客  |  2026-01-01\n\n[SPEAKER_00] 你好。"
    prompt = build_session_prompt(package)
    assert "会话内校订规范" in prompt
    assert "## Show Notes" in prompt
    assert "## 摘要" in prompt
    assert "## 核心观点" in prompt
    assert "## 问题与思考" in prompt
    assert "## 术语表" in prompt
    assert "## 人物简介" in prompt
    assert "## 全文转录" in prompt
    assert package in prompt  # 输入包原样嵌入


def test_validate_accepts_golden_structure():
    problems = validate_session_output(GOLDEN)
    assert problems == []


def test_validate_accepts_with_source_leniency():
    problems = validate_session_output(GOLDEN, source_text="原文" * 20)
    assert problems == []


def test_validate_rejects_missing_heading():
    bad = GOLDEN.replace("## 核心观点", "")
    problems = validate_session_output(bad)
    assert any("核心观点" in p for p in problems)


def test_validate_rejects_missing_summary():
    summary = "本期对谈亚马逊 AGI 查晟，围绕开源大模型、模型塌缩与人类品味在 AI 时代的角色展开。"
    bad = GOLDEN.replace(f"## 摘要\n\n{summary}\n\n", "")
    problems = validate_session_output(bad)
    assert any("摘要" in p for p in problems)


def test_validate_rejects_missing_questions():
    bad = GOLDEN.replace(
        "## 问题与思考\n\n"
        "1. **开源模型最大的结构性缺陷是什么？**\n\n"
        "   无法从用户使用回流训练数据，数据飞轮转不起来。\n\n"
        "2. **模型塌缩可控吗？**\n\n"
        "   关键在于对生成数据做质量过滤与多样性控制，维持多样性。\n\n",
        "",
    )
    problems = validate_session_output(bad)
    assert any("问题与思考" in p for p in problems)


def test_validate_accepts_empty_terms_glossary():
    # 残留术语表 0 条合法（不再强制 4-8）
    bad = GOLDEN.replace(
        "- **数据飞轮**：开源模型无法从用户回流数据的结构性缺陷。\n"
        "- **模型塌缩**：AI 用生成数据训练导致能力退化的现象。\n"
        "- **认知负债**：过度依赖 AI 导致思考能力下降。\n"
        "- **主权 AI**：各国训练符合自身价值观的大模型。\n"
        "- **品味（Taste）**：人类定义问题与指引优化的能力。\n",
        "",
    )
    problems = validate_session_output(bad)
    assert all("术语表" not in p for p in problems)


def test_validate_accepts_none_speaker_intro():
    bad = GOLDEN.replace(
        "**主播**：大卫翁，播客主理人\n**嘉宾**：查晟，亚马逊 AGI 优化与研究团队负责人",
        "（无）",
    )
    problems = validate_session_output(bad)
    assert all("人物简介" not in p for p in problems)


def test_validate_rejects_colon_after_key_point_topic():
    # 核心观点三段式：观点句、展开论述句、引用句顺次相接，主题句后不得用冒号引出展开
    bad = GOLDEN.replace(
        "- **模型塌缩不可避免。** 关键在于对 AI 数据的筛选。",
        "- **模型塌缩不可避免**：关键在于对 AI 数据的筛选。",
    )
    problems = validate_session_output(bad)
    assert any("冒号" in p for p in problems)


def test_validate_allows_colon_inside_key_point_topic():
    # 主题句内部必要冒号可保留（如「第三个时代：常驻同事」），整句仍以句号收尾
    ok = GOLDEN.replace(
        "- **模型塌缩不可避免。** 关键在于对 AI 数据的筛选。",
        "- **AI 产品的第三个时代：常驻同事。** 关键在于对 AI 数据的筛选。",
    )
    assert validate_session_output(ok) == []


def test_session_rules_require_no_colon_after_key_point_topic():
    assert "严禁用冒号把展开句挂在主题句后面" in SESSION_RULES


def test_validate_rejects_unbolded_question():
    # 问题与思考：问题须整句加粗、序号留在加粗外，否则与答案无法区分
    bad = GOLDEN.replace(
        "1. **开源模型最大的结构性缺陷是什么？**",
        "1. 开源模型最大的结构性缺陷是什么？",
    )
    problems = validate_session_output(bad)
    assert any("加粗" in p for p in problems)


def test_validate_accepts_bolded_question():
    # GOLDEN 已是加粗标准，确认成品通过
    assert validate_session_output(GOLDEN) == []


def test_session_rules_require_bolded_question():
    assert "问题整句加粗" in SESSION_RULES


def test_validate_rejects_malformed_terms_glossary():
    # 有条目时仍须为 **术语**：解释、句号结尾
    bad = GOLDEN.replace("- **数据飞轮**：", "- 数据飞轮：")
    problems = validate_session_output(bad)
    assert any("格式不符" in p for p in problems)

    bad2 = GOLDEN.replace("结构性缺陷。", "结构性缺陷")
    problems2 = validate_session_output(bad2)
    assert any("句号" in p for p in problems2)


def test_validate_rejects_leftover_speaker_labels():
    bad = GOLDEN.replace(
        "【主播】大家好", "[SPEAKER_00] 大家好"
    )
    problems = validate_session_output(bad)
    assert any("SPEAKER" in p for p in problems)


def test_validate_rejects_source_without_three_parts():
    bad = GOLDEN.replace(
        "> 来源：起朱楼宴宾客 | 174. 我们还能给算法当多久的品味老师？｜对谈亚马逊AGI查晟 | 2026-07-21",
        "> 来源：起朱楼宴宾客  |  2026-07-21",
    )
    problems = validate_session_output(bad)
    assert any("来源" in p for p in problems)


def test_validate_rejects_missing_show_notes():
    bad = GOLDEN.replace("## Show Notes\n\n这是一期对谈节目，嘉宾是亚马逊 AGI 优化与研究团队负责人查晟。\n\n", "")
    problems = validate_session_output(bad)
    assert any("Show Notes" in p for p in problems)


def test_validate_rejects_quote_intro_format():
    bad = GOLDEN.replace(
        "**主播**：大卫翁，播客主理人",
        "> **主播**：大卫翁，播客主理人",
    )
    problems = validate_session_output(bad)
    assert any("引用" in p for p in problems)


def test_validate_rejects_overly_short_transcript():
    short = GOLDEN.replace("【主播】大家好，欢迎来到我的博客节目。", "【主播】大家好。")
    problems = validate_session_output(short, source_text="原文" * 100)
    assert any("过度删减" in p for p in problems)


def test_validate_rejects_outline_section():
    bad = GOLDEN + "\n## 大纲\n\n- 一些要点\n"
    problems = validate_session_output(bad)
    assert any("大纲" in p for p in problems)


def test_validate_rejects_mermaid_diagram():
    bad = GOLDEN + "\n```mermaid\nmindmap\n  root\n```\n"
    problems = validate_session_output(bad)
    assert any("mermaid" in p for p in problems)


def test_strip_unwanted_sections_removes_outline_and_mermaid():
    from src.processor.session_edit import _strip_unwanted_sections

    md = (
        GOLDEN
        + "\n## 大纲\n\n- 要点\n\n```mermaid\nmindmap\n  root\n```\n\n"
        + "## 全文转录\n\n【主播】x。\n"
    )
    cleaned = _strip_unwanted_sections(md)
    assert "## 大纲" not in cleaned
    assert "mermaid" not in cleaned
    assert "【主播】x。" in cleaned


def test_validate_flags_speaker_label_mismatch():
    bad = GOLDEN.replace(
        "【嘉宾】谢谢，我最近在波士顿参加学术会议。",
        "【主播】谢谢，我是 ACE，最近在波士顿参加学术会议。",
    )
    problems = validate_session_output(bad)
    assert any("说话人标签错位" in p for p in problems)


def test_validate_flags_dunhao_between_quoted_items():
    bad = GOLDEN.replace(
        "本期对谈亚马逊 AGI 查晟，围绕开源大模型、模型塌缩与人类品味在 AI 时代的角色展开。",
        "本期对谈「开源」、「回流」与「模型塌缩」。",
    )
    problems = validate_session_output(bad)
    assert any("不应使用顿号" in p for p in problems)


def test_validate_flags_speaker_imbalance():
    # 两人对话里某人独占绝大多数段落，疑似 diarization 过度切分 / 标签归并错误
    md = (
        "# 标题\n\n> 来源：播客 | 标题 | 2026-08-28\n\n"
        "## Show Notes\n\n简介\n\n## 摘要\n\n本期聊求职。\n\n"
        "## 核心观点\n\n- **要点。** 证据\n\n"
        "## 问题与思考\n\n1. 找不到工作是谁的问题？\n\n   策略错位。\n\n"
        "## 术语表\n\n- **术语**：解释。\n\n"
        "## 人物简介\n\n**主播**：湫湫\n\n"
        "## 全文转录\n\n"
        + "\n\n".join(f"【嘉宾吱吱】第{i}句内容。" for i in range(9))
        + "\n\n【主播湫湫】最后一句话。"
    )
    problems = validate_session_output(md)
    assert any("占比失衡" in p for p in problems)


def test_validate_flags_unmerged_same_speaker_runs():
    # 同一说话人连续多段未合并（源转录按停顿切碎 / 同人被判成多簇）
    md = (
        "# 标题\n\n> 来源：播客 | 标题 | 2026-08-28\n\n"
        "## Show Notes\n\n简介\n\n## 摘要\n\n本期聊求职。\n\n"
        "## 核心观点\n\n- **要点。** 证据\n\n"
        "## 问题与思考\n\n1. 要不要逼自己学 AI？\n\n   先判断成本。\n\n"
        "## 术语表\n\n- **术语**：解释。\n\n"
        "## 人物简介\n\n**主播**：湫湫\n\n"
        "## 全文转录\n\n"
        + "\n\n".join(f"【湫湫】第{i}句。" for i in range(4))
        + "\n\n【小朱】插一句。"
    )
    problems = validate_session_output(md)
    assert any("未合并" in p for p in problems)


def test_session_rules_require_same_speaker_merge():
    # 校订规则须硬性要求「同一说话人相邻段落合并为一段」
    assert "同一说话人的相邻段落必须合并为一段" in SESSION_RULES


def test_session_rules_warn_mid_phrase_period():
    # 校订规则须提示「连贯短语中途被句号切断」类断句错误
    assert "好不好找工作" in SESSION_RULES


def test_required_headings_are_exported():
    assert REQUIRED_HEADINGS == (
        "## Show Notes",
        "## 摘要",
        "## 核心观点",
        "## 问题与思考",
        "## 术语表",
        "## 人物简介",
        "## 全文转录",
    )
