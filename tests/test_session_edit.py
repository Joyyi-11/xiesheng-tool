"""Tests for deterministic in-session transcript editing."""

from src.processor.session_edit import (
    REQUIRED_HEADINGS,
    SESSION_SPEC_VERSION,
    build_session_prompt,
    validate_session_output,
)

GOLDEN = """# 174. 我们还能给算法当多久的品味老师？｜对谈亚马逊AGI查晟

> 来源：起朱楼宴宾客 | 174. 我们还能给算法当多久的品味老师？｜对谈亚马逊AGI查晟 | 2026-07-21

# Show Notes

这是一期对谈节目，嘉宾是亚马逊 AGI 优化与研究团队负责人查晟。

## 内容提要

- **中国团队坚持开源大模型。** 但缺乏数据分发。
- **模型塌缩不可避免。** 关键在于对 AI 数据的筛选。
- **AI 无法替代人类品味。** 目前 AI 更需要的是 harness。

## 闪光语句

- **品味**：这是一种很稀缺的能力。
- **prompt**：这是一个 prompt，最终被 AI 写的。

## 大纲

```mermaid
mindmap
  root((品味老师))
    00:00 开源与数据飞轮
      开源模型无法回流数据
    08:00 模型塌缩的可控性
      质量过滤与多样性控制
    15:00 人类的品味角色
      定义问题与优化方向
    22:00 认知负债与监管缺位
    30:00 主权 AI 与文化同化
    40:00 通才与第一性原理
      通识教育与提问能力
```

## 关键词

- **数据飞轮**：开源模型无法从用户回流数据的结构性缺陷
- **模型塌缩**：AI 用生成数据训练导致能力退化的现象
- **认知负债**：过度依赖 AI 导致思考能力下降
- **主权 AI**：各国训练符合自身价值观的大模型
- **品味（Taste）**：人类定义问题与指引优化的能力

## 人物简介

**主播**：大卫翁，播客主理人
**嘉宾**：查晟，亚马逊 AGI 优化与研究团队负责人

## 全文转录

**主播**：大家好，欢迎来到我的博客节目。

**嘉宾**：谢谢，我最近在波士顿参加学术会议。
"""


def test_session_rules_are_versioned():
    assert SESSION_SPEC_VERSION >= 1


def test_build_session_prompt_wraps_package():
    package = "# 测试标题\n\n> 来源：测试播客  |  2026-01-01\n\n[SPEAKER_00] 你好。"
    prompt = build_session_prompt(package)
    assert "会话内校订规范" in prompt
    assert "# Show Notes" in prompt
    assert "## 内容提要" in prompt
    assert "## 闪光语句" in prompt
    assert "## 大纲" in prompt
    assert "## 关键词" in prompt
    assert "## 人物简介" in prompt
    assert "## 全文转录" in prompt
    assert "mermaid" in prompt  # 大纲要求 Mermaid mindmap
    assert package in prompt  # 输入包原样嵌入


def test_validate_accepts_golden_structure():
    problems = validate_session_output(GOLDEN)
    assert problems == []


def test_validate_accepts_with_source_leniency():
    problems = validate_session_output(GOLDEN, source_text="原文" * 20)
    assert problems == []


def test_validate_rejects_missing_heading():
    bad = GOLDEN.replace("## 闪光语句", "")
    problems = validate_session_output(bad)
    assert any("闪光语句" in p for p in problems)


def test_validate_rejects_too_few_outline_topics():
    bad = GOLDEN.replace(
        "    08:00 模型塌缩的可控性\n"
        "      质量过滤与多样性控制\n"
        "    15:00 人类的品味角色\n"
        "      定义问题与优化方向\n"
        "    22:00 认知负债与监管缺位\n"
        "    30:00 主权 AI 与文化同化\n",
        "",
    )
    problems = validate_session_output(bad)
    assert any("大纲" in p for p in problems)


def test_validate_rejects_outline_without_mermaid():
    bad = GOLDEN.replace("```mermaid\nmindmap\n  root((品味老师))\n", "")
    problems = validate_session_output(bad)
    assert any("mermaid" in p for p in problems)


def test_validate_rejects_too_few_keywords():
    bad = GOLDEN.replace(
        "- **模型塌缩**：AI 用生成数据训练导致能力退化的现象\n"
        "- **认知负债**：过度依赖 AI 导致思考能力下降\n"
        "- **主权 AI**：各国训练符合自身价值观的大模型\n"
        "- **品味（Taste）**：人类定义问题与指引优化的能力\n",
        "",
    )
    problems = validate_session_output(bad)
    assert any("关键词" in p for p in problems)


def test_validate_rejects_leftover_speaker_labels():
    bad = GOLDEN.replace(
        "**主播**：大家好", "[SPEAKER_00] 大家好"
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
    bad = GOLDEN.replace("# Show Notes\n\n这是一期对谈节目，嘉宾是亚马逊 AGI 优化与研究团队负责人查晟。\n\n", "")
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
    short = GOLDEN.replace("**主播**：大家好，欢迎来到我的博客节目。", "**主播**：大家好。")
    problems = validate_session_output(short, source_text="原文" * 100)
    assert any("过度删减" in p for p in problems)


def test_required_headings_are_exported():
    assert REQUIRED_HEADINGS == (
        "## 内容提要",
        "## 闪光语句",
        "## 大纲",
        "## 关键词",
        "## 人物简介",
        "## 全文转录",
    )
