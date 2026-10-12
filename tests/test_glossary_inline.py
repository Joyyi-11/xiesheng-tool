# -*- coding: utf-8 -*-
"""术语表三态判据（rules v28）的测试。

判据与确定性修复共用 src.processor.glossary_inline 一份实现，
校验闸门也引用同一份，故此处锁定行为即等于锁定三处口径。
"""

from src.processor.glossary_inline import (
    classify_terms,
    count_glossary_issues,
    fix_glossary_position,
    inline_glossary_terms,
    is_unedited,
    term_variants,
)

MD = """# 标题

## 摘要

摘要内容。

## 核心观点

- **第一条观点，讲的是通才。** 通才在 AI 时代更吃香，因为横向能力稀缺。
- **第二条观点，讲的是软技能。** 软技能成了硬通货，沟通与 EQ 由人来做。
- **第三条观点，讲的是三个同心圆。** what、how、why 三层里 AI 接管了 how。
  - **三个同心圆。** 外圈 what、中圈 how、内圈 why。

## 术语表

- **通才（Generalist）**：横向能力覆盖多个领域的人。
- **软技能**：批判性思维、沟通与 EQ 等人类特质。
- **三个同心圆**：外圈 what、中圈 how、内圈 why 的工作认知模型。
- **残留黑话**：与全部观点都无关联的行业术语。

## 问题与思考

- 你会选择通才路线吗？
"""


class TestClassify:
    """三态判定：dup / inline / keep。"""

    def test_three_states(self):
        kinds = {}
        for line, kind, tgt in classify_terms(MD):
            term = line.split("**")[1]
            kinds[term] = (kind, tgt)
        # 三个同心圆已内联在第三条观点下 → 重复
        assert kinds["三个同心圆"][0] == "dup"
        # 通才、软技能只在观点正文里出现 → 可内联
        assert kinds["通才（Generalist）"][0] == "inline"
        assert kinds["软技能"][0] == "inline"
        # 残留黑话两处都没有 → 保留
        assert kinds["残留黑话"][0] == "keep"

    def test_inline_target_is_first_occurrence(self):
        """归属取首次出现的那条观点，不另挑「更相关」的。"""
        for line, kind, tgt in classify_terms(MD):
            if "通才" in line:
                assert kind == "inline"
                assert tgt == 0      # 第一条观点
            if "软技能" in line:
                assert kind == "inline"
                assert tgt == 1      # 第二条观点

    def test_bracket_variant_locates(self):
        """术语表写「通才（Generalist）」，正文只写「通才」——靠归一化定位。"""
        assert "通才" in term_variants("通才（Generalist）")
        # 若不做归一化，这条会被误判成 keep（与观点无关联）而留在术语表
        kinds = {l.split("**")[1]: k for l, k, _ in classify_terms(MD)}
        assert kinds["通才（Generalist）"] == "inline"

    def test_count_matches_verdict(self):
        assert count_glossary_issues(MD) == 3      # 1 dup + 2 inline


class TestInline:
    """确定性修复与幂等。"""

    def test_dup_removed(self):
        new, d, i = inline_glossary_terms(MD)
        assert d == 1                              # 三个同心圆
        assert "**三个同心圆**：" not in new       # 术语表里的重复条目已删

    def test_inline_attached_to_point(self):
        new, d, i = inline_glossary_terms(MD)
        assert i == 2
        # 通才挂到第一条观点下、软技能挂到第二条观点下
        assert "  - **通才（Generalist）。**" in new
        assert "  - **软技能。**" in new
        lines = new.splitlines()
        # 子条目必须紧跟在它所归属的那条观点之后
        for n, l in enumerate(lines):
            if l.startswith("- **第二条观点"):
                assert lines[n + 1].startswith("  - **软技能。**")
                break
        else:
            raise AssertionError("未找到第二条观点")

    def test_keep_stays(self):
        new, d, i = inline_glossary_terms(MD)
        assert "- **残留黑话**：" in new           # 无关联的术语保留
        assert "## 术语表" in new                  # 还有条目，整节保留

    def test_idempotent(self):
        once, _, _ = inline_glossary_terms(MD)
        twice, d2, i2 = inline_glossary_terms(once)
        assert (d2, i2) == (0, 0)
        assert twice == once

    def test_section_dropped_when_empty(self):
        md = MD.replace("- **残留黑话**：与全部观点都无关联的行业术语。\n", "")
        new, d, i = inline_glossary_terms(md)
        assert "## 术语表" not in new              # 全部处理完 → 整节删除


class TestPosition:
    """术语表站位：紧跟核心观点、在问题与思考之前。"""

    WRONG = """## 核心观点

- **观点一。** 内容。

## 问题与思考

- 问题一？

## 术语表

- **残留黑话**：与全部观点都无关联的行业术语。

## 人物简介

- **某人**：身份。
"""

    def test_moved_before_questions(self):
        new = fix_glossary_position(self.WRONG)
        order = [l.strip() for l in new.splitlines() if l.startswith("## ")]
        assert order == ["## 核心观点", "## 术语表", "## 问题与思考", "## 人物简介"]

    def test_idempotent(self):
        once = fix_glossary_position(self.WRONG)
        assert fix_glossary_position(once) == once

    def test_already_correct_untouched(self):
        ok = """## 核心观点

- **观点一。** 内容。

## 术语表

- **残留黑话**：与全部观点都无关联的行业术语。

## 问题与思考

- 问题一？
"""
        assert fix_glossary_position(ok) == ok

    def test_content_preserved(self):
        """纯搬移：文字一个字都不该变。"""
        new = fix_glossary_position(self.WRONG)
        assert sorted(new.split()) == sorted(self.WRONG.split())


class TestUneditedGuard:
    """未校订占位稿不参与修复（否则把占位符搬来搬去）。"""

    PLACEHOLDER = """## 摘要

摘要。

## 核心观点

- **待校订。** 待提炼本期重要观点（每条以加粗主题句开头、句号结尾）。

## 术语表

- **待校订**：待提取无法归入核心观点的残留术语。
"""

    def test_detected(self):
        assert is_unedited(self.PLACEHOLDER)

    def test_skipped(self):
        assert classify_terms(self.PLACEHOLDER) == []
        new, d, i = inline_glossary_terms(self.PLACEHOLDER)
        assert (d, i) == (0, 0)
        assert new == self.PLACEHOLDER

    def test_normal_md_not_flagged(self):
        assert not is_unedited(MD)
