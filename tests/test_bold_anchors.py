"""Tests for deterministic transcript anchor bolding (src.processor.bold_anchors)."""

from src.processor.bold_anchors import bold_markdown, extract_candidates


def _md(transcript: str, kp: str = "- **观点句。**展开句。「模型塌缩不可避免，一定要自己筛数据」") -> str:
    return f"""# 标题
> 来源：播客 | 标题 | 2026-01-01

## Show Notes
sn

## 摘要
s

## 核心观点

{kp}

## 问题与思考

1. **怎么办？**
   答案。

## 人物简介

**主播**：谁

## 原文转录

{transcript}
"""


class TestExtractCandidates:
    def test_quote_and_term_kinds(self):
        upper = "## 核心观点\n\n- **P。**e。「一句够长的原话内容」\n  - **数据飞轮**：解释。\n"
        cands = extract_candidates(upper)
        assert ("一句够长的原话内容", "quote") in cands
        assert ("数据飞轮", "term") in cands

    def test_term_parenthetical_variants(self):
        # 「dogfooding（内部试用）」拆成英文词与中文释义两个候选，提高命中率
        upper = "## 术语表\n\n- **dogfooding（内部试用）**：解释。\n"
        cands = extract_candidates(upper)
        assert ("dogfooding", "term") in cands
        assert ("内部试用", "term") in cands

    def test_viewpoint_sentence_not_a_candidate(self):
        # 核心观点的观点句是提炼书面语，不作为回标候选
        upper = "## 核心观点\n\n- **模型塌缩不可避免。**关键在于筛选。\n"
        cands = [c for c, _ in extract_candidates(upper)]
        assert "模型塌缩不可避免。关键在于筛选。" not in cands


class TestBoldMarkdown:
    def test_bolds_quote_and_term(self):
        md = _md(
            "【主播】今天聊模型塌缩，模型塌缩不可避免，一定要自己筛数据。\n\n【嘉宾】数据飞轮很关键。",
            kp="- **模型塌缩不可避免。**关键在于筛选。「模型塌缩不可避免，一定要自己筛数据」\n  - **数据飞轮**：解释。",
        )
        new, anchors = bold_markdown(md)
        assert "**模型塌缩不可避免，一定要自己筛数据**" in new
        assert "**数据飞轮**" in new
        assert anchors

    def test_speaker_label_never_bolded(self):
        md = _md(
            "【主播】模型塌缩不可避免，一定要自己筛数据。",
            kp="- **P。**e。「模型塌缩不可避免，一定要自己筛数据」",
        )
        new, _ = bold_markdown(md)
        assert "【**" not in new

    def test_no_candidates_returns_unchanged(self):
        md = _md("【主播】转录内容。", kp="- **观点句。**展开句。")
        new, anchors = bold_markdown(md)
        assert new == md
        assert anchors == []

    def test_missing_quote_silently_skipped(self):
        # 引文在校订后措辞改变、未在转录中逐字出现时静默跳过，不改字
        md = _md("【主播】转录里根本没有那句话。", kp="- **P。**e。「完全不存在的原话内容在这里」")
        new, anchors = bold_markdown(md)
        assert anchors == []
        assert "**" not in new.split("## 原文转录")[1]

    def test_no_transcript_section_returns_unchanged(self):
        md = "# 标题\n\n## 核心观点\n\n- **P。**e。「一句够长的原话内容」\n"
        new, anchors = bold_markdown(md)
        assert new == md
        assert anchors == []

    def test_idempotent_on_already_bolded_transcript(self):
        # 渲染器已回标的成稿再来跑本脚本（会话链路交付前必跑），不得叠成 ****锚点****
        md = _md(
            "【主播】模型塌缩不可避免，一定要自己筛数据。\n\n【嘉宾】数据飞轮很关键。",
            kp="- **模型塌缩不可避免。**关键在于筛选。「模型塌缩不可避免，一定要自己筛数据」\n  - **数据飞轮**：解释。",
        )
        once, _ = bold_markdown(md)
        twice, _ = bold_markdown(once)
        assert twice == once
        assert "****" not in twice

    def test_already_bolded_anchor_not_rewrapped(self):
        md = _md(
            "【主播】今天聊**模型塌缩不可避免，一定要自己筛数据**这件事。",
            kp="- **模型塌缩不可避免。**关键在于筛选。「模型塌缩不可避免，一定要自己筛数据」",
        )
        new, anchors = bold_markdown(md)
        assert "****" not in new
        assert new.count("**模型塌缩不可避免，一定要自己筛数据**") == 1
