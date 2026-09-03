from src.models.schemas import KeyPoint, OutputDoc, QuestionItem
from src.processor.output_quality import assert_valid_output_doc, validate_output_doc


def _doc(**overrides):
    defaults = {
        "title": "标题",
        "podcast_name": "播客",
        "pub_date": "2026-01-01",
        "show_notes": "简介",
        "key_points": [KeyPoint("要点。", "证据。")],
        "summary": "摘要。",
        "questions": [QuestionItem("问题？", "回答。")],
        "full_text": "【主播】正文。",
        "speaker_intro": "",
    }
    defaults.update(overrides)
    return OutputDoc(**defaults)


def test_valid_document_has_no_problems():
    assert validate_output_doc(_doc()) == []


def test_missing_required_reading_aids_are_blocking():
    problems = validate_output_doc(
        _doc(summary="", key_points=[KeyPoint(" ", "证据。")], questions=[QuestionItem(" ", "回答。")])
    )
    assert "摘要为空" in problems
    assert "核心观点为空" not in problems
    assert "核心观点存在空主题句" in problems
    assert "问题与思考存在空问题" in problems


def test_empty_glossary_and_speaker_intro_are_allowed():
    assert validate_output_doc(_doc(keywords=[], speaker_intro="")) == []


def test_leftover_speaker_label_is_blocking():
    problems = validate_output_doc(_doc(full_text="[SPEAKER_01] 正文。"))
    assert problems == ["全文转录残留未映射说话人标签：[SPEAKER_01]"]


def test_assert_raises_with_combined_problems():
    try:
        assert_valid_output_doc(_doc(title=" ", full_text=""))
    except ValueError as exc:
        assert "缺少节目标题" in str(exc)
        assert "全文转录为空" in str(exc)
    else:
        raise AssertionError("expected ValueError")
