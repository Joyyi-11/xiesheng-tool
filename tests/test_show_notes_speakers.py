"""Show Notes 说话人解析单测（纯标准库，无需模型）。"""
from src.diarization.show_notes_speakers import parse_speakers_from_show_notes


def test_host_and_guest():
    notes = (
        "【主播】姜旭：《姜就一下》、《当你红了》主播。\n"
        "【嘉宾简介】吱吱：《庸人不扰》主播，退休媒体人/大厂运营。"
    )
    assert parse_speakers_from_show_notes(notes) == ["姜旭", "吱吱"]


def test_only_host_returns_none():
    # 只点名主播、无嘉宾，无法确定人数 → 交回 --speakers / 自动检测
    notes = "主播：王五，本期一个人聊。"
    assert parse_speakers_from_show_notes(notes) is None


def test_empty_returns_none():
    assert parse_speakers_from_show_notes("") is None
    assert parse_speakers_from_show_notes(None) is None


def test_role_suffix_and_brackets():
    # 「主理人」「客座」等角色词应被识别；名字后通常紧跟标点/《》作为边界
    notes = "主理人李四、客座嘉宾张三，复盘本期。"
    # 主理人李四 → 李四；客座嘉宾张三 → 张三
    assert parse_speakers_from_show_notes(notes) == ["李四", "张三"]
