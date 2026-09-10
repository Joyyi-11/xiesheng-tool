"""roster 增强（补散文/数词漏人）单元测试。

背景：guide 集 Show Notes 写「斯怡、机器坏人、404 三位」，但 _ROSTER_RE 只抓带角色
前缀的名字，404 在散文里被漏掉，导致 base_k 被低估成 2，cam++ 分不出第 3 人。
"""
import pytest

from src.diarization.speaker_resolver import (
    parse_roster_with_roles,
    parse_stated_speaker_count,
)

GUIDE_NOTES = """本期请到斯怡、机器坏人、404 三位「非典型文科生 AI 博主」，从被 AI 冲击、到用 AI 反击。

本场主持人
斯怡Jins：科技人文研究者、AI 自媒体博主
本场对话嘉宾
机器坏人：AI 产品经理、AI 自媒体博主
"""


def test_stated_count_prose():
    assert parse_stated_speaker_count("斯怡、机器坏人、404 三位") == 3


def test_stated_count_two():
    assert parse_stated_speaker_count("两位嘉宾对谈") == 2


def test_stated_count_arabic():
    assert parse_stated_speaker_count("本场共 5 位嘉宾") == 5


def test_stated_count_ordinal_excluded():
    # 「第N位」是序数不是人数，必须排除
    assert parse_stated_speaker_count("第三位出场的是主播") is None


def test_stated_count_none():
    assert parse_stated_speaker_count("这是一期单人独白") is None


def test_roster_catches_prose_name():
    roster = parse_roster_with_roles(GUIDE_NOTES)
    names = {n for _, n in roster}
    # 角色前缀抓到的两位
    assert "斯怡Jins" in names
    assert "机器坏人" in names
    # 散文漏网者 404 现在应被补进花名册（角色置空）
    assert "404" in names
    # base_k 应据此抬到 3（len(roster) >= 3）
    assert len(roster) >= 3


def test_roster_roles_preserved():
    roster = parse_roster_with_roles(GUIDE_NOTES)
    # parse_roster_with_roles 契约是 [(role, name), ...]，故反转为 name->role 再断言。
    roles = {n: r for r, n in roster}
    assert roles["斯怡Jins"] == "主持"
    assert roles["机器坏人"] == "嘉宾"
