from src.diarization.speaker_resolver import (
    detect_diarization_issues,
    parse_roster_with_roles,
    resolve_speaker_map,
    to_display_label,
)


def test_parse_roster_accepts_host_producer_role():
    roster = parse_roster_with_roles("主播简介：湫湫\n嘉宾：小朱\n主理人：连漪")
    assert ("主播", "湫湫") in roster
    assert ("嘉宾", "小朱") in roster
    assert ("主理", "连漪") in roster


def test_alias_is_normalized_before_merge_quality_gate():
    roster = [("主播", "湫湫"), ("嘉宾", "小朱")]
    segments = [{"speaker": "SPEAKER_00", "text": "我是小猪。"}]
    assert detect_diarization_issues(segments, roster) == []


def test_multiple_real_people_in_one_cluster_stay_unmapped():
    roster = [("主播", "湫湫"), ("嘉宾", "小朱")]
    segments = [{"speaker": "SPEAKER_00", "text": "我是湫湫。我是小朱。"}]
    mapping, issues = resolve_speaker_map(segments, roster)
    assert mapping == {}
    assert any("同时命中多个花名册真人" in issue for issue in issues)


def test_cluster_without_evidence_is_not_assigned_by_elimination():
    roster = [("主播", "湫湫"), ("嘉宾", "小朱")]
    segments = [{"speaker": "SPEAKER_01", "text": "今天聊一聊工作。"}]
    mapping, issues = resolve_speaker_map(segments, roster)
    assert mapping == {}
    assert any("无法可靠映射" in issue for issue in issues)


def test_same_guest_is_not_mapped_to_multiple_clusters():
    roster = [("主播", "湫湫"), ("嘉宾", "小朱")]
    segments = [
        {"speaker": "SPEAKER_00", "text": "我是小朱。"},
        {"speaker": "SPEAKER_01", "text": "我是小朱。"},
    ]
    mapping, issues = resolve_speaker_map(segments, roster)
    assert mapping == {"SPEAKER_00": "嘉宾小朱"}
    assert any("同时指向嘉宾 小朱" in issue for issue in issues)


class TestToDisplayLabel:
    """转写显示标签：只留短名，角色前缀整词剥离（不得劈出「【人Nina】」这类残名）。"""

    def test_strips_three_char_role_whole(self):
        assert to_display_label("主持人Nina") == "【Nina】"

    def test_strips_two_char_roles(self):
        assert to_display_label("主播连漪") == "【连漪】"
        assert to_display_label("嘉宾安部") == "【安部】"
        assert to_display_label("主理人小朱") == "【小朱】"

    def test_keeps_plain_name(self):
        assert to_display_label("湫湫") == "【湫湫】"

    def test_strips_paren_note_and_keeps_given_name(self):
        assert to_display_label("伦尼·拉奇茨基（Lenny's Podcast 主持）") == "【伦尼】"

    def test_empty_returns_empty(self):
        assert to_display_label("") == ""
