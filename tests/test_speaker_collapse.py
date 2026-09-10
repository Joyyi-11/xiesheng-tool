"""说话人塌缩检测（src/diarization/collapse_check.py）测试。

阈值是实测标定的，故每个判据都配一条「必须命中」与一条「必须不命中」的用例，
防止后续调阈值时把已修好的误报重新放进来。
"""

import pytest

from src.diarization.collapse_check import (
    DOMINANT_MIN_SEGMENTS,
    MIN_SEGMENTS,
    SINGLE_CLUSTER_SEGMENTS,
    cluster_distribution,
    detect_collapse,
    format_collapse_warning,
)


def _segs(*pairs) -> list[dict]:
    """pairs: (speaker, count) → 段列表。"""
    segs: list[dict] = []
    for speaker, count in pairs:
        segs.extend({"speaker": speaker} for _ in range(count))
    return segs


def test_cluster_distribution_sorted_by_count_desc():
    dist = cluster_distribution(_segs(("SPEAKER_00", 5), ("SPEAKER_01", 20)))
    assert list(dist) == ["SPEAKER_01", "SPEAKER_00"]
    assert dist["SPEAKER_01"] == 20


def test_empty_segments_no_reason():
    assert detect_collapse([]) == []


def test_below_min_segments_skipped():
    """段数太少不具备统计意义——全英 Vol.37 只有 1 段，不该告警。"""
    segs = _segs(("SPEAKER_00", MIN_SEGMENTS - 1))
    assert detect_collapse(segs) == []


def test_single_cluster_with_many_segments_is_flagged():
    """真实塌缩形态：Vol.014 672 段、触底反弹 1128 段、文科女+AI 1695 段全塌成 1 簇。"""
    segs = _segs(("SPEAKER_00", SINGLE_CLUSTER_SEGMENTS))
    reasons = detect_collapse(segs)
    assert len(reasons) == 1
    assert "1 个说话人簇" in reasons[0]
    # 文案必须自带「单口可忽略」，否则 #28 / Vol.552 这类真单口会误导人重跑
    assert "单口" in reasons[0]


def test_dominant_cluster_above_ratio_is_flagged():
    segs = _segs(("SPEAKER_00", 1000), ("SPEAKER_01", 20))
    reasons = detect_collapse(segs)
    assert any("独占" in r for r in reasons)


def test_talkative_guest_two_person_dialog_not_flagged():
    """#706 回归：嘉宾 1090 段 vs 主播 139 段（89%）是正常的 2 人对话，不是塌缩。

    占比门槛曾为 0.85 时误报此集，收紧到 0.95 后归零。本用例锁死该行为。
    """
    segs = _segs(("SPEAKER_01", 1090), ("SPEAKER_00", 139))
    assert detect_collapse(segs) == []


def test_balanced_multi_speaker_not_flagged():
    segs = _segs(("SPEAKER_00", 400), ("SPEAKER_01", 380), ("SPEAKER_02", 350))
    assert detect_collapse(segs) == []


def test_many_tiny_fragments_not_flagged():
    """过切分（多簇碎片段）不是塌缩，不该告警。"""
    segs = _segs(*[(f"SPEAKER_{i:02d}", 30) for i in range(20)])
    assert detect_collapse(segs) == []


def test_show_notes_roster_exceeds_clusters_is_flagged():
    segs = _segs(("SPEAKER_00", 300), ("SPEAKER_01", 300))
    reasons = detect_collapse(segs, expected_speakers=5)
    assert any("Show Notes" in r for r in reasons)


def test_show_notes_roster_matching_clusters_not_flagged():
    segs = _segs(("SPEAKER_00", 300), ("SPEAKER_01", 300))
    assert detect_collapse(segs, expected_speakers=3) == []


def test_dominant_rule_requires_absolute_segment_floor():
    """占比够高但主簇段数不足 DOMINANT_MIN_SEGMENTS 时不告警。"""
    segs = _segs(("SPEAKER_00", DOMINANT_MIN_SEGMENTS - 1), ("SPEAKER_01", 2))
    assert detect_collapse(segs) == []


def test_format_warning_empty_reasons_returns_empty():
    assert format_collapse_warning([]) == ""


def test_format_warning_includes_rerun_command():
    msg = format_collapse_warning(["测试理由"], "https://example.com/ep/abc")
    assert "测试理由" in msg
    # 坍塌告警的重跑命令已升级为只重分离、复用已有转录的 --refresh-diarization
    assert "--refresh-diarization" in msg
    assert "https://example.com/ep/abc" in msg


def test_format_warning_without_url_gives_placeholder():
    msg = format_collapse_warning(["测试理由"])
    assert "<url>" in msg
    assert "WeSpeaker" in msg
    assert "--refresh-diarization" in msg


@pytest.mark.parametrize("unknown_only", [True, False])
def test_unknown_speaker_segments_are_counted(unknown_only):
    """UNKNOWN 也计入分布，避免全 UNKNOWN 时误判为「形态正常」。"""
    segs = _segs(("UNKNOWN", 300)) if unknown_only else _segs(("SPEAKER_00", 300))
    assert detect_collapse(segs)
