"""分离前人数预检 ``suggest()`` 的判据单测。

所有标定数据来自四份真实样本（2026-10-10 实测），出处写在各用例 docstring 里。
关键是**这些样本同时包含正例与反例**——只测「双人能判出来」是不够的，必须保证
确凿单口不被误报成双人，否则 auto 模式会给用户制造新的麻烦。
"""

import pytest

from src.diarization.probe import MIN_CLUSTER_SHARE, SILHOUETTE_MIN, suggest

# #757 标定：Tibo（嘉宾，68%） + Lenny（主播，32%）。自动估 K 时命中
# diarize 的 sim_p10>=0.16 短路被判成 1 人，实际是双人访谈。
SWEEP_757 = {
    2: {"silhouette": 0.3098, "min_share": 0.3225, "shares": {0: 0.6775, 1: 0.3225}},
    3: {"silhouette": 0.1383, "min_share": 0.3144, "shares": {0: 0.3144, 1: 0.3225, 2: 0.3631}},
    4: {"silhouette": 0.0825, "min_share": 0.0850, "shares": {0: 0.3482, 1: 0.3279, 2: 0.0850, 3: 0.2389}},
}

# #28 标定：确凿单口（同一人从头讲到尾）。同样命中 sim_p10 短路（0.3127），
# 但 K=2 轮廓系数仅 0.0712 ——低于 SILHOUETTE_MIN，必须维持 cam++。
SWEEP_28 = {
    2: {"silhouette": 0.0712, "min_share": 0.3100, "shares": {0: 0.6900, 1: 0.3100}},
    3: {"silhouette": 0.0600, "min_share": 0.1000, "shares": {0: 0.6000, 1: 0.3000, 2: 0.1000}},
}


class TestSuggest:
    def test_two_speaker_interview_is_split(self):
        """#757：双人访谈必须被判成 wespeaker + 显式 K=2。

        显式给人数是绕开 sim_p10 短路的唯一办法，这也是本模块存在的理由。
        """
        got = suggest({"sweep": SWEEP_757, "hit_short_circuit": True, "sim_p10": 0.2362})
        assert got["backend"] == "wespeaker"
        assert got["num_speakers"] == 2

    def test_short_circuit_is_flagged_in_reason(self):
        """命中短路时要写明「自动估 K 不可信」，否则用户会以为后台另有依据。"""
        got = suggest({"sweep": SWEEP_757, "hit_short_circuit": True, "sim_p10": 0.2362})
        assert "sim_p10" in got["reason"]

    def test_no_short_circuit_note_when_not_hit(self):
        got = suggest({"sweep": SWEEP_757, "hit_short_circuit": False, "sim_p10": 0.0696})
        assert "sim_p10" not in got["reason"]

    def test_solo_episode_is_not_misreported(self):
        """#28：确凿单口不得误报 —— 这是最有价值的反例。

        只看「能不能把双人切出来」会让这条用例失效；低轮廓系数必须守住 cam++。
        """
        got = suggest({"sweep": SWEEP_28, "hit_short_circuit": True, "sim_p10": 0.3127})
        assert got["backend"] == "campplus"
        assert got["num_speakers"] is None

    def test_picks_argmax_not_largest_k(self):
        """多个 K 都达标时取轮廓系数最优，不是取最大 K。

        轮廓系数天然随 K 增大而下降，取最大 K 会被「刚好过线的细碎 K」骗到。
        """
        got = suggest({"sweep": SWEEP_757, "hit_short_circuit": True, "sim_p10": 0.2362})
        # K=3 / K=4 同样满足 min_share，但轮廓系数远低于 K=2
        assert got["num_speakers"] == 2

    def test_fragment_cluster_is_excluded(self):
        """#753：多出的簇只占 4%（< MIN_CLUSTER_SHARE）时应回到 K=2。

        该簇对应节目插入的乔布斯英文原声，不是第三位说话人。
        """
        sweep = {
            2: {"silhouette": 0.4014, "min_share": 0.2330, "shares": {0: 0.7670, 1: 0.2330}},
            3: {"silhouette": 0.4149, "min_share": 0.0398, "shares": {0: 0.7585, 1: 0.2017, 2: 0.0398}},
        }
        got = suggest({"sweep": sweep, "hit_short_circuit": False, "sim_p10": 0.0696})
        assert got["num_speakers"] == 2

    def test_silhouette_below_threshold_stays_camplus(self):
        sweep = {2: {"silhouette": SILHOUETTE_MIN - 0.01, "min_share": 0.4, "shares": {0: 0.6, 1: 0.4}}}
        got = suggest({"sweep": sweep, "hit_short_circuit": False, "sim_p10": 0.05})
        assert got["backend"] == "campplus"

    def test_min_share_below_threshold_stays_camplus(self):
        sweep = {2: {"silhouette": 0.5, "min_share": MIN_CLUSTER_SHARE - 0.01, "shares": {0: 0.97, 1: 0.04}}}
        got = suggest({"sweep": sweep, "hit_short_circuit": False, "sim_p10": 0.05})
        assert got["backend"] == "campplus"

    def test_empty_sweep_falls_back_to_camplus(self):
        """预检未产出任何候选（如 VAD 全静音）时不得升级后端。"""
        got = suggest({"sweep": {}, "hit_short_circuit": False, "sim_p10": None})
        assert got == dict(got, backend="campplus", num_speakers=None)

    def test_missing_sweep_key_is_tolerated(self):
        got = suggest({"hit_short_circuit": False})
        assert got["backend"] == "campplus"


class TestSuggestReason:
    """reason 是给用户看的，必须包含让人能复核的信息。"""

    def test_reason_mentions_silhouette_and_share(self):
        got = suggest({"sweep": SWEEP_757, "hit_short_circuit": True, "sim_p10": 0.2362})
        assert "轮廓系数" in got["reason"]
        assert "最小簇" in got["reason"]

    def test_minority_warns_about_possible_clip(self):
        """最小簇占比 <15% 时提示可能是插入的原声片段，便于映射阶段留意。"""
        sweep = {
            2: {"silhouette": 0.40, "min_share": 0.08, "shares": {0: 0.92, 1: 0.08}},
        }
        got = suggest({"sweep": sweep, "hit_short_circuit": False, "sim_p10": 0.05})
        assert got["num_speakers"] == 2
        assert "原声" in got["reason"]

    def test_reason_is_str_for_all_inputs(self):
        for probe_result in (
            {"sweep": SWEEP_757, "hit_short_circuit": True, "sim_p10": 0.2362},
            {"sweep": SWEEP_28, "hit_short_circuit": True, "sim_p10": 0.3127},
            {"sweep": {}, "hit_short_circuit": False, "sim_p10": None},
        ):
            assert isinstance(suggest(probe_result)["reason"], str)


@pytest.mark.parametrize("backend", ["campplus", "wespeaker"])
def test_result_shape(backend):
    """返回结构固定为这三个键，调用方按此取值。"""
    got = suggest({"sweep": SWEEP_757, "hit_short_circuit": True, "sim_p10": 0.2362})
    assert set(got) == {"backend", "num_speakers", "reason"}
