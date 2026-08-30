"""_align_speakers 强制 K 类聚类单测（mock cam++ SV，无需真实模型）。"""
from src.transcriber.funasr_transcriber import FunASRTranscriber


def _make_transcriber():
    # 构造器不加载模型；_ensure_sv_loaded 由测试替换
    return FunASRTranscriber()


def _fake_sv(sim_matrix):
    class FakeSV:
        def inference_sv(self, a, b):
            i = int(str(a).rsplit("_", 1)[1].split(".")[0])
            j = int(str(b).rsplit("_", 1)[1].split(".")[0])
            return {"scores": [sim_matrix[i][j]]}

    return FakeSV()


def _reps(n):
    return [{"key": (0, k), "audio": f"/tmp/clip_{k}.wav"} for k in range(n)]


# 4 个代表片段，真实 2 人：{0,1} 同类（相似度 0.9），{2,3} 同类，跨类 0.1
_SIM_2 = [
    [1.0, 0.9, 0.1, 0.1],
    [0.9, 1.0, 0.1, 0.1],
    [0.1, 0.1, 1.0, 0.9],
    [0.1, 0.1, 0.9, 1.0],
]


def test_align_forced_k_merges_oversegmented():
    # 已知 2 人时，即便 cam++ 把同人两段打成 0.1（远低于 0.31 门槛），
    # 也应强制聚成 2 类，而非碎成 4 类（vol.231 根因）。
    tr = _make_transcriber()
    tr._ensure_sv_loaded = lambda: _fake_sv(_SIM_2)
    mapping = tr._align_speakers(_reps(4), num_speakers=2)
    assert len(set(mapping.values())) == 2
    assert mapping[(0, 0)] == mapping[(0, 1)]
    assert mapping[(0, 2)] == mapping[(0, 3)]
    assert mapping[(0, 0)] != mapping[(0, 2)]


def test_align_none_keeps_greedy():
    # num_speakers=None 保持原贪心 + SV_THRESHOLD 行为
    tr = _make_transcriber()
    tr._ensure_sv_loaded = lambda: _fake_sv(_SIM_2)
    mapping = tr._align_speakers(_reps(4), num_speakers=None)
    assert len(set(mapping.values())) == 2


def test_kmedoids_basic():
    assign = FunASRTranscriber._kmedoids(_SIM_2, 2)
    assert len(set(assign)) == 2
    assert assign[0] == assign[1]
    assert assign[2] == assign[3]
