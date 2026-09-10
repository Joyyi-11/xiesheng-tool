"""_align_speakers 强制 K 类聚类单测（mock cam++ SV，无需真实模型）。"""
import numpy as np
import pytest

from src.transcriber.funasr_transcriber import FunASRTranscriber


def _make_transcriber():
    # 构造器不加载模型；_ensure_sv_loaded 由测试替换
    return FunASRTranscriber()


def _fake_sv(embeddings):
    """返回 mock 的 cam++ 模型：generate 返回对应片段的 speaker embedding。

    新实现对每个代表片段调用一次 ``generate(input=audio)`` 取 embedding，而非
    旧的 ``inference_sv(a, b)`` 两两打分。
    """

    class FakeSV:
        def generate(self, input=None, **kwargs):
            k = int(str(input).rsplit("_", 1)[1].split(".")[0])
            return {"spk_embedding": embeddings[k]}

    return FakeSV()


def _reps(n):
    return [{"key": (0, k), "audio": f"/tmp/clip_{k}.wav"} for k in range(n)]


def _embeddings_2pairs():
    """4 个代表片段、真实 2 人的 embedding。

    5 维构造：{0,1} 同类（余弦 ≈ 0.9），{2,3} 同类，跨类余弦 ≈ 0.01（非零，
    避开相似度矩阵退化自检的 0 值阈值）。
    """
    base = 0.1
    return [
        np.array([1.0, 0.0, 0.0, 0.0, base]),
        np.array([0.9, 0.4359, 0.0, 0.0, base]),
        np.array([0.0, 0.0, 1.0, 0.0, base]),
        np.array([0.0, 0.0, 0.9, 0.4359, base]),
    ]


# kmedoids 纯函数单测用的相似度矩阵：{0,1} 同类（0.9），{2,3} 同类，跨类 0.1
_SIM_2 = [
    [1.0, 0.9, 0.1, 0.1],
    [0.9, 1.0, 0.1, 0.1],
    [0.1, 0.1, 1.0, 0.9],
    [0.1, 0.1, 0.9, 1.0],
]


def test_align_forced_k_merges_oversegmented():
    # 已知 2 人时，即便 cam++ 把同人两段打成低相似度，也应强制聚成 2 类，
    # 而非碎成 4 类（vol.231 根因）。
    tr = _make_transcriber()
    tr._ensure_sv_loaded = lambda: _fake_sv(_embeddings_2pairs())
    mapping = tr._align_speakers(_reps(4), num_speakers=2)
    assert len(set(mapping.values())) == 2
    assert mapping[(0, 0)] == mapping[(0, 1)]
    assert mapping[(0, 2)] == mapping[(0, 3)]
    assert mapping[(0, 0)] != mapping[(0, 2)]


def test_align_none_keeps_greedy():
    # num_speakers=None 保持原贪心 + SV_THRESHOLD 行为
    tr = _make_transcriber()
    tr._ensure_sv_loaded = lambda: _fake_sv(_embeddings_2pairs())
    mapping = tr._align_speakers(_reps(4), num_speakers=None)
    assert len(set(mapping.values())) == 2


def test_kmedoids_basic():
    assign = FunASRTranscriber._kmedoids(_SIM_2, 2)
    assert len(set(assign)) == 2
    assert assign[0] == assign[1]
    assert assign[2] == assign[3]


def _sim_with_outlier():
    """3 人各 2 片段 + 1 孤立噪声点（与所有人相似度 0.05）。"""
    n = 7
    sim = [[0.1] * n for _ in range(n)]  # 默认异人 0.1
    for i in range(n):
        sim[i][i] = 1.0
    for a, b in ((0, 1), (2, 3), (4, 5)):  # 同人 0.9
        sim[a][b] = sim[b][a] = 0.9
    for i in range(n):  # 噪声点(6) 与所有人 0.05
        sim[i][6] = sim[6][i] = 0.05
    sim[6][6] = 1.0
    return sim


def test_kmedoids_does_not_pick_outlier_as_medoid():
    # 旧 max-min 初始化会把孤立噪声点（与所有人低相似度）选为 medoid、独占一簇，
    # 把真实说话人挤进剩余簇。修复后 3 个真实说话人应各自成簇（噪声可并入某簇，
    # 但不应独占一簇）。
    sim = _sim_with_outlier()
    assign = FunASRTranscriber._kmedoids(sim, 3)
    assert assign[0] == assign[1]  # 人 A
    assert assign[2] == assign[3]  # 人 B
    assert assign[4] == assign[5]  # 人 C
    assert len({assign[0], assign[2], assign[4]}) == 3  # 三人分到三个不同簇


def test_assert_no_chunk_collapse_raises_on_degenerate():
    # 全零矩阵（非对角）正是求职别慌一期的塌缩形态，必须显式拦截。
    n = 4
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim[i][i] = 1.0
    with pytest.raises(RuntimeError):
        FunASRTranscriber._assert_no_chunk_collapse(sim)


def test_assert_no_chunk_collapse_passes_on_healthy():
    # _SIM_2 非对角无精确 0，不应报错
    FunASRTranscriber._assert_no_chunk_collapse(_SIM_2)
