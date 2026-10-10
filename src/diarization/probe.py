"""说话人人数预检（probe）——**在跑分离之前**判断该用哪条路径、该指定几个人。

## 为什么需要它

此前只有「塌缩已发生后」的检测（``collapse_check.detect_collapse``）：必须先用
cam++ 跑完分离，发现簇数不对才告警，再人工用 wespeaker 重跑。这是事后补救，而且
连 wespeaker 的**自动路边也可能判断错**（见下）。本模块把判断提前到分离之前。

## 关键发现：自动估 K 有一条会把双人访谈判成单人的短路

``diarize.estimate_speakers()`` 开头有一段单说话人预检（实测源码）：:

    _single_speaker_sim_p10: float = 0.16
    if sim_p10 >= _single_speaker_sim_p10 and min_k <= 1:
        return 1, ... reason="cosine_similarity_single_speaker"

即：全体 embedding 两两余弦相似度的 10 分位一旦 >= 0.16，**直接返回 1，连
GMM-BIC 都不跑**。远程访谈播客普遍过这条线（#757 实测 0.2393），于是无论后端是
cam++ 还是 wespeaker，自动路径都会把它当单人。

绕法是给 ``min_k=2``：短路条件是 ``min_k <= 1``，给 2 就跳过短路、走真正的
GMM-BIC。**这就是为什么 #757 手动 ``--speakers 2`` 能切对，而自动路径塌成 1 簇。**

## 三层判据（从强到弱）

1. ``sweep`` 多 K 扫描：强制切成 K 个簇时的余弦**轮廓系数**与各簇占比。
   轮廓系数高 = embedding 真的分成几何上分离的两团。
2. ``forced_k``：``estimate_speakers(min_k=2)`` 绕开短路后的 GMM-BIC 结果。
3. ``sim_p10`` / ``auto_reason``：是否命中了上面那条短路（命中即说明不可信）。

``suggest()`` 只在「存在 K>=2 且轮廓系数达标、且每簇都不是碎片」时才主张升级到
wespeaker + 指定人数；否则维持 cam++。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# 轮廓系数门槛：低于此值说明簇在 embedding 空间里仍挤成一团，强制多 K 无意义。
SILHOUETTE_MIN = 0.12
# 真实说话人的最小段数占比：低于此值视为碎片簇而非一个人（#757 主播 Lenny 占 32%）。
MIN_CLUSTER_SHARE = 0.05
# diarize 包内短路阈值，仅用于标注解释。
SHORT_CIRCUIT_P10 = 0.16


def _l2norm(mat):
    import numpy as np

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def cosine_sim_p10(embeddings) -> float:
    """embedding 两两余弦相似度的 10 分位——`diarize` 用它做单说话人短路判据。"""
    import numpy as np

    X = _l2norm(np.asarray(embeddings, dtype="float32"))
    n = X.shape[0]
    if n < 2:
        return 0.0
    sim = X @ X.T
    mask = ~np.eye(n, dtype=bool)
    return float(np.percentile(sim[mask], 10))


def silhouette_cosine(embeddings, labels) -> float:
    """余弦距离下的轮廓系数均值；簇数 < 2 时返回 0.0。

    自己实现而非依赖 sklearn：speaker embedding 天然用余弦度量，直接算比降维到
    欧氏空间更准，也避免为单一指标引入重依赖。
    """
    import numpy as np

    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return 0.0
    X = _l2norm(np.asarray(embeddings, dtype="float32"))
    dist = 1.0 - X @ X.T
    unique = sorted(set(labels.tolist()))
    sils = []
    for i in range(len(labels)):
        own = labels[i]
        same = labels == own
        same_cnt = int(same.sum()) - 1
        if same_cnt <= 0:
            continue
        a_i = dist[i][same].sum() / same_cnt
        b_i = min(float(dist[i][labels == u].mean()) for u in unique if u != own)
        denom = max(a_i, b_i)
        sils.append(0.0 if denom == 0 else (b_i - a_i) / denom)
    return float(sum(sils) / len(sils)) if sils else 0.0


def _subsample_uniform(speech_segments, ratio: float):
    """按时间顺序均匀抽样 VAD 段。

    刻意不用随机抽样：估 K 对长时间跨度上的说话人分布敏感，均匀采样才能保留
    全片覆盖度（一个人可能只在前半段出现）。
    """
    if ratio >= 1.0:
        return list(speech_segments)
    k = max(1, int(len(speech_segments) * ratio))
    if k >= len(speech_segments):
        return list(speech_segments)
    step = len(speech_segments) / float(k)
    return [speech_segments[int(i * step)] for i in range(k)]


def probe_diarization(
    audio_path: str | Path,
    sample_ratio: float = 1.0,
    max_k: int = 6,
) -> dict:
    """跑 VAD + embedding + 多 K 扫描，返回人数与可分性诊断。

    只读音频、不写产物。``sample_ratio`` < 1 时按 VAD 段抽样（近似线性降本），
    用于「值不值得换后端」的快速预判。
    """
    from diarize import (
        cluster_speakers,
        estimate_speakers,
        extract_embeddings,
        run_vad,
    )

    audio_path = Path(audio_path)
    t0 = time.time()
    speech = run_vad(audio_path)
    t_vad = time.time() - t0

    kept = _subsample_uniform(speech, sample_ratio)
    t1 = time.time()
    emb, subs = extract_embeddings(audio_path, kept)
    t_emb = time.time() - t1

    sim_p10 = cosine_sim_p10(emb)

    # 两路估 K：auto 可能被短路，forced 绕开短路走真正的 GMM-BIC
    try:
        auto_k, auto_details = estimate_speakers(emb)
        auto_reason = getattr(auto_details, "reason", None)
    except Exception as exc:  # pragma: no cover - 依赖内部行为
        logger.warning("自动估 K 失败：%s", exc)
        auto_k, auto_reason = 1, "error"
    try:
        forced_k, forced_details = estimate_speakers(emb, min_k=2, max_k=max_k)
    except Exception as exc:  # pragma: no cover
        logger.warning("绕短路估 K 失败：%s", exc)
        forced_k, forced_details = max_k, None

    # 多 K 扫描：强制切 K 簇时的聚类质量
    total = len(subs) or 1
    sweep: dict[int, dict] = {}
    for k in range(2, max_k + 1):
        try:
            labels, _ = cluster_speakers(emb, num_speakers=k)
        except Exception as exc:  # pragma: no cover
            logger.warning("K=%d 聚类失败：%s", k, exc)
            continue
        counts: dict[int, int] = {}
        for lb in labels.tolist() if hasattr(labels, "tolist") else labels:
            counts[int(lb)] = counts.get(int(lb), 0) + 1
        shares = {c: round(v / total, 4) for c, v in sorted(counts.items())}
        sweep[k] = {
            "silhouette": round(silhouette_cosine(emb, labels), 4),
            "shares": shares,
            "min_share": min(shares.values()) if shares else 0.0,
        }

    return {
        "audio": str(audio_path),
        "vad_segments": len(speech),
        "embedded_segments": len(subs),
        "sample_ratio": sample_ratio,
        "sim_p10": round(sim_p10, 4),
        "hit_short_circuit": bool(auto_reason == "cosine_similarity_single_speaker"),
        "auto_k": int(auto_k),
        "auto_reason": auto_reason,
        "forced_k": int(forced_k),
        "sweep": sweep,
        "elapsed_vad_s": round(t_vad, 1),
        "elapsed_embed_s": round(t_emb, 1),
        "elapsed_total_s": round(time.time() - t0, 1),
    }


def suggest(probe_result: dict) -> dict:
    """按三层判据给出后端与人数建议，返回 ``(backend, num_speakers, reason)``。

    **取经过实测标定的 argmax，不是取最大 K**：轮廓系数天然随 K 增大而下降，取最大
    K 会被「刚好过线的细碎 K」骗到。两个标定样本：

    - #757（Tibo + Lenny，实测 68%/32%）：K=2 轮廓系数 0.3098 断层领先，K=3 骤降
      到 0.1383（-55%）——明显的肘部在 K=2；
    - #753（Ben Thompson 独讲 + 插入的乔布斯原声）：K=3 的轮廓系数仅比 K=2 高
      0.013，且多出的簇只占 4%（低于 ``MIN_CLUSTER_SHARE``）—— 属次要声源，排除。
    """
    sweep = probe_result.get("sweep") or {}

    # 先按「每簇都不是碎片」过滤，再取轮廓系数最优的 K
    candidates = [
        k
        for k in sweep
        if sweep[k]["silhouette"] >= SILHOUETTE_MIN
        and sweep[k]["min_share"] >= MIN_CLUSTER_SHARE
    ]
    hit = probe_result.get("hit_short_circuit")

    if candidates:
        k = max(candidates, key=lambda kk: sweep[kk]["silhouette"])
        info = sweep[k]
        minority = sorted(info["shares"].values())[0]
        note = (
            f"（注：自动估 K 命中 diarize 的 sim_p10>={SHORT_CIRCUIT_P10} 短路，"
            f"实为 sim_p10={probe_result.get('sim_p10')}，该结果不可信）"
            if hit
            else ""
        )
        warn = (
            f"；最小簇仅占 {minority:.1%}，可能是插入的原声片段而非第三位说话人，映射时留意"
            if minority < 0.15
            else ""
        )
        # 被更好的次要候选排除时说明理由
        extras = [
            f"K={kk}:{sweep[kk]['silhouette']:.2f}"
            for kk in sorted(sweep)
            if kk != k and kk >= k
        ]
        return {
            "backend": "wespeaker",
            "num_speakers": k,
            "reason": (
                f"声学可分 {k} 人：K={k} 轮廓系数 {info['silhouette']:.2f}"
                f"（其余候选 {' '.join(extras)}），最小簇占比 {info['min_share']:.1%}"
                f"——必须显式指定人数，否则会塌成 1 簇{note}{warn}"
            ),
        }

    second = sorted(sweep, reverse=True)[0] if sweep else None
    detail = ""
    if second:
        detail = (
            f"（最优候选 K={second} 轮廓系数仅 {sweep[second]['silhouette']:.2f}"
            f"，最小簇占比 {sweep[second]['min_share']:.1%}）"
        )
    return {
        "backend": "campplus",
        "num_speakers": None,
        "reason": f"未找到可分的多簇形态{detail}，维持 cam++ 自动标签",
    }


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="说话人人数预检（决定分离后端与人数）")
    ap.add_argument("audio", help="音频路径（16k mono WAV）")
    ap.add_argument("--sample-ratio", type=float, default=1.0, help="VAD 段抽样比例（降本）")
    ap.add_argument("--max-k", type=int, default=6, help="扫描的最大 K")
    args = ap.parse_args()

    res = probe_diarization(args.audio, args.sample_ratio, max_k=args.max_k)
    res["suggestion"] = suggest(res)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
