"""说话人塌缩检测 —— 识别 cam++（spk_model）的失效形态。

背景（25 集批量实测）：SenseVoice 的 ``spk_model`` 即 cam++ 声纹模型，按 VAD 段
做**局部**聚类，在圆桌 3+ 人场景会系统性塌缩——多位嘉宾被并进同一个
``[SPEAKER_XX]``。典型实测形态：

- Vol.014：672 段全部落在单一簇
- AI时代触底反弹：1128 段全部落在单一簇
- 文科女+AI：3 位嘉宾塌成 1 簇，共 1695 段

换 WeSpeaker（ResNet34-LM 全局 embedding + GMM BIC 自动估 K）后均可正确拆开。

**本模块只识别可疑形态、不改标签**：命中即告警并给出重跑命令，由人决定是否重跑
（wespeaker 单集要多花 7~11 分钟）。边界注意——单口节目天然只有 1 簇，会命中强
信号，属已知可忽略情形，故告警文案必须自带「确为单口可忽略」的说明。

判据（三条，任一命中即告警）：
1. 强信号：簇数 == 1 且段数 >= ``SINGLE_CLUSTER_SEGMENTS``。**真实塌缩几乎都是这
   一形态**（Vol.014 672 段、触底反弹 1128 段、文科女+AI 1695 段，全部塌成 1 簇）。
2. 中信号：主簇占比 >= ``DOMINANT_RATIO`` 且主簇段数 >= ``DOMINANT_MIN_SEGMENTS``
   ——捕「近乎全塌但残留零星碎片」的情形。
3. 弱信号：Show Notes 花名册人数 > 实际簇数 + 1。

阈值是实测标定而非拍脑袋：对 ``output/`` 下 49 个会话包全量扫描，占比门槛 0.85
会误报 #706（嘉宾 1090 段 vs 主播 139 段，89%，正常的「话痨嘉宾」形态），收紧到
0.95 后误报归零；簇数==1 命中的 #28 与 Vol.552 经核对确为单口（【Jessie】/【主播】
各 1 人），属设计内可忽略。**注意不可靠的降噪方向**：不能用「Show Notes 只识别 1 人」
来排除塌缩——触底反弹正是 Show Notes 漏抓嘉宾名、实际是 4 人圆桌。
"""

from __future__ import annotations

import logging
from collections import Counter

logger = logging.getLogger(__name__)

# 段数太少不具备统计意义（短音频/单段异常），直接跳过检测。
MIN_SEGMENTS = 50
# 簇数为 1 时的段数门槛：达到即判为强信号。
SINGLE_CLUSTER_SEGMENTS = 200
# 主簇占比阈值与绝对段数门槛（两条需同时满足）。
# 0.95 是实测标定值：0.85 会把「嘉宾话多 + 主播惜字」的 2 人正常对话误判为塌缩。
DOMINANT_RATIO = 0.95
DOMINANT_MIN_SEGMENTS = 300


def cluster_distribution(segments: list[dict]) -> dict[str, int]:
    """统计每个 ``[SPEAKER_XX]`` 的段数，按段数降序返回。"""
    counts = Counter(seg.get("speaker") or "UNKNOWN" for seg in segments)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def detect_collapse(
    segments: list[dict],
    expected_speakers: int | None = None,
) -> list[str]:
    """返回塌缩/可疑理由列表；空列表表示形态正常。

    ``expected_speakers`` 为 Show Notes 花名册识别到的人数，可缺省。
    """
    if not segments:
        return []

    dist = cluster_distribution(segments)
    total = sum(dist.values())
    if total < MIN_SEGMENTS:
        return []

    reasons: list[str] = []
    top_speaker, top_count = next(iter(dist.items()))

    if len(dist) == 1 and total >= SINGLE_CLUSTER_SEGMENTS:
        reasons.append(
            f"仅 1 个说话人簇却切出 {total} 段——圆桌节目若被 cam++ 并成一簇即属塌缩"
            f"（若本期确为单口，可忽略）"
        )
    elif (
        len(dist) > 1
        and top_count >= DOMINANT_MIN_SEGMENTS
        and top_count / total >= DOMINANT_RATIO
    ):
        reasons.append(
            f"{top_speaker} 独占 {top_count}/{total} 段"
            f"（{top_count / total:.0%}），其余 {len(dist) - 1} 簇合计仅 "
            f"{total - top_count} 段——疑似多位嘉宾被并进主簇"
        )

    if expected_speakers and len(dist) + 1 < expected_speakers:
        reasons.append(
            f"Show Notes 识别到 {expected_speakers} 位说话人，但只分出 {len(dist)} 簇"
        )

    return reasons


def format_collapse_warning(reasons: list[str], url: str = "") -> str:
    """把检测理由拼成可直接打印/记日志的告警文本（含重跑命令）。"""
    if not reasons:
        return ""
    detail = "；".join(reasons)
    rerun = "python -m src.main <url> --refresh-diarization"
    if url:
        rerun = f"python -m src.main {url} --refresh-diarization"
    return (
        f"说话人分离疑似塌缩：{detail}。"
        f"当前标签可能不可用于校订，确为单口可忽略；否则建议换 WeSpeaker 重跑（仅重分离、复用已有转录）：{rerun}"
    )
