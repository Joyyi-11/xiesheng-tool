"""Deterministic speaker resolution & diarization quality gate (no LLM needed).

Two failure modes of forced-K clustering are caught and reported here:

* MERGE — two acoustically similar speakers forced into one cluster. Detected by
  a self-reference contradiction: one cluster contains both "我是A" and "我是B"
  mapping to different real people (this episode: 小猪 + 秋秋 merged under K=3).
* OVER-SEG — one speaker split across several clusters (host fragmented).

Fixes applied downstream (in ``src/main.py``):

* MERGE -> raise K so similar voices separate (re-align only; the ASR chunk cache
  is reused, so this is seconds, not a re-transcribe), then merge fragments back
  by identity.
* OVER-SEG -> merge fragments that resolve to the same real person.

Speaker -> real-name mapping combines the Show Notes roster (host + guests) with
in-transcript self-references and host markers. Low-confidence clusters are left
as ``[SPEAKER_XX]`` and flagged — never silently mislabeled. The one thing this
guarantees is that the HOST is never wrongly attributed to a guest (the catastrophic
failure seen before).
"""
from __future__ import annotations

import re
from collections import defaultdict

_ROLE = r"(?:主[播持理]|嘉宾|客座)"
_NAME = r"[一-龥A-Za-z·]{1,6}"
# 角色词后捕获名字，并额外返回角色（show_notes_speakers 是其薄封装）。
_ROSTER_RE = re.compile(rf"{_ROLE}(?:简介|人)?(?:主[播持理]|嘉宾|客座|主持)?[：:\s]*({_NAME})")
_TRUNCATE = re.compile(r"(?:主[播持理]|嘉宾|客座|主持|与|和|及|等|携|邀请)")

# 自称之后紧接的词里，明显不是人名的填词，避免把「我是本科/一个」当名字。
_STOPWORDS = {
    "是", "与", "和", "等", "的", "为", "叫", "有", "及", "或", "也", "来",
    "一个", "这个", "那个", "他们", "她们", "我们", "你们", "因为", "所以",
    "本科", "学历", "文科", "腾讯", "微信", "在做", "当时", "现在",
}

# 自称提取：我是/我叫/我们是/我就是 X。X 后须紧跟标点/空白边界，
# 否则会把「我是出生于河南啊」整段误捕获为名字。限定 1-4 字（人名通常很短）。
_SELF_RE = re.compile(r"我(?:是|叫|们是|就是)\s*([一-龥A-Za-z·]{1,4})(?=[，。！？、；：\s])")
# 主播标记（强信号）：主持人才会「邀请/介绍」嘉宾。泛用的「两位」太弱
# （嘉宾也会说「两位」），不用于判定主播，避免把嘉宾碎片误判为主播。
_HOST_RE = re.compile(r"邀请了两位|邀请了|给大家介绍|今天邀请|咱们.?两位|两位嘉宾")

# 转写显示标签：仅「短名」+【】包裹，角色与外文姓氏不进前缀。
# 角色归属是 ## 人物简介 / Show Notes 的职责，外文按 · 取 given name
# （伦尼·拉奇茨基→伦尼）。这是显示格式唯一真相源，api / session 两链路共用，
# 保证产出等价、不依赖 LLM 是否「遵守约定」。
# 不能用 ^{_ROLE}：_ROLE 里的「主[播持理]」只吃两字，会把三字角色「主持人」劈成
# 「主持」+「人」，产出「【人Nina】」这类残名；此处须整词吞掉（含可选「人」）。
_ROLE_PREFIX = re.compile(r"^(?:主[播持理]人?|嘉宾|客座)")
_PAREN = re.compile(r"[（(][^）)]*[）)]")

# 口语自称与 Show Notes 花名册姓名的同音/昵称别名表。
# 解析时口语自称（如「小猪」「秋秋」）据此对齐到花名册正式姓名（「小朱」「湫湫」），
# 避免同一人因自称与花名册写法不同而落为 [SPEAKER_XX]。
# 扩充：发现新的自称/花名册错位时在此追加（如「小朱」在别期被叫作「朱哥」）。
COMMON_ALIASES = {
    "小猪": "小朱",
    "秋秋": "湫湫",
}


def to_display_label(canonical_name: str) -> str:
    """把 canonical 姓名渲染为转写用的显示标签：【短名】。

    - 剥角色前缀（主播/嘉宾…）；
    - 剥括号注释（如「（Lenny's Podcast 主持）」）；
    - 外文按 · 取 given name（伦尼·拉奇茨基→伦尼）；
    - 以【】包裹。

    角色元数据不出现在此（归 ## 人物简介 / Show Notes），避免每行前缀复读造成
    视觉冗余。这是显示格式唯一真相源，api / session 两链路都经此函数落地。
    """
    name = (canonical_name or "").strip()
    name = _ROLE_PREFIX.sub("", name)
    name = _PAREN.sub("", name).strip()
    if not name:
        return ""
    if "·" in name:
        name = name.split("·", 1)[0].strip()
    return f"【{name}】"


def _derive_aliases(roster: list[tuple[str, str]], segments: list[dict]) -> dict[str, str]:
    """从段自称推导别名：自称不在花名册但命中 COMMON_ALIASES 且目标在花名册，
    则对齐（如口语「小猪」→ 花名册「小朱」）。无匹配则返回空 dict。
    """
    aliases: dict[str, str] = {}
    roster_names = {n for _, n in roster}
    sig = _cluster_signals(segments)
    for spk, d in sig.items():
        for nm in d["self_names"]:
            if nm in roster_names:
                continue
            target = COMMON_ALIASES.get(nm)
            if target and target in roster_names:
                aliases[nm] = target
    return aliases


def parse_roster_with_roles(show_notes: str) -> list[tuple[str, str]]:
    """从 Show Notes 解析 [(role, name), ...]；role ∈ {主播,主持,嘉宾,客座}。

    无 Show Notes 或不足 2 人时返回空列表（交给显式 --speakers 或自动检测）。
    """
    if not show_notes:
        return []
    clean = re.sub(r"[【】]", "", show_notes)
    out: list[tuple[str, str]] = []
    for m in _ROSTER_RE.finditer(clean):
        # _ROLE 为非捕获组，唯一捕获组 group(1)=名字；角色取匹配首两字。
        role = m.group(0)[:2]
        name = _TRUNCATE.split(m.group(1))[0].strip()
        if len(name) < 2 or name in _STOPWORDS:
            continue
        if (role, name) not in out:
            out.append((role, name))
    return out


def _cluster_signals(segments: list[dict]) -> dict[str, dict]:
    """汇总每簇信号：自称集合、主播标记计数、拼接全文。"""
    sig: dict[str, dict] = defaultdict(lambda: {"self_names": set(), "host_markers": 0, "text": []})
    for s in segments:
        spk = s.get("speaker")
        if not spk:
            continue
        text = s.get("text", "")
        sig[spk]["text"].append(text)
        for sm in _SELF_RE.finditer(text):
            nm = sm.group(1).strip()
            if len(nm) >= 2 and nm not in _STOPWORDS:
                sig[spk]["self_names"].add(nm)
        sig[spk]["host_markers"] += len(_HOST_RE.findall(text))
    for spk in sig:
        sig[spk]["text"] = "\n".join(sig[spk]["text"])
    return sig


def detect_diarization_issues(
    segments: list[dict], roster: list[tuple[str, str]]
) -> list[str]:
    """返回问题清单（空 = 无合并矛盾/严重过切分）。

    仅报「合并矛盾」（需提高 K 重对齐）与「严重过切分」（簇数远超花名册）。
    主播碎片化本身由 merge 兜底，不在此报错。
    """
    issues: list[str] = []
    roster_names = {n for _, n in roster}
    aliases = _derive_aliases(roster, segments)
    sig = _cluster_signals(segments)
    for spk, d in sig.items():
        matched = {aliases.get(n, n) for n in d["self_names"]} & roster_names
        if len(matched) >= 2:
            issues.append(
                f"{spk} 同簇出现多个不同真人自称（{'、'.join(sorted(matched))}），"
                f"疑似 diarization 把声线相近的人合并，建议提高 K 重对齐"
            )
    spk_count = len(sig)
    if roster and spk_count > len(roster) + 1:
        issues.append(
            f"说话人簇数 {spk_count} 超过花名册人数 {len(roster)}，"
            f"疑似过切分（将按身份合并碎片）"
        )
    return issues


def resolve_speaker_map(
    segments: list[dict],
    roster: list[tuple[str, str]],
    aliases: dict[str, str] | None = None,
    cluster_overrides: dict[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """把簇映射到【身份姓名】的纯文字标签（不含方括号）。

    返回 ``(spk -> label, issues)``。低置信度簇不进入 map（保留 [SPEAKER_XX]）。

    映射优先级：⓪ 显式覆盖 cluster_overrides（经核验的过切分碎片合并，最高优先）；
    ① 精确自称匹配花名册；② 主播判定（自称=主播名，或含主播标记且无
    嘉宾自称冲突）。仍无法分配的簇保留标签并标红。
    ``aliases`` 用于昵称对齐（如小猪->小朱），提升全自动命中率。
    """
    aliases = aliases or {}
    issues: list[str] = []
    roster_names = {n for _, n in roster}
    host_names = {n for r, n in roster if r in ("主播", "主持", "主理")}
    guest_names = {n for r, n in roster if r in ("嘉宾", "客座")}
    sig = _cluster_signals(segments)

    spk_to_label: dict[str, str] = {}
    used_guests: set[str] = set()
    ambiguous_speakers: set[str] = set()

    # ⓪ 显式覆盖：经人工/LLM 核验的过切分碎片合并（如 SPEAKER_03->主播湫湫）。
    # 最高优先，跳过后续自动推断、绝不猜测。
    if cluster_overrides:
        for spk, lbl in cluster_overrides.items():
            spk_to_label[spk] = lbl
            role = lbl[:2] if lbl[:2] in ("主播", "主持", "主理", "嘉宾", "客座") else ""
            if role in ("嘉宾", "客座"):
                used_guests.add(lbl[2:])

    # Step 1: 精确自称匹配（含昵称别名）。多个真人命中时保留标签并告警。
    for spk in sorted(sig):
        d = sig[spk]
        candidates = {aliases.get(x, x) for x in d["self_names"]} & roster_names
        if len(candidates) > 1:
            ambiguous_speakers.add(spk)
            issues.append(
                f"{spk} 同时命中多个花名册真人（{'、'.join(sorted(candidates))}），"
                "保留 [SPEAKER_XX] 待人工复核"
            )
            continue
        if not candidates:
            continue
        nm = next(iter(candidates))
        if nm in used_guests:
            issues.append(f"{spk} 与其他簇同时指向嘉宾 {nm}，保留 [SPEAKER_XX] 待人工复核")
            continue
        role = next((r for r, n in roster if n == nm), "")
        spk_to_label[spk] = f"{role}{nm}"
        if role in ("嘉宾", "客座"):
            used_guests.add(nm)

    # Step 2: 主播判定
    for spk, d in sig.items():
        if spk in spk_to_label:
            continue
        if spk in ambiguous_speakers:
            continue
        self_in_host = {aliases.get(x, x) for x in d["self_names"]} & host_names
        if self_in_host:
            spk_to_label[spk] = f"主播{min(self_in_host)}"
            continue
        if d["host_markers"] >= 1 and not (d["self_names"] & guest_names):
            if len(host_names) == 1:
                spk_to_label[spk] = f"主播{next(iter(host_names))}"
                continue

    # 未获得可靠身份证据的簇一律不自动分配（不做嘉宾排除法猜测）。
    unassigned = [spk for spk in sig if spk not in spk_to_label]
    for spk in unassigned:
        if spk not in spk_to_label:
            issues.append(
                f"{spk} 无法可靠映射到花名册（自称={'/'.join(sorted(sig[spk]['self_names']) or ['无'])}，"
                f"主播标记={sig[spk]['host_markers']}），保留 [SPEAKER_XX] 待人工/LLM 复核"
            )
    return spk_to_label, issues


def apply_speaker_map(
    segments: list[dict], spk_to_label: dict[str, str]
) -> list[dict]:
    """返回新段列表：每段加 ``speaker_label``（【身份姓名】或 [SPEAKER_XX]），

    并合并相邻同标签段（gap<=0.2s）。未映射的簇保留 ``[SPEAKER_XX]``（带方括号，
    与历史会话包约定一致、可被校验器捕获），绝不改写为猜测的姓名。
    """
    out: list[dict] = []
    for s in segments:
        spk = s.get("speaker")
        if spk in spk_to_label:
            label = f"【{spk_to_label[spk]}】"
        elif spk:
            label = f"[{spk}]"
        else:
            label = ""
        ns = dict(s)
        ns["speaker_label"] = label
        out.append(ns)
    merged: list[dict] = []
    for s in out:
        lbl = s.get("speaker_label")
        prev = merged[-1] if merged else None
        can_merge = (
            prev is not None
            and prev.get("speaker_label") == lbl
            and "start_time" in s
            and "end_time" in prev
            and s["start_time"] - prev["end_time"] <= 0.2
        )
        if can_merge:
            prev["text"] = (prev["text"] + " " + s["text"]).strip()
            prev["end_time"] = s["end_time"]
        else:
            merged.append(dict(s))
    return merged


def resolve_and_label(
    segments: list[dict],
    show_notes: str,
    aliases: dict[str, str] | None = None,
    cluster_overrides: dict[str, str] | None = None,
) -> tuple[list[dict], dict[str, str], list[str]]:
    """一站式：解析花名册 -> 映射 -> 贴标 -> 合并。

    ``aliases`` 为 None 时自动从花名册 + 段自称推导同音/昵称别名
    （见 ``COMMON_ALIASES`` / ``_derive_aliases``），使口语自称对齐花名册姓名。
    ``cluster_overrides`` 为经核验的过切分碎片合并（最高优先，见 ``resolve_speaker_map``）。

    返回 (labeled_segments, spk_to_label, issues)。
    """
    roster = parse_roster_with_roles(show_notes)
    if aliases is None:
        aliases = _derive_aliases(roster, segments)
    spk_to_label, issues = resolve_speaker_map(
        segments, roster, aliases=aliases, cluster_overrides=cluster_overrides
    )
    labeled = apply_speaker_map(segments, spk_to_label)
    return labeled, spk_to_label, issues
