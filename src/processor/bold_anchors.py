"""成稿回标加粗：从上层段落提取锚点，在原文转录区做确定性加粗。

背景：会话链路（``--llm-mode session``）由 LLM 直接产出 Markdown，转录区的回标
加粗指望 LLM 自己执行，实测命中率极低——LLM 给的「原话」常被润色成书面语，
与口语转录逐字对不上（见 ``scripts/oneoff/backfill_anchors.py`` 的实测：15 条 0 命中）。
结果成稿的原文转录区一处加粗都没有，读者无法据此定位关键内容。

本模块不重跑 LLM：直接从成稿**已有的**上层内容里取候选锚点——核心观点的引文
（「」内）、缩进术语子条目、术语表词条——在转录区做「变长片段精确匹配」，
命中即用渲染器的确定性回标加粗（``src.renderer.markdown._apply_bold_anchors``）。

用法：
    python -m src.processor.bold_anchors "output/xxx.md"          # 干跑，只报命中
    python -m src.processor.bold_anchors "output/xxx.md" --apply  # 落盘（自动 .bak）
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from src.renderer.markdown import _apply_bold_anchors
from src.utils import TRANSCRIPT_HEADING_RE

# 只剥行首说话人标签（[SPEAKER_XX] / 【姓名】），保留正文与换行
TAG_RE = re.compile(r"^\s*\[?[^\]】]*[\]】]\s*")
PUNCT_RE = re.compile(r"[ \t　，。！？、；：,.!?;:\"'「」『』（）()《》\-—…]")
BOUNDARY_CHARS = " \n，。！？、；：,.!?;:（）()「」『』《》\"'"

MIN_FRAG = 8          # 长候选（引用原话）片段最短长度：再短容易命中无关位置
MAX_FRAG = 24         # 片段最长长度：与渲染器 anchor 上限（40）留余量
MIN_ANCHOR_NORM = 10  # 命中后的最小有效长度：过短无回标价值，宁缺勿滥
MIN_TERM_NORM = 2     # 术语词最短长度：术语通常只有几个字，走精确匹配
MAX_TERM_NORM = 12    # 术语词最长长度：与渲染器术语锚上限一致


def _section(md: str, name: str) -> str:
    """取 ``## name`` 小节正文（到下一个二级标题或文末）。"""
    m = re.search(rf"^##\s*{re.escape(name)}\s*$([\s\S]*?)(?=^##\s|\Z)", md, re.M)
    return m.group(1) if m else ""


def _term_variants(term: str) -> list[str]:
    """术语词变体：原词 + 括号前后拆分（「dogfooding（内部试用）」拆出两段）。"""
    t = term.strip().rstrip("：:")
    out = [t] if t else []
    m = re.match(r"^(.*?)[（(]([^）)]*)[）)]\s*$", t)
    if m:
        if m.group(1).strip():
            out.append(m.group(1).strip())
        if m.group(2).strip():
            out.append(m.group(2).strip())
    return out


def extract_candidates(upper_md: str) -> list[tuple[str, str]]:
    """从上层段落取候选锚点，返回 (文本, 类型) 列表。

    类型 ``quote``＝核心观点的引用原话（走变长片段匹配）；
    类型 ``term``＝缩进术语子条目与术语表词条（术语只有几个字，走精确匹配）。
    只取「核心观点」与「术语表」两节——观点句、问句是提炼书面语，逐字命中率极低，
    且不该被加粗；引用原话与术语才是回标对象。
    """
    cands: list[tuple[str, str]] = []
    kp = _section(upper_md, "核心观点")
    for q in re.findall(r"「([^」]+)」", kp):
        if q.strip():
            cands.append((q.strip(), "quote"))
    for t in re.findall(r"(?m)^\s+-\s+\*\*([^*]+)\*\*", kp):  # 缩进术语子条目
        cands.extend((v, "term") for v in _term_variants(t))
    gl = _section(upper_md, "术语表")
    for t in re.findall(r"(?m)^-\s+\*\*([^*]+)\*\*", gl):     # 术语表词条
        cands.extend((v, "term") for v in _term_variants(t))

    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for c, kind in cands:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            out.append((c, kind))
    return out


def _strip_tags(text: str) -> str:
    """剥掉每行行首的说话人标签（锚点须在剥标签后的正文里取）。"""
    return "\n".join(TAG_RE.sub("", ln) for ln in text.splitlines())


def _build_norm(raw: str) -> tuple[str, list[int]]:
    """去标点归一化 + 索引映射：norm[i] 对应 raw[idx_map[i]]。

    anchor 必须回 raw 取子串——渲染器在**带标点的原文**里做精确匹配，
    直接返回去标点后的片段会全部落空。
    """
    chars: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(raw):
        if not PUNCT_RE.match(ch):
            chars.append(ch)
            idx.append(i)
    return "".join(chars), idx


def _at_boundary(text: str, start: int, end: int) -> bool:
    """锚点两端须落在自然边界上，避免切出「odex 和 ChatGPT」这种半截词。"""
    if start > 0 and text[start - 1] not in BOUNDARY_CHARS:
        return False
    if end < len(text) and text[end] not in BOUNDARY_CHARS:
        return False
    return True


def find_anchor(cand: str, norm: str, idx_map: list[int], raw: str) -> str | None:
    """在原文找候选的逐字锚点（原文子串）；找不到返回 None。

    三道硬约束（同 backfill_anchors）：① 必须是原文子串 ② 不跨行 ③ 两端落自然边界。
    做法：把候选去标点，按 24→8 字降序切片在归一化原文里找首次精确命中，
    命中后贪心右扩展补回被步长跳过的字。
    """
    q = PUNCT_RE.sub("", cand)
    if len(q) < MIN_FRAG:
        return None
    hi = min(len(q), MAX_FRAG)
    for length in range(hi, MIN_FRAG - 1, -2):
        for start_q in range(0, len(q) - length + 1, 2):
            p = norm.find(q[start_q : start_q + length])
            if p < 0:
                continue
            k = length
            while (
                start_q + k < len(q)
                and p + k < len(norm)
                and norm[p + k] == q[start_q + k]
                and norm[p + k] != "\n"
            ):
                k += 1
            start, end = idx_map[p], idx_map[p + k - 1] + 1
            if k < MIN_ANCHOR_NORM:  # 太短没有回标价值（如「和 ChatGPT」）
                continue
            if start_q > 0 and not _at_boundary(raw, start, end):
                continue
            if "\n" in raw[start:end]:
                continue
            return raw[start:end].strip()
    return None


def find_term_anchor(cand: str, norm: str, idx_map: list[int], raw: str) -> str | None:
    """术语命中：术语只有几个字，直接做精确匹配（不跨行；两端不限边界）。"""
    q = PUNCT_RE.sub("", cand)
    if not (MIN_TERM_NORM <= len(q) <= MAX_TERM_NORM):
        return None
    p = norm.find(q)
    if p < 0:
        return None
    start, end = idx_map[p], idx_map[p + len(q) - 1] + 1
    if "\n" in raw[start:end]:
        return None
    return raw[start:end].strip()


def collect_anchors(md: str) -> list[str]:
    """从成稿取可在转录区命中的锚点列表（长锚优先）。"""
    m = TRANSCRIPT_HEADING_RE.search(md)
    if not m:
        return []
    upper, trans = md[: m.start()], md[m.start():]
    raw = _strip_tags(trans)
    norm, idx_map = _build_norm(raw)
    anchors: list[str] = []
    for cand, kind in extract_candidates(upper):
        a = (
            find_anchor(cand, norm, idx_map, raw)
            if kind == "quote"
            else find_term_anchor(cand, norm, idx_map, raw)
        )
        if a and a not in anchors:
            anchors.append(a)
    # 长锚优先（与渲染器一致：长锚先加粗，其子串短锚不再叠加）
    return sorted(anchors, key=len, reverse=True)


def bold_markdown(md: str) -> tuple[str, list[str]]:
    """返回 (加粗后的 md, 命中的锚点)；无可命中锚点时原样返回。"""
    anchors = collect_anchors(md)
    if not anchors:
        return md, []
    m = TRANSCRIPT_HEADING_RE.search(md)
    upper, trans = md[: m.start()], md[m.start():]
    return upper + _apply_bold_anchors(trans, anchors), anchors


def main() -> int:
    ap = argparse.ArgumentParser(description="成稿原文转录回标加粗（确定性，不调 LLM）")
    ap.add_argument("md_path", type=Path, help="成稿 .md 路径")
    ap.add_argument("--apply", action="store_true", help="写回文件（自动 .bak 备份）")
    args = ap.parse_args()

    if not args.md_path.exists():
        print(f"错误：找不到文件 {args.md_path}", file=sys.stderr)
        return 1
    md = args.md_path.read_text(encoding="utf-8")
    new_md, anchors = bold_markdown(md)
    print(f"文件: {args.md_path.name}")
    print(f"命中锚点: {len(anchors)}")
    for a in anchors:
        print(f"  OK  {a}")
    if not args.apply:
        print("\n（干跑，未落盘。加 --apply 生效）")
        return 0
    if new_md == md:
        print("\n转录区已是加粗状态，无需改动。")
        return 0
    bak = args.md_path.with_suffix(args.md_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(args.md_path, bak)
    args.md_path.write_text(new_md, encoding="utf-8")
    _m = TRANSCRIPT_HEADING_RE.search(new_md)
    trans = new_md[_m.start():] if _m else ""
    print(f"\n已加粗并写回 {args.md_path.name}（备份 {bak.name}）")
    print(f"转录区加粗 {trans.count('**') // 2} 处")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
