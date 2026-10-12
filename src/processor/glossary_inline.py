# -*- coding: utf-8 -*-
"""术语表三态判定与确定性内联。

规则出处：``rules.STRUCT_GLOSSARY``（v28 新增三态判据）。同一术语相对
「核心观点／问题与思考」只有三种处境，只能落一处、不得两处都写：

  1. **已内联**（术语名已在某条观点的缩进子条目里）→ 术语表重复，删除该条
  2. **可内联**（术语名出现在某条观点正文、但没有对应子条目）→ 就近内联，取首次出现的观点
  3. **无关联**（两处都没出现）→ 这才是术语表该收的残留硬术语，保留

本模块是这三条判据的**唯一实现**：校验闸门
（``session_edit.validate_session_output``）与确定性修复共用同一套判定，
避免两边各写一份导致口径漂移（教训见 anchor 失效那次：只改 md 没同步 handoff）。
"""

from __future__ import annotations

import re

# 术语名候选形态：主名（去括号）→ 括号内外文 → 原名 → 斜杠分段。
# 术语表常写成「抽象层级（abstraction layer）」，而正文里只出现「抽象层级」，
# 不归一化会把大量本可定位的术语误判成「无关联」。
_BRACKET_RE = re.compile(r"^(.+?)[（(](.+?)[）)]\s*$")

# 术语表条目：- **术语**：解释
_TERM_ITEM_RE = re.compile(r"\*\*(.+?)\*\*[:：]")
# 缩进子条目（观点下已内联的术语）
_SUB_ITEM_RE = re.compile(r"^\s+-\s+")
# 子条目的术语名：**超级个体。** 解释 —— 只取加粗部分，不能拿整行去匹配：
# 术语的解释文本里常出现别的术语名（如「运行框架」的解释里写了 ChatGPT Work），
# 拿整行做子串匹配会把这些无辜术语误判成「已内联的重复」。
_SUB_TERM_RE = re.compile(r"^\s+-\s+\*\*(.+?)\*\*")
# 顶级条目
_TOP_ITEM_RE = re.compile(r"^-\s+")

# 长度下限：避免「变化」「选择」这类两字常用词在正文出现就被误判为可内联
_MIN_TERM_LEN = 2


def term_variants(term: str) -> list[str]:
    """术语名的候选形态，按优先级排列（先主名、后外文、再原名）。"""
    out: list[str] = []
    m = _BRACKET_RE.match(term)
    if m:
        out.append(m.group(1).strip())
        out.append(m.group(2).strip())
    out.append(term)
    if "/" in term:
        out.extend(x.strip() for x in term.split("/"))
    seen: set[str] = set()
    res: list[str] = []
    for v in out:
        if v and v not in seen and len(v) >= _MIN_TERM_LEN:
            seen.add(v)
            res.append(v)
    return res


def _section(md: str, name: str) -> tuple[int, int, str] | None:
    """返回小节的 (标题起始, 正文结束, 正文)。找不到返回 None。"""
    m = re.search(r"^## " + re.escape(name) + r"\s*$", md, re.M)
    if not m:
        return None
    nxt = re.search(r"^## ", md[m.end():], re.M)
    end = m.end() + nxt.start() if nxt else len(md)
    return m.start(), end, md[m.end(): end]


# 未校订占位稿：- **待校订。** 待提炼本期重要观点……
# 这类稿的核心观点与术语表都还是模板占位，参与确定性修复只会把占位符搬来搬去，
# 产出垃圾且掩盖「这集根本没校订」的事实——一律跳过，交由校验器报未校订。
_PLACEHOLDER_RE = re.compile(r"^-\s+\*\*待校订[。.]?\*\*", re.M)


def is_unedited(md: str) -> bool:
    """核心观点是否仍是未校订占位稿。"""
    kp = _section(md, "核心观点")
    return bool(kp and _PLACEHOLDER_RE.search(kp[2]))


def _top_and_sub(body: str) -> tuple[list[str], list[str]]:
    """拆出小节的顶级条目正文与缩进子条目。"""
    lines = body.splitlines()
    top = [l for l in lines if _TOP_ITEM_RE.match(l)]
    sub = [l for l in lines if _SUB_ITEM_RE.match(l)]
    return top, sub


def classify_terms(md: str) -> list[tuple[str, str, int | None]]:
    """三态判定。

    返回 ``[(术语表原始行, 态, 目标条目序号)]``；态为
    ``"dup"``（已内联，术语表该条应删）/ ``"inline"``（可内联，给首次出现的
    条目序号）/ ``"keep"``（保留）。序号相对「核心观点 + 问题与思考」的
    顶级条目合并序列，核心观点在前。
    """
    if is_unedited(md):
        return []
    gl = _section(md, "术语表")
    if not gl:
        return []
    items = [l for l in gl[2].splitlines() if l.strip().startswith("- ")]
    if not items:
        return []

    kp = _section(md, "核心观点")
    qs = _section(md, "问题与思考")
    tops: list[str] = []
    subs: list[str] = []
    for s in (kp, qs):
        if s:
            t, sb = _top_and_sub(s[2])
            tops.extend(t)
            subs.extend(sb)
    # 只取子条目的术语名做比对（去掉加粗内末尾的句号）
    sub_names: set[str] = set()
    for l in subs:
        m = _SUB_TERM_RE.match(l)
        if m:
            sub_names.add(re.sub(r"[。.]\s*$", "", m.group(1).strip()))

    out: list[tuple[str, str, int | None]] = []
    for item in items:
        m = _TERM_ITEM_RE.search(item)
        if not m:
            out.append((item, "keep", None))
            continue
        term = m.group(1).strip()
        # ① 已内联：子条目里已有同名术语 → 术语表这条是重复
        if any(v in sub_names for v in term_variants(term)):
            out.append((item, "dup", None))
            continue
        # ② 可内联：术语名出现在某条顶级条目正文里 → 取首次出现那条
        hit: int | None = None
        for v in term_variants(term):
            for i, l in enumerate(tops):
                if v in l:
                    hit = i
                    break
            if hit is not None:
                break
        out.append((item, "inline" if hit is not None else "keep", hit))
    return out


def describe_target(md: str, idx: int) -> str:
    """把合并序号翻译成人话（核心观点第 N 条 / 问题与思考第 N 条）。"""
    kp = _section(md, "核心观点")
    n = len([l for l in kp[2].splitlines() if _TOP_ITEM_RE.match(l)]) if kp else 0
    if idx < n:
        return f"核心观点第 {idx + 1} 条"
    return f"问题与思考第 {idx - n + 1} 条"


def _as_sub_item(term_line: str) -> str:
    """术语表条目 → 观点下的缩进子条目形态（与渲染器一致：**术语。** 解释）。"""
    m = _TERM_ITEM_RE.search(term_line)
    if not m:
        return term_line
    term = m.group(1).strip()
    desc = term_line[m.end():].strip()
    return f"  - **{term}。** {desc}"


def inline_glossary_terms(md: str) -> tuple[str, int, int]:
    """确定性修复：删除重复的术语表条目、把可内联的挂到首次出现的观点下。

    返回 ``(新文本, 删除条数, 内联条数)``。幂等：无该缺陷时原样返回。
    """
    verdict = classify_terms(md)
    if not verdict:
        return md, 0, 0
    dup = [(line, tgt) for line, kind, tgt in verdict if kind == "dup"]
    inl = [(line, tgt) for line, kind, tgt in verdict if kind == "inline"]
    if not dup and not inl:
        return md, 0, 0

    # ---- ① 先内联：把术语挂到目标顶级条目下（该条目最后一个子条目之后）
    if inl:
        lines = md.splitlines()
        # 定位「核心观点 + 问题与思考」的顶级条目在全文中的行号
        tops_pos: list[int] = []
        in_sec = False
        for i, l in enumerate(lines):
            if re.match(r"^##\s+", l):
                in_sec = bool(re.match(r"^##\s+(核心观点|问题与思考)\s*$", l))
                continue
            if in_sec and _TOP_ITEM_RE.match(l):
                tops_pos.append(i)
        # 倒序插入，避免插入打乱后续行号
        for line, tgt in sorted(inl, key=lambda x: x[1] or 0, reverse=True):
            if tgt is None or tgt >= len(tops_pos):
                continue
            start = tops_pos[tgt]
            end = tops_pos[tgt + 1] if tgt + 1 < len(tops_pos) else None
            if end is None:
                # 最后一条：插到小节末尾前——找下一个 ## 标题或文末
                j = start + 1
                while j < len(lines) and not re.match(r"^##\s+", lines[j]):
                    j += 1
                end = j
            # 插到该条目的最后一个缩进子条目之后；无子条目则紧随条目行
            ins = start + 1
            while ins < end and (_SUB_ITEM_RE.match(lines[ins]) or not lines[ins].strip()):
                ins += 1
            lines.insert(ins, _as_sub_item(line))
        md = "\n".join(lines)

    # ---- ② 再删术语表里的重复/已内联条目
    gl = _section(md, "术语表")
    if gl:
        body = gl[2]
        drop = {line for line, _ in dup} | {line for line, _ in inl}
        kept: list[str] = []
        for l in body.splitlines():
            if l in drop:
                continue
            kept.append(l)
        new_body = "\n".join(kept)
        # 术语表被清空 → 整节删除（v26：无残留硬术语时不写该节）
        if not [l for l in kept if l.strip().startswith("- ")]:
            md = md[: gl[0]] + md[gl[1]:]
            md = re.sub(r"\n{3,}", "\n\n", md)
        else:
            md = md[: gl[0]] + "## 术语表\n" + new_body.lstrip("\n") + md[gl[1]:]
    return md, len(dup), len(inl)


def fix_glossary_position(md: str) -> str:
    """术语表须紧跟核心观点、位于问题与思考之前（v23 起的站位要求）。

    存量稿里术语表普遍被排到了问题与思考之后，属纯结构错位，
    整块搬移即可、不动任何文字。
    """
    gl = _section(md, "术语表")
    if not gl:
        return md
    qs = re.search(r"^## 问题与思考\s*$", md, re.M)
    if not qs or gl[0] < qs.start():
        return md                       # 没有「问题与思考」或位置本就正确
    block = md[gl[0]: gl[1]]
    md2 = md[: gl[0]] + md[gl[1]:]
    qs2 = re.search(r"^## 问题与思考\s*$", md2, re.M)
    if not qs2:                         # 理论不可达，保险起见
        return md
    md2 = md2[: qs2.start()] + block + md2[qs2.start():]
    return re.sub(r"\n{3,}", "\n\n", md2)


def drop_empty_glossary(md: str) -> str:
    """术语表没有有效条目时整节删除。

    存量稿里「无术语可写」被写成了占位条目 `- （无）`，既不符合
    `**术语**：解释` 格式、也不以句号结尾，会同时触发两条闸门报错。
    规则是「不存在这类词时术语表整节不写」，所以正确写法是删整节、不是留占位。
    """
    if is_unedited(md):
        return md                       # 未校订稿的占位条目不动
    gl = _section(md, "术语表")
    if not gl:
        return md
    valid = [
        l for l in gl[2].splitlines()
        if l.strip().startswith("- ") and _TERM_ITEM_RE.search(l)
    ]
    if valid:
        return md
    md2 = md[: gl[0]] + md[gl[1]:]
    return re.sub(r"\n{3,}", "\n\n", md2)


def count_glossary_issues(md: str) -> int:
    """待修条数（dup + inline），供校验闸门与 CLI 报告使用。"""
    return sum(1 for _, kind, _ in classify_terms(md) if kind in ("dup", "inline"))
