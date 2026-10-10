"""Text normalization helpers for transcript post-processing.

These enforce project house-style rules that should apply regardless of whether
the LLM post-processing step is used (e.g. when running with --llm-mode session and doing
manual in-session editing).

除 ``normalize_quotes``（转录落盘时调用）外，本模块还承载两条**纯机械**的成稿级
规范：``normalize_en_punct``（外文词后的半角标点转全角）与 ``merge_repeat_chars``
（≥3 连同字合并，拟声与应答叠用除外）。二者合为 ``normalize_markdown``，供：

    python -m src.processor.normalize "output/xxx.md"          # 干跑，只报改动数
    python -m src.processor.normalize "output/xxx.md" --apply  # 落盘（自动 .bak）

**为什么要有这个入口**（2026-09-15 连漪拍板，源自公开仓 issue #1 / #2）：这两条
规则此前只写在提示词层（``rules.EDIT_EN_PUNCT``）、或散落在每集复制的
``scripts/oneoff/proofread_*.py`` 里。实测结果——规则随脚本复制而漂移：带
``en_punct`` 的 118 / Vol.200 集残留 0 处，未带的 Vol.1 残留 91 处、#651 残留 74
处。同一批规则必须只实现一次、由代码确定性执行，不依赖 LLM 在长文里自觉守约，
也不靠每集脚本各自记得带上。新集校订的确定性清洗一律调用本模块，不在 oneoff
脚本里重复实现。
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Opening / closing Chinese corner brackets (直角双引号) 「」
CORNER_OPEN = "\u300c"   # 「
CORNER_CLOSE = "\u300d"  # 」

# ASCII straight double quote and common curly variants
_STRAIGHT = '"'
_CURLY_OPEN = "\u201c"   # "
_CURLY_CLOSE = "\u201d"  # "


def normalize_quotes(text: str) -> str:
    """Convert double quotes to Chinese corner brackets 「」.

    - ASCII straight quotes ``"..."`` become ``「...」``.
    - Curly double quotes ``"..."`` are first normalized to straight, then paired.
    - Quotes are paired sequentially: the first opens with ``「``, the next closes
      with ``」``, and so on.

    Only *paired* quotes are converted. A dangling (unpaired) straight quote is
    left as-is instead of being mis-paired, so we never emit a bare ``「`` or an
    odd mismatch across English/nested content.

    Per project house style, double quotes anywhere in the Chinese transcript use
    corner brackets; only genuinely English-native quotes (e.g. inside code/URLs)
    are expected to be handled by the LLM prompt, but in practice the transcription
    is Chinese so blanket conversion is safe and intended.
    """
    if not text:
        return text

    # Normalize curly variants to straight first for uniform pairing
    normalized = text.replace(_CURLY_OPEN, _STRAIGHT).replace(_CURLY_CLOSE, _STRAIGHT)

    out: list[str] = []
    i = 0
    n = len(normalized)
    while i < n:
        ch = normalized[i]
        if ch == _STRAIGHT:
            # Find the matching close quote; if missing, keep the raw quote.
            j = normalized.find(_STRAIGHT, i + 1)
            if j != -1:
                out.append(CORNER_OPEN)
                out.append(normalized[i + 1:j])
                out.append(CORNER_CLOSE)
                i = j + 1
            else:
                out.append(ch)
                i += 1
        else:
            out.append(ch)
            i += 1

    return "".join(out)


# ============================================================
# 外文词后的半角标点 → 中文全角（rules.EDIT_EN_PUNCT 的确定性落地）
# ============================================================

_EN_PUNCT_MAP = {",": "，", ".": "。", "?": "？", "!": "！"}

# 判据四层，宁可少转也不错转（错了要人工回头改，漏了下次跑还会补上）：
#   1. 标点后不紧跟字母或数字，也不接「空格 + 拉丁字母」——排除英文词内标点
#      （`Node.js` `v1.5`）与英文句内标点（`Talk is cheap, show me the code`，
#      这类整句引用常夹在含中文的行里，单靠「行含中文」拦不住）；
#   2. 标点所在的**整行含中文**才转——中文语境里被 ASR 落成半角的情形
#      （`gap,那个`、`token, 它并不会`、行尾的 `尽可能多的 Token.`）都命中；
#      纯英文行（`really cheeky knowledge.`）整行保持原样；
#   3. 已知拉丁缩写（下方 _ABBREV）单独保护，其标点不转（`e.g. 这种` 不动）；
#   4. Show Notes 小节整节跳过（节目源简介透传，不归校订管）。
#
# 三组捕获：字母 + 其后的加粗符 + 标点。**加粗符必须单列一组**——回标加粗常把
# 标点一起包进 `**`（实测 `那个 **Genspark**,他`），此时标点前的字符是 `*` 而非
# 字母，用 `(?<=[A-Za-z])` 会整片漏掉（实测 13 份稿共 38 处）。Python 的
# lookbehind 要求定长，故不能写成 `(?<=[A-Za-z]\*{0,2})`，改用捕获组收回。
#
# 曾试过「标点前至少两个字母」的判据，实测漏掉最常见的一类——ASR 把英文词切成
# 单字母（`c e o.` `p p t,`），故改为缩写白名单反向保护。
_EN_PUNCT_RE = re.compile(r"([A-Za-z])(\*{0,2})([,.?!])(?![A-Za-z0-9])(?!\s+[A-Za-z])")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 拉丁缩写：其句点属缩写本身，不能转成中文句号（转了就成 `e。g。`）。
_ABBREV = (
    "e.g.", "i.e.", "etc.", "vs.", "cf.", "et al.", "Mr.", "Mrs.", "Ms.",
    "Dr.", "St.", "No.", "U.S.", "U.K.", "Ph.D.", "a.m.", "p.m.",
    "Inc.", "Ltd.", "Co.", "Jr.", "Sr.",
)


def _is_abbrev(text: str, pos: int) -> bool:
    """标点 ``pos`` 是否属于某个已知拉丁缩写（取标点前的短窗口比对）。"""
    head = text[max(0, pos - 8): pos + 1].strip()
    return any(head.endswith(a) for a in _ABBREV)


def _line_has_cjk(text: str, pos: int) -> bool:
    """标点 ``pos`` 所在的整行是否含中文（决定该标点属中文还是英文体系）。"""
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    if end < 0:
        end = len(text)
    return bool(_CJK_RE.search(text[start:end]))


def _convertible(text: str, m: re.Match[str]) -> bool:
    """该字符位置上的半角标点是否应转全角（判据合一，供转换与计数复用）。"""
    pos = m.start(3)
    return not _is_abbrev(text, pos) and _line_has_cjk(text, pos)


def normalize_en_punct(text: str) -> str:
    """外文词/字母词后的半角标点转为中文全角（。，？！）。

    仅转换中文语境里的标点：标点后不紧跟字母/数字、所在行含中文、且不属已知
    拉丁缩写。英文词内标点（Node.js / v1.5）与纯英文句（lead on taste, ...）
    保持原样。标点前若夹着回标加粗符（`**Genspark**,`），一并识别并保留。
    """
    if not text:
        return text

    def _sub(m: re.Match[str]) -> str:
        ch = _EN_PUNCT_MAP[m.group(3)] if _convertible(text, m) else m.group(3)
        return m.group(1) + m.group(2) + ch

    return _EN_PUNCT_RE.sub(_sub, text)


# ============================================================
# ≥3 连相同汉字合并为一个（rules.EDIT_REDO 的确定性部分）
# ============================================================

# 拟声词与笑声：叠用本身是正确写法，合并会毁掉语义（哈哈哈 → 哈）。
_ONOMATOPOEIA = set(
    "哈呵嘻嘿噗哼嘎咚隆哗吱嗡轰呼啦咕嘟喵汪叮铛嘀嗒噼啪"
    "淅沥潺汩呜哇唧啾嗷嚎咩哞嘭砰唰飕簌啵"
)

# 应答与语气叠用：三连承担语气强度（对对对＝强烈认同），删除会改变含义。
# 这类不由机械脚本判，留给语义层（会话链路的 LLM）按上下文取舍。
_INTERJECTION = set("对是好行中嗯哦啊哎唉噢唔哟咧")

_KEEP_REPEAT = _ONOMATOPOEIA | _INTERJECTION

# 同一汉字连续 ≥3 次（含 3 次）；2 连不匹配——中文大量正常叠词（刚刚、看看、
# 慢慢）都是 2 连，合并必误伤。
_REPEAT3_RE = re.compile(r"([\u4e00-\u9fff])\1{2,}")


def merge_repeat_chars(text: str) -> str:
    """把连续 ≥3 个相同汉字合并为一个（口吃/ASR 切碎），白名单内保留原样。

    白名单＝拟声笑声 + 应答语气叠用（见 ``_KEEP_REPEAT``）。2 连一律不动。
    """
    if not text:
        return text

    def _sub(m: re.Match[str]) -> str:
        ch = m.group(1)
        return m.group(0) if ch in _KEEP_REPEAT else ch

    return _REPEAT3_RE.sub(_sub, text)


# ============================================================
# 中文读数日期 → 阿拉伯数字（2026年9月28日）
# ============================================================

# ASR 的逆文本正则化（ITN）常把日期落成中文读数，还会在年与月之间插口语逗号
# （实测 `二零二六年，九月二十八日星期一发布`）。成稿里保留中文读数既难扫读、
# 也不利于后续检索与引用，统一落成阿拉伯数字。
#
# ============================================================

# 只提年份、不带月日的情形（`二零二四年那篇生成式 AI 通往未来的桥`）。负向断言
# 让它让位给上面的完整日期规则，避免转出 `2024年五月` 这种年数字其余中文的混血写法。
_CN_YEAR_RE = re.compile(
    r"([零一二三四五六七八九]{4})年(?![，,]?\s*[一二三四五六七八九十]{1,3}月)"
)

# 只认「年+月+日」完整结构——孤立的中文数字（「一是……二是……」）不会被误伤；
# 年份限定为四字逐位读数（二零二六），口语的「两千零二十六年」暂不支持。
_CN_DATE_RE = re.compile(
    r"([零一二三四五六七八九]{4})年[，,]?\s*"
    r"([一二三四五六七八九十]{1,3})月[，,]?\s*"
    r"([一二三四五六七八九十]{1,3})日"
)
_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "两": 2,
}


def _cn_num_to_int(s: str) -> int | None:
    """中文数字 → int。支持逐位读数（二零二六→2026）与十进位（十二/二十八/九）。"""
    if not s:
        return None
    if all(ch in _CN_DIGITS for ch in s):
        return int("".join(str(_CN_DIGITS[ch]) for ch in s))
    if "十" in s and len(s) <= 3:
        left, _, right = s.partition("十")
        tens = 1 if left == "" else _CN_DIGITS.get(left)
        ones = 0 if right == "" else _CN_DIGITS.get(right)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return None


def normalize_cn_date(text: str) -> str:
    """``二零二六年，九月二十八日`` → ``2026年9月28日``。

    月份/日期越界（如听成「十三月」）或年份不在合理区间时**一律原样保留**——
    宁可漏转也不错转，错误的外形比难读更难发现。
    """
    if not text:
        return text

    def _sub(m: re.Match[str]) -> str:
        year, month, day = (
            _cn_num_to_int(m.group(1)),
            _cn_num_to_int(m.group(2)),
            _cn_num_to_int(m.group(3)),
        )
        if year is None or month is None or day is None:
            return m.group(0)
        if not (1000 <= year <= 2999 and 1 <= month <= 12 and 1 <= day <= 31):
            return m.group(0)
        return f"{year}年{month}月{day}日"

    def _sub_year(m: re.Match[str]) -> str:
        year = _cn_num_to_int(m.group(1))
        if year is None or not (1000 <= year <= 2999):
            return m.group(0)
        return f"{year}年"

    # 先转完整日期，再转孤立年份——第二步自然跳过已被转成阿拉伯数字的部分
    return _CN_YEAR_RE.sub(_sub_year, _CN_DATE_RE.sub(_sub, text))


def _iter_date_hits(text: str):
    """产出可转换的日期匹配：先完整日期，再孤立年份（负向断言排除已吃掉的部分）。"""
    consumed: list[tuple[int, int]] = []
    for m in _CN_DATE_RE.finditer(text or ""):
        year, month, day = (
            _cn_num_to_int(m.group(1)),
            _cn_num_to_int(m.group(2)),
            _cn_num_to_int(m.group(3)),
        )
        if (
            year is not None
            and month is not None
            and day is not None
            and 1000 <= year <= 2999
            and 1 <= month <= 12
            and 1 <= day <= 31
        ):
            consumed.append(m.span())
            yield m
    for m in _CN_YEAR_RE.finditer(text or ""):
        if any(m.start() >= s and m.end() <= e for s, e in consumed):
            continue
        year = _cn_num_to_int(m.group(1))
        if year is not None and 1000 <= year <= 2999:
            yield m


def count_cn_date(text: str) -> int:
    """中文读数日期/年份的可转换处数（越界不算），供干跑预览用。"""
    return sum(1 for _ in _iter_date_hits(text))


# ============================================================
# 成稿级入口：跳过 Show Notes（节目源简介透传，不归校订管）
# ============================================================

_SECTION_RE = re.compile(r"^(##\s+.+?)\s*$", re.MULTILINE)
_SKIP_SECTIONS = ("## Show Notes",)


def _clean_block(block: str) -> str:
    return merge_repeat_chars(normalize_cn_date(normalize_en_punct(block)))


def normalize_markdown(md: str) -> str:
    """对成稿跑两条确定性规范，``## Show Notes`` 小节原样透传。"""
    if not md:
        return md
    marks = list(_SECTION_RE.finditer(md))
    if not marks:
        return _clean_block(md)

    out: list[str] = [_clean_block(md[: marks[0].start()])]
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(md)
        head, body = m.group(1), md[m.end(): end]
        out.append(m.group(0))
        out.append(body if head.startswith(_SKIP_SECTIONS) else _clean_block(body))
    return "".join(out)


def count_text(text: str) -> tuple[int, int]:
    """返回单块文本的 (半角待转数, ≥3 连待合并数)。供校验器复核转录区用。"""
    if not text:
        return 0, 0
    en = sum(1 for m in _EN_PUNCT_RE.finditer(text) if _convertible(text, m))
    rp = sum(1 for m in _REPEAT3_RE.finditer(text) if m.group(1) not in _KEEP_REPEAT)
    return en, rp


def count_changes(md: str) -> tuple[int, int]:
    """返回成稿 (半角标点待转数, ≥3 连待合并数)，``## Show Notes`` 不计入。"""
    if not md:
        return 0, 0
    marks = list(_SECTION_RE.finditer(md))
    hits_en = hits_rp = 0
    blocks = [md[: marks[0].start()]] if marks else [md]
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(md)
        head, body = m.group(1), md[m.end(): end]
        if not head.startswith(_SKIP_SECTIONS):
            blocks.append(body)
    for b in blocks:
        en, rp = count_text(b)
        hits_en += en
        hits_rp += rp
    return hits_en, hits_rp


def count_all(md: str) -> dict[str, int]:
    """返回三类待改数量 ``{en_punct, repeat, cn_date}``，``## Show Notes`` 不计入。

    比 ``count_changes`` 多了日期项；单独开一个新函数而非改旧签名，避免牵动
    ``finish_phase_b`` / ``output_quality`` 等既有调用方。
    """
    if not md:
        return {"en_punct": 0, "repeat": 0, "cn_date": 0}
    marks = list(_SECTION_RE.finditer(md))
    blocks = [md[: marks[0].start()]] if marks else [md]
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(md)
        head, body = m.group(1), md[m.end(): end]
        if not head.startswith(_SKIP_SECTIONS):
            blocks.append(body)
    out = {"en_punct": 0, "repeat": 0, "cn_date": 0}
    for b in blocks:
        en, rp = count_text(b)
        out["en_punct"] += en
        out["repeat"] += rp
        out["cn_date"] += count_cn_date(b)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="成稿确定性规范化（外文词后半角标点转全角 + ≥3 连重复字合并 + 中文读数日期转阿拉伯数字，不调 LLM）"
    )
    ap.add_argument("md_path", type=Path, help="成稿 .md 路径")
    ap.add_argument("--apply", action="store_true", help="写回文件（自动 .bak 备份）")
    args = ap.parse_args()

    if not args.md_path.exists():
        print(f"错误：找不到文件 {args.md_path}", file=sys.stderr)
        return 1

    md = args.md_path.read_text(encoding="utf-8")
    new_md = normalize_markdown(md)
    counts = count_all(md)
    hits_en, hits_rp, hits_date = counts["en_punct"], counts["repeat"], counts["cn_date"]

    print(f"文件: {args.md_path.name}")
    print(f"待转半角标点: {hits_en} 处")
    print(f"待合并 ≥3 连重复字: {hits_rp} 处")
    print(f"待转中文读数日期: {hits_date} 处")
    if hits_en == 0 and hits_rp == 0 and hits_date == 0:
        print("\n无需改动。")
        return 0
    if not args.apply:
        print("\n（干跑，未落盘。加 --apply 生效）")
        return 0

    bak = args.md_path.with_suffix(args.md_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(args.md_path, bak)
    args.md_path.write_text(new_md, encoding="utf-8")
    print(f"\n已写回 {args.md_path.name}（备份 {bak.name}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
