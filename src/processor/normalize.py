"""Text normalization helpers for transcript post-processing.

These enforce project house-style rules that should apply regardless of whether
the LLM post-processing step is used (e.g. when running with --llm-mode session and doing
manual in-session editing).

除 ``normalize_quotes``（转录落盘时调用）外，本模块还承载四条**纯机械**的成稿级
规范：``normalize_en_punct``（外文词后的半角标点转全角）、``merge_repeat_chars``
（≥3 连同字合并，拟声与应答叠用除外）、``normalize_cn_date``（中文读数日期转
阿拉伯数字）与 ``normalize_quote_period``（核心观点句末引用原话补句末标点）。
四者合为 ``normalize_markdown``，供：

    python -m src.processor.normalize "output/xxx.md"          # 干跑，只报改动数
    python -m src.processor.normalize "output/xxx.md" --apply  # 落盘（自动 .bak）

**为什么要有这个入口**（2026-09-15 连漪拍板，源自公开仓 issue #1 / #2）：这两条
规则此前只写在提示词层（``rules.EDIT_EN_PUNCT``）、或散落在每集复制的
``scripts/oneoff/proofread_*.py`` 里。实测结果——规则随脚本复制而漂移：带
``en_punct`` 的 118 / Vol.200 集残留 0 处，未带的 Vol.1 残留 91 处、#651 残留 74
处。同一批规则必须只实现一次、由代码确定性执行，不依赖 LLM 在长文里自觉守约，
也不靠每集脚本各自记得带上。新集校订的确定性清洗一律调用本模块，不在 oneoff
脚本里重复实现。

``--apply`` 会顺带同步同名 ``_handoff.json`` 的 ``key_points[].quote``（``anchor``
不动）——核心观点是从 handoff 重渲染的，只改 md 会在下次重跑时被覆盖回去。
"""

from __future__ import annotations

import argparse
import json
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

# 汉字字元类（供下方标点规则拼字符组用；单个汉字的判断用已编译的 _CJK_RE）
_CJK_CHAR = "\u4e00-\u9fff"

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
# 标点清理（2026-10-11 定）：ASR 停顿与换行落成的空格 / 冗余 / 错位标点
# ============================================================
# 由来：全量 63 份成稿审计（排除 tmp_/lark-doc 临时件）暴露四类**可机械判定**的缺陷，
# 此前只能逐集手改。四类各自的标准写法写在各条规则上方。
#
# 共同原则：**只删除不该存在的空格与重复标点，从不新增标点、不改写措辞**——
# 与数量转换同守「宁可漏改也不错改」，错改会把原话改成别的意思。

# 全角停顿/句末标点，参与空格与堆叠收敛的那一组
_PUNCT_CJK = "，。！？、；："
_PUNCT_PAUSE = "，、；："          # 停顿类（句末前多余、句末后多余都要删）
_PUNCT_END = "。！？"             # 句末类（重复只留一个）

# ------------------------------------------------------------
# ① 停顿空格：ASR 在停顿处先留空格、后补全角标点，空格就夹在了标点两侧
# ------------------------------------------------------------
# 中文全角标点自带字身尾空，**其后不得再加空格**（中文—中文：必删）。
# 中文—西文之间是否留空格各家不一，按库内压倒性惯例取「不留」：实测
# 「标点后无空格 + 西文」1970 处 vs「标点后带空格 + 西文」132 处（93.7% 无空格）。
# 标点前出现空格（`种种困惑 。`、`3 分钟 ：`）同样是错的，一并删。
# 但有三类上下文必须放过（2026-10-11 全量干跑实测踩到的坑）：
#   a. 来源行的 ` | ` 字段分隔符（`判断力？ | 2026-08-31`）、Markdown 表格剩余留白；
#   b. 西文人名缩写（`R. F. Kuang` 被上一环 en_punct 转成 `R。 F。 Kuang` 后，
#      句点间的空格若再删就成了 `R。F。Kuang`，可读性反而变差）；
#   c. 英文句子/歌词整句被转成全角标点后的空格（`kid, you always have been` →
#      `kid， you always...`，此时空格是英文书写所必须）。
# 故拆成两条、判据更严：**后接汉字才一律删**；后接西文或数字仅在**标点前也是汉字**
# 时才删（西文环境的空格一律保留）。
_SPACE_AFTER_CJK_RE = re.compile(rf"([{_PUNCT_CJK}]) +(?=[{_CJK_CHAR}])")
_SPACE_AFTER_LATIN_RE = re.compile(rf"(?<=[{_CJK_CHAR}])([{_PUNCT_CJK}]) +(?=[A-Za-z0-9])")
_SPACE_BEFORE_RE = re.compile(rf"(?<=[{_CJK_CHAR}]) +(?=[{_PUNCT_CJK}])")

# ------------------------------------------------------------
# ② 堆叠标点：ASR 多标点连打 / 换行处前后标点叠加
# ------------------------------------------------------------
# 收敛方向一律**取首个标点**——前一个停顿决定了此处是什么性质的停顿：
#   「包括愤怒、，悲伤」保留「、」→「包括愤怒、悲伤」；「但是当前，，因为」留「，」。
# P2/P3 处理句末标点旁边的冗余停顿（「涉及面。，你肯定」「水电煤工人，。建筑工人」）；
# P4 处理句末标点连打（「nba。。」）。**异形组合不动**——「？！」「！？」是合法语气叠加。
_CLUSTER_STEPS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"[{_PUNCT_PAUSE}]+(?=[{_PUNCT_END}])"), ""),      # P2 句末标点前
    (re.compile(rf"(?<=[{_PUNCT_END}])[{_PUNCT_PAUSE}]+"), ""),     # P3 句末标点后
    (re.compile(r"([，、])[，、]+"), r"\1"),                          # P1 相邻同类停顿
    (re.compile(r"([。！？])\1+"), r"\1"),                           # P4 句末标点堆叠
)

# ------------------------------------------------------------
# ③ 中文语境里的半角残留
# ------------------------------------------------------------
# normalize_en_punct 只管「字母后」的半角标点（`Node.js` `v1.5` `Talk is cheap,`）；
# **汉字后**紧跟的半角（`循环在跑,并且`、`发推特, 他们`、`游说说:塞利米耶`、`聆听!`）
# 落在它的判据之外，实测 63 份稿共 12 处靠人工才发现。本规则补这一段。
#
# 判据（三层，一层比一层严，避免把西文整句误改成中文）：
#   1. 标点前一个字符必须是汉字；
#   2. 标点后第一个非空格字符必须是**汉字**；若已到行尾，则要求整行以汉字为主体
#      （该行汉字数 ≥ 拉丁字母数）——「何找到志同道合的搭子?」过，
#      「Special Guests: Kolento, 林桐宇, and 饶杰武.」不过（西文列举句，整句保持原样）；
#   3. 行尾的情形还要看行为主线，如上。
# 英文列举（`Kolento, 林桐宇, and`）、时间/版本号（`12:30` `v1.5`）因此都不会被碰。
_CJK_HALF_MAP = {",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "："}
_CJK_HALF_RE = re.compile(rf"([{_CJK_CHAR}])( *[,.?!;:])")

# 三点/六点省略号单独收：中文语境应作「……」（GB/T 15834 用两个省略号字符）。
# 要求标点前是汉字、其后不紧跟字母数字（`v1...` `wait...` 之类留给西文）。
_CJK_ELLIPSIS_RE = re.compile(rf"(?<=[{_CJK_CHAR}])\.{{3,}}(?![A-Za-z0-9])")

# ④ 句首错位标点：换行把标点甩到了下一行行首（`\n\n，包括其实线下`、`\n。相比之下`）
_LEAD_PAUSE_RE = re.compile(r"^([ \t>]*)[，、；：]+")


def _line_cjk_dominant(text: str, pos: int) -> bool:
    """``pos`` 所在整行是否以汉字为主体（汉字数 ≥ 拉丁字母数）。

    给「标点后已到行尾」的半角标点兜第 2 层判据：只有中文主体的行，
    其行末半角标点才是 ASR 漏网的中文标点。
    """
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    if end < 0:
        end = len(text)
    line = text[start:end]
    n_cjk = len(_CJK_RE.findall(line))
    n_lat = len(re.findall(r"[A-Za-z]", line))
    return n_cjk >= n_lat


def _cjk_half_ok(text: str, pos: int) -> bool:
    """``pos`` 处的半角标点是否处在中文语境里（判据合一，转换与计数共用）。"""
    i = pos + 1
    n = len(text)
    while i < n and text[i] in " \t":
        i += 1
    if i < n and _CJK_RE.match(text[i]):
        return True
    if i >= n or text[i] == "\n":
        return _line_cjk_dominant(text, pos)
    return False

def normalize_punct_space(text: str) -> str:
    """① 清理紧贴全角标点的多余空格（ASR 停顿空格残留）。

    「量化取样， 然后」→「量化取样，然后」；「种种困惑 。」→「种种困惑。」。
    中文—中文之间必删；中文标点—西文之间按库内 93.7% 的无空格惯例也删。
    """
    if not text:
        return text
    text = _SPACE_AFTER_CJK_RE.sub(r"\1", text)
    text = _SPACE_AFTER_LATIN_RE.sub(r"\1", text)
    return _SPACE_BEFORE_RE.sub("", text)


def normalize_punct_cluster(text: str) -> str:
    """② 收敛堆叠/重复的标点（ASR 多标点连打、换行处前后标点叠加）。"""
    if not text:
        return text
    for pat, repl in _CLUSTER_STEPS:
        text = pat.sub(repl, text)
    return text


def normalize_cjk_halfwidth(text: str) -> str:
    """③ 汉字后紧跟的半角标点转全角（``normalize_en_punct`` 漏网的那一半）。"""
    if not text:
        return text
    text = _CJK_ELLIPSIS_RE.sub("……", text)

    def _sub(m: re.Match[str]) -> str:
        pos = m.start(2) + len(m.group(2)) - 1
        if not _cjk_half_ok(text, pos):
            return m.group(0)
        return m.group(1) + " " * (len(m.group(2)) - 1) + _CJK_HALF_MAP[m.group(2)[-1]]

    return _CJK_HALF_RE.sub(_sub, text)


def normalize_orphan_punct(block: str) -> str:
    """④ 句首错位标点：换行把标点甩到下一行行首时收回。

    两种形态，处置不同：
      - 行首是停顿标点（`，包括其实线下`、`> ，以及`）——段落绝无可能以逗号起头，
        直接删（`->` 引用标记与缩进保留）；
      - 行首是句末标点（`\n。相比之下`、`\n。值得欣慰的是`）——换行是软换行，
        与前一行合并，让句号回到前一句末尾。前一行已是完整句（以句末标点收尾）
        或是列表项、标题时不合并，交给人工看。
    """
    if not block:
        return block
    out: list[str] = []
    for line in block.split("\n"):
        # 形态一：行首停顿标点 → 直接删（保留缩进与 > 引用标记）
        line = _LEAD_PAUSE_RE.sub(r"\1", line)
        # 判空必须用显式长度判断：`"" in "。！？"` 恒为 True（空串是任何串的子串），
        # 会让末尾空行被误判成「以句末标点开头」而与前一行合并，吃掉文末换行。
        lead = line.lstrip()
        if lead and lead[0] in _PUNCT_END:
            prev = out[-1].rstrip() if out else ""
            joinable = (
                prev
                and not prev[-1:] in _PUNCT_END
                and prev[-1:] not in "…”）」】"
                and not re.match(r"^\s*([-*>#|]|\d+[.)、])", prev)
            )
            if joinable:
                out[-1] = prev + lead
                continue
        out.append(line)
    return "\n".join(out)


def normalize_punct(text: str) -> str:
    """四类标点缺陷一并清（①→④ 的执行顺序写在函数体内）。

    顺序有讲究：先做行结构上的句首错位收回（否则换行会把标点拆到两行、后续
    规则看不见），再补汉字后半角（在此之前正常化，好让全角标点随后参与聚类），
    最后是聚类与空格清理（收尾清理前面几步留下的新例子）。
    """
    if not text:
        return text
    text = normalize_orphan_punct(text)
    text = normalize_cjk_halfwidth(text)
    text = normalize_punct_cluster(text)
    return normalize_punct_space(text)


def count_punct(text: str) -> int:
    """待清标点缺陷数（四类合计）。转换与计数共用同一组 pattern，避免口径漂移。"""
    if not text:
        return 0
    n = (
        len(_SPACE_AFTER_CJK_RE.findall(text))
        + len(_SPACE_AFTER_LATIN_RE.findall(text))
        + len(_SPACE_BEFORE_RE.findall(text))
    )
    n += len(_CJK_ELLIPSIS_RE.findall(text))
    n += sum(len(p.findall(text)) for p, _r in _CLUSTER_STEPS)
    n += sum(1 for m in _CJK_HALF_RE.finditer(text) if _cjk_half_ok(text, m.start(2) + len(m.group(2)) - 1))
    for line in text.split("\n"):
        if _LEAD_PAUSE_RE.match(line):
            n += 1
    return n


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
# 中文读数数量 → 阿拉伯数字（十六家→16家、十二亿用户→12亿用户）
# ============================================================

# ASR 常把数量落成中文读数，难扫读、不利检索。连漪 2026-10-11 拍板：数量尽量用
# 阿拉伯数字，一眼就能锁定那个数。与日期规则的区别：日期只认「年+月+日/年」结构，
# 本条覆盖一切「中文数字 + 量词/名词」的数量（十六家、十二亿用户、三百六十五、两万…）。
#
# 防误伤（宁可漏转也不错转）：
#   - 只转含进位字（十/百/千/万/亿）的读数——所有精确中文数量 ≥10 都含进位字；
#     而概数「两三」「七八」「三四」「三五」只由两个单字凑成、不含进位字，自然跳过；
#   - 概数前缀（数/近/约/上/成/超）后的读数也跳过（「数百万」「近百」保留中文）；
#   - 后接 月/日/号/年 的读数交给日期规则（「十三月」「十二月」不转，避免覆盖日期守卫）；
#   - 后接非中文（标点/换行/结尾）的读数不转——「他是个二百五。」不转；
#   - 单字读数（一/三/六…）本就不含进位字，已被首条规则兜住；
#   - 「二百五」既含进位字又确为成语，列入黑名单整体跳过。
_CN_RUN_RE = re.compile(r"[零一二三四五六七八九十百千万亿两]+")
_CN_SCALE = frozenset("十百千万亿")          # 进位字：精确数量 ≥10 必含其一
_CN_APPROX_PREFIX = frozenset("数近约上成超余")  # 概数前缀：其后读数不转
_BLOCKLIST_Q = ("万一", "千万", "二百五")     # 两字非数量词（倘若/务必）+ 含进位字的成语
_Q_SCALE = ("万", "亿")


def _cn_quantity_to_int(s: str) -> int | None:
    """完整中文数字 → int（支持 十百千万亿 进位，如 十二亿→1200000000）。

    遇非数字字符返回 None（放弃该读数，交由上层原样保留）。
    """
    if not s:
        return None
    total = section = number = 0
    for ch in s:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch == "十":
            number = 1 if number == 0 else number
            section += number * 10
            number = 0
        elif ch == "百":
            section += (1 if number == 0 else number) * 100
            number = 0
        elif ch == "千":
            section += (1 if number == 0 else number) * 1000
            number = 0
        elif ch == "万":
            section = (section + number) * 10000
            total += section
            section = number = 0
        elif ch == "亿":
            section = (section + number) * 100000000
            total += section
            section = number = 0
        else:
            return None
    return total + section + number


def _cn_number_sub(m: re.Match[str]) -> str:
    run = m.group(0)
    if run in _BLOCKLIST_Q or len(run) < 2:
        return run
    # 只转含进位字的读数：精确中文数量 ≥10 必含 十/百/千/万/亿；
    # 概数「两三」「七八」等只由单字凑成、不含进位字，跳过（保留中文概数）。
    if not any(ch in _CN_SCALE for ch in run):
        return run
    # 非法读数（实为两个相邻数量黏连、或「X 或 Y」概数范围）一律原样保留，宁可漏转：
    #   - 含重复进位字（两个 十/百/千/万/亿）：「五十五十」「三十万二十万」是两个数量被
    #     贪心正则黏在一起，转出来会变成 100 / 300020万；
    #   - 含连续单字读数（两个数字字紧挨、中间无进位字）：「七八十」「两三百」「三四万」
    #     「一两千」是「X 或 Y」概数范围，不是单一数值，转成单值会丢掉区间。
    _CN_DIGIT_ONLY = frozenset("零一二三四五六七八九两")
    _prev_digit = False
    _seen_scale = set()
    for _c in run:
        if _c in _CN_SCALE:
            if _c in _seen_scale:
                return run                      # 重复进位字 → 两个数量黏连
            _seen_scale.add(_c)
            _prev_digit = False
        elif _c in _CN_DIGIT_ONLY:
            if _prev_digit:
                return run                      # 连续单字读数 → 概数范围
            _prev_digit = True
        else:
            return run                          # 不应出现（_CN_RUN_RE 只匹配这些字）
    # 万/亿 后紧跟裸数字且未显式带 千/百/十：可能是省略千位（一万五 = 1万5千）
    # 或新数量（一亿两 = 1亿 + 两周），空后缀（一万五 结尾的五）也在此拦；
    # 两者都无法安全解析，整串保守跳过，宁可让「一万五」留中文，也不腐蚀成
    # 10005 / 100000002。带 千/百/十 的显式低位（一万五千 = 15000）不在此列。
    # 注意：后缀为空时必须显式判等——空串是任何串的子串，``"" not in "千百十"`` 会误判为 False。
    for _k, _c in enumerate(run):
        if _c in "万亿" and _k + 1 < len(run) and run[_k + 1] in _CN_DIGIT_ONLY:
            _after = run[_k + 2] if _k + 2 < len(run) else ""
            if _after == "" or _after not in "千百十":
                return run
    # 「亿万」是词（亿万富翁）不是数量，亿后紧跟万也跳过
    if "亿万" in run:
        return run
    text, i = m.string, m.end()
    # 概数前缀（数/近/约/上…）：「数百万」「近百」等保留中文读数
    if m.start() > 0 and text[m.start() - 1] in _CN_APPROX_PREFIX:
        return run
    # 后接字符约束：末尾（无后接）→ 转；后接 月/日/号/年 → 交日期规则；
    # 后接其它非中文（拉丁/标点/数字）→ 不转（避免「十六 3 家」这类或孤立读数）。
    # 后接中文（量词/名词/助词）→ 转。
    if i < len(text):
        nxt = text[i]
        if nxt in "月日号年" or not _CJK_RE.match(nxt):
            return run
    # 尾部 万/亿 作为单位保留（十二亿→12亿）；其余出整数（三百六十五→365）
    tail = re.fullmatch(r"(.+?)([万亿])$", run)
    if tail:
        head = _cn_quantity_to_int(tail.group(1))
        return run if head is None else f"{head}{tail.group(2)}"
    v = _cn_quantity_to_int(run)
    return run if v is None else str(v)


def normalize_cn_number(text: str) -> str:
    """中文读数数量转阿拉伯数字（保留万/亿单位），避开日期与成语。"""
    if not text:
        return text
    return _CN_RUN_RE.sub(_cn_number_sub, text)


def count_cn_number(text: str) -> int:
    """可转换的中文读数数量处数（供干跑预览用）。"""
    if not text:
        return 0
    n = 0
    for m in _CN_RUN_RE.finditer(text):
        if _cn_number_sub(m) != m.group(0):
            n += 1
    return n


# ============================================================
# 原文转录：话轮末逗号转句号（rules 话轮边界即句末，2026-10-11 连漪拍板）
# ============================================================

# 成因：ASR 在话轮边界把一句话切开，导致「一个人讲完却用逗号收尾」的不自然形态
# （下一行本就是另一位说话人，并不存在真正的跨话轮续句）。话轮边界应视作句末，
# 逗号收尾须转句号。只在「## 原文转录」段生效。
_TURN_RE = re.compile(r"^(【[^】]*】)(.*)$")


def normalize_turn_terminal(block: str) -> str:
    """原文转录每个【说话人】话轮若以逗号（，或 ,）收尾，改为句号（。）。"""
    if not block:
        return block
    out: list[str] = []
    for line in block.split("\n"):
        stripped = line.rstrip()
        if _TURN_RE.match(stripped) and stripped.endswith(("，", ",")):
            fixed = stripped[:-1] + "。"
            line = fixed + line[len(stripped):]
        out.append(line)
    return "\n".join(out)


# ============================================================
# 核心观点：句末引用原话自带句末标点（rules.STRUCT_KEY_POINT 的确定性落地）
# ============================================================

# 校订出来的核心观点常以一句引用原话收尾，而那句引语不带句末标点：
#
#     ……因为流量会真的涌进来。「互联网上的大多数操作都会由 Agent 来完成」
#
# 读到那里会断住——它看着像半句话。完整引用应是「……由 Agent 来完成。」，
# 句号在引号**之内**（疑问句用问号）。
#
# 不能对全文盲补：核心观点里大量引语是嵌在句中的短语（工作的基本单位从
# 「职位」变为「任务」），补句号就成了病句。所以判据刻意窄——**只看位于句末
# （引号后即换行/条目结尾）且自身不以句末标点收尾的引语**，这一类机械可判，
# 不会误伤嵌入式短语。
#
# 以承接连词（而/但/却/以及……）开头的引语是**残句**：原话本身就从半句开始，
# 补上句号仍不是完整句子。这类不自动改，只报出来交人工回原文补全。
_QUOTE_RE = re.compile(r"「([^」]*)」")
_QUOTE_END_PUNCT = "。？！.?!"
_QUOTE_INTERROGATIVE = ("吗", "呢", "？", "?")
_QUOTE_LEADING_CONNECTORS = ("而", "但", "却", "以及", "还是", "或", "所以", "就是")

# 只对这一节生效：条款写在 STRUCT_KEY_POINT 下，别处（摘要、Show Notes）不适用
_KEY_POINT_HEAD = "## 核心观点"


def _quote_end_mark(quote: str) -> str:
    """疑问句用问号，其余用句号——只补句号会把「……要我试着修一下吗」压成陈述句。"""
    return "？" if quote.endswith(_QUOTE_INTERROGATIVE) else "。"


def _quote_at_sentence_end(block: str, end: int) -> bool:
    """``」`` 之后紧接换行或块结束 → 该引语独立处于句末。

    不能用 ``(. )`` 去接尾部字符：``.`` 不匹配换行，会把句末那处整片漏掉。
    """
    return end >= len(block) or block[end] in "\r\n"


def _iter_bad_quotes(block: str):
    """产出块内「位于句末且缺句末标点」的引语 ``(match, quote)``。"""
    for m in _QUOTE_RE.finditer(block or ""):
        quote = m.group(1)
        if not quote or quote[-1] in _QUOTE_END_PUNCT:
            continue
        if not _quote_at_sentence_end(block, m.end()):
            continue
        if block[m.end(): m.end() + 1] in ("。", "？", "！"):
            continue
        yield m, quote


def scan_quote_period(block: str) -> tuple[list[str], list[str]]:
    """返回 ``(可自动补句号的完整句, 需人工判断的承接残句)``。"""
    fixable: list[str] = []
    manual: list[str] = []
    for _m, quote in _iter_bad_quotes(block):
        if quote in fixable or quote in manual:
            continue
        (manual if quote.startswith(_QUOTE_LEADING_CONNECTORS) else fixable).append(quote)
    return fixable, manual


def normalize_quote_period(block: str) -> str:
    """核心观点块里句末的引用原话补上句末标点（写进引号内）。承接残句不动。"""
    if not block:
        return block
    out: list[str] = []
    last = 0
    done: set[str] = set()
    for m, quote in _iter_bad_quotes(block):
        if quote.startswith(_QUOTE_LEADING_CONNECTORS) or quote in done:
            continue
        # m.end() 落在后引号之后：回退一格把句末标点放进引号内，避免写成「…」。」
        out.append(block[last: m.end() - 1])
        out.append(_quote_end_mark(quote) + CORNER_CLOSE)
        last = m.end()
        done.add(quote)  # 同名第二次出现按嵌入式短语处理，不改
    out.append(block[last:])
    return "".join(out)


def count_quote_period(block: str) -> int:
    """待补句末标点的引语处数（含承接残句——脚本不改，但要报出来给人看）。"""
    return sum(1 for _ in _iter_bad_quotes(block or ""))


def sync_handoff_quotes(doc: dict, quotes: list[str]) -> int:
    """同步 handoff 的 ``key_points[].quote`` 句末标点，返回改动处数。

    ``anchor`` 一律不动：它是原文转录里的加粗回标锚点，必须与转录逐字一致。
    只改 md 而不改 handoff，下次 Phase B 从 handoff 重渲染上层章节时会被覆盖回去。
    """
    targets = set(quotes)
    hits = 0
    for kp in doc.get("key_points") or []:
        q = kp.get("quote")
        if isinstance(q, str) and q in targets:
            kp["quote"] = q + _quote_end_mark(q)
            targets.discard(q)
            hits += 1
    return hits


def repair_stale_anchors(doc: dict, md_text: str) -> int:
    """把已无法在成稿里定位的 ``anchor`` 重新对齐，返回修复条数。

    ``anchor`` 是转录逐字锚点，理想情况应与 md 一字不差；但成稿侧已被确定性清洗过
    （数量转阿拉伯、标点清理、话轮末句号），anchor 若仍是清洗前的形态就会**失效**——
    ``bold_anchors`` 回标时定位不到，引用链静默断掉（实测 63 集里已断了 9 处，
    其中 6 处是上一轮只转 md 未同步 anchor 留下的旧伤）。

    只给**定位不到的** anchor 施加同一套清洗（能定位的一律不动）：逐字保真优先于美观，
    最小干预优先于统一格式。
    """
    n = 0
    for sec in ("key_points", "questions"):
        for item in doc.get(sec) or []:
            v = item.get("anchor")
            if not isinstance(v, str) or not v or v in md_text:
                continue
            for cand in (
                normalize_punct(v),
                normalize_cn_number(normalize_punct(v)),
                merge_repeat_chars(normalize_cn_number(normalize_punct(v))),
            ):
                if cand in md_text:
                    item["anchor"] = cand
                    n += 1
                    break
    return n


def sync_handoff_punct(doc: dict) -> int:
    """把 handoff 文本字段的标点清理一遍，返回改动处数。

    字段集合与 ``_sync_handoff_numbers`` 一致（``point/evidence/quote/question/answer``），
    ``anchor`` 交由 :func:`repair_stale_anchors` 按需最小修复，不在此统一改。
    """
    n = 0
    for kp in doc.get("key_points") or []:
        for key in ("point", "evidence", "quote"):
            v = kp.get(key)
            if isinstance(v, str) and v:
                new = normalize_punct(v)
                if new != v:
                    kp[key] = new
                    n += 1
    for q in doc.get("questions") or []:
        for key in ("question", "answer"):
            v = q.get(key)
            if isinstance(v, str) and v:
                new = normalize_punct(v)
                if new != v:
                    q[key] = new
                    n += 1
    return n


def _sync_handoff_numbers(doc: dict) -> int:
    """把 handoff 文本字段的中文读数数量转阿拉伯数字，返回改动处数。

    原因同 ``sync_handoff_quotes``：handoff 是上层章节的重渲染源，只改 md 不改
    handoff 会在下次 Phase B 时被覆盖回去。``anchor`` 不动（须与转录逐字一致）。
    """
    n = 0
    for kp in doc.get("key_points") or []:
        for key in ("point", "evidence", "quote"):
            v = kp.get(key)
            if isinstance(v, str) and v:
                new = normalize_cn_number(v)
                if new != v:
                    kp[key] = new
                    n += 1
    for q in doc.get("questions") or []:
        for key in ("question", "answer"):
            v = q.get(key)
            if isinstance(v, str) and v:
                new = normalize_cn_number(v)
                if new != v:
                    q[key] = new
                    n += 1
    return n


# ============================================================
# 成稿级入口：跳过 Show Notes（节目源简介透传，不归校订管）
# ============================================================

_SECTION_RE = re.compile(r"^(##\s+.+?)\s*$", re.MULTILINE)
_SKIP_SECTIONS = ("## Show Notes",)


def _iter_blocks(md: str) -> list[tuple[str, str, str]]:
    """按 ``## 标题`` 切块，产出 ``[(标题原文, 标题文本, 正文), ...]``。

    前言的标题为 ``""``。标题原文单独保留（含行尾空白）——拼接时原样回填，
    清洗只作用于正文，避免切块改写把格式弄乱。
    """
    marks = list(_SECTION_RE.finditer(md))
    if not marks:
        return [("", "", md)]
    blocks = [("", "", md[: marks[0].start()])]
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(md)
        blocks.append((m.group(0), m.group(1), md[m.end(): end]))
    return blocks


def _clean_block(block: str, head: str = "") -> str:
    block = merge_repeat_chars(
        normalize_punct(
            normalize_cn_date(normalize_cn_number(normalize_en_punct(block)))
        )
    )
    if head == _KEY_POINT_HEAD:
        block = normalize_quote_period(block)
    elif head in ("## 原文转录", "## 全文转录"):
        block = normalize_turn_terminal(block)
    return block


def normalize_markdown(md: str) -> str:
    """对成稿跑确定性规范，``## Show Notes`` 小节原样透传。"""
    if not md:
        return md
    out: list[str] = []
    for raw_head, head, body in _iter_blocks(md):
        out.append(raw_head)
        out.append(body if head.startswith(_SKIP_SECTIONS) else _clean_block(body, head))
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
    hits_en = hits_rp = 0
    for _raw, head, body in _iter_blocks(md):
        if head.startswith(_SKIP_SECTIONS):
            continue
        en, rp = count_text(body)
        hits_en += en
        hits_rp += rp
    return hits_en, hits_rp


def count_all(md: str) -> dict[str, int]:
    """返回待改数量 ``{en_punct, repeat, cn_date, cn_number, quote_period, punct}``，``## Show Notes`` 不计入。

    比 ``count_changes`` 多了日期、数量、引语句末标点与标点清理四项；单独开一个新
    函数而非改旧签名，避免牵动 ``finish_phase_b`` / ``output_quality`` 等既有调用方。
    """
    if not md:
        return {
            "en_punct": 0, "repeat": 0, "cn_date": 0, "cn_number": 0,
            "quote_period": 0, "punct": 0,
        }
    out = {
        "en_punct": 0, "repeat": 0, "cn_date": 0, "cn_number": 0,
        "quote_period": 0, "punct": 0,
    }
    for _raw, head, body in _iter_blocks(md):
        if head.startswith(_SKIP_SECTIONS):
            continue
        en, rp = count_text(body)
        out["en_punct"] += en
        out["repeat"] += rp
        out["cn_date"] += count_cn_date(body)
        out["cn_number"] += count_cn_number(body)
        out["punct"] += count_punct(body)
        if head == _KEY_POINT_HEAD:
            out["quote_period"] += count_quote_period(body)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="成稿确定性规范化（外文词后半角标点转全角 + ≥3 连重复字合并 + 中文读数日期转阿拉伯数字 + 中文读数数量转阿拉伯数字 + 原文转录话轮末逗号转句号 + 核心观点引语句末标点，不调 LLM）"
    )
    ap.add_argument("md_path", type=Path, help="成稿 .md 路径")
    ap.add_argument("--apply", action="store_true", help="写回文件（自动 .bak 备份）")
    ap.add_argument(
        "--no-handoff",
        action="store_true",
        help="不改同名 _handoff.json（默认会同步 key_points[].quote）",
    )
    args = ap.parse_args()

    if not args.md_path.exists():
        print(f"错误：找不到文件 {args.md_path}", file=sys.stderr)
        return 1

    md = args.md_path.read_text(encoding="utf-8")
    new_md = normalize_markdown(md)
    counts = count_all(md)
    hits_en, hits_rp = counts["en_punct"], counts["repeat"]
    hits_date, hits_number = counts["cn_date"], counts["cn_number"]
    hits_quote = counts["quote_period"]
    hits_punct = counts["punct"]

    # 承接残句不自动改，单独列出来交人工回原文补全
    quote_manual: list[str] = []
    quote_fixable: list[str] = []
    for _raw, head, body in _iter_blocks(md):
        if head == _KEY_POINT_HEAD:
            quote_fixable, quote_manual = scan_quote_period(body)

    print(f"文件: {args.md_path.name}")
    print(f"待转半角标点: {hits_en} 处")
    print(f"待合并 ≥3 连重复字: {hits_rp} 处")
    print(f"待转中文读数日期: {hits_date} 处")
    print(f"待转中文读数数量: {hits_number} 处")
    print(f"待补引语句末标点: {hits_quote} 处")
    print(f"待清标点缺陷(空格/堆叠/半角/错位): {hits_punct} 处")
    if quote_manual:
        print(f"\n需人工判断的承接残句 {len(quote_manual)} 处（不自动补）：")
        for q in quote_manual:
            print(f"  「{q}」")

    if (
        hits_en == 0 and hits_rp == 0 and hits_date == 0 and hits_number == 0
        and hits_punct == 0 and not quote_fixable
    ):
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

    if args.no_handoff:
        return 0
    handoff = args.md_path.with_name(args.md_path.name[: -len(".md")] + "_handoff.json")
    if not handoff.exists():
        return 0
    doc = json.loads(handoff.read_text(encoding="utf-8"))
    n = sync_handoff_quotes(doc, quote_fixable)
    n_num = _sync_handoff_numbers(doc)
    n_punct = sync_handoff_punct(doc)
    n_anchor = repair_stale_anchors(doc, new_md)
    if n or n_num or n_punct or n_anchor:
        h_bak = handoff.with_suffix(handoff.suffix + ".bak")
        if not h_bak.exists():
            shutil.copy2(handoff, h_bak)
        handoff.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        parts = []
        if n:
            parts.append(f"quote {n} 处")
        if n_num:
            parts.append(f"数量 {n_num} 处")
        if n_punct:
            parts.append(f"标点 {n_punct} 处")
        if n_anchor:
            parts.append(f"失效 anchor 重定位 {n_anchor} 处")
        print(f"已同步 {handoff.name} 的 {'、'.join(parts)}（备份 {h_bak.name}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
