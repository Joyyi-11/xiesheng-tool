"""从 Show Notes 解析预期说话人数与名单，为 diarization 提供 K 锚点。

仅靠 ASR 说话人分离（cam++）对两人相似声线极易过度切分——vol.231 一期两人
对话被切成 14 个伪说话人，正是根因。Show Notes 通常写明主播与嘉宾，可稳定得到
「有几人、分别是谁」，让下游强制聚成 K 类、LLM 贴名有锚点。

纯标准库实现，无 funasr / diarize 重依赖，可脱离模型独立单测。
"""
from __future__ import annotations

import re

_ROLE = r"(?:主[播持理]|嘉宾|客座)"
# 角色词后可选「简介 / 人」等后缀；再允许接一个可选角色词（如「客座嘉宾张三」
# 里的「嘉宾」），最后捕获名字。名字只取 CJK 字母/点号（纯词符），遇标点、
# 书名号、顿号、冒号即停，避免把边界字符吞进名字。
_NAME = r"[一-龥A-Za-z·]{1,6}"
_PATTERN = re.compile(rf"{_ROLE}(?:简介|人)?(?:主[播持理]|嘉宾|客座|主持)?[：:\s]*({_NAME})")

# 名字后若紧跟角色词或连接词（无标点边界时），截断到第一个出现处。
_TRUNCATE = re.compile(r"(?:主[播持理]|嘉宾|客座|主持|与|和|及|等|携|邀请)")

# 角色词后易误捕获的填充词，不作为说话人名。
_STOPWORDS = {"是", "与", "和", "等", "的", "为", "叫", "有", "及", "或", "也", "来"}


def parse_speakers_from_show_notes(show_notes: str) -> list[str] | None:
    """返回 Show Notes 中识别到的说话人名字列表；不足 2 人时返回 None。

    None 表示无法可靠确定人数，交给显式 ``--speakers`` 或自动检测。
    多嘉宾（「嘉宾：张三、李四」）可能漏识别后用「、」连接的名字，此类节目
    仍需手动 ``--speakers``；本解析面向「主播 + 单嘉宾」的常规两人播客。
    """
    if not show_notes:
        return None
    clean = re.sub(r"[【】]", "", show_notes)
    names: list[str] = []
    for m in _PATTERN.finditer(clean):
        name = _TRUNCATE.split(m.group(1))[0].strip()
        if len(name) < 2 or name in _STOPWORDS:
            continue
        if name not in names:
            names.append(name)
    return names if len(names) >= 2 else None
