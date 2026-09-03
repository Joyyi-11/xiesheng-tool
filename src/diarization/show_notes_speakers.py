"""从 Show Notes 解析预期说话人数与名单，为 diarization 提供 K 锚点。

实现收敛为 `speaker_resolver.parse_roster_with_roles` 的薄封装：花名册正则与角色
解析口径统一在 speaker_resolver 一处维护，本模块仅保留对外 API 与单测契约
（返回名字列表；不足 2 人返回 None）。
"""
from src.diarization.speaker_resolver import parse_roster_with_roles


def parse_speakers_from_show_notes(show_notes: str) -> list[str] | None:
    """返回 Show Notes 中识别到的说话人名字列表；不足 2 人时返回 None。

    None 表示无法可靠确定人数，交给显式 ``--speakers`` 或自动检测。
    多嘉宾（「嘉宾：张三、李四」）可能漏识别后用「、」连接的名字，此类节目
    仍需手动 ``--speakers``；本解析面向「主播 + 单嘉宾」的常规两人播客。
    """
    roster = parse_roster_with_roles(show_notes)
    names: list[str] = []
    for _, name in roster:
        if name not in names:
            names.append(name)
    return names if len(names) >= 2 else None
