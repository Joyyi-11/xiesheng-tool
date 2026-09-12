# -*- coding: utf-8 -*-
"""Vol.1《AI最前沿的人已经不聊大模型了》会话包确定性校订。

流程：读 _diarized.txt → 说话人映射 + 同人合段 + 标点/填充词清理 + 英文碎词拼回
→ 组装 v23 固定结构 .md（上层六段由本脚本嵌入）→ 落盘。
落盘后另跑：
    python -m src.processor.bold_anchors <md> --apply
    python -m src.processor.session_edit --check <md>
"""
from __future__ import annotations

import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
SRC = BASE / "output" / "Vol.1 AI最前沿的人已经不聊大模型了_diarized.txt"
OUT = BASE / "output" / "Vol.1 AI最前沿的人已经不聊大模型了.md"

TITLE = "Vol.1 AI最前沿的人已经不聊大模型了"
PODCAST = "易论AI"
DATE = "2026-09-09"

SPK = {
    "SPEAKER_00": "歸藏",
    "SPEAKER_01": "易亚婷",
    "SPEAKER_02": "橘子",
    "SPEAKER_03": "李继刚",
}

# POST_MAP：英文碎词/同音错字拼回（specific 在前，general 在后）。
# 仅对转录区生效；无法可靠还原的碎片原样保留，不臆造。
POST_MAP = [
    # 人名/自称误识
    ("纪刚", "李继刚"), ("金刚", "李继刚"), ("基刚", "李继刚"), ("技纲", "李继刚"),
    ("藏师傅", "歸藏"), ("丧师傅", "歸藏"), ("赞师傅", "歸藏"), ("归杖", "歸藏"),
    # 英文专名碎片
    ("f d一", "FDE"), ("f d e", "FDE"),
    ("sars斯", "SaaS"), ("sars", "SaaS"),
    ("a 零", "AI"), ("a键", "AI"), ("a i", "AI"),
    ("a 策", "Agent"), ("a 这策", "Agent"), ("a 症", "Agent"),
    ("a 正", "Agent"), ("a 政策", "Agent"), ("a 与 a", "Agent 与 Agent"),
    ("a agent", "Agent"), ("a 帧", "Agent"), ("a 政势", "Agent"),
    ("a j 加 i m", "Agent 加 IM"), ("a 加 x", "Agent 加 X"),
    ("i m 加 agent", "IM 加 Agent"), (" i m", " IM"), ("x 加 agent", "X 加 Agent"),
    ("agentent", "Agent"), ("agentt", "Agent"),
    (" engine 产品", " AI 产品"), (" engine ", " AI "), (" edit ", " AI "), (" ed ", " AI "),
    ("gi upub", "GitHub"), ("giub", "GitHub"), ("gi 艺术", "GitHub issue"),
    ("the get up", "Git"), ("get", "Git"),
    ("e秀和 p r", "issue 和 PR"), ("e秀", "issue"), (" p r", " PR"),
    ("select co", "Slack"), ("select", "Slack"),
    ("very coding", "vibe coding"), ("curs", "Cursor"), ("斯啊到 code", "Cursor"),
    ("co拉", "Cola"), ("cola", "Cola"), ("col ", "Cola "),
    (" coco", " Coze"), ("cocoa", "Coze"),
    ("dash视报", "dashboard"), ("电视报道", "dashboard"),
    ("op不", "Obsidian"), ("op x", "Obsidian"), ("op 不", "Obsidian"),
    ("非洲", "Manus"),
    ("电影贺", "电影《Her》"),
    ("土 b", "to B"), ("垂雷", "垂直"),
    ("生下文", "上下文"), ("剩下文", "上下文"), ("上下门", "上下文"),
    ("上下位", "上下文"),
    ("删文", "上下文"), ("删下文", "上下文"),
    ("日门网盘", "日历、网盘"),
    ("jaman", "Gemini"), ("j密", "Gemini"),
    ("m c p", "MCP"),
    ("g g p t", "GPT"), ("g p t", "GPT"), ("code袋头", "ChatGPT"),
    ("讯机", "虚拟机"),
    ("飞触", "飞书"), ("飞洲", "飞书"), ("飞叔", "飞书"), ("飞数", "飞书"),
    ("飞叔那个录音豆", "飞书妙记"),
    ("信息减房", "信息茧房"),
    ("group bo", "Genspark"), ("grow board", "Genspark"), ("growthbo", "Genspark"),
    ("k 那个 growthbo", "Genspark"),
    ("d b c", "DeepSeek"), ("deeps", "DeepSeek"), ("deept", "DeepSeek"),
    ("deep c", "DeepSeek"), ("d斯", "DeepSeek"), ("第 stick", "DeepSeek"), ("d v c", "DeepSeek"),
    ("英维达", "英伟达"), ("it且", "GPU"),
    ("露娜", "Llama"), ("luna", "Llama"),
    ("so塔", "SOTA"), ("sort 机", "SOTA"), ("sort 模型", "SOTA"),
    ("a m c", "API"),
    ("op 四点六", "o3"), ("四点六", "4o"),
    ("cloud code", "Claude Code"), ("cloud cloud", "Claude"), ("cloud", "Claude"),
    ("奥夫斯斯五", "Opus"), ("奥夫斯五", "Opus"),
    ("克拉", "Cola"), ("也那", "也就是"), ("太接了", "太远了"),
    ("空共血糖", "空腹血糖"),
    ("膳网", "释放"), ("回验实验", "实验"), ("发验", "实验"),
    ("宣间段", "时间段"), ("自滋药", "滋补药"),
    ("经线", "经期"), ("止性的", "镇静的"),
    ("harnessness", "harness"), ("ha尔nes斯", "harness"), ("honeynes", "honest"),
    ("结偶", "解耦"),
    ("多林国", "多邻国"),
    ("二一二一年", "2021 年"), ("二二年", "2022 年"), ("三零一年", "2030 年"),
]


def clean_text(t: str) -> str:
    t = t.strip()
    # 1) 去标点前后空格（须在合并重标点之前）
    t = re.sub(r"\s*([，。！？、；：])\s*", r"\1", t)
    # 2) 合并重复标点
    t = re.sub(r"([，。！？、；：])\1+", r"\1", t)
    # 3) 删无意义语气词（仅独立成词的 呃/嗯/啊/哎/唉/哦）
    t = re.sub(r"(^|[，。！？、；：\s])(呃|嗯|啊|哎|唉|哦)\s*", r"\1", t)
    # 4) 再合并一次（删词可能产生新重标点）
    t = re.sub(r"([，。！？、；：])\1+", r"\1", t)
    # 5) 清「。，」「。、」
    t = re.sub(r"([。！？])[，、]+", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def build_transcript_body(diarized_text: str) -> str:
    lines = diarized_text.splitlines()
    tr_start = next(i for i, l in enumerate(lines) if l.startswith("# 转录全文"))
    tr_lines = lines[tr_start + 1:]

    groups: list[list[str]] = []  # [(speaker, [body,...])]
    cur_spk = None
    for l in tr_lines:
        m = re.match(r"\[(SPEAKER_\d+)\]\s*(.*)", l)
        if not m:
            continue
        spk, body = m.group(1), m.group(2).strip()
        if spk == cur_spk:
            groups[-1][1].append(body)
        else:
            groups.append([spk, [body]])
            cur_spk = spk

    paras = []
    for spk, bodies in groups:
        merged = "".join(bodies)
        cleaned = clean_text(merged)
        if not cleaned:
            continue
        name = SPK.get(spk, spk)
        paras.append(f"【{name}】{cleaned}")

    body = "\n\n".join(paras)
    for pat, rep in POST_MAP:
        body = body.replace(pat, rep)
    return body


def extract_show_notes(diarized_text: str) -> str:
    lines = diarized_text.splitlines()
    sn = next(i for i, l in enumerate(lines) if l.strip() == "# Show Notes")
    tr = next(i for i, l in enumerate(lines) if l.startswith("# 转录全文"))
    return "\n".join(lines[sn + 1:tr]).strip()


UPPER = """## Show Notes

{show_notes}

## 摘要

本期是播客《易论AI》「锵锵四人行」第一期，主理人易亚婷与李继刚、橘子、歸藏三位 AI 创业者，围绕「这半年对 AI 的判断有什么变化」展开圆桌对谈。话题从 AI 落地为何要靠「服务裹着能力」、创业里的「齿轮速度差」，一路聊到 Agent 作为软件入口的形态之争、内容消费侧被忽视的 AI 渗透，以及个人上下文该如何维护才不会把人拉回过去的平均值。

## 核心观点

- **服务裹着能力，而非能力裹着服务。**李继刚认为企业买了账号和 Token 不等于拿到结果，模型能力与业务之间的落差要靠服务去填补，所谓 FDE 就是把这个落差填平。「服务裹着能力，而不是能力裹着服务。」
- **软件已死，AI 交付的是服务。**橘子指出 SaaS 与 AI 卖的都不是软件而是服务，产品必须自带一套 FDE、把最佳实践封装进去，用户才感知到价值而非只看到功能列表。「软件已经死了。」
- **上下文是要维护的车，不是增值资产。**易亚婷提出个人上下文积累越多越要主动划边界、定期清理，否则 AI 会把你拉回过去的平均值，而非带来新的认知。「上下文是一辆要维护的车。」
  - **信息茧房**：把 AI 当唯一信源、只在自己的闭环里打转，会陷入互相夸对方、越聊越觉得彼此都对的认知困境。
- **技术齿轮与制度齿轮存在速度差。**李继刚用组织内层层齿轮的比喻解释创业摩擦——技术几月一变、制度与利益分配层层滞后，服务正是在填平这道速度差。「小齿轮大齿轮它速度不匹配。」
- **Agent 加 X 比 X 加 Agent 后劲更足。**李继刚判断入口之争向左走向右走，但让一切功能化散为零、以对话为界面入口的「Agent 加 X」更贴近未来人机协作的姿态。「Agent 加 X 是一条后劲更足的路。」
- **记忆的价值在「网」不在堆。**众人讨论个人 Agent 上下文闭环与 A-to-A 社交，指出当前缺的是连接一个个 Agent 的「那张网」，且人始终更偏爱与人连接而非与别人的 Agent 连接。「缺的是那张网。」
- **模型价格两极分化。**橘子判断 SOTA 模型恒定高位、主流模型指数下降，Llama 已满足九成需求但顶尖智能模型仍贵，而缓存率决定 API 的真实性价比。「已经能够满足百分之九十人的需求。」

## 术语表

- **FDE**：Forward Deployed Engineer（前沿部署工程师）的缩写，指带着服务直接进入企业、现场摸排并沉淀最佳实践，把「模型能力」与「业务结果」之间的落差填平。
- **SaaS**：Software as a Service（软件即服务），本期中橘子用以说明「卖的是服务不是软件」，AI 同理。
- **MCP**：Model Context Protocol（模型上下文协议），OpenAI、谷歌等开放的标准连接协议，让 Agent 能直接调用邮件、日历、网盘等外部服务。
- **Genspark**：一款以纯对话为界面、背后连接永久云端虚拟机与各服务的 AI Agent 产品，橘子与歸藏以此讨论 C 端 Agent 的极简形态。

## 问题与思考

1. **当 AI 能力相同，企业为什么还是用不起来？**李继刚（43AI 合伙人）指出落差不在产品功能，而在「模型能力」与「业务结果」之间的服务缺口——同样一款 Agent，不同人的使用效果差异巨大；培训来不及，靠 FDE 带着服务现场摸排出最佳实践，才是填平落差的第三条路。

2. **Agent 加 X 与 X 加 Agent，哪条路能走到终局？**李继刚认为以 Agent 为入口、把各类功能化散为零的「Agent 加 X」后劲更足；橘子（Cola、ListenHub 创始人）则提醒，旧产品的组织结构、用户习惯与付费链路都让「X 加 Agent」别扭且难以原生，但两条路目前都未分胜负。

3. **我们天天研究的 AI，和大众正在消费的 AI 是同一个吗？**歸藏（AI 创业者）观察到行业圈子在谈效率与 coding，而短剧、游戏、陪伴等消费侧 AI 的渗透率增长远快于工具侧；李继刚坦言自己刷短剧时也被钩子一级级勾着走，两个世界默认彼此不存在。

4. **个人上下文到底该不该无限堆积？**易亚婷（易论AI 主理人）发现上下文越多，AI 越容易把她拉回过去的平均值；三人共识是上下文要像车一样维护——划边界、做索引、按项目隔离，而不是无状态地往里堆。

## 人物简介

**易亚婷**：易论AI 主理人。
**李继刚**：43AI 合伙人，长期研究人如何与 AI 协作。
**橘子**：Cola、ListenHub 创始人。
**歸藏**：AI 创业者，Skill 玩得很深，过去做内容，今年开始用内容带产品。

## 原文转录

"""


def main() -> None:
    diarized = SRC.read_text(encoding="utf-8")
    show_notes = extract_show_notes(diarized)
    transcript = build_transcript_body(diarized)

    md = f"# {TITLE}\n\n> 来源：{PODCAST} | {TITLE} | {DATE}\n\n"
    md += UPPER.format(show_notes=show_notes)
    md += transcript + "\n"

    OUT.write_text(md, encoding="utf-8")
    print(f"[OK] 已写出 {OUT.name}（{len(md)} 字符，转录区 {len(transcript)} 字符）")


if __name__ == "__main__":
    main()
