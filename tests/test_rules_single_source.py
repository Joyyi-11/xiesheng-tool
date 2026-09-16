"""规则单一权威源（src.processor.rules）的同源保证测试。

背景（连漪 2026-09-07 拍板）：session 链路与 api 链路本质都调用大模型，
**编辑口径必须完全一致**——规则文字只存在于 rules.py，两条链路的提示词
都引用同一批常量组装，杜绝两份文本人工同步漂移（v17 冒号事故即此类）。

本文件验证：
1. 版本同源：session/api 的版本号都引用 RULES_SPEC_VERSION；
2. 片头精华条款（EDIT_OPENING）在共享总表里、且被两条链路引用；
3. 核心观点口径（STRUCT_KEY_POINT）含「片头精华/金句预告是提炼素材」说明；
4. 两链路提示词实际都包含共享条款文字（不是只 import 不用）。
"""

from src.processor import rules
from src.processor.prompt import CLEAN_SYSTEM_PROMPT, STRUCT_SYSTEM_PROMPT
from src.processor.session_edit import SESSION_SPEC_VERSION, SESSION_RULES


def test_edit_opening_is_shared_and_listed_once():
    # 片头精华条款必须在共享 EDIT_RULES 中（连漪：除调用方式外其余都应一样）
    assert rules.EDIT_OPENING in rules.EDIT_RULES
    # 不得重复定义：EDIT_OPENING 文字只在总表出现一次（曾有两份并存互相遮蔽）
    assert rules.EDIT_RULES.count(rules.EDIT_OPENING) == 1
    # 完整版含「金句预告」字样（片头精华是核心观点/亮点句的提炼素材）
    assert "片头精华/金句预告/串台预告不区分说话人" in rules.EDIT_OPENING


def test_opening_rule_reaches_both_chains():
    # session 链路提示词包含片头精华条款（编号化共享条款之一）
    assert "片头精华/金句预告/串台预告不区分说话人" in SESSION_RULES
    # api CLEAN 提示词包含同一条款
    assert "片头精华/金句预告/串台预告不区分说话人" in CLEAN_SYSTEM_PROMPT
    # 片头精华在共享条款中的位置与会话链路编号一致（共享第 9 条 → 会话规则第 9 条）
    idx = rules.EDIT_RULES.index(rules.EDIT_OPENING) + 1
    assert f"{idx}. **片头精华" in SESSION_RULES


def test_struct_key_point_mentions_opening_as_source_material():
    # 连漪：片头精华可作为核心观点、亮点句的参考来源
    assert "片头精华/金句预告是提炼素材" in rules.STRUCT_KEY_POINT
    # 两链路的结构化提示词都引用该口径
    assert "片头精华/金句预告是提炼素材" in STRUCT_SYSTEM_PROMPT
    assert "片头精华/金句预告是提炼素材" in SESSION_RULES


def test_versions_track_shared_rules_version():
    # session 版本号引用 rules；api 的 CLEAN/STRUCT 缓存版本同源（见 llm_processor）
    assert SESSION_SPEC_VERSION == rules.RULES_SPEC_VERSION
    assert rules.RULES_SPEC_VERSION == 24


def test_en_punct_and_repeat_rules_are_deterministic_backed():
    """v24：半角标点与 ≥3 连重复字两条须写明「由确定性脚本兜底」。

    回归背景（2026-09-15 公开仓 issue #1 / #2）：两条规则原文只写在提示词层，
    实测随每集复制的 oneoff 校订脚本漂移——带 en_punct 的集残留 0 处、未带的
    Vol.1 残留 91 处。条款须显式指向 src.processor.normalize，否则又会退回
    「指望 LLM 在长文里自觉执行」。
    """
    assert "src.processor.normalize" in rules.EDIT_EN_PUNCT
    assert "src.processor.normalize" in rules.EDIT_REDO
    # 两条都得进共享总表并抵达会话链路
    assert rules.EDIT_EN_PUNCT in rules.EDIT_RULES
    assert rules.EDIT_REDO in rules.EDIT_RULES
    assert "src.processor.normalize" in SESSION_RULES


def test_fidelity_allows_filler_removal_with_conservation_judge():
    """v24：EDIT_FIDELITY 必须开出「删口语噪音」例外口，且带信息量守恒判据。

    回归背景（issue #2）：原文三条互相打架——FIDELITY 只准修正错字（堵住删噪）、
    FILLER 授权删除、REDO 把「重写表达」明确禁止。开了口子就必须同时给判据，
    否则会滑向「为通顺而压缩概要」。
    """
    assert "口语噪音" in rules.EDIT_FIDELITY
    assert "信息量守恒" in rules.EDIT_FIDELITY


def test_anchor_rule_reaches_both_links():
    """回标锚点口径必须进两条链路——否则会话链路转录区零回标、与 api 出品不一致。

    回归背景：STRUCT_ANCHORS 曾被当作「未使用 import」从 session_edit 清理掉，
    导致会话链路完全没有回标加粗口径（quote 又是润色版、逐字匹配必然失败）。
    """
    assert "回标加粗" in rules.STRUCT_ANCHORS
    # 锚点必须是原文逐字子串（这是回标能命中的前提）
    assert "逐字存在" in rules.STRUCT_ANCHORS
    # 两链路都得有：api 走 STRUCT 提示词，session 走 SESSION_RULES
    assert "回标加粗" in STRUCT_SYSTEM_PROMPT
    assert "回标加粗" in SESSION_RULES
    # 共享条款必须机制中立（不得只描述 api 的渲染层实现）
    assert "渲染层" not in rules.STRUCT_ANCHORS
    # 但须说明两链路落地等价（会话链路改用确定性回标脚本，不再靠 LLM 自觉加粗）
    assert "bold_anchors" in rules.STRUCT_ANCHORS
    assert "bold_anchors" in SESSION_RULES


def test_session_rules_embed_shared_edit_rules_textually():
    # SESSION_RULES 必须包含共享校订条款的实际文字（非仅占位）
    for rule in rules.EDIT_RULES:
        # 每条共享条款的核心标题词应出现在组装后的 SESSION_RULES 里
        head = rule.split("：")[0].split(":")[0].strip("** \n")
        if len(head) >= 4:  # 跳过过短的标题词，避免误匹配
            assert head in SESSION_RULES, f"共享条款未进 SESSION_RULES: {head}"


def test_struct_rules_reach_session_structure_section():
    # 会话链路「输出结构」引用共享 STRUCT 口径（摘要/核心观点/问题/术语表/人物简介）
    for text in (
        "基于全文但不照搬原文",
        "观点句必须是完整陈述句",
        "首次出现须写明「姓名（身份/头衔）」",
        "排除播客名、节目名、嘉宾姓名、平台名",
        "正文格式（**姓名**：简介）",
    ):
        assert text in SESSION_RULES
