"""批量给「核心观点」里**句末的引语**补句末标点（薄封装，规则只在 normalize 里实现一次）。

## 它解决什么

校订出来的核心观点常以一句引用原话收尾，但那句引语**不带句末标点**：

    ……因为流量会真的涌进来。「互联网上的大多数操作都会由 Agent 来完成」

读到这一句会断住——它看起来像半句话。书面里完整的引用应当是：

    ……因为流量会真的涌进来。「互联网上的大多数操作都会由 Agent 来完成。」

## 本脚本只做「批量」，不做规则

规则与修复逻辑在 ``src.processor.normalize``（``normalize_quote_period`` /
``scan_quote_period`` / ``sync_handoff_quotes``），本文件**不再自己实现判据**——
确定性规则只实现一次是本项目的硬规矩（见 SKILL.md「易错点」）。这里只补上
normalize 单文件 CLI 没有的能力：一次扫完 ``output/`` 下全部成稿。

单集预览/落盘直接用 normalize（它还会同步 ``_handoff.json`` 的 quote）：

    D:/Python311/python.exe -m src.processor.normalize "output/<prefix>.md" --apply

## 判据（照搬 normalize，此处仅说明）

只认**位于句末**（引号后即换行/条目结尾）且自身不以句末标点收尾的引语；
嵌在句中的短语引述（工作的基本单位从「职位」变为「任务」）不带标点，不误伤。
疑问句补问号不补句号。

以承接连词（而/但/却/以及/还是/或/所以/就是）开头的引语是**残句**，原话本身
就从半句开始，补句号仍不是完整句——脚本不自动改，只列出来交人工**回原文补全**
（例：「而不只是每秒多发一个 prompt」→ 原文是「但我认为这才是 AI 能带来的好处，
而不只是每秒多发一个 prompt。」）。

用法：
    D:/Python311/python.exe scripts/oneoff/fix_upper_quote_period.py "#757"       # 干跑
    D:/Python311/python.exe scripts/oneoff/fix_upper_quote_period.py "#757" --apply
    D:/Python311/python.exe scripts/oneoff/fix_upper_quote_period.py --all --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 脚本位于 scripts/oneoff/，运行时不带仓库根到 sys.path，这里手动注入，
# 才能 `from src.processor.normalize import ...`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT = Path("output")


def process(md_path: Path, apply: bool) -> tuple[int, int, list[str]]:
    """处理单份成稿，返回 ``(md 改动行数, handoff 同步处数, 待人工补全的残句)``。"""
    # 延迟导入：本脚本在 output/ 目录语境下跑，路径与 normalize 的 CLI 一致
    from src.processor.normalize import (
        _iter_blocks,
        _KEY_POINT_HEAD,
        normalize_markdown,
        scan_quote_period,
        sync_handoff_quotes,
    )

    text = md_path.read_text(encoding="utf-8")
    fixable: list[str] = []
    manual: list[str] = []
    for _raw, head, body in _iter_blocks(text):
        if head == _KEY_POINT_HEAD:
            fixable, manual = scan_quote_period(body)
    if not fixable:
        return 0, 0, manual

    new_text = normalize_markdown(text)
    handoff = md_path.with_name(md_path.name[: -len(".md")] + "_handoff.json")
    h_hits = 0
    doc = None
    if handoff.exists():
        doc = json.loads(handoff.read_text(encoding="utf-8"))
        h_hits = sync_handoff_quotes(doc, fixable)

    md_diff = sum(1 for a, b in zip(text.splitlines(), new_text.splitlines()) if a != b)
    if apply:
        if new_text != text:
            md_path.write_text(new_text, encoding="utf-8")
        if doc is not None and h_hits:
            handoff.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    return md_diff, h_hits, manual


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefix", nargs="?", help="成稿文件名前缀，如 #757")
    ap.add_argument("--all", action="store_true", help="处理 output/ 下全部成稿")
    ap.add_argument("--apply", action="store_true", help="落盘（默认干跑）")
    args = ap.parse_args()

    if not args.all and not args.prefix:
        ap.error("需要给出前缀，或用 --all")

    files = (
        sorted(OUTPUT.glob("*.md"))
        if args.all
        else sorted(OUTPUT.glob(f"{args.prefix}*.md"))
    )
    files = [f for f in files if not f.name.endswith(("_clean_tmp.md", "_final.md", ".bak.md"))]

    total_md = total_h = 0
    manual_all: list[tuple[str, str]] = []
    for f in files:
        md_diff, h_hits, manual = process(f, args.apply)
        total_md += md_diff
        total_h += h_hits
        for q in manual:
            manual_all.append((f.name, q))
        if md_diff or h_hits or manual:
            flag = "已改" if args.apply else "干跑"
            print(f"[{flag}] {f.name[:44]:46s} md {md_diff} 行 / handoff {h_hits} 处")

    print()
    print(f"合计：md {total_md} 行、handoff {total_h} 处  {'（已落盘）' if args.apply else '（干跑，未改）'}")
    if manual_all:
        print(f"\n需人工判断的承接残句 {len(manual_all)} 处（未自动补，回原文补整句）：")
        for name, q in manual_all:
            print(f"  {name[:40]:42s} 「{q}」")
    return 0


if __name__ == "__main__":
    sys.exit(main())
