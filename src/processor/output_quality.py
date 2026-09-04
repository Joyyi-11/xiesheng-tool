"""Final-output quality gates shared by the API and session pipelines."""

from src.models.schemas import OutputDoc
from src.utils import SPEAKER_LABEL_RE


def validate_output_doc(doc: OutputDoc) -> list[str]:
    """Return blocking problems in a structured output document.

    The glossary is intentionally allowed to be empty: the current v13 spec
    treats residual terms as optional. Speaker introduction is also optional at
    the model level; the renderer emits an explicit ``(none)`` section.
    """
    problems: list[str] = []

    if not doc.title.strip():
        problems.append("缺少节目标题")
    if not doc.podcast_name.strip():
        problems.append("缺少播客名称")
    if not doc.pub_date.strip():
        problems.append("缺少播出日期")
    if not doc.full_text.strip():
        problems.append("原文转录为空")
    if not doc.summary.strip():
        problems.append("摘要为空")
    if not doc.key_points:
        problems.append("核心观点为空")
    if any(not kp.point.strip() for kp in doc.key_points):
        problems.append("核心观点存在空主题句")
    if not doc.questions:
        problems.append("问题与思考为空")
    if any(not q.question.strip() for q in doc.questions):
        problems.append("问题与思考存在空问题")

    leftover_speakers = sorted(set(SPEAKER_LABEL_RE.findall(doc.full_text)))
    if leftover_speakers:
        problems.append("原文转录残留未映射说话人标签：" + "、".join(leftover_speakers))

    return problems


def assert_valid_output_doc(doc: OutputDoc) -> None:
    problems = validate_output_doc(doc)
    if problems:
        raise ValueError("LLM 结构化输出未通过成品校验：" + "；".join(problems))
