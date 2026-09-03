"""Tests for the pydantic struct models used to parse LLM output."""

from src.models.llm_struct import parse_struct, struct_to_doc


class TestParseStruct:
    def test_parses_valid_payload(self):
        struct = parse_struct(
            {
                "key_points": [{"point": "观点", "evidence": "证据"}],
                "speaker_intro": "**主播**：Nina",
                "speaker_mapping": {"SPEAKER_00": "Nina"},
                "keywords": [{"key": " AI ", "desc": " d "}],
                "summary": " 本期讲求职 ",
                "questions": [{"question": " 找不到工作是谁的问题？ ", "answer": " 策略错位。 "}],
            }
        )
        assert struct.key_points[0].point == "观点"
        assert struct.speaker_intro == "**主播**：Nina"
        assert struct.speaker_mapping == {"SPEAKER_00": "Nina"}
        assert struct.keywords[0].key == "AI"
        assert struct.keywords[0].desc == "d"
        assert struct.summary == "本期讲求职"
        assert struct.questions[0].question == "找不到工作是谁的问题？"
        assert struct.questions[0].answer == "策略错位。"

    def test_none_returns_empty_struct(self):
        struct = parse_struct(None)
        assert struct.key_points == []
        assert struct.keywords == []
        assert struct.questions == []
        assert struct.speaker_mapping == {}
        assert struct.speaker_intro == ""

    def test_malformed_items_dropped_but_rest_kept(self):
        struct = parse_struct(
            {
                "key_points": [{"point": "好"}, "bad", {"evidence": "no point"}],
                "keywords": ["bad", {"key": "", "desc": "空"}],
                "questions": ["bad", {"question": ""}, {"question": "有效问题", "answer": "整理答案"}],
                "speaker_intro": 123,
                "speaker_mapping": "not-a-dict",
            }
        )
        assert [k.point for k in struct.key_points] == ["好", ""]
        assert len(struct.keywords) == 0
        assert [q.question for q in struct.questions] == ["有效问题"]
        assert struct.questions[0].answer == "整理答案"
        assert struct.speaker_intro == ""
        assert struct.speaker_mapping == {}

    def test_caps_detected_lists(self):
        many = [{"point": f"p{i}"} for i in range(50)]
        struct = parse_struct({"key_points": many})
        assert len(struct.key_points) == 20  # MAX_KEY_POINTS（不限条数，宽容上限）

    def test_caps_questions_at_max(self):
        many = [{"question": f"q{i}"} for i in range(20)]
        struct = parse_struct({"questions": many})
        assert len(struct.questions) == 5  # MAX_QUESTIONS（限制上限）


class TestStructToDoc:
    def test_round_trips_clean_payload(self):
        raw = {
            "key_points": [{"point": "观点", "evidence": "证据", "quote": "原话"}],
            "speaker_intro": "intra",
            "speaker_mapping": {"SPEAKER_00": "A"},
            "keywords": [{"key": "词", "desc": "d"}],
            "summary": " 本期讲求职 ",
            "questions": [{"question": "找不到工作是谁的问题？", "answer": "策略错位。"}],
        }
        doc = struct_to_doc(parse_struct(raw))
        assert doc["key_points"][0]["point"] == "观点"
        assert doc["key_points"][0]["quote"] == "原话"
        assert doc["speaker_mapping"] == {"SPEAKER_00": "A"}
        assert doc["keywords"][0]["key"] == "词"
        assert doc["summary"] == "本期讲求职"
        assert doc["questions"] == [{"question": "找不到工作是谁的问题？", "answer": "策略错位。"}]

    def test_strips_whitespace_and_drops_invalid(self):
        doc = struct_to_doc(
            parse_struct(
                {
                "key_points": [{"point": "  a  ", "evidence": "  e  "}, {"point": "", "evidence": "x"}],
                "keywords": [{"key": "", "desc": "d"}],
                }
            )
        )
        assert doc["key_points"] == [{"point": "a", "evidence": "e", "quote": "", "terms": []}]
        assert doc["keywords"] == []
