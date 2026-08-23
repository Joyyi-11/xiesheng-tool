"""Tests for the pydantic struct models used to parse LLM output."""

from src.models.llm_struct import parse_struct, struct_to_doc


class TestParseStruct:
    def test_parses_valid_payload(self):
        struct = parse_struct(
            {
                "key_points": [{"point": "观点", "evidence": "证据"}],
                "highlight_quotes": [" 闪光句 "],
                "speaker_intro": "**主播**：Nina",
                "speaker_mapping": {"SPEAKER_00": "Nina"},
                "keywords": [{"key": " AI ", "desc": " d "}],
                "outline": [{"title": " 主题 ", "start": "00:05:00", "points": [" 子点1 ", "子点2"]}],
            }
        )
        assert struct.key_points[0].point == "观点"
        assert struct.highlight_quotes == ["闪光句"]
        assert struct.speaker_intro == "**主播**：Nina"
        assert struct.speaker_mapping == {"SPEAKER_00": "Nina"}
        assert struct.keywords[0].key == "AI"
        assert struct.keywords[0].desc == "d"
        assert struct.outline[0].title == "主题"
        assert struct.outline[0].start == "00:05:00"
        assert struct.outline[0].points == ["子点1", "子点2"]

    def test_none_returns_empty_struct(self):
        struct = parse_struct(None)
        assert struct.key_points == []
        assert struct.keywords == []
        assert struct.outline == []
        assert struct.speaker_mapping == {}
        assert struct.speaker_intro == ""

    def test_malformed_items_dropped_but_rest_kept(self):
        struct = parse_struct(
            {
                "key_points": [{"point": "好"}, "bad", {"evidence": "no point"}],
                "highlight_quotes": ["好", 123, "   ", "坏"],
                "keywords": ["bad", {"key": "", "desc": "空"}],
                "outline": ["bad", {"title": ""}, {"title": "有效主题", "points": ["好", 123, ""]}],
                "speaker_intro": 123,
                "speaker_mapping": "not-a-dict",
            }
        )
        assert [k.point for k in struct.key_points] == ["好", ""]
        assert struct.highlight_quotes == ["好", "123", "坏"]
        assert len(struct.keywords) == 0
        assert [o.title for o in struct.outline] == ["有效主题"]
        assert struct.outline[0].points == ["好", "123"]
        assert struct.speaker_intro == ""
        assert struct.speaker_mapping == {}

    def test_caps_detected_lists(self):
        many = [{"point": f"p{i}"} for i in range(50)]
        quotes = [f"q{i}" for i in range(50)]
        struct = parse_struct({"key_points": many, "highlight_quotes": quotes})
        assert len(struct.key_points) == 20  # MAX_KEY_POINTS（不限条数，宽容上限）
        assert len(struct.highlight_quotes) == 20  # MAX_QUOTES（不限条数，宽容上限）

    def test_outline_start_parsing(self):
        struct = parse_struct(
            {"outline": [
                {"title": "含时", "start": "01:05:00"},
                {"title": "秒数", "start": 349},
                {"title": "无时", "start": None},
            ]}
        )
        assert struct.outline[0].start == "01:05:00"
        assert struct.outline[1].start == 349
        assert struct.outline[2].start is None


class TestStructToDoc:
    def test_round_trips_clean_payload(self):
        raw = {
            "key_points": [{"point": "观点", "evidence": "证据"}],
            "highlight_quotes": ["引文"],
            "speaker_intro": "intra",
            "speaker_mapping": {"SPEAKER_00": "A"},
            "keywords": [{"key": "词", "desc": "d"}],
            "outline": [{"title": "主题", "start": "00:01:00", "points": ["子点"]}],
        }
        doc = struct_to_doc(parse_struct(raw))
        assert doc["key_points"][0]["point"] == "观点"
        assert doc["highlight_quotes"] == ["引文"]
        assert doc["speaker_mapping"] == {"SPEAKER_00": "A"}
        assert doc["keywords"][0]["key"] == "词"
        assert doc["outline"] == [{"title": "主题", "start": "00:01:00", "points": ["子点"]}]

    def test_strips_whitespace_and_drops_invalid(self):
        doc = struct_to_doc(
            parse_struct(
                {
                    "key_points": [{"point": "  a  ", "evidence": "  e  "}, {"point": "", "evidence": "x"}],
                    "highlight_quotes": ["  x  ", ""],
                    "keywords": [{"key": "", "desc": "d"}],
                }
            )
        )
        assert doc["key_points"] == [{"point": "a", "evidence": "e"}]
        assert doc["highlight_quotes"] == ["x"]
        assert doc["keywords"] == []
