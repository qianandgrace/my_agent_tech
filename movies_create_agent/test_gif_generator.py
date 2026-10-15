"""gif_generator.py 纯逻辑函数的单元测试。

跑法: python -m pytest movies_create_agent/test_gif_generator.py
"""
import io
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gif_generator import assemble_gif, extract_image_url, parse_numbered_prompts


def _png_bytes(color):
    img = Image.new("RGB", (10, 10), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestParseNumberedPrompts:
    def test_parses_standard_numbered_list(self):
        text = "Here are the prompts:\n1. A cat at a desk\n2. The cat dips the quill\n3. The cat writes\n4. The cat pauses\n5. The cat seals the letter"
        result = parse_numbered_prompts(text)
        assert result == [
            "A cat at a desk",
            "The cat dips the quill",
            "The cat writes",
            "The cat pauses",
            "The cat seals the letter",
        ]

    def test_handles_markdown_bold_numbers(self):
        text = "**1.** First scene\n**2.** Second scene\n**3.** Third\n**4.** Fourth\n**5.** Fifth"
        result = parse_numbered_prompts(text)
        assert result[0] == "First scene"
        assert len(result) == 5

    def test_handles_parenthesis_numbers(self):
        text = "1) Alpha\n2) Beta\n3) Gamma\n4) Delta\n5) Epsilon"
        result = parse_numbered_prompts(text)
        assert result[4] == "Epsilon"

    def test_ignores_intro_and_blank_lines(self):
        text = "Sure! Based on the plot:\n\n1. One\n2. Two\n3. Three\n4. Four\n5. Five\n\nAnything else?"
        result = parse_numbered_prompts(text)
        assert result == ["One", "Two", "Three", "Four", "Five"]

    def test_raises_when_fewer_than_expected(self):
        text = "1. Only\n2. Three\n3. Items"
        with pytest.raises(ValueError, match="Expected 5 prompts, got 3"):
            parse_numbered_prompts(text)

    def test_expected_count_is_configurable(self):
        text = "1. A\n2. B\n3. C"
        assert parse_numbered_prompts(text, expected=3) == ["A", "B", "C"]


class TestExtractImageUrl:
    # MaaS 网关 chat/completions 返回的真实结构（wan2.7-image）
    SAMPLE_RESPONSE = {
        "request_id": "132f4677",
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "image", "image": "https://example.com/img.png"}],
                    },
                    "finish_reason": "stop",
                }
            ],
            "finished": True,
        },
        "usage": {"image_count": 1},
    }

    def test_extracts_url_from_gateway_response(self):
        assert extract_image_url(self.SAMPLE_RESPONSE) == "https://example.com/img.png"

    def test_returns_none_when_no_image_part(self):
        data = {"output": {"choices": [{"message": {"content": [{"type": "text", "text": "no image"}]}}]}}
        assert extract_image_url(data) is None

    def test_returns_none_on_empty_response(self):
        assert extract_image_url({}) is None


class TestAssembleGif:
    def test_creates_playable_gif_with_all_frames(self):
        frames = [_png_bytes((255, 0, 0)), _png_bytes((0, 255, 0)), _png_bytes((0, 0, 255))]
        gif = assemble_gif(frames, frame_ms=500)
        assert isinstance(gif, bytes)
        img = Image.open(io.BytesIO(gif))
        assert img.format == "GIF"
        assert img.n_frames == 3
        assert img.info.get("loop") == 0

    def test_returns_none_for_empty_frames(self):
        assert assemble_gif([]) is None

    def test_resizes_large_frames_to_max_size(self):
        img = Image.new("RGB", (2048, 2048), (0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        gif = assemble_gif([buf.getvalue()], max_size=512)
        out = Image.open(io.BytesIO(gif))
        assert out.size == (512, 512)

    def test_keeps_small_frames_untouched(self):
        gif = assemble_gif([_png_bytes((255, 0, 0))], max_size=512)
        out = Image.open(io.BytesIO(gif))
        assert out.size == (10, 10)
