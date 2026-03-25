"""Tests for the CLI interface."""

import io
import json
import sys

import pytest

from chunkweaver.cli import main


def _capture_stdout(argv: list[str]) -> str:
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        main(argv)
    finally:
        sys.stdout = old
    return buf.getvalue()


def _capture_stderr(argv: list[str]) -> str:
    buf = io.StringIO()
    old = sys.stderr
    sys.stderr = buf
    try:
        main(argv)
    finally:
        sys.stderr = old
    return buf.getvalue()


class TestCLIText:
    def test_file_input(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("# Title\nContent here.\n# Next\nMore content.")

        output = _capture_stdout(
            [
                str(f),
                "--size", "5000",
                "--boundaries", r"^#{1,6}\s",
                "--min-size", "0",
                "--overlap", "0",
            ]
        )
        assert "Title" in output
        assert "Content" in output


class TestCLIJson:
    def test_json_output(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("Hello world.\n\nAnother paragraph.")

        output = _capture_stdout(
            [str(f), "--format", "json", "--size", "5000", "--overlap", "0"]
        )
        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "text" in data[0]
        assert "start" in data[0]

    def test_json_output_includes_boundary_level(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("Hello world.\n\nAnother paragraph.")

        output = _capture_stdout(
            [str(f), "--format", "json", "--size", "5000", "--overlap", "0"]
        )
        data = json.loads(output)
        assert "boundary_level" in data[0]

    def test_jsonl_output(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("# A\nContent.\n# B\nMore.")

        output = _capture_stdout(
            [
                str(f),
                "--format", "jsonl",
                "--size", "5000",
                "--boundaries", r"^#{1,6}\s",
                "--min-size", "0",
                "--overlap", "0",
            ]
        )
        lines = output.strip().split("\n")
        assert len(lines) >= 2
        for line in lines:
            obj = json.loads(line)
            assert "text" in obj


class TestCLIDetectBoundaries:
    def test_detect_mode(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("Preamble\nArticle 1\nStuff\nArticle 2\nMore")

        output = _capture_stdout(
            [str(f), "--detect-boundaries", "--boundaries", r"^Article\s+\d+"]
        )
        assert "Article 1" in output
        assert "Article 2" in output


class TestCLIPreset:
    def test_preset_flag(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("# Heading\nText.\n## Sub\nMore text.")

        output = _capture_stdout(
            [
                str(f),
                "--preset", "markdown",
                "--size", "5000",
                "--min-size", "0",
                "--overlap", "0",
            ]
        )
        assert "Heading" in output


# -------------------------------------------------------------------
# --export-dir
# -------------------------------------------------------------------


class TestCLIExportDir:
    def test_export_creates_chunk_files(self, tmp_path):
        src = tmp_path / "input.txt"
        src.write_text("# A\nFirst section.\n# B\nSecond section.")
        out = tmp_path / "chunks"

        _capture_stderr(
            [
                str(src),
                "--export-dir", str(out),
                "--boundaries", r"^#{1,6}\s",
                "--size", "5000",
                "--min-size", "0",
                "--overlap", "0",
            ]
        )

        files = sorted(out.iterdir())
        assert len(files) >= 2
        assert files[0].name == "chunk-0000.txt"
        assert files[1].name == "chunk-0001.txt"
        content = files[0].read_text(encoding="utf-8")
        assert "First section" in content or "A" in content

    def test_export_summary_on_stderr(self, tmp_path):
        src = tmp_path / "input.txt"
        src.write_text("Hello world. Some content here.")
        out = tmp_path / "chunks"

        stderr = _capture_stderr(
            [str(src), "--export-dir", str(out), "--size", "5000", "--overlap", "0"]
        )
        assert "Exported" in stderr
        assert "chunks" in stderr

    def test_export_errors_on_nonempty_dir(self, tmp_path):
        src = tmp_path / "input.txt"
        src.write_text("Hello world.")
        out = tmp_path / "chunks"
        out.mkdir()
        (out / "existing.txt").write_text("already here")

        with pytest.raises(SystemExit):
            _capture_stderr(
                [str(src), "--export-dir", str(out), "--size", "5000", "--overlap", "0"]
            )

    def test_export_creates_dir_if_missing(self, tmp_path):
        src = tmp_path / "input.txt"
        src.write_text("Some content for chunking.")
        out = tmp_path / "nested" / "chunks"

        _capture_stderr(
            [str(src), "--export-dir", str(out), "--size", "5000", "--overlap", "0"]
        )
        assert out.is_dir()
        assert any(out.iterdir())


# -------------------------------------------------------------------
# --recommend --format json
# -------------------------------------------------------------------


class TestCLIRecommendJson:
    def test_recommend_json_valid(self, tmp_path):
        src = tmp_path / "input.txt"
        src.write_text(
            "Article 1\nSome legal text about obligations.\n\n"
            "Article 2\nMore legal text about rights and duties.\n"
        )

        output = _capture_stdout(
            [str(src), "--recommend", "--format", "json"]
        )
        data = json.loads(output)
        assert "suggested_target_size" in data
        assert "recommended_presets" in data
        assert "char_count" in data

    def test_recommend_text_still_works(self, tmp_path):
        src = tmp_path / "input.txt"
        src.write_text("Hello world.\n\nAnother paragraph.\n")

        output = _capture_stdout([str(src), "--recommend"])
        assert "chunkweaver recommend" in output


# -------------------------------------------------------------------
# --inspect --format json
# -------------------------------------------------------------------


class TestCLIInspectJson:
    def test_inspect_json_valid(self, tmp_path):
        src = tmp_path / "input.txt"
        src.write_text(
            "# Intro\nSome text here that is long enough to be a chunk.\n\n"
            "# Methods\nMore text describing methodology and approach.\n"
        )

        output = _capture_stdout(
            [
                str(src),
                "--inspect",
                "--format", "json",
                "--size", "5000",
                "--overlap", "0",
            ]
        )
        data = json.loads(output)
        assert "chunk_count" in data
        assert "fallback_ratio" in data
        assert "boundary_counts" in data

    def test_inspect_text_still_works(self, tmp_path):
        src = tmp_path / "input.txt"
        src.write_text("Hello world.\n\nAnother paragraph.\n")

        output = _capture_stdout(
            [str(src), "--inspect", "--size", "5000", "--overlap", "0"]
        )
        assert "chunkweaver inspect" in output
