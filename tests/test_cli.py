"""Tests for the CLI interface."""

import json
import os
import tempfile

from chunkweaver.cli import main


class TestCLIText:
    def test_file_input(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("# Title\nContent here.\n# Next\nMore content.")

        import io
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main([str(f), "--size", "5000", "--boundaries", r"^#{1,6}\s",
                  "--min-size", "0", "--overlap", "0"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "Title" in output
        assert "Content" in output


class TestCLIJson:
    def test_json_output(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("Hello world.\n\nAnother paragraph.")

        import io
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main([str(f), "--format", "json", "--size", "5000", "--overlap", "0"])
        finally:
            sys.stdout = old_stdout

        data = json.loads(captured.getvalue())
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "text" in data[0]
        assert "start" in data[0]

    def test_jsonl_output(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("# A\nContent.\n# B\nMore.")

        import io
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main([str(f), "--format", "jsonl", "--size", "5000",
                  "--boundaries", r"^#{1,6}\s", "--min-size", "0", "--overlap", "0"])
        finally:
            sys.stdout = old_stdout

        lines = captured.getvalue().strip().split("\n")
        assert len(lines) >= 2
        for line in lines:
            obj = json.loads(line)
            assert "text" in obj


class TestCLIDetectBoundaries:
    def test_detect_mode(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("Preamble\nArticle 1\nStuff\nArticle 2\nMore")

        import io
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main([str(f), "--detect-boundaries", "--boundaries", r"^Article\s+\d+"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "Article 1" in output
        assert "Article 2" in output


class TestCLIPreset:
    def test_preset_flag(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("# Heading\nText.\n## Sub\nMore text.")

        import io
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main([str(f), "--preset", "markdown", "--size", "5000",
                  "--min-size", "0", "--overlap", "0"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "Heading" in output
