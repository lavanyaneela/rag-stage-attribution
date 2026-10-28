"""The report is derived from experiment outputs, so deriving it is tested."""

from pathlib import Path

from rag_stage_attribution.reporting import EXPERIMENTS_DIR, write_report


def test_the_report_renders_from_the_committed_experiment_outputs(tmp_path):
    out = tmp_path / "report.md"
    write_report(out, EXPERIMENTS_DIR)
    text = out.read_text()
    assert "## E1" in text and "## E5" in text
    assert "| arm |" in text


def test_the_report_quotes_no_absolute_duration(tmp_path):
    out = tmp_path / "report.md"
    write_report(out, EXPERIMENTS_DIR)
    text = out.read_text()
    assert " ms" not in text
    assert "asserted by a test" in text


def test_the_committed_report_matches_a_fresh_render():
    committed = Path(EXPERIMENTS_DIR) / "report.md"
    if not committed.exists():
        import pytest

        pytest.skip("report not rendered yet")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "report.md"
        write_report(fresh, EXPERIMENTS_DIR)
        assert fresh.read_text() == committed.read_text()
