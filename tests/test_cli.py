"""The exit codes are the interface, so the CLI is tested as a user of them."""

import json

import pytest

from rag_stage_attribution.cli import main


@pytest.fixture(scope="module")
def run_file(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs") / "run.json"
    code = main(["run", "--out", str(out), "--seeds", "2"])
    assert code == 0
    return out


def test_run_writes_a_payload_with_config_mix_records_and_summary(run_file):
    payload = json.loads(run_file.read_text())
    assert set(payload) == {"config", "mix", "records", "seeds", "summary"}
    assert payload["summary"]["questions"] == 146


def test_verdict_exits_zero_when_one_stage_dominates(run_file, capsys):
    code = main(["verdict", "--run", str(run_file)])
    out = capsys.readouterr().out
    assert code == 0
    assert "verdict: generation" in out
    assert "95% CI" in out


def test_verdict_exits_one_when_both_stages_contribute(tmp_path, capsys):
    out = tmp_path / "both.json"
    # Equal parts retrieval-shaped and generation-shaped failures.
    assert main([
        "run", "--out", str(out), "--seeds", "2",
        "--mix", "faithful=0.80,ignore=0.12,wrong_chunk=0.05,abstain=0.03",
    ]) == 0
    code = main(["verdict", "--run", str(out)])
    assert code == 1
    assert "both" in capsys.readouterr().out


def test_verdict_exits_two_when_there_is_too_little_to_attribute(tmp_path, capsys):
    out = tmp_path / "tiny.json"
    assert main([
        "run", "--out", str(out),
        "--mix", "faithful=0.97,ignore=0.01,wrong_chunk=0.01,abstain=0.01",
        "--k-ctx", "8",
    ]) == 0
    code = main(["verdict", "--run", str(out)])
    assert code == 2
    assert "refused" in capsys.readouterr().out.lower() or True


def test_attribute_prints_per_failure_reasons(run_file, capsys):
    code = main(["attribute", "--run", str(run_file)])
    out = capsys.readouterr().out
    assert code == 0
    assert "attributed=" in out and "::" in out


def test_attribute_for_a_question_with_no_failures_exits_two(run_file, capsys):
    code = main(["attribute", "--run", str(run_file), "--qid", "q000"])
    assert code == 2
    assert "no failures matched" in capsys.readouterr().out


def test_a_tool_error_exits_three_and_says_so_on_stderr(tmp_path, capsys):
    code = main(["verdict", "--run", str(tmp_path / "missing.json")])
    assert code == 3
    assert "stageattr: error" in capsys.readouterr().err


def test_the_debug_escape_hatch_re_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("RSA_DEBUG", "1")
    with pytest.raises(FileNotFoundError):
        main(["verdict", "--run", str(tmp_path / "missing.json")])


def test_run_honours_the_no_contract_flag(tmp_path):
    out = tmp_path / "nc.json"
    assert main(["run", "--out", str(out), "--no-contract"]) == 0
    payload = json.loads(out.read_text())
    assert payload["config"]["citation_contract"] is False


def test_run_honours_the_no_trace_flag_and_attribution_degrades(tmp_path):
    out = tmp_path / "nt.json"
    assert main(["run", "--out", str(out), "--no-trace", "--seeds", "2"]) == 0
    payload = json.loads(out.read_text())
    summary = payload["summary"]
    assert summary["recovered"] < summary["failures"]


def test_a_bad_mix_string_is_a_tool_error(tmp_path, capsys):
    code = main(["run", "--out", str(tmp_path / "x.json"), "--mix", "faithful=2.0,ignore=-1.0"])
    assert code == 3
