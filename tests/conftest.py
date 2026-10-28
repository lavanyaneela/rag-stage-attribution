"""Shared fixtures. Resources are built once per session because the fixture
loading they do is the expensive part, and every test that mutates anything
works on copies in tmp_path, never on the committed fixture."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rag_stage_attribution.config import PipelineConfig
from rag_stage_attribution.corpus import load_corpus, load_golden, load_parametric
from rag_stage_attribution.runner import build_resources

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture(scope="session")
def docs(cfg):
    return load_corpus(cfg.corpus_dir)


@pytest.fixture(scope="session")
def golden(cfg, docs):
    return load_golden(cfg.golden_path, docs)


@pytest.fixture(scope="session")
def bank(cfg, golden):
    return load_parametric(cfg.parametric_path, golden)


@pytest.fixture(scope="session")
def resources(cfg):
    return build_resources(cfg)


@pytest.fixture(scope="session")
def resources_no_contract():
    return build_resources(PipelineConfig(citation_contract=False))


@pytest.fixture()
def corpus_copy(tmp_path, cfg) -> Path:
    """A mutable copy of the committed corpus tree for corruption tests."""
    dest = tmp_path / "fixtures"
    shutil.copytree(cfg.root / "fixtures" / "corpus", dest / "corpus")
    shutil.copytree(cfg.root / "fixtures" / "golden", dest / "golden")
    shutil.copytree(cfg.root / "fixtures" / "parametric", dest / "parametric")
    return dest


def rewrite_jsonl(path: Path, transform) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = transform(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
