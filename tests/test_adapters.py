"""The live adapter must stay off the default path and keep its shape."""

import importlib
import sys

import pytest

from rag_stage_attribution.adapters.azure_openai import (
    REQUIRED_ENV,
    SYSTEM_PROMPT,
    build_messages,
    parse_citations,
)


def test_nothing_on_the_default_path_imports_the_live_adapter(resources):
    # Simulate the 'live' extra being absent entirely: the openai package is
    # unimportable, and the whole measured surface must keep working.
    blocked = sys.modules.pop("openai", None)
    sys.modules["openai"] = None  # type: ignore[assignment]
    try:
        from rag_stage_attribution.injection import InjectionMix
        from rag_stage_attribution.runner import run_suite

        records = run_suite(resources, InjectionMix(), seed=5)
        assert len(records) == len(resources.golden)
    finally:
        if blocked is not None:
            sys.modules["openai"] = blocked
        else:
            sys.modules.pop("openai", None)


def test_importing_the_adapter_module_itself_needs_no_openai_package():
    module = importlib.import_module("rag_stage_attribution.adapters.azure_openai")
    assert module.SYSTEM_PROMPT


def test_the_adapter_refuses_to_start_without_its_environment(monkeypatch):
    from rag_stage_attribution.adapters.azure_openai import AzureOpenAIAdapter

    for name in REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="environment variables"):
        AzureOpenAIAdapter()


def test_the_prompt_carries_the_citation_contract():
    assert "CITATIONS:" in SYSTEM_PROMPT
    assert "only" in SYSTEM_PROMPT


def test_build_messages_blocks_every_context_chunk_with_its_id(golden):
    messages = build_messages(golden[0], [("doc:p01", "text one"), ("doc:p02", "text two")])
    user = messages[1]["content"]
    assert "[doc:p01]" in user and "[doc:p02]" in user
    assert golden[0].question in user


def test_parse_citations_reads_only_known_chunk_ids():
    answer = "The limit is $25.\nCITATIONS: doc:p01 doc:p09"
    body, cited = parse_citations(answer, {"doc:p01"})
    assert body == "The limit is $25."
    assert cited == ("doc:p01",)


def test_parse_citations_without_the_marker_returns_no_citations():
    body, cited = parse_citations("just an answer", {"doc:p01"})
    assert body == "just an answer" and cited == ()


def test_parse_citations_handles_the_none_form():
    body, cited = parse_citations("answer\nCITATIONS: none", {"doc:p01"})
    assert cited == ()
