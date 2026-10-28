"""The rewrite is pinned by the embedding fixture, so its behaviour is law."""

from rag_stage_attribution.retrieval.rewrite import REWRITE_TABLE, rewrite_query, rewrite_table_sha


def test_known_acronyms_are_expanded_in_place():
    assert rewrite_query("how much PTO do I get") == "how much PTO paid time off do I get"


def test_expansion_keeps_the_original_token():
    out = rewrite_query("is VPN required")
    assert "VPN" in out and "virtual private network" in out


def test_rewriting_twice_equals_rewriting_once():
    once = rewrite_query("PTO and SEV1 and VPN rules")
    assert rewrite_query(once) == once


def test_an_acronym_inside_a_word_is_left_alone():
    assert rewrite_query("the POTATO question") == "the POTATO question"


def test_a_query_without_acronyms_passes_through_unchanged():
    query = "how long do I have to turn in a purchase"
    assert rewrite_query(query) == query


def test_longer_acronyms_win_over_their_prefixes():
    # PO is a prefix of nothing here, but SEV1 must not be eaten by a shorter key.
    out = rewrite_query("a SEV1 and a PO")
    assert "severity one incident" in out and "purchase order" in out


def test_the_table_sha_is_stable_for_a_fixed_table():
    assert rewrite_table_sha() == rewrite_table_sha()


def test_the_fixture_manifest_pins_the_current_table(cfg):
    import json

    manifest = json.loads((cfg.embeddings_dir / "manifest.json").read_text())
    assert manifest["rewrite_table_sha"] == rewrite_table_sha()


def test_every_expansion_starts_with_its_own_acronym():
    assert all(expansion.startswith(key) for key, expansion in REWRITE_TABLE.items())
