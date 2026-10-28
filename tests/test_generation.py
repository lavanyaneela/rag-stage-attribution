"""The stand-in is the experiment's actor; its contract is pinned per mode."""

import random

import pytest

from rag_stage_attribution.generation import ABSTAIN_TEXT, SeededStandIn, paraphrase
from rag_stage_attribution.generation.standin import CITE_GOLDEN_ANYWAY, MODES
from rag_stage_attribution.scoring import is_correct


def _context_for(resources, qid, with_golden=True):
    golden_ids = resources.golden_ids_by_qid[qid]
    ctx = [
        (cid, text) for cid, text in resources.chunk_texts.items() if cid not in golden_ids
    ][:4]
    if with_golden and not any(cid in golden_ids for cid, _ in ctx):
        gid = next(iter(golden_ids))
        ctx[-1] = (gid, resources.chunk_texts[gid])
    return ctx


@pytest.fixture(scope="module")
def question(resources):
    return resources.golden[0]


def test_paraphrase_never_touches_digits_or_currency():
    rng = random.Random(3)
    sentence = "Receipts are required for any expense above $25 within 30 days."
    for _ in range(20):
        out = paraphrase(sentence, rng)
        assert "$25" in out and "30" in out


def test_paraphrase_protects_the_callers_phrase_word_for_word():
    rng = random.Random(5)
    sentence = "Business class requires VP approval regardless of duration."
    for _ in range(20):
        out = paraphrase(sentence, rng, protect="VP approval")
        assert "VP approval" in out


def test_paraphrase_actually_drifts_for_some_seed():
    sentence = "Expense reports are approved by the direct manager and audited quarterly."
    outs = {paraphrase(sentence, random.Random(seed)) for seed in range(12)}
    assert len(outs) > 1
    assert any(out != sentence for out in outs)


def test_paraphrase_is_deterministic_for_a_fixed_seed():
    sentence = "Claims submitted after 30 days require written approval."
    assert paraphrase(sentence, random.Random(9)) == paraphrase(sentence, random.Random(9))


def test_an_unknown_mode_is_rejected(resources, question):
    standin = SeededStandIn(resources.bank, True, seed=1)
    with pytest.raises(ValueError, match="unknown stand-in mode"):
        standin.generate(question, [], frozenset(), "hallucinate")


def test_faithful_with_the_golden_chunk_present_is_always_scored_correct(resources):
    standin = SeededStandIn(resources.bank, True, seed=4)
    for q in resources.golden:
        ctx = _context_for(resources, q.qid, with_golden=True)
        out = standin.generate(q, ctx, resources.golden_ids_by_qid[q.qid], "faithful")
        assert is_correct(out.answer, q.answer), q.qid
        assert out.citations and out.citations[0] in resources.golden_ids_by_qid[q.qid]


def test_faithful_without_the_golden_chunk_answers_from_the_best_available(resources, question):
    standin = SeededStandIn(resources.bank, True, seed=4)
    ctx = _context_for(resources, question.qid, with_golden=False)
    out = standin.generate(question, ctx, resources.golden_ids_by_qid[question.qid], "faithful")
    assert out.citations and out.citations[0] in {cid for cid, _ in ctx}
    assert not is_correct(out.answer, question.answer)


def test_faithful_with_an_empty_context_abstains(resources, question):
    standin = SeededStandIn(resources.bank, True, seed=4)
    out = standin.generate(question, [], resources.golden_ids_by_qid[question.qid], "faithful")
    assert out.answer == ABSTAIN_TEXT


def test_ignore_answers_are_never_scored_correct(resources):
    standin = SeededStandIn(resources.bank, True, seed=4)
    for q in resources.golden:
        ctx = _context_for(resources, q.qid, with_golden=True)
        out = standin.generate(q, ctx, resources.golden_ids_by_qid[q.qid], "ignore")
        assert not is_correct(out.answer, q.answer), q.qid


def test_ignore_without_the_contract_cites_nothing(resources, question):
    standin = SeededStandIn(resources.bank, False, seed=4)
    ctx = _context_for(resources, question.qid, with_golden=True)
    out = standin.generate(question, ctx, resources.golden_ids_by_qid[question.qid], "ignore")
    assert out.citations == ()


def test_ignore_under_the_contract_sometimes_cites_the_chunk_it_ignored(resources):
    standin = SeededStandIn(resources.bank, True, seed=11)
    golden_cites = other_cites = 0
    for q in resources.golden:
        golden_ids = resources.golden_ids_by_qid[q.qid]
        ctx = _context_for(resources, q.qid, with_golden=True)
        out = standin.generate(q, ctx, golden_ids, "ignore")
        assert len(out.citations) == 1
        if out.citations[0] in golden_ids:
            golden_cites += 1
        else:
            other_cites += 1
    share = golden_cites / (golden_cites + other_cites)
    assert 0.05 < share < 0.6, f"cite-golden share {share} far from {CITE_GOLDEN_ANYWAY}"


def test_wrong_chunk_cites_a_non_golden_chunk_under_the_contract(resources, question):
    standin = SeededStandIn(resources.bank, True, seed=4)
    golden_ids = resources.golden_ids_by_qid[question.qid]
    ctx = _context_for(resources, question.qid, with_golden=True)
    out = standin.generate(question, ctx, golden_ids, "wrong_chunk")
    assert out.citations and out.citations[0] not in golden_ids


def test_wrong_chunk_with_an_empty_context_abstains(resources, question):
    standin = SeededStandIn(resources.bank, True, seed=4)
    out = standin.generate(question, [], resources.golden_ids_by_qid[question.qid], "wrong_chunk")
    assert out.answer == ABSTAIN_TEXT


def test_abstain_declines_and_cites_nothing(resources, question):
    standin = SeededStandIn(resources.bank, True, seed=4)
    ctx = _context_for(resources, question.qid, with_golden=True)
    out = standin.generate(question, ctx, resources.golden_ids_by_qid[question.qid], "abstain")
    assert out.answer == ABSTAIN_TEXT and out.citations == ()


def test_the_stand_in_is_deterministic_per_seed(resources, question):
    ctx = _context_for(resources, question.qid, with_golden=True)
    golden_ids = resources.golden_ids_by_qid[question.qid]
    a = SeededStandIn(resources.bank, True, seed=8).generate(question, ctx, golden_ids, "ignore")
    b = SeededStandIn(resources.bank, True, seed=8).generate(question, ctx, golden_ids, "ignore")
    assert a == b


def test_mode_names_match_the_injection_vocabulary():
    assert MODES == ("faithful", "ignore", "wrong_chunk", "abstain")
