"""Correctness is a string decision, so its edges get their own tests."""

from rag_stage_attribution.scoring import is_correct, normalise


def test_containment_is_on_word_boundaries_so_it_does_not_match_inside_items():
    # The bug this pins: 'IT' normalises to 'it', which is a substring of
    # 'items'; plain containment scored 'Items are approved' as naming IT.
    assert not is_correct("Items outside the catalog need a ticket", "IT")
    assert is_correct("These are approved by IT, not the manager", "IT")


def test_a_payload_that_closes_a_sentence_still_matches():
    # The bug this pins: '$25.' kept its sentence period in the token, so a
    # faithful answer whose fact ended the sentence failed the match.
    assert is_correct("Receipts are needed above $25.", "$25")


def test_interior_dots_in_amounts_survive_normalisation():
    assert normalise("$0.62 per mile.") == "$0.62 per mile"
    assert is_correct("the rate is $0.62 per mile", "$0.62")


def test_case_and_punctuation_do_not_matter():
    assert is_correct("WITHIN 30 DAYS!", "30 days")


def test_percent_signs_are_payload():
    assert is_correct("the canary serves 5 percent of traffic", "5 percent")
    assert normalise("20%") == "20%"


def test_a_wrong_number_never_matches():
    assert not is_correct("the limit is 45 days", "30 days")


def test_an_abstention_never_matches_a_payload():
    assert not is_correct("I cannot find this in the provided context.", "$25")
