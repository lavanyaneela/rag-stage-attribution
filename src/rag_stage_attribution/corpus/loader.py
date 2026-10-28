"""Load the committed corpus and golden set, and refuse the broken ones.

Consistency checks live here, on the read path, not inside any one consumer's
constructor. A golden span that does not appear exactly once in its document
would silently corrupt every downstream measurement, so loading is the moment
to catch it: everything that reads the corpus reads it through this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class CorpusError(ValueError):
    """The committed fixture contradicts itself."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    paragraphs: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


@dataclass(frozen=True)
class GoldenQuestion:
    qid: str
    question: str
    doc_id: str
    span: str  # an exact sentence from the document
    answer: str  # the key phrase a correct answer must contain


def load_corpus(corpus_dir: Path) -> list[Document]:
    docs: list[Document] = []
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise CorpusError(f"no corpus documents under {corpus_dir}")
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        if not lines or not lines[0].startswith("# "):
            raise CorpusError(f"{path.name}: first line must be a '# ' title")
        title = lines[0][2:].strip()
        body = "\n".join(lines[1:]).strip()
        paragraphs = tuple(p.strip() for p in body.split("\n\n") if p.strip())
        if len(paragraphs) < 3:
            raise CorpusError(f"{path.name}: a document needs at least 3 paragraphs")
        docs.append(Document(doc_id=path.stem, title=title, paragraphs=paragraphs))
    return docs


def load_golden(golden_path: Path, docs: list[Document]) -> list[GoldenQuestion]:
    by_id = {d.doc_id: d for d in docs}
    questions: list[GoldenQuestion] = []
    seen: set[str] = set()
    for line in golden_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        q = GoldenQuestion(row["qid"], row["question"], row["doc_id"], row["span"], row["answer"])
        if q.qid in seen:
            raise CorpusError(f"{q.qid}: duplicate qid in golden set")
        seen.add(q.qid)
        doc = by_id.get(q.doc_id)
        if doc is None:
            raise CorpusError(f"{q.qid}: unknown doc_id {q.doc_id!r}")
        occurrences = doc.text.count(q.span)
        if occurrences != 1:
            raise CorpusError(
                f"{q.qid}: span occurs {occurrences} times in {q.doc_id}, must be exactly 1"
            )
        hosting = [p for p in doc.paragraphs if q.span in p]
        if len(hosting) != 1:
            raise CorpusError(f"{q.qid}: span must sit inside exactly one paragraph")
        if q.answer not in q.span:
            raise CorpusError(f"{q.qid}: answer {q.answer!r} is not a substring of its span")
        questions.append(q)
    if not questions:
        raise CorpusError(f"golden set at {golden_path} is empty")
    return questions


def load_parametric(parametric_path: Path, questions: list[GoldenQuestion]) -> dict[str, dict]:
    """The stand-in's committed wrong beliefs, one per golden question.

    The attributor never reads this file. It is the injector's script, and
    giving the judge the defendant's diary would make every measurement here
    circular. The runner enforces that separation with a test, not a comment.
    """
    from rag_stage_attribution.scoring import is_correct

    qids = {q.qid for q in questions}
    answers = {q.qid: q.answer for q in questions}
    bank: dict[str, dict] = {}
    for line in parametric_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        qid = row["qid"]
        if qid not in qids:
            raise CorpusError(f"parametric entry for unknown qid {qid!r}")
        if is_correct(row["statement"], answers[qid]) or is_correct(
            row["wrong_answer"], answers[qid]
        ):
            raise CorpusError(f"{qid}: parametric belief contains the true answer")
        bank[qid] = {
            "wrong_answer": row["wrong_answer"],
            "preamble": row["preamble"],
            "statement": row["statement"],
        }
    missing = qids - set(bank)
    if missing:
        raise CorpusError(f"parametric bank missing qids: {sorted(missing)[:3]}")
    return bank
