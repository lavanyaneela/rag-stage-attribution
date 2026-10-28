# ADR-003: Chunking strategy, and why the golden label is a span

Status: accepted.

## Context

Chunking changes everything downstream: what the retriever can find, what
rank the right content sits at, and therefore which stage appears to fail.
A repository about blame cannot treat the chunker as a preprocessing detail,
and it cannot hard-wire golden labels to chunk ids, because chunk ids are an
artifact of the strategy under test.

## Decision

Three strategies ship and E4 measures the blame mix under each:

- paragraph: one chunk per paragraph, the default. 112 chunks on this
  corpus; the golden content for a question is exactly one chunk.
- window2: two paragraphs with stride one. Interior paragraphs appear in
  two chunks, so golden membership is a set, which is the honest shape once
  any overlap exists.
- fixed160: 160 word windows with stride 120, ignoring paragraph breaks.
  The strategy that can cut a sentence away from its context, priced rather
  than avoided.

The golden label is a span: an exact sentence of the source document. Golden
chunks are recomputed per strategy as the chunks containing that sentence,
on whitespace normalised text because fixed windows rejoin words. The window
and stride are chosen so the overlap exceeds the longest golden span, a
property a test pins, so a span can never fall between windows for the
committed fixture. A span no chunk contains routes the question to the
refusal path (exit code 2), because retrieval cannot be judged against a
chunk that does not exist.

## The default, and what it costs

Paragraph chunking is the default because the corpus is authored policy
text whose paragraphs are already topical units, and because a one chunk
golden set makes every rank statement in the trace unambiguous. The cost is granularity: 112 paragraph chunks make ranking
harder than 44 fixed-window chunks, and E4 shows the price as a higher retrieval share of blame, with 10.2% of blame
on retrieval misses under paragraph while fixed160 moves it to 8.4%. Bigger
chunks retrieve easier and bury the payload deeper; the blame mix is where
that tradeoff becomes visible.
