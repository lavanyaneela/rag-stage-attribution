# Failure modes

What goes wrong, how it presents, and what the system does about it. The
convention throughout: a wrong answer is exit 1 territory, an unanswerable
question is exit 2 territory, and a crash is exit 3; the one thing never
allowed is a confident number standing on broken evidence.

## Fixture contradictions

| Failure | Presents as | Behaviour |
| --- | --- | --- |
| Golden span absent or duplicated in its document | corrupted fixture edit | loader refuses at read time with the qid and the count |
| Parametric belief contains the true answer | a "wrong" belief that scores correct | loader refuses; the bank may never be accidentally right |
| Chunker splits a golden span (possible for a foreign corpus) | no golden chunk under a strategy | the question routes to the refusal path; verdict exits 2 naming golden coverage |
| Embedding fixture drifted from corpus, questions, or rewrite table | silent stale vectors | store refuses on load via manifest digests; the error names the capture command |
| Fixture vectors not unit length | cosine scores quietly wrong | store refuses; digests cannot catch this, so the norm check is separate |

## Measurement integrity

| Failure | Presents as | Behaviour |
| --- | --- | --- |
| A document quotes a figure the code no longer produces | stale README claim | `check_numbers` fails the build on the exact sentence |
| A number in the README or the guide matches no metric value | an invented figure | the reverse pass flags it, including single digits outside anchors; a number equal to a measured value passes on value alone, the stated limit |
| An absolute duration lands in any tracked markdown | machine dependent claim | the duration sweep fails the build |
| The checker itself regresses | green runs on false claims | `prove_checker` breaks a figure on purpose in CI and must see it fail |
| Metrics regeneration is not deterministic | churn on every verify | a test regenerates metrics.json and asserts byte equality |

## Attribution limits

| Failure | Presents as | Behaviour |
| --- | --- | --- |
| Golden chunk ignored from the context tail | generation fault | known blind spot; ADR-004 rule may say retrieval_rank at low confidence; E2 prices it |
| Generator cites the chunk it ignored | contradictory evidence | citation signal neutralised, support evidence decides, confidence drops |
| Answer supported by nothing provided | fabrication | generation verdict via the unsupported floor |
| Too few attributable failures for a verdict | small sample | exit 2 with the policy floor named |
| Too many unattributable failures | evidence poverty | exit 2 with the share and the line |

## Operational

| Failure | Presents as | Behaviour |
| --- | --- | --- |
| make folding exit codes | every failure looks like exit 2 | documented: automation branches on `stageattr`, not on make |
| Crash inside the CLI | exit 3 with a one line error | `RSA_DEBUG=1` re-raises with the traceback |
| Live adapter without its environment | partial configuration | constructor refuses, naming the missing variables |
