"""Capture real embeddings once, commit them, and never embed again.

This is the only file in the repository that touches the network, and it is
behind the 'capture' extra so a fresh clone can run everything else without
it. It downloads the MiniLM encoder from Hugging Face, embeds every chunk of
every chunking strategy plus every golden query in its rewritten form, L2
normalises, and writes .npz fixtures with a manifest that pins the model, the
input digests, and the rewrite table. Excluded from coverage because CI is
offline; the store verifies its outputs on every single load instead.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_FILES = ("tokenizer.json", "onnx/model.onnx")


def _embed(texts: list[str], tokenizer, session) -> np.ndarray:
    enc = tokenizer.encode_batch(texts)
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    out = session.run(
        None,
        {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)},
    )[0]
    weights = mask[..., None].astype(np.float32)
    pooled = (out * weights).sum(axis=1) / weights.sum(axis=1)
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled.astype(np.float32)


def _batched(texts: list[str], tokenizer, session, batch: int = 32) -> np.ndarray:
    parts = [_embed(texts[i : i + batch], tokenizer, session) for i in range(0, len(texts), batch)]
    return np.concatenate(parts, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("fixtures/embeddings"))
    args = parser.parse_args()

    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    from rag_stage_attribution.config import CHUNKING_STRATEGIES, PipelineConfig
    from rag_stage_attribution.corpus import chunk_corpus, load_corpus, load_golden
    from rag_stage_attribution.embeddings.store import text_digest
    from rag_stage_attribution.retrieval.rewrite import rewrite_query, rewrite_table_sha

    local = {name: hf_hub_download(MODEL_REPO, name) for name in MODEL_FILES}
    model_sha = hashlib.sha256(Path(local["onnx/model.onnx"]).read_bytes()).hexdigest()[:16]
    tokenizer = Tokenizer.from_file(local["tokenizer.json"])
    tokenizer.enable_padding()
    tokenizer.enable_truncation(256)
    session = ort.InferenceSession(local["onnx/model.onnx"], providers=["CPUExecutionProvider"])

    cfg = PipelineConfig()
    docs = load_corpus(cfg.corpus_dir)
    golden = load_golden(cfg.golden_path, docs)
    args.out.mkdir(parents=True, exist_ok=True)

    chunk_digests = {}
    for strategy in CHUNKING_STRATEGIES:
        chunks = chunk_corpus(docs, strategy)
        vectors = _batched([c.text for c in chunks], tokenizer, session)
        np.savez_compressed(
            args.out / f"chunks_{strategy}.npz",
            ids=np.array([c.chunk_id for c in chunks]),
            vectors=vectors,
        )
        chunk_digests[strategy] = text_digest([c.text for c in chunks])
        print(f"captured {strategy}: {vectors.shape[0]} chunks x {vectors.shape[1]} dims")

    rewritten = [rewrite_query(q.question) for q in golden]
    qvecs = _batched(rewritten, tokenizer, session)
    np.savez_compressed(
        args.out / "queries.npz",
        ids=np.array([q.qid for q in golden]),
        vectors=qvecs,
    )
    print(f"captured queries: {qvecs.shape[0]} x {qvecs.shape[1]}")

    manifest = {
        "model_repo": MODEL_REPO,
        "model_onnx_sha256_16": model_sha,
        "pooling": "attention-mask weighted mean, L2 normalised",
        "max_tokens": 256,
        "dims": int(qvecs.shape[1]),
        "captured_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
        "rewrite_table_sha": rewrite_table_sha(),
        "chunk_digests": chunk_digests,
        "query_digest": text_digest(rewritten),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("manifest written")


if __name__ == "__main__":
    main()
