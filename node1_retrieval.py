#!/usr/bin/env python3
# ===========================================================
# FIX OpenMP conflicts on macOS (must be before ANY imports!)
# ===========================================================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ===========================================================

import gc
import time
import logging
import sqlite3
from queue import Queue, Empty
from threading import Thread
from typing import List, Dict, Optional

import faiss
import numpy as np
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests

# ---------------- CONFIG ----------------
NODE_1_IP = os.environ.get("NODE_1_IP", "localhost:8001")
NODE_2_IP = os.environ.get("NODE_2_IP", "localhost:8002")
NODE_0_IP = os.environ.get("NODE_0_IP", "localhost:8000")

FAISS_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "documents")
DOCUMENT_DB = os.path.join(DOCUMENTS_DIR, "documents.db")

RETRIEVAL_K = 10
RERANKER_MODEL = "BAAI/bge-reranker-base"
TRUNCATE_LENGTH = 512

NODE1_WAIT_TIME = float(os.environ.get("NODE1_WAIT_TIME", "0.02"))
BATCH_MAX = 8
WAIT_TIME = max(0.0, NODE1_WAIT_TIME)
MAX_FAISS_SUBBATCH = max(1, int(os.environ.get("NODE1_MAX_FAISS_SUBBATCH", 2)))

logger = logging.getLogger("Node1")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# ---------------- GLOBALS ----------------
faiss_index = None
tokenizer = None
reranker = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

request_queue = Queue()


# ===========================================================
# Load FAISS
# ===========================================================
def load_faiss():
    global faiss_index

    if not os.path.exists(FAISS_PATH):
        logger.error(f"[Node1] FAISS index missing: {FAISS_PATH}")
        raise SystemExit(1)

    logger.info(f"[Node1] Loading FAISS index from {FAISS_PATH} ...")
    faiss_index = faiss.read_index(FAISS_PATH)
    logger.info(f"[Node1] FAISS ready: {faiss_index.ntotal} vectors")

    gc.collect()


# ===========================================================
# Load Reranker
# ===========================================================
def load_reranker():
    global tokenizer, reranker

    logger.info(f"[Node1] Loading reranker: {RERANKER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    reranker = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL).to(
        device
    )
    reranker.eval()
    logger.info("[Node1] Reranker loaded.")


# ===========================================================
# Document Fetch from SQLite
# ===========================================================
def fetch_docs(doc_id_lists: List[List[int]]) -> List[List[Dict]]:
    if not os.path.exists(DOCUMENT_DB):
        logger.error(f"[Node1] SQLite DB missing: {DOCUMENT_DB}")
        return [[] for _ in doc_id_lists]

    conn = sqlite3.connect(DOCUMENT_DB)
    cur = conn.cursor()

    out = []
    for ids in doc_id_lists:
        docs = []
        for did in ids:
            cur.execute(
                "SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?",
                (int(did),),
            )
            row = cur.fetchone()
            if row:
                docs.append(
                    {
                        "doc_id": row[0],
                        "title": row[1],
                        "content": row[2],
                        "category": row[3],
                    }
                )
        out.append(docs)

    conn.close()
    return out


# ===========================================================
# Rerank
# ===========================================================
def rerank_batch(queries: List[str], docs_batch: List[List[Dict]]) -> List[List[Dict]]:
    outputs = []

    for query, docs in zip(queries, docs_batch):
        if not docs:
            outputs.append([])
            continue

        pairs = [[query, d["content"]] for d in docs]

        inputs = tokenizer(
            pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=TRUNCATE_LENGTH,
        ).to(device)

        with torch.no_grad():
            scores = reranker(**inputs).logits.squeeze(-1).cpu().numpy().tolist()

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        outputs.append([d for d, _ in ranked])

    return outputs


# ===========================================================
# Build context for LLM
# ===========================================================
def build_context(docs: List[Dict], top_k=3):
    return "\n".join([f"- {d['title']}: {d['content'][:200]}" for d in docs[:top_k]])


# ===========================================================
# Worker Thread
# ===========================================================
def worker_loop():
    logger.info("[Node1] Worker started.")

    while True:
        first = request_queue.get()
        if first is None:
            break

        batch = [first]
        first_enqueue_ts = first["timestamp"]
        t0 = time.time()
        logger.info(
            "[Node1] Worker picked first request_id=%s queue_size=%d enqueue_lag=%.3fs",
            first["request_id"],
            request_queue.qsize(),
            time.time() - first_enqueue_ts,
        )

        # opportunistic batching
        while len(batch) < BATCH_MAX:
            try:
                remain = WAIT_TIME - (time.time() - t0)
                if remain <= 0:
                    break
                nxt = request_queue.get(timeout=remain)
                batch.append(nxt)
            except Empty:
                break

        logger.info(
            "[Node1] Worker processing batch_size=%d ids=%s",
            len(batch),
            [b["request_id"] for b in batch],
        )

        queries = [x["query"] for x in batch]
        embeddings = np.array([x["embedding"] for x in batch], dtype="float32")

        contexts: List[Optional[str]] = [None] * len(batch)
        llm_duration: Optional[float] = None
        processing_start = time.time()

        for start in range(0, len(batch), MAX_FAISS_SUBBATCH):
            end = min(start + MAX_FAISS_SUBBATCH, len(batch))
            sub_queries = queries[start:end]
            sub_embeddings = embeddings[start:end]

            logger.info(
                "[Node1] Running FAISS search subbatch=%d-%d size=%d",
                start,
                end,
                len(sub_queries),
            )
            _, idxs = faiss_index.search(sub_embeddings, RETRIEVAL_K)

            docs_batch = fetch_docs(idxs.tolist())

            logger.info(
                "[Node1] Reranking documents for subbatch size=%d", len(sub_queries)
            )
            reranked = rerank_batch(sub_queries, docs_batch)

            chunk_contexts = [build_context(d) for d in reranked]
            for offset, ctx in enumerate(chunk_contexts):
                contexts[start + offset] = ctx

        if any(ctx is None for ctx in contexts):
            logger.error("[Node1] Missing context for some requests; marking as ERROR")
            contexts = [
                ctx if ctx is not None else "Context generation failed" for ctx in contexts
            ]

        payload = {"queries": queries, "contexts": contexts}
        try:
            llm_start = time.time()
            r = requests.post(f"http://{NODE_2_IP}/generate", json=payload, timeout=300)
            responses = r.json().get("responses", [])
            llm_duration = time.time() - llm_start
            logger.info(
                "[Node1] Node2 inference batch size=%d time=%.3fs",
                len(batch),
                llm_duration,
            )
        except Exception as e:
            logger.error(f"[Node1] ERROR sending to Node2: {e}")
            responses = ["ERROR"] * len(batch)
            llm_duration = None

        if len(responses) != len(batch):
            logger.error(
                "[Node1] Node2 response count mismatch expected=%d got=%d",
                len(batch),
                len(responses),
            )
            responses = (responses + ["ERROR"] * len(batch))[: len(batch)]

        # Callback Node0
        for req, out in zip(batch, responses):
            cb = {
                "request_id": req["request_id"],
                "generated_text": out,
                "llm_duration": llm_duration,
            }
            try:
                requests.post(f"http://{NODE_0_IP}/node2_callback", json=cb)
            except Exception as e:
                logger.error(f"[Node1] Callback Node0 error: {e}")

        for _ in batch:
            request_queue.task_done()

        total_processing_time = time.time() - processing_start
        logger.info(
            "[Node1] Batch finished size=%d total_time=%.3fs llm_time=%s",
            len(batch),
            total_processing_time,
            f"{llm_duration:.3f}s" if llm_duration is not None else "n/a",
        )


# ===========================================================
# API
# ===========================================================
@app.route("/search_and_rerank_batch", methods=["POST"])
def handle_batch():
    data = request.json or {}
    reqs = data.get("requests", [])

    logger.info(
        "[Node1] Received batch payload size=%d keys=%s",
        len(reqs),
        list(data.keys()),
    )

    for r in reqs:
        request_queue.put(
            {
                "request_id": r["request_id"],
                "query": r["query"],
                "embedding": r["embedding"],
                "timestamp": r["timestamp"],
            }
        )

    logger.info("[Node1] Queue size after enqueue=%d", request_queue.qsize())

    return jsonify({"status": "ok"})


# ===========================================================
# Main
# ===========================================================
def main():
    load_faiss()
    load_reranker()

    worker = Thread(target=worker_loop, daemon=True)
    worker.start()

    host, port = NODE_1_IP.split(":")
    logger.info(f"[Node1] Running on {host}:{port}")
    app.run(host=host, port=int(port), threaded=True)


if __name__ == "__main__":
    main()
