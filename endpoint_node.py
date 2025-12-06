#!/usr/bin/env python3
import os
import time
import logging
import threading
from queue import Queue, Empty
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import requests
import sqlite3
import torch

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline as hf_pipeline,
)

# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Node0")

app = Flask(__name__)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
NODE_0_IP = os.environ.get("NODE_0_IP", "localhost:8000")
NODE_1_IP = os.environ.get("NODE_1_IP", "localhost:8001")
NODE_2_IP = os.environ.get("NODE_2_IP", "localhost:8002")

FAISS_NODES = [NODE_1_IP, NODE_2_IP]  # round robin

DOCUMENTS_DB = os.environ.get("DOCUMENTS_DB", "documents/documents.db")

EMBED_BATCH_SIZE = 32
EMBED_BATCH_TIMEOUT = 0.05

LLM_BATCH_SIZE = 4  # optional batching

MAX_QUEUE_SIZE = 2000

# ------------------------------------------------------------
# GLOBAL STATE
# ------------------------------------------------------------
embedding_queue = Queue(maxsize=MAX_QUEUE_SIZE)
callback_queue = Queue(maxsize=MAX_QUEUE_SIZE)

results: Dict[str, Dict[str, Any]] = {}
results_lock = threading.Lock()

pending_queries: Dict[str, str] = {}
pending_q_lock = threading.Lock()

rr_idx = 0
rr_lock = threading.Lock()

# Models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_INT = 0 if torch.cuda.is_available() else -1

embedder = None
reranker_model = None
reranker_tok = None
llm_model = None
llm_tok = None
sent_pipe = None
safe_pipe = None


# ------------------------------------------------------------
# ROUND ROBIN
# ------------------------------------------------------------
def pick_faiss_node():
    global rr_idx
    with rr_lock:
        node = FAISS_NODES[rr_idx % len(FAISS_NODES)]
        rr_idx += 1
    return node


# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------
def load_models():
    global \
        embedder, \
        reranker_tok, \
        reranker_model, \
        llm_tok, \
        llm_model, \
        sent_pipe, \
        safe_pipe

    logger.info("Loading embedding model...")
    embedder = SentenceTransformer(
        "BAAI/bge-base-en-v1.5", device=("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger.info("Loading reranker...")
    reranker_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    reranker_model = (
        AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
        .to(DEVICE)
        .eval()
    )

    logger.info("Loading LLM...")
    llm_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    llm_model = (
        AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            dtype=dtype,
            use_cache=True,
        )
        .to(DEVICE)
        .eval()
    )

    logger.info("Loading sentiment & safety...")
    sent_pipe = hf_pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=DEVICE_INT,
    )
    safe_pipe = hf_pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=DEVICE_INT,
    )

    logger.info("All models loaded.")


# ------------------------------------------------------------
# FETCH DOCUMENTS
# ------------------------------------------------------------
def fetch_docs(doc_ids: List[int]):
    db = DOCUMENTS_DB
    if not os.path.exists(db):
        return []

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    placeholders = ",".join("?" for _ in doc_ids)
    q = f"SELECT doc_id, title, content FROM documents WHERE doc_id IN ({placeholders})"
    cur.execute(q, doc_ids)

    out = {row["doc_id"]: dict(row) for row in cur.fetchall()}
    conn.close()

    return [out[i] for i in doc_ids if i in out]


# ------------------------------------------------------------
# RERANK
# ------------------------------------------------------------
def rerank(query: str, docs: List[Dict]):
    if not docs:
        return []

    pairs = [[query, d["content"]] for d in docs]
    with torch.no_grad():
        tok = reranker_tok(
            pairs, truncation=True, padding=True, return_tensors="pt"
        ).to(DEVICE)
        logits = reranker_model(**tok).logits.squeeze(-1).cpu().numpy()

    scored = sorted(zip(docs, logits.tolist()), key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored]


# ------------------------------------------------------------
# LLM GENERATION
# ------------------------------------------------------------
def llm_generate(query: str, docs: List[Dict]):
    ctx = "\n".join([f"- {d['title']}: {d['content'][:200]}" for d in docs[:3]])

    messages = [
        {"role": "system", "content": "Answer as 'Answer: <final>'"},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"},
    ]

    text = llm_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = llm_tok([text], return_tensors="pt", truncation=True, max_length=512).to(
        DEVICE
    )

    with torch.no_grad():
        ids = llm_model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.01,
            pad_token_id=llm_tok.eos_token_id,
        )

    new_ids = ids[:, inputs.input_ids.shape[1] :]
    return llm_tok.batch_decode(new_ids, skip_special_tokens=True)[0]


# ------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------
def analyze(text: str):
    sent = sent_pipe(text[:512])[0]
    safe = safe_pipe(text[:512])[0]

    star_to_sent = {
        "1 star": "very negative",
        "2 stars": "negative",
        "3 stars": "neutral",
        "4 stars": "positive",
        "5 stars": "very positive",
    }

    sentiment = star_to_sent.get(sent["label"], "neutral")
    toxic = safe["score"] > 0.5

    return sentiment, toxic


# ------------------------------------------------------------
# EMBEDDING WORKER
# ------------------------------------------------------------
def embed_worker():
    logger.info("embed_worker started.")

    while True:
        batch = []
        try:
            first = embedding_queue.get()
            if first is None:
                break
            batch.append(first)

            start = time.time()
            while len(batch) < EMBED_BATCH_SIZE:
                remain = EMBED_BATCH_TIMEOUT - (time.time() - start)
                if remain <= 0:
                    break
                try:
                    batch.append(embedding_queue.get(timeout=remain))
                except Empty:
                    break

            batch_size = len(batch)
            logger.info(f"[Node0] Embedding batch size={batch_size}")

            queries = [b["query"] for b in batch]
            req_ids = [b["request_id"] for b in batch]

            t0 = time.time()
            embs = embedder.encode(
                queries, normalize_embeddings=True, convert_to_numpy=True
            )
            logger.info(
                "[Node0] Embedding finished batch_size=%d time=%.3fs",
                batch_size,
                time.time() - t0,
            )

            # round robin 发送到 FAISS node
            for i, r in enumerate(batch):
                node = pick_faiss_node()
                host, port = node.split(":")
                url = f"http://{host}:{port}/search"

                payload = {
                    "request_id": r["request_id"],
                    "embeddings": embs[i].tolist(),
                }

                try:
                    resp = requests.post(url, json=payload, timeout=3)
                    if resp.status_code != 202:
                        logger.error(
                            "[Node0] FAISS POST failed req_id=%s url=%s status=%s body=%s",
                            r["request_id"],
                            url,
                            resp.status_code,
                            resp.text,
                        )
                except Exception as e:
                    logger.error(
                        "[Node0] Error sending req_id=%s to %s: %s",
                        r["request_id"],
                        url,
                        e,
                    )
                    with results_lock:
                        results[r["request_id"]] = {"success": False, "error": str(e)}

            for _ in batch:
                embedding_queue.task_done()

        except Exception as e:
            logger.error(f"embed_worker error: {e}")


# ------------------------------------------------------------
# CALLBACK WORKER (LLM + RERANK + ANALYSIS)
# ------------------------------------------------------------
def callback_worker():
    logger.info("callback_worker (BATCH MODE) started.")

    BATCH_SIZE = 16  # 👈 你可以调大，比如 32
    BATCH_TIMEOUT = 0.05  # 👈 典型 50ms opportunistic batching

    while True:
        batch = []
        try:
            # ------- 1) blocking get（第一个 callback）-------
            first = callback_queue.get()
            if first is None:
                break
            batch.append(first)

            start_t = time.time()

            # ------- 2) opportunistic batching -------
            while len(batch) < BATCH_SIZE:
                remain = BATCH_TIMEOUT - (time.time() - start_t)
                if remain <= 0:
                    break

                try:
                    nxt = callback_queue.get(timeout=remain)
                    if nxt is None:
                        break
                    batch.append(nxt)
                except Empty:
                    break

            batch_size = len(batch)
            logger.info(f"Callback batch size = {batch_size}")

            # --------------------------------------------------------
            # batch 分解：收集所有 query + doc_ids
            # --------------------------------------------------------
            request_ids = [b["request_id"] for b in batch]
            queries = []
            doc_ids_batch = []

            with pending_q_lock:
                for item in batch:
                    req_id = item["request_id"]
                    q = pending_queries.get(req_id, None)
                    queries.append(q)
                    # Each FAISS result is List[List[int]] ，此处取第一个
                    doc_ids_batch.append(item["doc_ids"][0] if item["doc_ids"] else [])

            # --------------------------------------------------------
            # 3) fetch documents（批量）
            # --------------------------------------------------------
            t_fetch = time.time()
            all_docs_batch = []
            for doc_ids in doc_ids_batch:
                all_docs_batch.append(fetch_docs(doc_ids))
            logger.info(
                "[Node0] Fetch docs batch_size=%d time=%.3fs",
                batch_size,
                time.time() - t_fetch,
            )

            # --------------------------------------------------------
            # 4) rerank（批量）
            # --------------------------------------------------------
            t_rerank = time.time()
            reranked_batch = []
            for q, docs in zip(queries, all_docs_batch):
                reranked_batch.append(rerank(q, docs))
            logger.info(
                "[Node0] Rerank batch_size=%d time=%.3fs",
                batch_size,
                time.time() - t_rerank,
            )

            # --------------------------------------------------------
            # 5) LLM inference（批量）
            # --------------------------------------------------------
            t_llm = time.time()
            answers = []
            for q, docs in zip(queries, reranked_batch):
                answers.append(llm_generate(q, docs))
            logger.info(
                "[Node0] LLM batch_size=%d time=%.3fs",
                batch_size,
                time.time() - t_llm,
            )

            # --------------------------------------------------------
            # 6) sentiment + toxicity（批量并行）
            # --------------------------------------------------------
            t_analysis = time.time()
            truncated = [a[:512] for a in answers]

            def run_sentiment():
                return sent_pipe(truncated)

            def run_safety():
                return safe_pipe(truncated)

            with ThreadPoolExecutor(max_workers=2) as executor:
                sentiment_future = executor.submit(run_sentiment)
                safety_future = executor.submit(run_safety)
                sent_raw = sentiment_future.result()
                safe_raw = safety_future.result()
            logger.info(
                "[Node0] Analysis batch_size=%d time=%.3fs",
                batch_size,
                time.time() - t_analysis,
            )

            sent_map = {
                "1 star": "very negative",
                "2 stars": "negative",
                "3 stars": "neutral",
                "4 stars": "positive",
                "5 stars": "very positive",
            }

            sentiments = [sent_map.get(x["label"], "neutral") for x in sent_raw]
            toxics = ["true" if x["score"] > 0.5 else "false" for x in safe_raw]

            # --------------------------------------------------------
            # 7) 写入结果
            # --------------------------------------------------------
            logger.info(
                "[Node0] Callback batch size=%d complete total_time=%.3fs",
                batch_size,
                time.time() - start_t,
            )

            with results_lock:
                for rid, ans, s, tox in zip(request_ids, answers, sentiments, toxics):
                    results[rid] = {
                        "success": True,
                        "generated_response": ans,
                        "sentiment": s,
                        "is_toxic": tox,
                    }

        except Exception as e:
            logger.error(f"callback_worker batch error: {e}")

        finally:
            # 标记 batch 所有任务 done()
            for _ in batch:
                callback_queue.task_done()


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.route("/query", methods=["POST"])
def query_api():
    data = request.json or {}
    req_id = data.get("request_id")
    query = data.get("query")

    if not req_id or not query:
        return jsonify({"error": "missing request_id or query"}), 400

    with pending_q_lock:
        pending_queries[req_id] = query

    with results_lock:
        results.pop(req_id, None)

    embedding_queue.put({"request_id": req_id, "query": query})
    start = time.time()

    # wait for callback_worker to finish
    while time.time() - start < 300:
        with results_lock:
            if req_id in results:
                return jsonify(results.pop(req_id)), 200
        time.sleep(0.05)

    return jsonify({"error": "timeout"}), 504


@app.route("/retrieval_callback", methods=["POST"])
def retrieval_callback():
    data = request.json or {}
    rid = data.get("request_id")
    success = data.get("success")
    doc_ids = data.get("doc_ids", [])

    if not rid:
        return jsonify({"error": "missing request_id"}), 400

    if not success:
        with results_lock:
            results[rid] = {
                "success": False,
                "error": data.get("error", "retrieval failed"),
            }
        return jsonify({"status": "ok"}), 200

    callback_queue.put(
        {
            "request_id": rid,
            "doc_ids": doc_ids,
        }
    )

    return jsonify({"status": "queued"}), 200


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    load_models()

    threading.Thread(target=embed_worker, daemon=True).start()
    threading.Thread(target=callback_worker, daemon=True).start()

    host, port = NODE_0_IP.split(":")
    app.run(host=host, port=int(port), threaded=True)


if __name__ == "__main__":
    main()
