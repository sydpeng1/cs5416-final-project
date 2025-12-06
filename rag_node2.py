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
logger = logging.getLogger("Node2")

app = Flask(__name__)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
NODE_0_IP = os.environ.get("NODE_0_IP", "localhost:8000")
NODE_2_IP = os.environ.get("NODE_2_IP", "localhost:8002")

DOCUMENTS_DB = os.environ.get("DOCUMENTS_DB", "documents/documents.db")

BATCH_SIZE = 8
BATCH_TIMEOUT = 0.05

MAX_QUEUE_SIZE = 2000

# ------------------------------------------------------------
# GLOBAL STATE
# ------------------------------------------------------------
callback_queue = Queue(maxsize=MAX_QUEUE_SIZE)

# Models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_INT = 0 if torch.cuda.is_available() else -1

reranker_model = None
reranker_tok = None
llm_model = None
llm_tok = None
sent_pipe = None
safe_pipe = None


# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------
def load_models():
    global reranker_tok, reranker_model, llm_tok, llm_model, sent_pipe, safe_pipe

    logger.info("Loading reranker...")
    reranker_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    reranker_model = (
        AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
        .to(DEVICE)
        .eval()
    )

    logger.info("Loading LLM...")
    llm_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    llm_tok.padding_side = 'left'
    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token
    
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    llm_model = (
        AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=dtype,
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

    logger.info("All models loaded on Node2.")


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


def llm_generate_batch(queries: List[str], docs_batch: List[List[Dict]]) -> List[str]:
    if not queries:
        return []
    
    all_texts = []
    for q, docs in zip(queries, docs_batch):
        ctx = "\n".join([f"- {d['title']}: {d['content'][:200]}" for d in docs[:3]])
        messages = [
            {"role": "system", "content": "Answer as 'Answer: <final>'"},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"},
        ]
        text = llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_texts.append(text)
    
    inputs = llm_tok(
        all_texts, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512, 
        padding=True
    ).to(DEVICE)
    
    with torch.no_grad():
        ids = llm_model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.01,
            pad_token_id=llm_tok.eos_token_id,
            do_sample=False,
        )
    
    input_len = inputs.input_ids.shape[1]
    new_ids = ids[:, input_len:]
    answers = llm_tok.batch_decode(new_ids, skip_special_tokens=True)
    
    return answers



def callback_worker():
    logger.info("Node2 callback_worker (BATCH MODE with true batch LLM) started.")

    node0_callback_url = f"http://{NODE_0_IP}/final_callback"
    logger.info(f"Node2 will callback Node0 at: {node0_callback_url}")

    while True:
        batch = []
        try:
            first = callback_queue.get()
            if first is None:
                break
            batch.append(first)

            start_t = time.time()

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
            logger.info(f"[Node2] Callback batch size = {batch_size}")

            request_ids = [b["request_id"] for b in batch]
            queries = [b["query"] for b in batch]
            doc_ids_batch = []

            for item in batch:
                doc_ids_batch.append(item["doc_ids"][0] if item["doc_ids"] else [])

            t_fetch = time.time()
            all_docs_batch = []
            for doc_ids in doc_ids_batch:
                all_docs_batch.append(fetch_docs(doc_ids))
            logger.info(
                "[Node2] Fetch docs batch_size=%d time=%.3fs",
                batch_size,
                time.time() - t_fetch,
            )

            t_rerank = time.time()
            reranked_batch = []
            for q, docs in zip(queries, all_docs_batch):
                reranked_batch.append(rerank(q, docs))
            logger.info(
                "[Node2] Rerank batch_size=%d time=%.3fs",
                batch_size,
                time.time() - t_rerank,
            )

            t_llm = time.time()
            answers = llm_generate_batch(queries, reranked_batch)
            logger.info(
                "[Node2] LLM (true batch) batch_size=%d time=%.3fs",
                batch_size,
                time.time() - t_llm,
            )

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
                "[Node2] Analysis batch_size=%d time=%.3fs",
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

            logger.info(
                "[Node2] Callback batch size=%d complete total_time=%.3fs",
                batch_size,
                time.time() - start_t,
            )

            def send_callback(rid, ans, s, tox):
                payload = {
                    "request_id": rid,
                    "success": True,
                    "generated_response": ans,
                    "sentiment": s,
                    "is_toxic": tox,
                }
                try:
                    resp = requests.post(node0_callback_url, json=payload, timeout=10)
                    logger.info(
                        f"[Node2] Callback to Node0 for {rid} status={resp.status_code}"
                    )
                except Exception as e:
                    logger.error(f"[Node2] Callback Node0 failed for {rid}: {e}")
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [
                    executor.submit(send_callback, rid, ans, s, tox)
                    for rid, ans, s, tox in zip(request_ids, answers, sentiments, toxics)
                ]
                for f in futures:
                    f.result()

        except Exception as e:
            logger.error(f"callback_worker batch error: {e}")
            for item in batch:
                try:
                    payload = {
                        "request_id": item["request_id"],
                        "success": False,
                        "error": str(e),
                    }
                    requests.post(node0_callback_url, json=payload, timeout=5)
                except:
                    pass

        finally:
            for _ in batch:
                callback_queue.task_done()


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "node": 2}), 200


@app.route("/retrieval_callback", methods=["POST"])
def retrieval_callback():
    data = request.json or {}
    rid = data.get("request_id")
    success = data.get("success")
    doc_ids = data.get("doc_ids", [])
    query = data.get("query", "")

    if not rid:
        return jsonify({"error": "missing request_id"}), 400

    if not success:
        node0_callback_url = f"http://{NODE_0_IP}/final_callback"
        try:
            payload = {
                "request_id": rid,
                "success": False,
                "error": data.get("error", "retrieval failed"),
            }
            requests.post(node0_callback_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to notify Node0 of failure for {rid}: {e}")
        return jsonify({"status": "ok"}), 200

    callback_queue.put(
        {
            "request_id": rid,
            "doc_ids": doc_ids,
            "query": query,
        }
    )

    return jsonify({"status": "queued"}), 200


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    load_models()

    threading.Thread(target=callback_worker, daemon=True).start()

    host, port = NODE_2_IP.split(":")
    logger.info(f"Node2 starting on {host}:{port}")
    app.run(host=host, port=int(port), threaded=True)


if __name__ == "__main__":
    main()