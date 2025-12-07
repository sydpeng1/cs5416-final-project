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
from metrics import MetricsCollector, StepSampler, StepMetrics, NodeMonitor

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

# Metrics
metrics = MetricsCollector(
    enable_metrics=True,
    metrics_file_path=f"metrics_node0.jsonl",
    metrics_summary_file_path=f"metrics_summary_node0.jsonl",
    immediate_flush=True,
)
node_monitor = None


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
    t0 = time.perf_counter()
    with StepSampler() as s_emb_load:
        embedder = SentenceTransformer(
            "BAAI/bge-base-en-v1.5", device=("cuda" if torch.cuda.is_available() else "cpu")
        )
    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="generate_embeddings.load_model",
            node_number=0,
            timestamp=time.time(),
            batch_size=0,
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s_emb_load, 'rss_samples_mb', []),
            rss_aggregates=getattr(s_emb_load, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass

    logger.info("Loading reranker...")
    t0 = time.perf_counter()
    with StepSampler() as s_rerank_load:
        reranker_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        reranker_model = (
            AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
            .to(DEVICE)
            .eval()
        )
    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="rerank.load_model",
            node_number=0,
            timestamp=time.time(),
            batch_size=0,
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s_rerank_load, 'rss_samples_mb', []),
            rss_aggregates=getattr(s_rerank_load, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass

    logger.info("Loading LLM...")
    llm_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    t0 = time.perf_counter()
    with StepSampler() as s_llm_load:
        llm_model = (
            AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct",
                dtype=dtype,
                use_cache=True,
            )
            .to(DEVICE)
            .eval()
        )
    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="generate_responses.load_model",
            node_number=0,
            timestamp=time.time(),
            batch_size=0,
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s_llm_load, 'rss_samples_mb', []),
            rss_aggregates=getattr(s_llm_load, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass

    logger.info("Loading sentiment & safety...")
    t0 = time.perf_counter()
    with StepSampler() as s_analysis_load:
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
    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="analysis.load_model",
            node_number=0,
            timestamp=time.time(),
            batch_size=0,
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s_analysis_load, 'rss_samples_mb', []),
            rss_aggregates=getattr(s_analysis_load, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass

    logger.info("All models loaded.")


# ------------------------------------------------------------
# FETCH DOCUMENTS
# ------------------------------------------------------------
def fetch_docs(doc_ids: List[int]):
    db = DOCUMENTS_DB
    if not os.path.exists(db):
        return []

    t0 = time.perf_counter()
    with StepSampler() as s:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

    placeholders = ",".join("?" for _ in doc_ids)
    q = f"SELECT doc_id, title, content FROM documents WHERE doc_id IN ({placeholders})"
    cur.execute(q, doc_ids)

    out = {row["doc_id"]: dict(row) for row in cur.fetchall()}
    conn.close()
    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="fetch_documents",
            node_number=0,
            timestamp=time.time(),
            batch_size=len(doc_ids),
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s, 'rss_samples_mb', []),
            rss_aggregates=getattr(s, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass
    return [out[i] for i in doc_ids if i in out]


# ------------------------------------------------------------
# RERANK
# ------------------------------------------------------------
def rerank(query: str, docs: List[Dict]):
    if not docs:
        return []

    t0 = time.perf_counter()
    with StepSampler() as s:
        pairs = [[query, d["content"]] for d in docs]
        with torch.no_grad():
            tok = reranker_tok(
                pairs, truncation=True, padding=True, return_tensors="pt"
            ).to(DEVICE)
            logits = reranker_model(**tok).logits.squeeze(-1).cpu().numpy()

    scored = sorted(zip(docs, logits.tolist()), key=lambda x: x[1], reverse=True)
    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="rerank_documents",
            node_number=0,
            timestamp=time.time(),
            batch_size=len(docs),
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s, 'rss_samples_mb', []),
            rss_aggregates=getattr(s, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass
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

    t0 = time.perf_counter()
    with StepSampler() as s:
        with torch.no_grad():
            ids = llm_model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.01,
                pad_token_id=llm_tok.eos_token_id,
            )

    new_ids = ids[:, inputs.input_ids.shape[1] :]
    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="generate_responses",
            node_number=0,
            timestamp=time.time(),
            batch_size=1,
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s, 'rss_samples_mb', []),
            rss_aggregates=getattr(s, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass
    return llm_tok.batch_decode(new_ids, skip_special_tokens=True)[0]


# ------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------
def analyze(text: str):
    t0 = time.perf_counter()
    with StepSampler() as s_sent:
        sent = sent_pipe(text[:512])[0]
    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="analyze_sentiment",
            node_number=0,
            timestamp=time.time(),
            batch_size=1,
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s_sent, 'rss_samples_mb', []),
            rss_aggregates=getattr(s_sent, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass

    t0 = time.perf_counter()
    with StepSampler() as s_safe:
        safe = safe_pipe(text[:512])[0]
    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="safety_filter",
            node_number=0,
            timestamp=time.time(),
            batch_size=1,
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s_safe, 'rss_samples_mb', []),
            rss_aggregates=getattr(s_safe, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass

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
            # queue wait
            try:
                metrics.record_step(StepMetrics(
                    step_name="request_queue_wait",
                    node_number=0,
                    timestamp=time.time(),
                    batch_size=1,
                    request_ids=[first["request_id"]],
                    duration_ms=(time.time() - time.time()) * 0.0,
                    rss_samples_mb=[],
                    rss_aggregates={}
                ))
                metrics.flush()
            except Exception:
                pass

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

            t0 = time.perf_counter()
            with StepSampler() as s_emb:
                embs = embedder.encode(
                    queries, normalize_embeddings=True, convert_to_numpy=True
                )
            logger.info(
                "[Node0] Embedding finished batch_size=%d time=%.3fs",
                batch_size,
                time.perf_counter() - t0,
            )
            try:
                metrics.record_step(StepMetrics(
                    step_name="generate_embeddings",
                    node_number=0,
                    timestamp=time.time(),
                    batch_size=batch_size,
                    request_ids=req_ids,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    rss_samples_mb=getattr(s_emb, 'rss_samples_mb', []),
                    rss_aggregates=getattr(s_emb, 'rss_aggregates', {})
                ))
                metrics.flush()
            except Exception:
                pass

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
            t_fetch = time.perf_counter()
            all_docs_batch = []
            for doc_ids in doc_ids_batch:
                all_docs_batch.append(fetch_docs(doc_ids))
            logger.info(
                "[Node0] Fetch docs batch_size=%d time=%.3fs",
                batch_size,
                time.perf_counter() - t_fetch,
            )

            # --------------------------------------------------------
            # 4) rerank（批量）
            # --------------------------------------------------------
            t_rerank = time.perf_counter()
            reranked_batch = []
            for q, docs in zip(queries, all_docs_batch):
                reranked_batch.append(rerank(q, docs))
            logger.info(
                "[Node0] Rerank batch_size=%d time=%.3fs",
                batch_size,
                time.perf_counter() - t_rerank,
            )

            # --------------------------------------------------------
            # 5) LLM inference（批量）
            # --------------------------------------------------------
            t_llm = time.perf_counter()
            answers = []
            for q, docs in zip(queries, reranked_batch):
                answers.append(llm_generate(q, docs))
            logger.info(
                "[Node0] LLM batch_size=%d time=%.3fs",
                batch_size,
                time.perf_counter() - t_llm,
            )

            # --------------------------------------------------------
            # 6) sentiment + toxicity（批量并行）
            # --------------------------------------------------------
            t_analysis = time.perf_counter()
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
                time.perf_counter() - t_analysis,
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
            try:
                metrics.record_step(StepMetrics(
                    step_name="node0_total",
                    node_number=0,
                    timestamp=time.time(),
                    batch_size=batch_size,
                    request_ids=request_ids,
                    duration_ms=(time.time() - start_t) * 1000.0,
                    rss_samples_mb=[],
                    rss_aggregates={}
                ))
                metrics.flush()
            except Exception:
                pass

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
                res = results.pop(req_id)
                try:
                    metrics.record_step(StepMetrics(
                        step_name="request_total",
                        node_number=0,
                        timestamp=time.time(),
                        batch_size=1,
                        request_ids=[req_id],
                        duration_ms=(time.time() - start) * 1000.0,
                        rss_samples_mb=[],
                        rss_aggregates={}
                    ))
                    metrics.flush()
                except Exception:
                    pass
                return jsonify(res), 200
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

    try:
        metrics.record_step(StepMetrics(
            step_name="request_queue_wait",
            node_number=0,
            timestamp=time.time(),
            batch_size=1,
            request_ids=[rid] if rid else [],
            duration_ms=0.0,
            rss_samples_mb=[],
            rss_aggregates={}
        ))
        metrics.flush()
    except Exception:
        pass
    return jsonify({"status": "queued"}), 200


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    load_models()

    threading.Thread(target=embed_worker, daemon=True).start()
    threading.Thread(target=callback_worker, daemon=True).start()

    try:
        global node_monitor
        node_monitor = NodeMonitor(node_number=0, metrics_file_path=f"metrics_node0.jsonl", sample_interval_s=0.02)
        node_monitor.start()
        metrics.start()
    except Exception:
        pass

    host, port = NODE_0_IP.split(":")
    app.run(host=host, port=int(port), threaded=True)


if __name__ == "__main__":
    main()
