import gc
import logging
import os
import sys
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

import faiss
import numpy as np
import sqlite3
import torch
from flask import Flask, request, jsonify
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

TOTAL_NODES = int(os.environ.get("TOTAL_NODES", 1))
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 1))
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
DOCUMENTS_DB = os.environ.get("DOCUMENTS_DB", "documents/documents.db")

NODE_1_IP_RAW = os.environ.get("NODE_1_IP", "localhost:8001")
NODE_0_IP_RAW = os.environ.get("NODE_0_IP", "localhost:8000")
NODE_2_IP_RAW = os.environ.get("NODE_2_IP", "localhost:8002")

CALLBACK_NODES = [NODE_0_IP_RAW, NODE_2_IP_RAW]
CALLBACK_PATH = "/retrieval_callback"

RETRIEVAL_K = 10

BATCH_SIZE = 8
BATCH_TIMEOUT = 0.05
MAX_QUEUE_SIZE = 500

request_queue = Queue(maxsize=MAX_QUEUE_SIZE)

rr_callback_idx = 0
rr_callback_lock = threading.Lock()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

index = None
reranker_model = None
reranker_tok = None


@dataclass
class SearchRequest:
    request_id: str
    embeddings: np.ndarray
    query: str
    timestamp: float


def pick_callback_node():
    global rr_callback_idx
    with rr_callback_lock:
        node = CALLBACK_NODES[rr_callback_idx % len(CALLBACK_NODES)]
        rr_callback_idx += 1
    return node


def load_index():
    global index
    logger.info(f"Attempting to load FAISS index from: {FAISS_INDEX_PATH}")

    if not os.path.exists(FAISS_INDEX_PATH):
        logger.error(f"FATAL: FAISS index file not found at {FAISS_INDEX_PATH}")
        sys.exit(1)

    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        logger.info("FAISS Index loaded successfully.")
        logger.info(f"   - Vectors: {index.ntotal}")
        logger.info(f"   - Dimensions: {index.d}")
        gc.collect()
    except Exception as e:
        logger.error(f"FATAL: Failed to read FAISS index: {e}")
        sys.exit(1)


def load_reranker():
    global reranker_tok, reranker_model
    logger.info("Loading reranker model...")
    reranker_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    reranker_model = (
        AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
        .to(DEVICE)
        .eval()
    )
    logger.info("Reranker model loaded.")


def fetch_docs_bulk(all_doc_ids: List[int]) -> Dict[int, Dict]:
    if not all_doc_ids:
        return {}
    
    db = DOCUMENTS_DB
    if not os.path.exists(db):
        return {}

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    unique_ids = list(set(all_doc_ids))
    placeholders = ",".join("?" for _ in unique_ids)
    q = f"SELECT doc_id, title, content FROM documents WHERE doc_id IN ({placeholders})"
    cur.execute(q, unique_ids)

    out = {row["doc_id"]: dict(row) for row in cur.fetchall()}
    conn.close()

    return out


def get_docs_for_request(doc_ids: List[int], docs_cache: Dict[int, Dict]) -> List[Dict]:
    return [docs_cache[i] for i in doc_ids if i in docs_cache]


def rerank_batch(queries: List[str], docs_batch: List[List[Dict]]) -> List[List[Dict]]:
    if not queries:
        return []
    
    all_pairs = []
    pair_mapping = []
    
    for req_idx, (query, docs) in enumerate(zip(queries, docs_batch)):
        for doc_idx, doc in enumerate(docs):
            all_pairs.append([query, doc["content"]])
            pair_mapping.append((req_idx, doc_idx))
    
    if not all_pairs:
        return [[] for _ in queries]
    
    with torch.no_grad():
        tok = reranker_tok(
            all_pairs, truncation=True, padding=True, return_tensors="pt"
        ).to(DEVICE)
        all_logits = reranker_model(**tok).logits.squeeze(-1).cpu().numpy()
    
    if all_logits.ndim == 0:
        all_logits = np.array([float(all_logits)])
    
    request_scores: List[List[Tuple[int, float]]] = [[] for _ in queries]
    for pair_idx, (req_idx, doc_idx) in enumerate(pair_mapping):
        request_scores[req_idx].append((doc_idx, float(all_logits[pair_idx])))
    
    reranked_batch = []
    for req_idx, docs in enumerate(docs_batch):
        if not docs:
            reranked_batch.append([])
            continue
        scores = request_scores[req_idx]
        sorted_indices = sorted(scores, key=lambda x: x[1], reverse=True)
        reranked = [docs[idx] for idx, _ in sorted_indices]
        reranked_batch.append(reranked)
    
    return reranked_batch


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {"status": "healthy", "node": NODE_NUMBER, "total_nodes": TOTAL_NODES}
    ), 200


def send_callback(callback_node, payload):
    callback_url = f"http://{callback_node}{CALLBACK_PATH}"
    try:
        resp = requests.post(callback_url, json=payload, timeout=10)
        logger.info(
            f"Callback to {callback_node} for {payload['request_id']} status={resp.status_code}"
        )
        return True
    except Exception as e:
        logger.error(f"Callback failed for {payload['request_id']}: {e}")
        return False


def worker_loop():
    logger.info("Node 1 Worker thread started. Waiting for search requests...")
    logger.info(f"Node1 will round-robin callback to: {CALLBACK_NODES}")

    while True:
        batch = []
        try:
            first_req = request_queue.get()
            if first_req is None:
                break
            batch.append(first_req)

            queue_size = request_queue.qsize()

            if queue_size > BATCH_SIZE * 3:
                dynamic_timeout = 0.005
            elif queue_size > BATCH_SIZE:
                dynamic_timeout = BATCH_TIMEOUT / 3
            else:
                dynamic_timeout = BATCH_TIMEOUT

            start_wait = time.time()
            while len(batch) < BATCH_SIZE:
                remaining = dynamic_timeout - (time.time() - start_wait)
                if remaining <= 0:
                    break

                try:
                    req = request_queue.get(timeout=remaining)
                    if req is None:
                        break
                    batch.append(req)
                except Empty:
                    break

            batch_size = len(batch)
            batch_start = time.time()
            logger.info(f"Processing batch of {batch_size} search requests")

            if index is None:
                logger.error("Index not loaded!")
                with ThreadPoolExecutor(max_workers=16) as executor:
                    for req_dict in batch:
                        callback_node = pick_callback_node()
                        payload = {
                            "request_id": req_dict["request_id"],
                            "success": False,
                            "error": "Index not ready",
                            "docs": [],
                            "query": req_dict.get("query", ""),
                        }
                        executor.submit(send_callback, callback_node, payload)
                for _ in batch:
                    request_queue.task_done()
                continue

            collect_start = time.time()
            all_embeddings = []
            request_mapping = []

            for req_dict in batch:
                try:
                    embeddings = np.array(req_dict["embeddings"], dtype="float32")
                    if embeddings.ndim == 1:
                        embeddings = embeddings.reshape(1, -1)

                    req_id = req_dict["request_id"]
                    query = req_dict.get("query", "")
                    num_embeddings = embeddings.shape[0]

                    for i in range(num_embeddings):
                        all_embeddings.append(embeddings[i])
                        request_mapping.append((req_id, i, query))

                    logger.info(f"Request {req_id}: {num_embeddings} embeddings")

                except Exception as e:
                    logger.error(
                        f"Error processing request {req_dict['request_id']}: {e}"
                    )
                    callback_node = pick_callback_node()
                    payload = {
                        "request_id": req_dict["request_id"],
                        "success": False,
                        "error": str(e),
                        "docs": [],
                        "query": req_dict.get("query", ""),
                    }
                    send_callback(callback_node, payload)

            if all_embeddings:
                logger.info(
                    "Collected %d embeddings from batch in %.3fs",
                    len(all_embeddings),
                    time.time() - collect_start,
                )
                batch_embeddings = np.vstack(all_embeddings)

                t0 = time.time()
                _, indices = index.search(batch_embeddings, RETRIEVAL_K)
                search_time = time.time() - t0

                logger.info(
                    f"Batch search completed in {search_time:.3f}s for {len(all_embeddings)} queries"
                )

                request_results = {}
                for (req_id, emb_idx, query), doc_ids in zip(request_mapping, indices):
                    if req_id not in request_results:
                        request_results[req_id] = {"doc_ids": [], "query": query}
                    request_results[req_id]["doc_ids"].extend(doc_ids.tolist())

                t_fetch = time.time()
                all_doc_ids = []
                for req_id, data in request_results.items():
                    all_doc_ids.extend(data["doc_ids"])
                
                docs_cache = fetch_docs_bulk(all_doc_ids)
                
                queries_list = []
                docs_batch = []
                req_ids_list = list(request_results.keys())
                
                for req_id in req_ids_list:
                    data = request_results[req_id]
                    queries_list.append(data["query"])
                    docs_batch.append(get_docs_for_request(data["doc_ids"], docs_cache))
                
                logger.info(
                    "Fetch docs batch_size=%d unique_docs=%d time=%.3fs",
                    batch_size, len(docs_cache), time.time() - t_fetch,
                )

                t_rerank = time.time()
                reranked_batch = rerank_batch(queries_list, docs_batch)
                logger.info(
                    "Rerank batch_size=%d time=%.3fs",
                    batch_size, time.time() - t_rerank,
                )

                callback_start = time.time()
                callback_tasks = []
                for req_id, query, reranked_docs in zip(req_ids_list, queries_list, reranked_batch):
                    callback_node = pick_callback_node()
                    top_docs = reranked_docs[:3]
                    payload = {
                        "request_id": req_id,
                        "success": True,
                        "docs": top_docs,
                        "query": query,
                    }
                    callback_tasks.append((callback_node, payload))
                
                with ThreadPoolExecutor(max_workers=32) as executor:
                    futures = [
                        executor.submit(send_callback, node, payload)
                        for node, payload in callback_tasks
                    ]
                    for f in futures:
                        f.result()
                
                logger.info(
                    "Callbacks finished (parallel) for batch_size=%d time=%.3fs",
                    batch_size,
                    time.time() - callback_start,
                )
                logger.info(
                    "Batch end batch_size=%d total_time=%.3fs",
                    batch_size,
                    time.time() - batch_start,
                )

            for _ in batch:
                request_queue.task_done()

        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            try:
                with ThreadPoolExecutor(max_workers=16) as executor:
                    for req_dict in batch:
                        callback_node = pick_callback_node()
                        payload = {
                            "request_id": req_dict["request_id"],
                            "success": False,
                            "error": str(e),
                            "docs": [],
                            "query": req_dict.get("query", ""),
                        }
                        executor.submit(send_callback, callback_node, payload)
            finally:
                for _ in batch:
                    request_queue.task_done()


@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.json
        if not data or "embeddings" not in data or "request_id" not in data:
            return jsonify({"error": "Missing request_id or embeddings"}), 400

        request_id = data["request_id"]
        query = data.get("query", "")

        try:
            request_queue.put(
                {
                    "request_id": request_id,
                    "embeddings": data["embeddings"],
                    "query": query,
                    "timestamp": time.time(),
                },
                timeout=2,
            )
        except:
            return jsonify({"error": "Queue full, service overloaded"}), 503

        return jsonify({"status": "accepted", "request_id": request_id}), 202

    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


def main():
    load_index()
    load_reranker()

    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()
    logger.info("Node 1 worker thread started")

    hostname = NODE_1_IP_RAW.split(":")[0]
    port = int(NODE_1_IP_RAW.split(":")[1]) if ":" in NODE_1_IP_RAW else 8001

    logger.info(f"Node 1 Retrieval Service starting on {hostname}:{port}")

    import logging as flask_logging

    flask_log = flask_logging.getLogger("werkzeug")
    flask_log.setLevel(flask_logging.ERROR)

    app.run(
        host=hostname,
        port=port,
        threaded=True,
        processes=1,
        use_reloader=False,
        debug=False,
    )


if __name__ == "__main__":
    main()