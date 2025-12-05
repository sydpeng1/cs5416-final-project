#!/usr/bin/env python3
import os
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

# -------------------------
# Config
# -------------------------
TOTAL_NODES = int(os.environ.get("TOTAL_NODES", 3))
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
NODE_0_IP = os.environ.get("NODE_0_IP", "127.0.0.1:8000")
NODE_1_IP = os.environ.get("NODE_1_IP", "127.0.0.1:8001")
NODE_2_IP = os.environ.get("NODE_2_IP", "127.0.0.1:8002")

BATCH_MAX_SIZE = 8
BATCH_MAX_WAIT = 0.10  # opportunistic batching, 100ms
TRUNCATE_LENGTH = 256


# -------------------------
# Flask + Global State
# -------------------------
app = Flask(__name__)

request_queue: "Queue[Dict[str, Any]]" = Queue()
results: Dict[str, Dict[str, Any]] = {}
waiters: Dict[str, threading.Condition] = {}
request_start_times: Dict[str, float] = {}

results_lock = threading.Lock()


@dataclass
class PipelineRequest:
    request_id: str
    query: str
    timestamp: float


# -------------------------
# Node0 Pipeline Class
# -------------------------
class Node0Pipeline:
    def __init__(self):
        print("[Node0] Initializing pipeline...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Node0] Using device: {self.device}")

        print("[Node0] Loading embedder...")
        self.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=self.device)

        print("[Node0] Loading analysis models...")
        d = 0 if torch.cuda.is_available() else -1

        self.sentiment_pipe = hf_pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=d,
        )

        self.safety_pipe = hf_pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=d,
        )

        self.node1_url = f"http://{NODE_1_IP}/search_and_rerank_batch"
        print(f"[Node0] Node1 URL = {self.node1_url}")

    # -------------------------
    # Process a batch
    # -------------------------
    def run_batch(self, reqs: List[PipelineRequest]):
        if not reqs:
            return

        queries = [r.query for r in reqs]
        request_ids = [r.request_id for r in reqs]

        print(f"\n[Node0] Processing batch size={len(reqs)}")

        # 1. Embedding
        t0 = time.time()
        emb = self.embedder.encode(
            queries, normalize_embeddings=True, convert_to_numpy=True
        )
        print(f"[Node0] Embedding time: {time.time() - t0:.3f}s")

        # 2. Send batch to Node1
        import requests

        embeddings = emb.tolist()

        print(
            f"[Node0] Sending batch -> Node1 count={len(reqs)} ids={request_ids} emb_dim={len(embeddings[0]) if embeddings else 0}"
        )

        payload = {
            "requests": [
                {
                    "request_id": req.request_id,
                    "query": req.query,
                    "embedding": embeddings[i],
                    "timestamp": req.timestamp,
                }
                for i, req in enumerate(reqs)
            ]
        }

        try:
            requests.post(self.node1_url, json=payload, timeout=300)
        except Exception as e:
            print(f"[Node0] ERROR sending to Node1: {e}")


pipeline = Node0Pipeline()


# -------------------------
# Worker Thread (Batching)
# -------------------------
def batch_worker():
    print("[Node0] Batch worker started")

    while True:
        try:
            first = request_queue.get()
            if first is None:
                break

            batch = [first]
            start_t = first["timestamp"]

            while len(batch) < BATCH_MAX_SIZE:
                try:
                    remain = BATCH_MAX_WAIT - (time.time() - start_t)
                    if remain <= 0:
                        break
                    nxt = request_queue.get(timeout=remain)
                    if nxt is None:
                        break
                    batch.append(nxt)
                except Empty:
                    break

            reqs = [
                PipelineRequest(i["request_id"], i["query"], i["timestamp"])
                for i in batch
            ]

            pipeline.run_batch(reqs)

            for _ in batch:
                request_queue.task_done()

        except Exception as e:
            print(f"[Node0] Worker error: {e}")


# -------------------------
# Node2 Callback Handler
# -------------------------
@app.route("/node2_callback", methods=["POST"])
def node2_callback():
    data = request.json or {}
    req_id = data.get("request_id")
    generated = data.get("generated_text", "")
    llm_duration = data.get("llm_duration")

    if not req_id:
        return jsonify({"error": "missing request_id"}), 400

    # --- local sentiment & safety ---
    truncated = generated[:TRUNCATE_LENGTH]

    sent_raw = pipeline.sentiment_pipe([truncated])[0]
    tox_raw = pipeline.safety_pipe([truncated])[0]

    sentiment_map = {
        "1 star": "very negative",
        "2 stars": "negative",
        "3 stars": "neutral",
        "4 stars": "positive",
        "5 stars": "very positive",
    }
    sentiment = sentiment_map.get(sent_raw["label"], "neutral")
    is_toxic = tox_raw["score"] > 0.5

    with results_lock:
        results[req_id] = {
            "request_id": req_id,
            "text": generated,
            "sentiment": sentiment,
            "is_toxic": is_toxic,
            "success": True,
            "llm_duration": llm_duration,
        }

        cond = waiters.pop(req_id, None)

    if cond:
        with cond:
            cond.notify()

    if llm_duration is not None:
        print(f"[Node0] Callback result stored for {req_id} llm_time={llm_duration:.3f}s")
    else:
        print(f"[Node0] Callback result stored for {req_id}")
    return jsonify({"status": "ok"})


# -------------------------
# API: Client sends queries
# -------------------------
@app.route("/query", methods=["POST"])
def query():
    data = request.json or {}
    req_id = data.get("request_id")
    text = data.get("query")

    if not req_id or not text:
        return jsonify({"error": "missing request_id or query"}), 400

    cond = threading.Condition()

    start_time = time.time()
    with results_lock:
        waiters[req_id] = cond
        request_start_times[req_id] = start_time

    request_queue.put({"request_id": req_id, "query": text, "timestamp": start_time})

    result = None

    # ----- wait for Node2 result -----
    with cond:
        cond.wait(timeout=300)

    with results_lock:
        if req_id in results:
            result = results.pop(req_id)

    total_time = None
    with results_lock:
        start = request_start_times.pop(req_id, None)

    if result:
        if start is not None:
            total_time = time.time() - start
            llm_time = result.get("llm_duration")
            if llm_time is not None:
                print(
                    f"[Node0] Request {req_id} completed total_time={total_time:.3f}s llm_time={llm_time:.3f}s"
                )
            else:
                print(f"[Node0] Request {req_id} completed total_time={total_time:.3f}s")
        return jsonify(result), 200

    return jsonify({"error": "timeout"}), 504


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "node": NODE_NUMBER}), 200


# -------------------------
# Main
# -------------------------
def main():
    worker = threading.Thread(target=batch_worker, daemon=True)
    worker.start()

    host, port = NODE_0_IP.split(":")
    print(f"[Node0] Listening on {host}:{port}")
    app.run(host=host, port=int(port), threaded=True)


if __name__ == "__main__":
    main()
