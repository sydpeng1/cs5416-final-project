#!/usr/bin/env python3
import os
import time
import threading
from queue import Queue
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

# ----------------- Env & Config -----------------
TOTAL_NODES = int(os.environ.get("TOTAL_NODES", 1))
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
NODE_0_IP = os.environ.get("NODE_0_IP", "127.0.0.1:8000")
NODE_1_IP = os.environ.get("NODE_1_IP", "127.0.0.1:8001")

CONFIG = {
    "faiss_dim": 768,
    "max_tokens": 128,
    "retrieval_k": 10,
    "truncate_length": 512,
}

BATCH_MAX_SIZE = 8
BATCH_MAX_WAIT = 0.05  # seconds

app = Flask(__name__)

request_queue: "Queue[Dict[str, Any]]" = Queue()
results: Dict[str, Dict[str, Any]] = {}
results_lock = threading.Lock()


@dataclass
class PipelineRequest:
    request_id: str
    query: str
    timestamp: float


class Node0Pipeline:
    """
    Node0:
    - 接收 /query
    - 对 query 做 embedding
    - 批量把 embedding+query 发给 Node1
    - 等 Node2 回调 /node2_callback 后，由 /query 返回给 client
    """

    def __init__(self):
        self.device = torch.device("cpu")
        print(f"[Node0] Initializing on {self.device}")
        self.embedding_model_name = "BAAI/bge-base-en-v1.5"
        print(f"[Node0] Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name).to(
            self.device
        )

    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def send_to_node1(self, requests: List[PipelineRequest], embeddings: np.ndarray):
        payload = {
            "requests": [
                {
                    "request_id": r.request_id,
                    "query": r.query,
                    "embedding": embeddings[i].tolist(),
                }
                for i, r in enumerate(requests)
            ],
            "retrieval_k": CONFIG["retrieval_k"],
        }

        url = f"http://{NODE_1_IP}/search_and_rerank_batch"
        try:
            resp = requests.post(url, json=payload, timeout=300)
            if resp.status_code != 200:
                print(f"[Node0] Node1 returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Node0] Error sending batch to Node1: {e}")

    def process_batch(self, batch_items: List[Dict[str, Any]]):
        if not batch_items:
            return

        reqs = [
            PipelineRequest(
                request_id=item["request_id"],
                query=item["query"],
                timestamp=item["timestamp"],
            )
            for item in batch_items
        ]
        queries = [r.query for r in reqs]

        print(f"\n[Node0] Processing batch size={len(reqs)}")
        t0 = time.time()

        print("[Node0] Step 1: Generating embeddings...")
        embeddings = self._generate_embeddings_batch(queries)

        print("[Node0] Step 2: Sending batch to Node1...")
        self.send_to_node1(reqs, embeddings)

        elapsed = time.time() - t0
        print(f"[Node0] Batch dispatched to Node1 in {elapsed:.2f}s")


pipeline = Node0Pipeline()


def batch_worker():
    while True:
        item = request_queue.get()
        if item is None:
            break

        batch = [item]
        first_t = item["timestamp"]

        # opportunistic batching
        while len(batch) < BATCH_MAX_SIZE:
            try:
                wait_left = BATCH_MAX_WAIT - (time.time() - first_t)
                if wait_left <= 0:
                    break
                nxt = request_queue.get(timeout=wait_left)
                if nxt is None:
                    break
                batch.append(nxt)
            except Exception:
                break

        try:
            pipeline.process_batch(batch)
        except Exception as e:
            print(f"[Node0] Error processing batch: {e}")
        finally:
            for _ in batch:
                request_queue.task_done()


# ------------- Node2 回调入口：Node2 调这个 -------------
@app.route("/node2_callback", methods=["POST"])
def node2_callback():
    data = request.json or {}
    request_id = data.get("request_id")

    if not request_id:
        return jsonify({"error": "missing request_id"}), 400

    with results_lock:
        results[request_id] = {
            "request_id": request_id,
            "generated_response": data.get("generated_response", ""),
            "sentiment": data.get("sentiment", "neutral"),
            "is_toxic": data.get("is_toxic", "false"),
        }

    print(f"[Node0] Received callback for {request_id} from Node2")
    return jsonify({"status": "ok"}), 200


# ------------- client 调用入口 -------------
@app.route("/query", methods=["POST"])
def handle_query():
    data = request.json or {}
    request_id = data.get("request_id")
    query = data.get("query")

    if not request_id or not query:
        return jsonify({"error": "missing request_id or query"}), 400

    # 清理旧结果（避免复用）
    with results_lock:
        if request_id in results:
            results.pop(request_id)

    print(f"[Node0] Enqueue request {request_id}")
    request_queue.put(
        {"request_id": request_id, "query": query, "timestamp": time.time()}
    )

    # 等待 Node2 回调写入 results[request_id]
    timeout = 300
    start = time.time()
    while True:
        with results_lock:
            if request_id in results:
                res = results.pop(request_id)
                print(f"[Node0] Returning result for {request_id} to client")
                return jsonify(res), 200

        if time.time() - start > timeout:
            print(f"[Node0] Timeout waiting for Node2 for {request_id}")
            return jsonify({"error": "timeout"}), 504

        time.sleep(0.05)


@app.route("/health", methods=["GET"])
def health():
    return (
        jsonify({"status": "healthy", "node": NODE_NUMBER, "total_nodes": TOTAL_NODES}),
        200,
    )


def main():
    print("=" * 60)
    print("NODE0 FRONTEND + EMBEDDING + CALLBACK HANDLER")
    print("=" * 60)
    print(f"[Node0] Listening on {NODE_0_IP}, calling Node1 at {NODE_1_IP}")

    worker = threading.Thread(target=batch_worker, daemon=True)
    worker.start()

    host, port = NODE_0_IP.split(":")
    app.run(host=host, port=int(port), threaded=True)


if __name__ == "__main__":
    main()
