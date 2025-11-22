#!/usr/bin/env python3
import os
import time
import sqlite3
import threading
from queue import Queue
from typing import Dict, Any, List

import numpy as np
import torch
import faiss
import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------- Env & Config -------------
TOTAL_NODES = int(os.environ.get("TOTAL_NODES", 1))
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 1))
NODE_1_IP = os.environ.get("NODE_1_IP", "127.0.0.1:8001")
NODE_2_IP = os.environ.get("NODE_2_IP", "127.0.0.1:8002")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "documents/")

CONFIG = {
    "faiss_index_path": FAISS_INDEX_PATH,
    "documents_path": DOCUMENTS_DIR,
    "faiss_dim": 768,
    "retrieval_k": 10,
    "truncate_length": 512,
}

BATCH_MAX_SIZE = 16
BATCH_MAX_WAIT = 0.05

app = Flask(__name__)

request_queue: "Queue[Dict[str, Any]]" = Queue()


class Node1Retrieval:
    """
    Node1:
    - 收到 Node0 的 embedding + query
    - FAISS 搜索 + SQLite 取文档 + Rerank
    - 把 (query, reranked_docs) 发给 Node2
    - 不负责等待、也不直接回给 Node0
    """

    def __init__(self):
        self.device = torch.device("cpu")
        print(f"[Node1] Initializing on {self.device}")
        print(f"[Node1] FAISS index: {FAISS_INDEX_PATH}")
        print(f"[Node1] Documents DB dir: {DOCUMENTS_DIR}")

        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError("[Node1] FAISS index not found.")
        print("[Node1] Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        self.reranker_model_name = "BAAI/bge-reranker-base"
        print(f"[Node1] Loading reranker: {self.reranker_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.reranker_model_name
        ).to(self.device)
        self.model.eval()

        self.db_path = os.path.join(DOCUMENTS_DIR, "documents.db")

    def _faiss_search_batch(self, embeddings: np.ndarray) -> List[List[int]]:
        embeddings = embeddings.astype("float32")
        _, indices = self.index.search(embeddings, CONFIG["retrieval_k"])
        return [row.tolist() for row in indices]

    def _fetch_documents_batch(
        self, doc_id_batches: List[List[int]]
    ) -> List[List[Dict[str, Any]]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        documents_batch: List[List[Dict[str, Any]]] = []

        for doc_ids in doc_id_batches:
            docs = []
            for doc_id in doc_ids:
                cursor.execute(
                    "SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?",
                    (doc_id,),
                )
                result = cursor.fetchone()
                if result:
                    docs.append(
                        {
                            "doc_id": result[0],
                            "title": result[1],
                            "content": result[2],
                            "category": result[3],
                        }
                    )
            documents_batch.append(docs)

        conn.close()
        return documents_batch

    def _rerank_documents_batch(
        self, queries: List[str], documents_batch: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        reranked_batches: List[List[Dict[str, Any]]] = []

        for query, docs in zip(queries, documents_batch):
            if not docs:
                reranked_batches.append([])
                continue

            pairs = [[query, d["content"]] for d in docs]
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=CONFIG["truncate_length"],
                ).to(self.device)
                scores = self.model(**inputs, return_dict=True).logits.view(-1).float()

            doc_scores = list(zip(docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([d for d, _ in doc_scores])

        return reranked_batches

    def send_to_node2(
        self, req_ids: List[str], queries: List[str], docs_batch: List[List[Dict]]
    ):
        payload = {
            "requests": [
                {"request_id": rid, "query": q, "documents": docs}
                for rid, q, docs in zip(req_ids, queries, docs_batch)
            ]
        }
        url = f"http://{NODE_2_IP}/generate_batch"
        try:
            resp = requests.post(url, json=payload, timeout=300)
            if resp.status_code != 200:
                print(f"[Node1] Node2 returned {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Node1] Error sending batch to Node2: {e}")

    def process_batch(self, batch_items: List[Dict[str, Any]]):
        if not batch_items:
            return

        req_ids = [it["request_id"] for it in batch_items]
        queries = [it["query"] for it in batch_items]
        embeddings = np.array([it["embedding"] for it in batch_items], dtype="float32")

        print(f"\n[Node1] Processing batch size={len(batch_items)}")
        t0 = time.time()

        # FAISS
        doc_ids_batch = self._faiss_search_batch(embeddings)
        # Fetch docs
        docs_batch = self._fetch_documents_batch(doc_ids_batch)
        # Rerank
        reranked_batch = self._rerank_documents_batch(queries, docs_batch)
        # Send to Node2
        print(f"[Node1] Step: Sending batch to Node2, request_ids={req_ids}")
        self.send_to_node2(req_ids, queries, reranked_batch)

        elapsed = time.time() - t0
        print(f"[Node1] Batch dispatched to Node2 in {elapsed:.2f}s")


node1 = Node1Retrieval()


def batch_worker():
    while True:
        item = request_queue.get()
        if item is None:
            break

        batch = [item]
        first_t = item["timestamp"]

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
            node1.process_batch(batch)
        except Exception as e:
            print(f"[Node1] Error processing batch: {e}")
        finally:
            for _ in batch:
                request_queue.task_done()


@app.route("/search_and_rerank_batch", methods=["POST"])
def search_and_rerank_batch():
    data = request.json or {}
    reqs = data.get("requests", [])
    if not reqs:
        return jsonify({"results": []}), 200

    for r in reqs:
        request_queue.put(
            {
                "request_id": r["request_id"],
                "query": r["query"],
                "embedding": r["embedding"],
                "timestamp": time.time(),
            }
        )

    # Node1 不等 Node2，直接 ACK
    return jsonify({"status": "accepted"}), 200


@app.route("/health", methods=["GET"])
def health():
    return (
        jsonify({"status": "healthy", "node": NODE_NUMBER, "total_nodes": TOTAL_NODES}),
        200,
    )


def main():
    print("=" * 60)
    print("NODE1 RETRIEVAL (FAISS + DOCS + RERANK → Node2)")
    print("=" * 60)
    print(f"[Node1] Listening on {NODE_1_IP}, calling Node2 at {NODE_2_IP}")

    worker = threading.Thread(target=batch_worker, daemon=True)
    worker.start()

    host, port = NODE_1_IP.split(":")
    app.run(host=host, port=int(port), threaded=True)


if __name__ == "__main__":
    main()
