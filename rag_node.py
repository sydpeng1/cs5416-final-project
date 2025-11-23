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
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 1))  # 1 或 2
NODE_0_IP = os.environ.get("NODE_0_IP", "127.0.0.1:8000")
NODE_1_IP = os.environ.get("NODE_1_IP", "127.0.0.1:8001")
NODE_2_IP = os.environ.get("NODE_2_IP", "127.0.0.1:8002")

# 自己监听的 IP/端口（同一份代码在 node1 / node2 跑）
MY_IP = NODE_1_IP if NODE_NUMBER == 1 else NODE_2_IP

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


class RagNode:
    """
    Node1 / Node2 通用：
    - 收到 Node0 的 embedding + query
    - FAISS 搜索 + SQLite 取文档 + Rerank
    - 把 (request_id, query, reranked_docs, nodeId) 回调给 Node0 的 /callback
    """

    def __init__(self):
        self.device = torch.device("cpu")
        print(f"[RAG Node {NODE_NUMBER}] Initializing on {self.device}")
        print(f"[RAG Node {NODE_NUMBER}] FAISS index: {FAISS_INDEX_PATH}")
        print(f"[RAG Node {NODE_NUMBER}] Documents DB dir: {DOCUMENTS_DIR}")

        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError("[RAG Node] FAISS index not found.")
        print("[RAG Node] Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        self.reranker_model_name = "BAAI/bge-reranker-base"
        print(f"[RAG Node] Loading reranker: {self.reranker_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.reranker_model_name
        ).to(self.device)
        self.model.eval()

        self.db_path = os.path.join(DOCUMENTS_DIR, "documents.db")

    # ---------- FAISS + SQLite + rerank ----------
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
                    "SELECT doc_id, title, content, category "
                    "FROM documents WHERE doc_id = ?",
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

    # ---------- 回调 Node0 ----------
    def send_callback_to_node0(
        self, req_ids: List[str], queries: List[str], docs_batch: List[List[Dict]]
    ):
        url = f"http://{NODE_0_IP}/callback"
        for rid, q, docs in zip(req_ids, queries, docs_batch):
            payload = {
                "request_id": rid,
                "query": q,
                "documents": docs,
                "nodeId": NODE_NUMBER,
            }
            try:
                resp = requests.post(url, json=payload, timeout=30)
                if resp.status_code != 200:
                    print(
                        f"[RAG Node {NODE_NUMBER}] Callback to Node0 failed "
                        f"for {rid}: {resp.status_code} {resp.text}"
                    )
                else:
                    print(f"[RAG Node {NODE_NUMBER}] Callback OK for {rid} → Node0")
            except Exception as e:
                print(
                    f"[RAG Node {NODE_NUMBER}] Error callback to Node0 for {rid}: {e}"
                )

    # ---------- 整个 batch 流程 ----------
    def process_batch(self, batch_items: List[Dict[str, Any]]):
        if not batch_items:
            return

        req_ids = [it["request_id"] for it in batch_items]
        queries = [it["query"] for it in batch_items]
        embeddings = np.array([it["embedding"] for it in batch_items], dtype="float32")

        print(f"\n[RAG Node {NODE_NUMBER}] Processing batch size={len(batch_items)}")
        t0 = time.time()

        # 1. FAISS
        doc_ids_batch = self._faiss_search_batch(embeddings)
        # 2. Fetch docs
        docs_batch = self._fetch_documents_batch(doc_ids_batch)
        # 3. Rerank
        reranked_batch = self._rerank_documents_batch(queries, docs_batch)
        # 4. Callback Node0（RAG 结果）
        self.send_callback_to_node0(req_ids, queries, reranked_batch)

        elapsed = time.time() - t0
        print(f"[RAG Node {NODE_NUMBER}] Batch finished in {elapsed:.2f}s")


rag_node = RagNode()


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
            rag_node.process_batch(batch)
        except Exception as e:
            print(f"[RAG Node {NODE_NUMBER}] Error processing batch: {e}")
        finally:
            for _ in batch:
                request_queue.task_done()


# ---------- Node0 调用入口 ----------
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

    # 异步处理，立即 ACK
    return jsonify({"status": "accepted"}), 200


@app.route("/health", methods=["GET"])
def health():
    return (
        jsonify({"status": "healthy", "node": NODE_NUMBER, "total_nodes": TOTAL_NODES}),
        200,
    )


def main():
    print("=" * 60)
    print(f"RAG NODE {NODE_NUMBER} (FAISS + DOCS + RERANK → callback Node0)")
    print("=" * 60)
    print(
        f"[RAG Node {NODE_NUMBER}] Listening on {MY_IP}, callback Node0 at {NODE_0_IP}"
    )

    worker = threading.Thread(target=batch_worker, daemon=True)
    worker.start()

    host, port = MY_IP.split(":")
    app.run(host=host, port=int(port), threaded=True)


if __name__ == "__main__":
    main()
