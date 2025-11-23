#!/usr/bin/env python3
import os
import gc
import time

import threading
from queue import Queue
from dataclasses import dataclass
from typing import Dict, Any, List

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline as hf_pipeline

import numpy as np
import torch
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor


# ----------------- Env & Config -----------------
TOTAL_NODES = int(os.environ.get("TOTAL_NODES", 1))
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
NODE_0_IP = os.environ.get("NODE_0_IP", "127.0.0.1:8000")
NODE_1_IP = os.environ.get("NODE_1_IP", "127.0.0.1:8001")
NODE_2_IP = os.environ.get("NODE_2_IP", "127.0.0.1:8002")
rag_nodes = [NODE_1_IP, NODE_2_IP]
next_rag_node = 0

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
llm_queue = Queue()
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

        self.llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.safety_model_name = "unitary/toxic-bert"

        print("[Node0] Loading LLM...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.llm_model.eval()

        print("[Node0] Loading sentiment & safety...")
        self.sentiment = hf_pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=-1,
        )
        self.safety = hf_pipeline(
            "text-classification",
            model=self.safety_model_name,
            device=-1,
        )

    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def send_to_rag_node(self, reqs: List[PipelineRequest], embeddings: np.ndarray):
        global next_rag_node, rag_nodes

        payload = {
            "requests": [
                {
                    "request_id": r.request_id,
                    "query": r.query,
                    "embedding": embeddings[i].tolist(),
                }
                for i, r in enumerate(reqs)
            ],
            "retrieval_k": CONFIG["retrieval_k"],
        }

        # Pick next RAG node (round robin)
        target_ip = rag_nodes[next_rag_node]
        next_rag_node = (next_rag_node + 1) % len(rag_nodes)

        url = f"http://{target_ip}/search_and_rerank_batch"
        print(f"[Node0] Sending batch to RAG node: {target_ip}")

        import requests

        try:
            resp = requests.post(url, json=payload, timeout=300)
            if resp.status_code != 200:
                print(f"[Node0] RAG node returned {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Node0] Error sending batch to RAG node {target_ip}: {e}")

    def _generate_responses_batch(
        self, queries: List[str], documents_batch: List[List[Dict]]
    ) -> List[str]:
        """Step 6: Generate LLM responses for each query in the batch"""
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            dtype=torch.float16,
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        responses = []
        for query, documents in zip(queries, documents_batch):
            context = "\n".join(
                [f"- {doc['title']}: {doc['content'][:200]}" for doc in documents[:3]]
            )
            messages = [
                {
                    "role": "system",
                    "content": "When given Context and Question, reply as 'Answer: <final answer>' only.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                },
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=CONFIG["max_tokens"],
                temperature=0.01,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]
            responses.append(response)
        del model, tokenizer
        gc.collect()
        return responses

    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Step 7: Analyze sentiment for each generated response"""
        classifier = hf_pipeline(
            "sentiment-analysis", model=self.sentiment_model_name, device=self.device
        )
        truncated_texts = [text[: CONFIG["truncate_length"]] for text in texts]
        raw_results = classifier(truncated_texts)
        sentiment_map = {
            "1 star": "very negative",
            "2 stars": "negative",
            "3 stars": "neutral",
            "4 stars": "positive",
            "5 stars": "very positive",
        }
        sentiments = []
        for result in raw_results:
            sentiments.append(sentiment_map.get(result["label"], "neutral"))
        del classifier
        gc.collect()
        return sentiments

    def _filter_response_safety_batch(self, texts: List[str]) -> List[bool]:
        """Step 8: Filter responses for safety for each entry in the batch"""
        classifier = hf_pipeline(
            "text-classification", model=self.safety_model_name, device=self.device
        )
        truncated_texts = [text[: CONFIG["truncate_length"]] for text in texts]
        raw_results = classifier(truncated_texts)
        toxicity_flags = []
        for result in raw_results:
            toxicity_flags.append(result["score"] > 0.5)
        del classifier
        gc.collect()
        return toxicity_flags

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

        print("[Node0] Step 2: Sending batch to RAG nodes...")
        self.send_to_rag_node(reqs, embeddings)

        elapsed = time.time() - t0
        print(f"[Node0] Batch dispatched to RAG Nodes in {elapsed:.2f}s")


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


def llm_worker():
    BATCH_SIZE = 4  # LLM batch size

    while True:
        job = llm_queue.get()
        if job is None:
            break

        # ---- Build a batch ----
        items = [job]
        try:
            while len(items) < BATCH_SIZE:
                nxt = llm_queue.get_nowait()
                items.append(nxt)
        except:
            pass

        # Collect queries & docs
        queries = [it["query"] for it in items]
        docs_batch = [it["documents"] for it in items]
        req_ids = [it["request_id"] for it in items]

        answers = []

        # ============================================================
        # 1) LLM GENERATION (batch)
        # ============================================================
        for query, docs in zip(queries, docs_batch):
            # Build context
            context = "\n".join(
                [f"- {d['title']}: {d['content'][:200]}" for d in docs[:3]]
            )

            messages = [
                {
                    "role": "system",
                    "content": "When given Context and Question, reply as 'Answer: <final answer>' only.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                },
            ]

            # Template → tokens
            text = pipeline.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = pipeline.llm_tokenizer([text], return_tensors="pt").to(
                pipeline.llm_model.device
            )

            # Run LLM
            with torch.no_grad():
                gen_ids = pipeline.llm_model.generate(
                    **inputs,
                    max_new_tokens=CONFIG["max_tokens"],
                    temperature=0.01,
                    pad_token_id=pipeline.llm_tokenizer.eos_token_id,
                )

            # Remove prompt tokens
            gen_ids = gen_ids[:, inputs.input_ids.shape[1] :]

            # Decode
            answer = pipeline.llm_tokenizer.batch_decode(
                gen_ids, skip_special_tokens=True
            )[0]

            answers.append(answer)

        # ============================================================
        # 2) sentiment batch & safety batch → RUN IN PARALLEL
        # ============================================================

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_sent = ex.submit(pipeline._analyze_sentiment_batch, answers)
            fut_toxic = ex.submit(pipeline._filter_response_safety_batch, answers)

            sentiments = fut_sent.result()
            toxic_flags = fut_toxic.result()

        # ============================================================
        # 3) Write back final batch results into results{}
        # ============================================================
        for rid, ans, sent, tox in zip(req_ids, answers, sentiments, toxic_flags):
            with results_lock:
                results[rid] = {
                    "request_id": rid,
                    "generated_response": ans,
                    "sentiment": sent,
                    "is_toxic": "true" if tox else "false",
                }

            print(f"[Node0] Finished LLM + sentiment + safety for {rid}")

        # mark queue tasks
        for _ in items:
            llm_queue.task_done()


# ------------- Node0 回调入口 -------------
@app.route("/callback", methods=["POST"])
def node_callback():
    data = request.json or {}
    request_id = data.get("request_id")
    query = data.get("query")
    documents = data.get("documents")
    nodeId = data.get("nodeId")

    if not request_id:
        return jsonify({"error": "missing request_id"}), 400

    # Instead of final result, push into LLM queue:
    llm_queue.put(
        {
            "request_id": request_id,
            "query": query,
            "documents": documents,
            "timestamp": time.time(),
        }
    )

    print(f"[Node0] Received callback from {nodeId} → queued for LLM: {request_id}")
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

    LLM_WORKERS = 3
    for i in range(LLM_WORKERS):
        t = threading.Thread(target=llm_worker, daemon=True)
        t.start()
        print(f"[Node0] LLM worker {i + 1} started!")

    host, port = NODE_0_IP.split(":")
    app.run(host=host, port=int(port), threaded=True)


if __name__ == "__main__":
    main()
