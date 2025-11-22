#!/usr/bin/env python3
import os
import time
import threading
from queue import Queue
from typing import Dict, Any, List

import torch
import requests
from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline as hf_pipeline,
)

# -------- Env & Config --------
TOTAL_NODES = int(os.environ.get("TOTAL_NODES", 1))
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 2))
NODE_2_IP = os.environ.get("NODE_2_IP", "127.0.0.1:8002")
NODE_0_IP = os.environ.get("NODE_0_IP", "127.0.0.1:8000")

CONFIG = {
    "max_tokens": 128,
    "truncate_length": 512,
}

BATCH_MAX_SIZE = 8
BATCH_MAX_WAIT = 0.05

app = Flask(__name__)

request_queue: "Queue[Dict[str, Any]]" = Queue()


class Node2Generation:
    """
    Node2:
    - 从 Node1 收 request (query + documents)
    - LLM 生成 + Sentiment + Toxicity
    - 结果直接 POST 回 Node0 的 /node2_callback
    """

    def __init__(self):
        self.device = torch.device("cpu")
        print(f"[Node2] Initializing on {self.device}")

        self.llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.safety_model_name = "unitary/toxic-bert"

        print(f"[Node2] Loading LLM: {self.llm_model_name}")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)
        self.llm_model.eval()

        print(f"[Node2] Loading sentiment model: {self.sentiment_model_name}")
        self.sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=-1,
        )

        print(f"[Node2] Loading safety model: {self.safety_model_name}")
        self.safety_pipeline = hf_pipeline(
            "text-classification",
            model=self.safety_model_name,
            device=-1,
        )

        self.sentiment_map = {
            "1 star": "very negative",
            "2 stars": "negative",
            "3 stars": "neutral",
            "4 stars": "positive",
            "5 stars": "very positive",
        }

    def _generate_responses_batch(
        self, items: List[Dict[str, Any]], max_tokens: int
    ) -> List[str]:
        responses: List[str] = []

        for it in items:
            query = it["query"]
            docs = it.get("documents", [])
            context = "\n".join(
                [
                    f"- {d.get('title', '')}: {d.get('content', '')[:200]}"
                    for d in docs[:3]
                ]
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
            text = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.llm_tokenizer([text], return_tensors="pt").to(
                self.llm_model.device
            )

            with torch.no_grad():
                gen_ids = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.01,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )

            gen_ids = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
            ]
            resp = self.llm_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
            responses.append(resp)

        return responses

    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        truncated = [t[: CONFIG["truncate_length"]] for t in texts]
        raw = self.sentiment_pipeline(truncated)
        sentiments: List[str] = []
        for r in raw:
            sentiments.append(self.sentiment_map.get(r["label"], "neutral"))
        return sentiments

    def _safety_filter_batch(self, texts: List[str]) -> List[bool]:
        truncated = [t[: CONFIG["truncate_length"]] for t in texts]
        raw = self.safety_pipeline(truncated)
        return [r["score"] > 0.5 for r in raw]

    def send_callback_to_node0(self, rid: str, result: Dict[str, Any]):
        url = f"http://{NODE_0_IP}/node2_callback"
        payload = {
            "request_id": rid,
            "generated_response": result["generated_response"],
            "sentiment": result["sentiment"],
            "is_toxic": result["is_toxic"],
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code != 200:
                print(
                    f"[Node2] Callback to Node0 failed for {rid}: "
                    f"{resp.status_code} {resp.text}"
                )
        except Exception as e:
            print(f"[Node2] Error callback to Node0 for {rid}: {e}")

    def process_batch(self, batch_items: List[Dict[str, Any]]):
        if not batch_items:
            return

        req_ids = [it["request_id"] for it in batch_items]

        print(f"\n[Node2] Processing batch size={len(batch_items)}")
        t0 = time.time()

        responses = self._generate_responses_batch(batch_items, CONFIG["max_tokens"])
        sentiments = self._analyze_sentiment_batch(responses)
        toxic_flags = self._safety_filter_batch(responses)

        elapsed = time.time() - t0
        print(f"[Node2] Batch completed in {elapsed:.2f}s")

        # 对每个请求发回调到 Node0
        for rid, resp, sent, tox in zip(req_ids, responses, sentiments, toxic_flags):
            result = {
                "generated_response": resp,
                "sentiment": sent,
                "is_toxic": "true" if tox else "false",
            }
            print(f"[Node2] Sending batch to Node0, request_ids={req_ids}")
            self.send_callback_to_node0(rid, result)


node2 = Node2Generation()


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
            node2.process_batch(batch)
        except Exception as e:
            print(f"[Node2] Error processing batch: {e}")
        finally:
            for _ in batch:
                request_queue.task_done()


@app.route("/generate_batch", methods=["POST"])
def generate_batch():
    data = request.json or {}
    reqs = data.get("requests", [])
    if not reqs:
        return jsonify({"results": []}), 200

    for r in reqs:
        request_queue.put(
            {
                "request_id": r["request_id"],
                "query": r["query"],
                "documents": r.get("documents", []),
                "timestamp": time.time(),
            }
        )

    # Node2 也异步处理，立即 ACK
    return jsonify({"status": "accepted"}), 200


@app.route("/health", methods=["GET"])
def health():
    return (
        jsonify({"status": "healthy", "node": NODE_NUMBER, "total_nodes": TOTAL_NODES}),
        200,
    )


def main():
    print("=" * 60)
    print("NODE2 GENERATION (LLM + SENTIMENT + SAFETY → callback Node0)")
    print("=" * 60)
    print(f"[Node2] Listening on {NODE_2_IP}, callback Node0 at {NODE_0_IP}")

    worker = threading.Thread(target=batch_worker, daemon=True)
    worker.start()

    host, port = NODE_2_IP.split(":")
    app.run(host=host, port=int(port), threaded=True)


if __name__ == "__main__":
    main()
