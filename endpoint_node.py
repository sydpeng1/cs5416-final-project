#!/usr/bin/env python3
import os
import time
import logging
import threading
from queue import Queue, Empty
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import requests
import torch

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline as hf_pipeline,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Node0")

app = Flask(__name__)

NODE_0_IP = os.environ.get("NODE_0_IP", "localhost:8000")
NODE_1_IP = os.environ.get("NODE_1_IP", "localhost:8001")
NODE_2_IP = os.environ.get("NODE_2_IP", "localhost:8002")

FAISS_NODE = NODE_1_IP

EMBED_BATCH_SIZE = 32
EMBED_BATCH_TIMEOUT = 0.05

CALLBACK_BATCH_SIZE = 8
CALLBACK_BATCH_TIMEOUT = 0.05

NUM_CALLBACK_WORKERS = 2

MAX_QUEUE_SIZE = 2000

embedding_queue = Queue(maxsize=MAX_QUEUE_SIZE)
callback_queue = Queue(maxsize=MAX_QUEUE_SIZE)

results: Dict[str, Dict[str, Any]] = {}
results_lock = threading.Lock()

result_events: Dict[str, threading.Event] = {}
events_lock = threading.Lock()

pending_queries: Dict[str, str] = {}
pending_q_lock = threading.Lock()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_INT = 0 if torch.cuda.is_available() else -1

embedder = None
llm_model = None
llm_tok = None
sent_pipe = None
safe_pipe = None


def load_models():
    global embedder, llm_tok, llm_model, sent_pipe, safe_pipe

    logger.info("Loading embedding model...")
    embedder = SentenceTransformer(
        "BAAI/bge-base-en-v1.5", device=("cuda" if torch.cuda.is_available() else "cpu")
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

    logger.info("All models loaded.")


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


def set_result(req_id: str, result: Dict):
    with results_lock:
        results[req_id] = result
    
    with events_lock:
        if req_id in result_events:
            result_events[req_id].set()


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

            t0 = time.time()
            embs = embedder.encode(
                queries, normalize_embeddings=True, convert_to_numpy=True
            )
            logger.info(
                "[Node0] Embedding finished batch_size=%d time=%.3fs",
                batch_size,
                time.time() - t0,
            )

            def send_to_node1(idx, r):
                host, port = FAISS_NODE.split(":")
                url = f"http://{host}:{port}/search"
                payload = {
                    "request_id": r["request_id"],
                    "embeddings": embs[idx].tolist(),
                    "query": r["query"],
                }
                try:
                    resp = requests.post(url, json=payload, timeout=3)
                    if resp.status_code != 202:
                        logger.error(
                            "[Node0] FAISS POST failed req_id=%s status=%s",
                            r["request_id"],
                            resp.status_code,
                        )
                except Exception as e:
                    logger.error("[Node0] Error sending req_id=%s: %s", r["request_id"], e)
                    set_result(r["request_id"], {"success": False, "error": str(e)})

            with ThreadPoolExecutor(max_workers=32) as executor:
                for i, r in enumerate(batch):
                    executor.submit(send_to_node1, i, r)

            for _ in batch:
                embedding_queue.task_done()

        except Exception as e:
            logger.error(f"embed_worker error: {e}")


def callback_worker(worker_id: int):
    logger.info(f"callback_worker-{worker_id} started.")

    while True:
        batch = []
        try:
            first = callback_queue.get()
            if first is None:
                break
            batch.append(first)

            start_t = time.time()

            while len(batch) < CALLBACK_BATCH_SIZE:
                remain = CALLBACK_BATCH_TIMEOUT - (time.time() - start_t)
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
            logger.info(f"[Node0-Worker{worker_id}] Callback batch size = {batch_size}")

            request_ids = [b["request_id"] for b in batch]
            queries = [b["query"] for b in batch]
            docs_batch = [b["docs"] for b in batch]

            t_llm = time.time()
            answers = llm_generate_batch(queries, docs_batch)
            logger.info(
                "[Node0-Worker%d] LLM (true batch) batch_size=%d time=%.3fs",
                worker_id, batch_size, time.time() - t_llm,
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
                "[Node0-Worker%d] Analysis batch_size=%d time=%.3fs",
                worker_id, batch_size, time.time() - t_analysis,
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
                "[Node0-Worker%d] Callback batch complete total_time=%.3fs",
                worker_id, time.time() - start_t,
            )

            for rid, ans, s, tox in zip(request_ids, answers, sentiments, toxics):
                set_result(rid, {
                    "success": True,
                    "generated_response": ans,
                    "sentiment": s,
                    "is_toxic": tox,
                })

        except Exception as e:
            logger.error(f"callback_worker-{worker_id} batch error: {e}")
            for item in batch:
                set_result(item["request_id"], {"success": False, "error": str(e)})

        finally:
            for _ in batch:
                callback_queue.task_done()


@app.route("/query", methods=["POST"])
def query_api():
    data = request.json or {}
    req_id = data.get("request_id")
    query = data.get("query")

    if not req_id or not query:
        return jsonify({"error": "missing request_id or query"}), 400

    event = threading.Event()
    with events_lock:
        result_events[req_id] = event

    with pending_q_lock:
        pending_queries[req_id] = query

    with results_lock:
        results.pop(req_id, None)

    embedding_queue.put({"request_id": req_id, "query": query})

    finished = event.wait(timeout=300)

    with events_lock:
        result_events.pop(req_id, None)

    if finished:
        with results_lock:
            result = results.pop(req_id, None)
        if result:
            return jsonify(result), 200
        return jsonify({"error": "result not found"}), 500

    return jsonify({"error": "timeout"}), 504


@app.route("/retrieval_callback", methods=["POST"])
def retrieval_callback():
    data = request.json or {}
    rid = data.get("request_id")
    success = data.get("success")
    docs = data.get("docs", [])
    query = data.get("query", "")

    if not rid:
        return jsonify({"error": "missing request_id"}), 400

    if not success:
        set_result(rid, {
            "success": False,
            "error": data.get("error", "retrieval failed"),
        })
        return jsonify({"status": "ok"}), 200

    callback_queue.put({
        "request_id": rid,
        "docs": docs,
        "query": query,
    })

    return jsonify({"status": "queued"}), 200


@app.route("/final_callback", methods=["POST"])
def final_callback():
    data = request.json or {}
    rid = data.get("request_id")
    success = data.get("success", False)

    if not rid:
        return jsonify({"error": "missing request_id"}), 400

    if success:
        set_result(rid, {
            "success": True,
            "generated_response": data.get("generated_response", ""),
            "sentiment": data.get("sentiment", "neutral"),
            "is_toxic": data.get("is_toxic", "false"),
        })
    else:
        set_result(rid, {
            "success": False,
            "error": data.get("error", "processing failed"),
        })

    return jsonify({"status": "ok"}), 200


def main():
    load_models()

    threading.Thread(target=embed_worker, daemon=True).start()
    
    for i in range(NUM_CALLBACK_WORKERS):
        threading.Thread(target=callback_worker, args=(i,), daemon=True).start()
    
    logger.info(f"Started {NUM_CALLBACK_WORKERS} callback workers")

    host, port = NODE_0_IP.split(":")
    app.run(host=host, port=int(port), threaded=True)


if __name__ == "__main__":
    main()