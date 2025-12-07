import gc
import logging
import os
import sys
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass

import faiss
import numpy as np
from flask import Flask, request, jsonify
import requests  # ✨ 新增
from metrics import MetricsCollector, StepSampler, StepMetrics, NodeMonitor

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
TOTAL_NODES = int(os.environ.get("TOTAL_NODES", 1))
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 1))
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")

NODE_1_IP_RAW = os.environ.get("NODE_1_IP", "localhost:8001")
NODE_0_IP_RAW = os.environ.get("NODE_0_IP", "localhost:8000")  # ✨ 新增：Node0 地址
NODE0_CALLBACK_PATH = "/retrieval_callback"  # ✨ 新增：回调路径

RETRIEVAL_K = 10

# Async batch processing configuration
BATCH_SIZE = 8
BATCH_TIMEOUT = 0.05  # 50ms - very fast for retrieval
MAX_QUEUE_SIZE = 500

# Global state
request_queue = Queue(maxsize=MAX_QUEUE_SIZE)
metrics = MetricsCollector(
    enable_metrics=True,
    metrics_file_path=f"metrics_node{NODE_NUMBER}.jsonl",
    metrics_summary_file_path=f"metrics_summary_node{NODE_NUMBER}.jsonl",
    immediate_flush=True,
)
node_monitor = None


@dataclass
class SearchRequest:
    request_id: str
    embeddings: np.ndarray
    timestamp: float


def load_index():
    global index
    logger.info(f"Attempting to load FAISS index from: {FAISS_INDEX_PATH}")

    if not os.path.exists(FAISS_INDEX_PATH):
        logger.error(f"FATAL: FAISS index file not found at {FAISS_INDEX_PATH}")
        sys.exit(1)

    try:
        t0 = time.perf_counter()
        with StepSampler() as s:
            index = faiss.read_index(FAISS_INDEX_PATH)
        t1 = time.perf_counter()
        try:
            metrics.record_step(StepMetrics(
                step_name="faiss_search.load_index",
                node_number=NODE_NUMBER,
                timestamp=time.time(),
                batch_size=0,
                request_ids=[],
                duration_ms=(t1 - t0) * 1000.0,
                rss_samples_mb=getattr(s, 'rss_samples_mb', []),
                rss_aggregates=getattr(s, 'rss_aggregates', {})
            ))
            metrics.flush()
        except Exception:
            pass
        logger.info("FAISS Index loaded successfully.")
        logger.info(f"   - Vectors: {index.ntotal}")
        logger.info(f"   - Dimensions: {index.d}")

        gc.collect()

    except Exception as e:
        logger.error(f"FATAL: Failed to read FAISS index: {e}")
        sys.exit(1)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {"status": "healthy", "node": NODE_NUMBER, "total_nodes": TOTAL_NODES}
    ), 200


def worker_loop():
    """Background worker that processes batched search requests and callback Node0"""
    logger.info("Node 1 Worker thread started. Waiting for search requests...")

    node0_base_url = f"http://{NODE_0_IP_RAW}{NODE0_CALLBACK_PATH}"
    logger.info(f"Node1 will callback Node0 at: {node0_base_url}")

    while True:
        batch = []
        try:
            first_req = request_queue.get()
            if first_req is None:  # Shutdown signal
                break
            batch.append(first_req)
            # queue wait for first
            try:
                metrics.record_step(StepMetrics(
                    step_name="request_queue_wait",
                    node_number=NODE_NUMBER,
                    timestamp=time.time(),
                    batch_size=1,
                    request_ids=[first_req["request_id"]],
                    duration_ms=(time.time() - first_req["timestamp"]) * 1000.0,
                    rss_samples_mb=[],
                    rss_aggregates={}
                ))
                metrics.flush()
            except Exception:
                pass

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
                    # queue wait for subsequent
                    try:
                        metrics.record_step(StepMetrics(
                            step_name="request_queue_wait",
                            node_number=NODE_NUMBER,
                            timestamp=time.time(),
                            batch_size=1,
                            request_ids=[req["request_id"]],
                            duration_ms=(time.time() - req["timestamp"]) * 1000.0,
                            rss_samples_mb=[],
                            rss_aggregates={}
                        ))
                        metrics.flush()
                    except Exception:
                        pass
                except Empty:
                    break

            batch_size = len(batch)
            batch_start = time.time()
            logger.info(f"Processing batch of {batch_size} search requests")

            if index is None:
                logger.error("Index not loaded!")
                # 通知 Node0 失败
                for req_dict in batch:
                    payload = {
                        "request_id": req_dict["request_id"],
                        "success": False,
                        "error": "Index not ready",
                        "doc_ids": [],
                    }
                    try:
                        requests.post(node0_base_url, json=payload, timeout=5)
                    except Exception as e2:
                        logger.error(
                            f"Callback Node0 failed for {req_dict['request_id']}: {e2}"
                        )
                for _ in batch:
                    request_queue.task_done()
                continue

            collect_start = time.time()
            all_embeddings = []
            request_mapping = []  # (request_id, embedding_index)

            for req_dict in batch:
                try:
                    embeddings = np.array(req_dict["embeddings"], dtype="float32")
                    if embeddings.ndim == 1:
                        embeddings = embeddings.reshape(1, -1)

                    req_id = req_dict["request_id"]
                    num_embeddings = embeddings.shape[0]

                    for i in range(num_embeddings):
                        all_embeddings.append(embeddings[i])
                        request_mapping.append((req_id, i))

                    logger.info(f"Request {req_id}: {num_embeddings} embeddings")

                except Exception as e:
                    logger.error(
                        f"Error processing request {req_dict['request_id']}: {e}"
                    )
                    # 通知 Node0 单个请求失败
                    payload = {
                        "request_id": req_dict["request_id"],
                        "success": False,
                        "error": str(e),
                        "doc_ids": [],
                    }
                    try:
                        requests.post(node0_base_url, json=payload, timeout=5)
                    except Exception as e2:
                        logger.error(
                            f"Callback Node0 failed for {req_dict['request_id']}: {e2}"
                        )

            if all_embeddings:
                logger.info(
                    "Collected %d embeddings from batch in %.3fs",
                    len(all_embeddings),
                    time.time() - collect_start,
                )
                batch_embeddings = np.vstack(all_embeddings)

                t0 = time.perf_counter()
                with StepSampler() as s:
                    _, indices = index.search(batch_embeddings, RETRIEVAL_K)
                t1 = time.perf_counter()
                search_time = t1 - t0
                try:
                    metrics.record_step(StepMetrics(
                        step_name="faiss_search.search",
                        node_number=NODE_NUMBER,
                        timestamp=time.time(),
                        batch_size=len(all_embeddings),
                        request_ids=[rid for rid, _ in request_mapping],
                        duration_ms=(t1 - t0) * 1000.0,
                        rss_samples_mb=getattr(s, 'rss_samples_mb', []),
                        rss_aggregates=getattr(s, 'rss_aggregates', {})
                    ))
                    metrics.flush()
                except Exception:
                    pass

                logger.info(
                    f"Batch search completed in {search_time:.3f}s for {len(all_embeddings)} queries"
                )

                # Group by request_id
                request_results = {}  # req_id -> [list_of_docid_lists]
                for (req_id, emb_idx), doc_ids in zip(request_mapping, indices):
                    if req_id not in request_results:
                        request_results[req_id] = []
                    request_results[req_id].append(doc_ids.tolist())

                # ✨ 对每个 request_id 调用 Node0 callback
                callback_start = time.time()
                for req_id, doc_ids_list in request_results.items():
                    payload = {
                        "request_id": req_id,
                        "success": True,
                        "doc_ids": doc_ids_list,  # List[List[int]]
                    }
                    try:
                        resp = requests.post(node0_base_url, json=payload, timeout=10)
                        logger.info(
                            f"Callback to Node0 for {req_id} status={resp.status_code}"
                        )
                    except Exception as e:
                        logger.error(f"Callback Node0 failed for {req_id}: {e}")
                logger.info(
                    "Callbacks to Node0 finished for batch_size=%d time=%.3fs",
                    batch_size,
                    time.time() - callback_start,
                )
                logger.info(
                    "Batch end batch_size=%d total_time=%.3fs",
                    batch_size,
                    time.time() - batch_start,
                )
                try:
                    metrics.record_step(StepMetrics(
                        step_name="node1_total",
                        node_number=NODE_NUMBER,
                        timestamp=time.time(),
                        batch_size=batch_size,
                        request_ids=[req_dict["request_id"] for req_dict in batch],
                        duration_ms=(time.time() - batch_start) * 1000.0,
                        rss_samples_mb=[],
                        rss_aggregates={}
                    ))
                    metrics.flush()
                except Exception:
                    pass

            for _ in batch:
                request_queue.task_done()

        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            # 这里没法知道 batch 里的具体错误，只能 best-effort 通知
            try:
                for req_dict in batch:
                    payload = {
                        "request_id": req_dict["request_id"],
                        "success": False,
                        "error": str(e),
                        "doc_ids": [],
                    }
                    try:
                        requests.post(node0_base_url, json=payload, timeout=5)
                    except Exception as e2:
                        logger.error(
                            f"Callback Node0 failed for {req_dict['request_id']}: {e2}"
                        )
            finally:
                for _ in batch:
                    request_queue.task_done()


@app.route("/search", methods=["POST"])
def search():
    """
    Fire-and-forget 式接口：
      - Node0 带 request_id + embeddings 调用 /search
      - Node1 把请求塞进队列，立即返回 202
      - Node1 异步处理完后，调用 Node0 的 /retrieval_callback
    """
    try:
        data = request.json
        if not data or "embeddings" not in data or "request_id" not in data:
            return jsonify({"error": "Missing request_id or embeddings"}), 400

        request_id = data["request_id"]

        try:
            request_queue.put(
                {
                    "request_id": request_id,
                    "embeddings": data["embeddings"],
                    "timestamp": time.time(),
                },
                timeout=2,
            )
        except:
            return jsonify({"error": "Queue full, service overloaded"}), 503

        # 不再等待结果，直接告诉 Node0：已接受
        return jsonify({"status": "accepted", "request_id": request_id}), 202

    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


index = None


def main():
    global node_monitor
    load_index()
    try:
        node_monitor = NodeMonitor(node_number=NODE_NUMBER, metrics_file_path=f"metrics_node{NODE_NUMBER}.jsonl", sample_interval_s=0.02)
        node_monitor.start()
        metrics.start()
    except Exception:
        pass

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
