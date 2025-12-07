import gc
import logging
import os
import sys

import faiss
import numpy as np
from flask import Flask, request, jsonify
from metrics import MetricsCollector, StepSampler, StepMetrics, NodeMonitor
import time

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 1))
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faiss_index.bin')
NODE_1_IP_RAW = os.environ.get('NODE_1_IP', 'localhost:8001')
RETRIEVAL_K = 10

metrics = MetricsCollector(
    enable_metrics=True,
    metrics_file_path=f"metrics_node{NODE_NUMBER}.jsonl",
    metrics_summary_file_path=f"metrics_summary_node{NODE_NUMBER}.jsonl",
    immediate_flush=True,
)
node_monitor = None


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
        logger.info(f"FAISS Index loaded successfully.")
        logger.info(f"   - Vectors: {index.ntotal}")
        logger.info(f"   - Dimensions: {index.d}")

        # Force GC to clear temporary loading buffers
        gc.collect()

    except Exception as e:
        logger.error(f"FATAL: Failed to read FAISS index: {e}")
        sys.exit(1)


# Routes
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'node': NODE_NUMBER,
        'total_nodes': TOTAL_NODES
    }), 200


@app.route('/search', methods=['POST'])
def search():
    if index is None:
        return jsonify({'error': 'Index not ready'}), 503

    try:
        req_start = time.perf_counter()
        data = request.json
        if not data or "embeddings" not in data:
            return jsonify({'error': 'Missing input embeddings'}), 400

        # 1. Convert input vectors to Numpy
        query_embeddings = np.array(data.get('embeddings'), dtype='float32')

        # 2. Perform search
        t0 = time.perf_counter()
        with StepSampler() as s:
            _, indices = index.search(query_embeddings, RETRIEVAL_K)
        t1 = time.perf_counter()
        try:
            metrics.record_step(StepMetrics(
                step_name="faiss_search.search",
                node_number=NODE_NUMBER,
                timestamp=time.time(),
                batch_size=len(query_embeddings) if query_embeddings is not None else 0,
                request_ids=data.get('request_ids', []),
                duration_ms=(t1 - t0) * 1000.0,
                rss_samples_mb=getattr(s, 'rss_samples_mb', []),
                rss_aggregates=getattr(s, 'rss_aggregates', {})
            ))
        except Exception:
            pass

        # 3. Serialize output (Numpy arrays not JSON serializable, so convert to list)
        resp = jsonify({
            'doc_ids': indices.tolist()
        }), 200
        try:
            metrics.record_step(StepMetrics(
                step_name="node1_total",
                node_number=NODE_NUMBER,
                timestamp=time.time(),
                batch_size=len(query_embeddings) if query_embeddings is not None else 0,
                request_ids=data.get('request_ids', []),
                duration_ms=(time.perf_counter() - req_start) * 1000.0,
                rss_samples_mb=[],
                rss_aggregates={}
            ))
            metrics.flush()
        except Exception:
            pass
        return resp

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


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

    hostname = NODE_1_IP_RAW.split(':')[0]
    port = int(NODE_1_IP_RAW.split(':')[1]) if ':' in NODE_1_IP_RAW else 8001

    logger.info(f"Node 1 Retrieval Service starting on {hostname}:{port}")
    app.run(host=hostname, port=port, threaded=True)


if __name__ == '__main__':
    main()