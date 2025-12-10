import logging
import os
import time
from flask import Flask, request, jsonify
from inference_engine import InferenceWorker
from metrics import MetricsCollector, StepSampler, StepMetrics, NodeMonitor

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- CONFIGURATION ---
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 2))
NODE_2_IP_RAW = os.environ.get('NODE_2_IP', 'localhost:8002')

# Matches the logic in Node 0.
LLM_GEN_MICROBATCH_SIZE = int(os.environ.get("LLM_GEN_MICROBATCH_SIZE", "1"))

# Shared Worker Instance
worker = None
metrics = MetricsCollector(
    enable_metrics=True,
    metrics_file_path=f"metrics_node{NODE_NUMBER}.jsonl",
    metrics_summary_file_path=f"metrics_summary_node{NODE_NUMBER}.jsonl",
    immediate_flush=True,
)
node_monitor = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    # If worker is initialized, we are ready to serve
    status = 'healthy' if worker else 'loading'
    return jsonify({
        'status': status,
        'node': NODE_NUMBER,
        'total_nodes': TOTAL_NODES
    }), 200


@app.route('/generate', methods=['POST'])
def generate():
    """
    Main Endpoint.
    Input: { "queries": ["..."], "doc_ids": [[1,2], [3,4]] }
    Output: { "results": [ { "text": "...", "sentiment": "...", "is_toxic": "..." } ] }
    """
    try:
        data = request.json
        queries = data.get('queries')
        doc_ids_batch = data.get('doc_ids')
        request_ids = data.get('request_ids', [])

        if not queries or not doc_ids_batch:
            return jsonify({'error': 'Missing queries or doc_ids'}), 400

        logger.info(f"Received Batch: {len(queries)} items")

        if worker is None:
            return jsonify({'error': 'Worker not initialized'}), 503

        # This runs Fetch -> Rerank -> Generate -> Analyze with instrumentation
        batch_size = len(queries)
        req_start = time.perf_counter()

        # Fetch documents
        t0 = time.perf_counter()
        with StepSampler() as s_fetch:
            docs = worker.fetch_documents(doc_ids_batch)
        t1 = time.perf_counter()
        try:
            m = StepMetrics(
                step_name="fetch_documents",
                node_number=NODE_NUMBER,
                timestamp=time.time(),
                batch_size=batch_size,
                request_ids=request_ids if isinstance(request_ids, list) else [],
                duration_ms=(t1 - t0) * 1000.0,
                rss_samples_mb=getattr(s_fetch, 'rss_samples_mb', []),
                rss_aggregates=getattr(s_fetch, 'rss_aggregates', {})
            )
            metrics.record_step(m)
        except Exception:
            pass

        # Rerank
        t0 = time.perf_counter()
        with StepSampler() as s_rerank:
            ranked = worker.rerank(queries, docs)
        t1 = time.perf_counter()
        try:
            m = StepMetrics(
                step_name="rerank_documents",
                node_number=NODE_NUMBER,
                timestamp=time.time(),
                batch_size=batch_size,
                request_ids=request_ids if isinstance(request_ids, list) else [],
                duration_ms=(t1 - t0) * 1000.0,
                rss_samples_mb=getattr(s_rerank, 'rss_samples_mb', []),
                rss_aggregates=getattr(s_rerank, 'rss_aggregates', {})
            )
            metrics.record_step(m)
        except Exception:
            pass

        # Generate
        t0 = time.perf_counter()
        with StepSampler() as s_gen:
            texts = worker.generate(queries, ranked)
        t1 = time.perf_counter()
        try:
            m = StepMetrics(
                step_name="generate_responses",
                node_number=NODE_NUMBER,
                timestamp=time.time(),
                batch_size=batch_size,
                request_ids=request_ids if isinstance(request_ids, list) else [],
                duration_ms=(t1 - t0) * 1000.0,
                rss_samples_mb=getattr(s_gen, 'rss_samples_mb', []),
                rss_aggregates=getattr(s_gen, 'rss_aggregates', {})
            )
            metrics.record_step(m)
        except Exception:
            pass

        # Analyze (sentiment + safety)
        t0 = time.perf_counter()
        with StepSampler() as s_an:
            final_results = worker.analyze(texts)
        t1 = time.perf_counter()
        try:
            m = StepMetrics(
                step_name="analyze_sentiment",
                node_number=NODE_NUMBER,
                timestamp=time.time(),
                batch_size=batch_size,
                request_ids=request_ids if isinstance(request_ids, list) else [],
                duration_ms=(t1 - t0) * 1000.0,
                rss_samples_mb=getattr(s_an, 'rss_samples_mb', []),
                rss_aggregates=getattr(s_an, 'rss_aggregates', {})
            )
            metrics.record_step(m)
        except Exception:
            pass
        # node2_total
        try:
            m_total = StepMetrics(
                step_name="node2_total",
                node_number=NODE_NUMBER,
                timestamp=time.time(),
                batch_size=batch_size,
                request_ids=request_ids if isinstance(request_ids, list) else [],
                duration_ms=(time.perf_counter() - req_start) * 1000.0,
                rss_samples_mb=[],
                rss_aggregates={}
            )
            metrics.record_step(m_total)
            metrics.flush()
        except Exception:
            pass

        return jsonify({'results': final_results}), 200

    except Exception as e:
        logger.error(f"Inference Error: {e}")
        return jsonify({'error': str(e)}), 500


def main():
    global worker
    global node_monitor

    # Initialize the heavy worker once at startup
    # This loads LLM, Reranker, Sentiment, and Safety models
    logger.info(f"Initializing Inference Engine (MicroBatch={LLM_GEN_MICROBATCH_SIZE})...")
    worker = InferenceWorker(micro_batch_size=LLM_GEN_MICROBATCH_SIZE)

    try:
        node_monitor = NodeMonitor(node_number=NODE_NUMBER, metrics_file_path=f"metrics_node{NODE_NUMBER}.jsonl", sample_interval_s=0.02)
        node_monitor.start()
    except Exception:
        pass
    try:
        metrics.start()
    except Exception:
        pass

    # Start Server
    hostname = NODE_2_IP_RAW.split(':')[0]
    port = int(NODE_2_IP_RAW.split(':')[1]) if ':' in NODE_2_IP_RAW else 8002

    logger.info(f"Node 2 Inference Service starting on {hostname}:{port}")
    app.run(host=hostname, port=port, threaded=True)


if __name__ == "__main__":
    main()
