import logging
import os
from flask import Flask, request, jsonify
from inference_engine import InferenceWorker

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- CONFIGURATION ---
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 2))
NODE_2_IP_RAW = os.environ.get('NODE_2_IP', 'localhost:8002')

# Matches the logic in Node 0.
GPU_MICRO_BATCH_SIZE = int(os.environ.get("GPU_MICRO_BATCH_SIZE", "1"))

# Shared Worker Instance
worker = None


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

        if not queries or not doc_ids_batch:
            return jsonify({'error': 'Missing queries or doc_ids'}), 400

        logger.info(f"Received Batch: {len(queries)} items")

        if worker is None:
            return jsonify({'error': 'Worker not initialized'}), 503

        # This runs Fetch -> Rerank -> Generate -> Analyze
        final_results = worker.run_pipeline(queries, doc_ids_batch)

        return jsonify({'results': final_results}), 200

    except Exception as e:
        logger.error(f"Inference Error: {e}")
        return jsonify({'error': str(e)}), 500


def main():
    global worker

    # Initialize the heavy worker once at startup
    # This loads LLM, Reranker, Sentiment, and Safety models
    logger.info("Initializing Inference Engine...")
    worker = InferenceWorker(micro_batch_size=GPU_MICRO_BATCH_SIZE)

    # Start Server
    hostname = NODE_2_IP_RAW.split(':')[0]
    port = int(NODE_2_IP_RAW.split(':')[1]) if ':' in NODE_2_IP_RAW else 8002

    logger.info(f"Node 2 Inference Service starting on {hostname}:{port}")
    app.run(host=hostname, port=port, threaded=True)


if __name__ == "__main__":
    main()
