import gc
import logging
import os
import sys

import faiss
import numpy as np
from flask import Flask, request, jsonify

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


def load_index():
    global index
    logger.info(f"Attempting to load FAISS index from: {FAISS_INDEX_PATH}")

    if not os.path.exists(FAISS_INDEX_PATH):
        logger.error(f"FATAL: FAISS index file not found at {FAISS_INDEX_PATH}")
        sys.exit(1)

    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
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
        data = request.json
        if not data or "embeddings" not in data:
            return jsonify({'error': 'Missing input embeddings'}), 400

        # 1. Convert input vectors to Numpy
        query_embeddings = np.array(data.get('embeddings'), dtype='float32')

        # 2. Perform search
        _, indices = index.search(query_embeddings, RETRIEVAL_K)

        # 3. Serialize output (Numpy arrays not JSON serializable, so convert to list)
        return jsonify({
            'doc_ids': indices.tolist()
        }), 200

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


index = None


def main():
    load_index()

    hostname = NODE_1_IP_RAW.split(':')[0]
    port = int(NODE_1_IP_RAW.split(':')[1]) if ':' in NODE_1_IP_RAW else 8001

    logger.info(f"Node 1 Retrieval Service starting on {hostname}:{port}")
    app.run(host=hostname, port=port, threaded=True)


if __name__ == '__main__':
    main()
