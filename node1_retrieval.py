import gc
import logging
import os
import sys
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import List

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

# Async batch processing configuration
BATCH_SIZE = 32
BATCH_TIMEOUT = 0.05  # 50ms - very fast for retrieval
MAX_QUEUE_SIZE = 500

# Global state
request_queue = Queue(maxsize=MAX_QUEUE_SIZE)
results = {}
results_lock = threading.Lock()


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


def worker_loop():
    """Background worker that processes batched search requests"""
    logger.info("Node 1 Worker thread started. Waiting for search requests...")
    
    while True:
        batch = []
        try:
            # 1. Blocking get - wait for first request
            first_req = request_queue.get()
            if first_req is None:  # Shutdown signal
                break
            batch.append(first_req)
            
            # 2. Dynamic batch collection - adjust timeout based on queue size
            queue_size = request_queue.qsize()
            
            # Node 1 is fast (retrieval), so we can be more aggressive with batching
            if queue_size > BATCH_SIZE * 3:
                dynamic_timeout = 0.005  # 5ms - process immediately under high load
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
                except Empty:
                    break
            
            # 3. Process batch
            logger.info(f"Processing batch of {len(batch)} search requests")
            
            if index is None:
                logger.error("Index not loaded!")
                with results_lock:
                    for req_dict in batch:
                        results[req_dict['request_id']] = {
                            'error': 'Index not ready',
                            'success': False
                        }
                for _ in batch:
                    request_queue.task_done()
                continue
            
            # Convert all embeddings to numpy array
            # Track which embeddings belong to which request
            all_embeddings = []
            request_mapping = []  # (request_id, embedding_index)
            
            for req_dict in batch:
                try:
                    embeddings = np.array(req_dict['embeddings'], dtype='float32')
                    # Handle both single embedding and batch of embeddings
                    if embeddings.ndim == 1:
                        embeddings = embeddings.reshape(1, -1)
                    
                    req_id = req_dict['request_id']
                    num_embeddings = embeddings.shape[0]
                    
                    for i in range(num_embeddings):
                        all_embeddings.append(embeddings[i])
                        request_mapping.append((req_id, i))
                    
                    logger.info(f"Request {req_id}: {num_embeddings} embeddings")
                    
                except Exception as e:
                    logger.error(f"Error processing request {req_dict['request_id']}: {e}")
                    with results_lock:
                        results[req_dict['request_id']] = {
                            'error': str(e),
                            'success': False
                        }
            
            if all_embeddings:
                # Stack all embeddings into single numpy array
                batch_embeddings = np.vstack(all_embeddings)
                
                # Single batch search
                t0 = time.time()
                _, indices = index.search(batch_embeddings, RETRIEVAL_K)
                search_time = time.time() - t0
                
                logger.info(f"Batch search completed in {search_time:.3f}s for {len(all_embeddings)} queries")
                
                # Group results by request_id
                request_results = {}
                for (req_id, emb_idx), doc_ids in zip(request_mapping, indices):
                    if req_id not in request_results:
                        request_results[req_id] = []
                    request_results[req_id].append(doc_ids.tolist())
                
                # Store results
                with results_lock:
                    for req_id, doc_ids_list in request_results.items():
                        results[req_id] = {
                            'doc_ids': doc_ids_list,  # List of lists
                            'success': True
                        }
                        logger.info(f"Stored result for {req_id}: {len(doc_ids_list)} doc_id lists")
            
            # Mark tasks as done
            for _ in batch:
                request_queue.task_done()
        
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            # Mark failed requests
            with results_lock:
                for req_dict in batch:
                    if req_dict['request_id'] not in results:
                        results[req_dict['request_id']] = {
                            'error': str(e),
                            'success': False
                        }
            for _ in batch:
                request_queue.task_done()


@app.route('/search', methods=['POST'])
def search():
    """Async endpoint - enqueues request and polls for result"""
    try:
        data = request.json
        if not data or "embeddings" not in data:
            return jsonify({'error': 'Missing input embeddings'}), 400
        
        # Generate request ID
        request_id = data.get('request_id', f"search_{time.time()}_{id(data)}")
        
        # Check if already processed (deduplication)
        with results_lock:
            if request_id in results:
                result = results.pop(request_id)
                if result['success']:
                    return jsonify({'doc_ids': result['doc_ids']}), 200
                else:
                    return jsonify({'error': result.get('error', 'Unknown error')}), 500
        
        # Enqueue request
        try:
            request_queue.put({
                'request_id': request_id,
                'embeddings': data['embeddings'],
                'timestamp': time.time()
            }, timeout=2)
        except:
            return jsonify({'error': 'Queue full, service overloaded'}), 503
        
        # Poll for result
        timeout = 30  # 30 seconds for retrieval
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            with results_lock:
                if request_id in results:
                    result = results.pop(request_id)
                    if result['success']:
                        return jsonify({'doc_ids': result['doc_ids']}), 200
                    else:
                        return jsonify({'error': result.get('error', 'Unknown error')}), 500
            
            time.sleep(0.01)  # Poll every 10ms
        
        return jsonify({'error': 'Search timeout'}), 504
    
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        return jsonify({'error': str(e)}), 500


index = None


def main():
    load_index()
    
    # Start background worker thread
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()
    logger.info("Node 1 worker thread started")

    hostname = NODE_1_IP_RAW.split(':')[0]
    port = int(NODE_1_IP_RAW.split(':')[1]) if ':' in NODE_1_IP_RAW else 8001

    logger.info(f"Node 1 Retrieval Service starting on {hostname}:{port}")
    
    # Disable Flask request logging for better performance
    import logging as flask_logging
    flask_log = flask_logging.getLogger('werkzeug')
    flask_log.setLevel(flask_logging.ERROR)
    
    app.run(
        host=hostname, 
        port=port, 
        threaded=True,
        processes=1,
        use_reloader=False,
        debug=False
    )


if __name__ == '__main__':
    main()