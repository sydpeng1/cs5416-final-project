import os
import time
import threading
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import torch
from queue import Queue, Empty
from typing import List, Dict, Any
from dataclasses import dataclass
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 0))
NODE_0_IP_RAW = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP_RAW = os.environ.get('NODE_1_IP', 'localhost:8001')  # Default to 8001
NODE_2_IP_RAW = os.environ.get('NODE_2_IP', 'localhost:8002')  # Default to 8002

BATCH_SIZE = 32  # Increased from 16 for better throughput
BATCH_TIMEOUT = 0.1  # Reduced from 0.5s for lower latency
TRUNCATE_LENGTH = 512
MAX_QUEUE_SIZE = 1000  # Prevent memory overflow under extreme load

# Flask App
app = Flask(__name__)
request_queue = Queue(maxsize=MAX_QUEUE_SIZE)
results = {}
results_lock = threading.Lock()

# Performance optimization: disable Flask request logging for production
import logging as flask_logging
flask_log = flask_logging.getLogger('werkzeug')
flask_log.setLevel(flask_logging.ERROR)


@dataclass
class PipelineRequest:
    request_id: str
    query: str
    timestamp: float


@dataclass
class PipelineResponse:
    request_id: str
    generated_response: str
    sentiment: str
    is_toxic: str
    processing_time: float


class DistributedPipeline:
    """
    Orchestrator Node (Node 0):
    1. Local: Embed Query
    2. Remote (Node 1): Search Index -> Get Doc IDs
    3. Remote (Node 2): Fetch + Rerank + Generate -> Get Text
    4. Local: Sentiment + Safety Analysis
    """
    def __init__(self):
        # HuggingFace pipelines expect an integer: 0 for GPU, -1 for CPU
        self.device = 0 if torch.cuda.is_available() else -1

        # SentenceTransformer expects a string: "cuda" or "cpu"
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
   
        logger.info(f"Initializing Distributed Pipeline on Node {NODE_NUMBER}")
        logger.info(f"Compute Device: {self.device_str} (ID: {self.device})")

        # Service URLs
        n1_host = NODE_1_IP_RAW.split(':')[0]
        n1_port = int(NODE_1_IP_RAW.split(':')[1]) if ':' in NODE_1_IP_RAW else 8001
        self.retrieval_url = f"http://{n1_host}:{n1_port}/search"

        n2_host = NODE_2_IP_RAW.split(':')[0]
        n2_port = int(NODE_2_IP_RAW.split(':')[1]) if ':' in NODE_2_IP_RAW else 8002
        self.inference_url = f"http://{n2_host}:{n2_port}/generate"

        logger.info(f"Target Node 1 (Retrieval): {self.retrieval_url}")
        logger.info(f"Target Node 2 (Inference): {self.inference_url}")
        
        # Validate URLs
        if not self.retrieval_url.startswith('http://'):
            logger.warning(f"Invalid retrieval URL: {self.retrieval_url}")
        if not self.inference_url.startswith('http://'):
            logger.warning(f"Invalid inference URL: {self.inference_url}")
        
        # Create HTTP session with connection pooling and retry strategy
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        
        # Mount adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry_strategy
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info("HTTP connection pool initialized (10 connections, max 20)")

        # Load Lightweight Models
        logger.info("Loading Embedder (Local)...")
        self.embedder = SentenceTransformer('BAAI/bge-base-en-v1.5', device=self.device_str)
        
        # Optimize embedder for batch processing
        if self.device_str == "cuda":
            self.embedder = self.embedder.half()  # Use float16 on GPU for speed
            logger.info("Embedder optimized with float16 precision")

        logger.info("Loading Analysis Models (Local)...")
        self.sentiment_pipe = hf_pipeline("sentiment-analysis",
                                          model='nlptown/bert-base-multilingual-uncased-sentiment',
                                          device=self.device)

        self.safety_pipe = hf_pipeline("text-classification",
                                       model='unitary/toxic-bert',
                                       device=self.device)

        logger.info("Node 0 Ready.")

    def process_batch(self, batch_requests: List[PipelineRequest]) -> List[PipelineResponse]:
        if not batch_requests: return []

        start_time = time.time()
        batch_size = len(batch_requests)
        queries = [req.query for req in batch_requests]
        logger.info(f"--- Processing Batch of {batch_size} Requests ---")

        try:
            # --- STEP 1: Embedding (Local) ---
            t0 = time.time()
            embeddings = self.embedder.encode(queries, normalize_embeddings=True, convert_to_numpy=True)
            logger.info(f"[1] Embeddings: {time.time() - t0:.3f}s")

            # --- STEP 2: Retrieval (Node 1) ---
            t0 = time.time()
            # Convert numpy to list for JSON serialization
            # Add request_id for better tracking in Node 1's async queue
            batch_id = f"batch_{int(time.time()*1000)}_{id(batch_requests)}"
            payload_n1 = {
                'embeddings': embeddings.tolist(),
                'request_id': f"{batch_id}_retrieval"
            }
            resp_n1 = self.session.post(self.retrieval_url, json=payload_n1, timeout=30)
            resp_n1.raise_for_status()
            doc_ids_batch = resp_n1.json()['doc_ids']
            logger.info(f"[2] Retrieval (Node 1): {time.time() - t0:.3f}s")
            
            # Validate doc_ids_batch format
            if not isinstance(doc_ids_batch, list):
                logger.error(f"Invalid doc_ids format from Node 1: {type(doc_ids_batch)}")
                raise ValueError(f"Expected list, got {type(doc_ids_batch)}")
            
            # Ensure doc_ids_batch is a list of lists
            if doc_ids_batch and not isinstance(doc_ids_batch[0], list):
                logger.warning(f"doc_ids_batch[0] is not a list, wrapping it. Type: {type(doc_ids_batch[0])}")
                doc_ids_batch = [doc_ids_batch]
            
            logger.info(f"[2] doc_ids_batch shape: {len(doc_ids_batch)} queries, "
                       f"each with {len(doc_ids_batch[0]) if doc_ids_batch else 0} docs")

            # --- STEP 3: Generation (Node 2) ---
            t0 = time.time()
            payload_n2 = {
                'queries': queries,
                'doc_ids': doc_ids_batch,
                'request_id': f"{batch_id}_inference"
            }
            logger.info(f"[3] Sending to Node 2 at {self.inference_url}")
            logger.info(f"[3] Payload: {len(queries)} queries, {len(doc_ids_batch)} doc_id lists")
            
            try:
                resp_n2 = self.session.post(self.inference_url, json=payload_n2, timeout=300)
                resp_n2.raise_for_status()
                responses_text = resp_n2.json()['responses']
                logger.info(f"[3] Generation (Node 2): {time.time() - t0:.3f}s")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"[3] Connection error to Node 2 at {self.inference_url}: {e}")
                raise
            except requests.exceptions.HTTPError as e:
                logger.error(f"[3] HTTP error from Node 2: {e}")
                logger.error(f"[3] Response content: {resp_n2.text if 'resp_n2' in locals() else 'N/A'}")
                raise

            # --- STEP 4: Analysis (Local) - PARALLEL EXECUTION ---
            t0 = time.time()
            # Truncate for BERT models to prevent errors
            truncated_texts = [t[:TRUNCATE_LENGTH] for t in responses_text]

            # Run Sentiment & Safety in PARALLEL using threads
            from concurrent.futures import ThreadPoolExecutor
            
            def run_sentiment():
                return self.sentiment_pipe(truncated_texts)
            
            def run_safety():
                return self.safety_pipe(truncated_texts)
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                sentiment_future = executor.submit(run_sentiment)
                safety_future = executor.submit(run_safety)
                
                raw_sentiments = sentiment_future.result()
                raw_safety = safety_future.result()

            # Parse Results
            sentiment_map = {
                '1 star': 'very negative', '2 stars': 'negative',
                '3 stars': 'neutral', '4 stars': 'positive', '5 stars': 'very positive'
            }
            final_sentiments = [sentiment_map.get(r['label'], 'neutral') for r in raw_sentiments]
            is_toxic_flags = [r['score'] > 0.5 for r in raw_safety]

            logger.info(f"[4] Analysis (Parallel): {time.time() - t0:.3f}s")

            # --- Assemble Responses ---
            pipeline_responses = []
            total_duration = time.time() - start_time

            for i, req in enumerate(batch_requests):
                # Individual request latency (approximate based on batch end)
                req_latency = time.time() - req.timestamp

                pipeline_responses.append(PipelineResponse(
                    request_id=req.request_id,
                    generated_response=responses_text[i],
                    sentiment=final_sentiments[i],
                    is_toxic="true" if is_toxic_flags[i] else "false",
                    processing_time=req_latency
                ))

            logger.info(f"Batch completed in {total_duration:.3f}s")
            return pipeline_responses

        except Exception as e:
            logger.error(f"Batch Failed: {e}")
            return []


pipeline_instance = None


def worker_loop():
    global pipeline_instance
    logger.info("Worker thread started. Waiting for requests...")

    while True:
        batch = []
        try:
            # 1. Blocking Get (Wait for first item)
            first_req = request_queue.get()
            if first_req is None: break  # Shutdown signal
            batch.append(first_req)

            # 2. Dynamic Batch Collection Strategy
            # Adjust timeout based on queue size for better throughput/latency balance
            queue_size = request_queue.qsize()
            
            # If queue is building up, reduce timeout to process faster
            # If queue is empty, use longer timeout to gather more requests
            if queue_size > BATCH_SIZE * 2:
                # High load: process immediately to reduce queue
                dynamic_timeout = 0.01  # 10ms
            elif queue_size > BATCH_SIZE:
                # Medium load: short timeout
                dynamic_timeout = BATCH_TIMEOUT / 2
            else:
                # Low load: normal timeout to gather batch
                dynamic_timeout = BATCH_TIMEOUT
            
            start_wait = time.time()
            while len(batch) < BATCH_SIZE:
                # Calculate remaining time in timeout window
                remaining = dynamic_timeout - (time.time() - start_wait)
                if remaining <= 0: break

                try:
                    # Non-blocking get (or very short timeout)
                    req = request_queue.get(timeout=remaining)
                    batch.append(req)
                except Empty:
                    break

            # 3. Convert dicts to Request Objects
            req_objects = [PipelineRequest(r['request_id'], r['query'], r['timestamp']) for r in batch]

            # 4. Process Batch
            responses = pipeline_instance.process_batch(req_objects)

            # 5. Store Results & Mark Task Done
            with results_lock:
                for res in responses:
                    results[res.request_id] = {
                        'request_id': res.request_id,
                        'generated_response': res.generated_response,
                        'sentiment': res.sentiment,
                        'is_toxic': res.is_toxic,
                        'success': True
                    }

            # If batch failed (empty response), mark errors
            if not responses:
                with results_lock:
                    for r in batch:
                        if r['request_id'] not in results:
                            results[r['request_id']] = {'error': 'Pipeline Error', 'success': False}

            # Notify queue we are done
            for _ in batch:
                request_queue.task_done()

        except Exception as e:
            logger.error(f"Worker Loop Error: {e}")


# Routes
@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        req_id = data.get('request_id')
        query = data.get('query')

        if not req_id or not query:
            return jsonify({'error': 'Missing params'}), 400

        # Check cache/duplicate
        with results_lock:
            if req_id in results:
                return jsonify(results[req_id]), 200

        # Enqueue with timeout to prevent blocking indefinitely
        try:
            request_queue.put({
                'request_id': req_id,
                'query': query,
                'timestamp': time.time()
            }, timeout=5)
        except:
            return jsonify({'error': 'Service overloaded'}), 503

        # Wait for result (Polling)
        timeout = 300
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            with results_lock:
                if req_id in results:
                    return jsonify(results.pop(req_id)), 200
            time.sleep(0.05)  # Check every 50ms

        return jsonify({'error': 'Timeout'}), 504

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'node': NODE_NUMBER,
        'total_nodes': TOTAL_NODES
    }), 200


def main():
    global pipeline_instance

    # Initialize Pipeline (Loads local models)
    pipeline_instance = DistributedPipeline()

    # Start Worker Thread
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

    # Start Server
    hostname = NODE_0_IP_RAW.split(':')[0]
    port = int(NODE_0_IP_RAW.split(':')[1]) if ':' in NODE_0_IP_RAW else 8000

    logger.info(f"Node 0 Orchestrator listening on {hostname}:{port}")
    # Optimized Flask configuration for production
    app.run(
        host=hostname, 
        port=port, 
        threaded=True,
        processes=1,  # Single process with threading for shared memory
        use_reloader=False,  # Disable reloader in production
        debug=False  # Disable debug mode
    )


if __name__ == '__main__':
    main()