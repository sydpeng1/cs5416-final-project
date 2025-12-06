import os
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor

import requests
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

BATCH_SIZE = 16
BATCH_TIMEOUT = 0.1
TRUNCATE_LENGTH = 512
MAX_WORKERS = 2

# Flask App
app = Flask(__name__)
request_queue = Queue()
results = {}
results_lock = threading.Lock()


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
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_str = "cuda"
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.device_str = "mps"
        else:
            self.device = torch.device('cpu')
            self.device_str = "cpu"

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

        # Load Lightweight Models
        logger.info("Loading Embedder (Local)...")
        self.embedder = SentenceTransformer('BAAI/bge-base-en-v1.5', device=self.device_str)

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
            payload_n1 = {'embeddings': embeddings.tolist()}
            resp_n1 = requests.post(self.retrieval_url, json=payload_n1, timeout=30)
            resp_n1.raise_for_status()
            doc_ids_batch = resp_n1.json()['doc_ids']
            logger.info(f"[2] Retrieval (Node 1): {time.time() - t0:.3f}s")

            # --- STEP 3: Generation (Node 2) ---
            t0 = time.time()
            payload_n2 = {
                'queries': queries,
                'doc_ids': doc_ids_batch
            }
            resp_n2 = requests.post(self.inference_url, json=payload_n2, timeout=300)
            resp_n2.raise_for_status()
            responses_text = resp_n2.json()['responses']
            logger.info(f"[3] Generation (Node 2): {time.time() - t0:.3f}s")

            # --- STEP 4: Analysis (Local) ---
            t0 = time.time()
            # Truncate for BERT models to prevent errors
            truncated_texts = [t[:TRUNCATE_LENGTH] for t in responses_text]

            # Run Sentiment & Safety in parallel conceptually (sequential here for simplicity)
            raw_sentiments = self.sentiment_pipe(truncated_texts)
            raw_safety = self.safety_pipe(truncated_texts)

            # Parse Results
            sentiment_map = {
                '1 star': 'very negative', '2 stars': 'negative',
                '3 stars': 'neutral', '4 stars': 'positive', '5 stars': 'very positive'
            }
            final_sentiments = [sentiment_map.get(r['label'], 'neutral') for r in raw_sentiments]
            is_toxic_flags = [r['score'] > 0.5 for r in raw_safety]

            logger.info(f"[4] Analysis: {time.time() - t0:.3f}s")

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

    # Maximum 4 batches "in-flight
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    while True:
        batch = []
        try:
            # 1. Blocking Get (Wait for first item)
            first_req = request_queue.get()
            if first_req is None: break  # Shutdown signal
            batch.append(first_req)

            # 2. Opportunistic Collection
            # Try to grab more items if they are immediately available (up to BATCH_SIZE)
            start_wait = time.time()
            while len(batch) < BATCH_SIZE:
                # Calculate remaining time in timeout window
                remaining = BATCH_TIMEOUT - (time.time() - start_wait)
                if remaining <= 0: break

                try:
                    # Non-blocking get (or very short timeout)
                    req = request_queue.get(timeout=remaining)
                    batch.append(req)
                except Empty:
                    break

            # 3. Convert dicts to Request Objects
            req_objects = [PipelineRequest(r['request_id'], r['query'], r['timestamp']) for r in batch]

            # 4. SUBMIT TO THREAD POOL (Non-blocking!)
            # The main loop immediately goes back to top to fetch next batch
            executor.submit(run_batch_task, pipeline_instance, req_objects)

        except Exception as e:
            logger.error(f"Worker Loop Error: {e}")


def run_batch_task(pipeline: DistributedPipeline, batch_reqs: List[PipelineRequest]):
    """Helper to run processing in a separate thread and save results."""
    try:
        # This blocks THIS thread, but not the main worker loop
        responses = pipeline.process_batch(batch_reqs)

        # Store results
        with results_lock:
            for res in responses:
                results[res.request_id] = {
                    'request_id': res.request_id,
                    'generated_response': res.generated_response,
                    'sentiment': res.sentiment,
                    'is_toxic': res.is_toxic,
                    'success': True
                }

        # Handle total failures (empty response list)
        if not responses:
            with results_lock:
                for r in batch_reqs:
                    if r.request_id not in results:
                        results[r.request_id] = {'error': 'Pipeline Error', 'success': False}

    except Exception as e:
        logger.error(f"Thread Error: {e}")
        with results_lock:
            for r in batch_reqs:
                if r.request_id not in results:
                    results[r.request_id] = {'error': str(e), 'success': False}
    finally:
        for _ in batch_reqs:
            request_queue.task_done()

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

        # Enqueue
        request_queue.put({
            'request_id': req_id,
            'query': query,
            'timestamp': time.time()
        })

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
    app.run(host=hostname, port=port, threaded=True)


if __name__ == '__main__':
    main()
