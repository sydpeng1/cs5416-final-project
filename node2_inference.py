import gc
import logging
import os
import sqlite3
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import List, Dict

import torch
from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 2))
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')
NODE_2_IP_RAW = os.environ.get('NODE_2_IP', 'localhost:8002')

RERANKER_MODEL_NAME = 'BAAI/bge-reranker-base'
LLM_MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
MAX_TOKENS = 128
TRUNCATE_LENGTH = 512

# Async batch processing configuration
BATCH_SIZE = 16  # Smaller for GPU-heavy inference
BATCH_TIMEOUT = 0.2  # 200ms - balance between latency and batching
MAX_QUEUE_SIZE = 200

# Global state
request_queue = Queue(maxsize=MAX_QUEUE_SIZE)
results = {}
results_lock = threading.Lock()


@dataclass
class InferenceRequest:
    request_id: str
    queries: List[str]
    doc_ids: List[List[int]]
    timestamp: float

# Device Setup (Use GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global Model State
reranker_tokenizer = None
reranker_model = None
llm_tokenizer = None
llm_model = None


def load_models():
    """Loads heavy models into memory at startup."""
    global reranker_tokenizer, reranker_model, llm_tokenizer, llm_model

    logger.info(f"Node 2: Loading models on {device}...")

    # 1. Load Reranker
    logger.info(f"Loading Reranker: {RERANKER_MODEL_NAME}")
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME).to(device)
    reranker_model.eval()

    # 2. Load LLM
    logger.info(f"Loading LLM: {LLM_MODEL_NAME}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    # Use float16 if on GPU for memory savings, otherwise float32 for CPU compatibility
    model_dtype = torch.float16 if device.type == 'cuda' else torch.float32

    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        dtype=model_dtype,
        use_cache=True  # Enable KV-cache for faster generation
    ).to(device)
    llm_model.eval()
    
    # Enable optimizations
    if device.type == 'cuda':
        # Use torch.compile for PyTorch 2.0+ (optional, may take time to compile)
        # llm_model = torch.compile(llm_model, mode="reduce-overhead")
        logger.info("GPU optimizations enabled (KV-cache, float16)")
    else:
        logger.info("CPU mode - using float32 for compatibility")

    logger.info("All models loaded successfully.")


def fetch_documents(doc_ids_batch: List[List[int]]) -> List[List[Dict]]:
    """Fetch actual text content from SQLite using IDs with optimized batch query."""
    db_path = os.path.join(DOCUMENTS_DIR, 'documents.db')

    if not os.path.exists(db_path):
        logger.error(f"Documents DB not found at {db_path}")
        return [[] for _ in doc_ids_batch]

    # Optimize SQLite connection
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Faster column access
    cursor = conn.cursor()

    batch_docs = []
    for doc_ids in doc_ids_batch:
        if not doc_ids:
            batch_docs.append([])
            continue
            
        # Batch fetch all docs at once using IN clause (much faster than individual queries)
        placeholders = ','.join('?' * len(doc_ids))
        query = f'SELECT doc_id, title, content, category FROM documents WHERE doc_id IN ({placeholders})'
        cursor.execute(query, doc_ids)
        
        # Create lookup dict for preserving order
        results_dict = {}
        for row in cursor.fetchall():
            results_dict[row['doc_id']] = {
                'doc_id': row['doc_id'],
                'title': row['title'],
                'content': row['content'],
                'category': row['category']
            }
        
        # Preserve original order
        docs = [results_dict[doc_id] for doc_id in doc_ids if doc_id in results_dict]
        batch_docs.append(docs)

    conn.close()
    return batch_docs


def rerank_documents(queries: List[str], documents_batch: List[List[Dict]]) -> List[List[Dict]]:
    """Rerank retrieved documents using the Cross-Encoder model with TRUE batching."""
    reranked_batches = []
    
    # Collect all pairs across all queries with metadata for reconstruction
    all_pairs = []
    pair_metadata = []  # (query_idx, doc_idx)
    
    for query_idx, (query, documents) in enumerate(zip(queries, documents_batch)):
        if not documents:
            reranked_batches.append([])
            continue
            
        for doc_idx, doc in enumerate(documents):
            all_pairs.append([query, doc['content']])
            pair_metadata.append((query_idx, doc_idx, doc))
    
    if not all_pairs:
        return [[] for _ in queries]
    
    # Batch process ALL pairs at once
    with torch.no_grad():
        inputs = reranker_tokenizer(
            all_pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=TRUNCATE_LENGTH
        ).to(device)
        
        # Single forward pass for all pairs
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
    
    # Reconstruct per-query results
    query_results = {i: [] for i in range(len(queries))}
    for score_val, (query_idx, doc_idx, doc) in zip(scores, pair_metadata):
        query_results[query_idx].append((doc, score_val.item()))
    
    # Sort each query's documents by score
    for query_idx in range(len(queries)):
        if query_idx in query_results and query_results[query_idx]:
            sorted_docs = sorted(query_results[query_idx], key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in sorted_docs])
        else:
            reranked_batches.append([])
    
    return reranked_batches


def generate_responses(queries: List[str], documents_batch: List[List[Dict]]) -> List[str]:
    """Generate final answers using the LLM with TRUE batching."""
    if not queries:
        return []
    
    # Prepare all prompts
    all_texts = []
    for query, documents in zip(queries, documents_batch):
        # Format Context (Use top 3 reranked docs)
        context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc in documents[:3]])
        
        messages = [
            {"role": "system", "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        
        text = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        all_texts.append(text)
    
    # Batch tokenization with padding
    model_inputs = llm_tokenizer(
        all_texts, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=TRUNCATE_LENGTH
    ).to(device)
    
    # Batch generation with optimizations
    with torch.no_grad():
        generated_ids = llm_model.generate(
            **model_inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.01,
            pad_token_id=llm_tokenizer.eos_token_id,
            do_sample=False,  # Greedy decoding for consistency
            use_cache=True,   # Use KV-cache (enabled by default, but explicit here)
            num_beams=1,      # Greedy search (fastest)
            early_stopping=False  # Not needed for greedy
        )
    
    # Decode only the new tokens for each sequence
    input_lengths = model_inputs.input_ids.shape[1]
    responses = []
    for i, output_ids in enumerate(generated_ids):
        # Extract only newly generated tokens
        new_tokens = output_ids[input_lengths:]
        response = llm_tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)
    
    return responses


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
    """Background worker that processes batched inference requests"""
    logger.info("Node 2 Worker thread started. Waiting for inference requests...")
    
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
            
            # Node 2 is slow (inference), balance batching vs latency carefully
            if queue_size > BATCH_SIZE * 2:
                # High load: process quickly to prevent queue overflow
                dynamic_timeout = 0.05  # 50ms
            elif queue_size > BATCH_SIZE / 2:
                # Medium load: short timeout
                dynamic_timeout = BATCH_TIMEOUT / 2
            else:
                # Low load: wait longer to gather larger batch (GPU efficiency)
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
            batch_size = sum(len(req_dict['queries']) for req_dict in batch)
            logger.info(f"Processing batch of {len(batch)} requests ({batch_size} total queries)")
            
            # Flatten all queries and doc_ids from the batch
            all_queries = []
            all_doc_ids = []
            request_metadata = []  # Track which queries belong to which request
            
            for req_dict in batch:
                queries = req_dict['queries']
                doc_ids = req_dict['doc_ids']
                
                # Ensure queries and doc_ids are lists
                if not isinstance(queries, list):
                    queries = [queries]
                if not isinstance(doc_ids, list):
                    doc_ids = [doc_ids]
                
                # Ensure doc_ids is a list of lists
                # If doc_ids[0] is an integer, it means doc_ids is a single list of IDs
                if doc_ids and isinstance(doc_ids[0], int):
                    doc_ids = [doc_ids]  # Wrap in another list
                
                for i, (query, doc_id_list) in enumerate(zip(queries, doc_ids)):
                    all_queries.append(query)
                    # Ensure doc_id_list is a list
                    if isinstance(doc_id_list, int):
                        doc_id_list = [doc_id_list]
                    all_doc_ids.append(doc_id_list)
                    request_metadata.append((req_dict['request_id'], i))
            
            try:
                # Process all queries together
                t_start = time.time()
                
                # 1. Fetch Documents
                documents_batch = fetch_documents(all_doc_ids)
                
                # 2. Rerank
                reranked_docs = rerank_documents(all_queries, documents_batch)
                
                # 3. Generate
                all_responses = generate_responses(all_queries, reranked_docs)
                
                processing_time = time.time() - t_start
                logger.info(f"Batch processing completed in {processing_time:.3f}s")
                
                # 4. Group responses back by request_id
                response_groups = {}
                for (req_id, idx), response in zip(request_metadata, all_responses):
                    if req_id not in response_groups:
                        response_groups[req_id] = {}
                    response_groups[req_id][idx] = response
                
                # 5. Store results
                with results_lock:
                    for req_id, response_dict in response_groups.items():
                        # Convert dict to list in order
                        ordered_responses = [response_dict[i] for i in sorted(response_dict.keys())]
                        results[req_id] = {
                            'responses': ordered_responses,
                            'success': True
                        }
            
            except Exception as e:
                import traceback
                logger.error(f"Batch processing error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error(f"Request data: queries={[type(req_dict.get('queries')) for req_dict in batch]}, "
                           f"doc_ids={[type(req_dict.get('doc_ids')) for req_dict in batch]}")
                with results_lock:
                    for req_dict in batch:
                        results[req_dict['request_id']] = {
                            'error': str(e),
                            'success': False
                        }
            
            # Mark tasks as done
            for _ in batch:
                request_queue.task_done()
        
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            with results_lock:
                for req_dict in batch:
                    if req_dict['request_id'] not in results:
                        results[req_dict['request_id']] = {
                            'error': str(e),
                            'success': False
                        }
            for _ in batch:
                request_queue.task_done()


@app.route('/generate', methods=['POST'])
def generate():
    """Async endpoint - enqueues request and polls for result"""
    try:
        data = request.json
        queries = data.get('queries')
        doc_ids_batch = data.get('doc_ids')

        if not queries or not doc_ids_batch:
            return jsonify({'error': 'Missing queries or doc_ids'}), 400
        
        # Validate and normalize input format
        if not isinstance(queries, list):
            queries = [queries]
        
        if not isinstance(doc_ids_batch, list):
            return jsonify({'error': f'doc_ids must be a list, got {type(doc_ids_batch)}'}), 400
        
        # Ensure doc_ids_batch is a list of lists
        if doc_ids_batch and isinstance(doc_ids_batch[0], int):
            logger.warning(f"Received doc_ids as single list, wrapping it")
            doc_ids_batch = [doc_ids_batch]
        
        # Validate length match
        if len(queries) != len(doc_ids_batch):
            return jsonify({'error': f'Mismatch: {len(queries)} queries but {len(doc_ids_batch)} doc_id lists'}), 400
        
        logger.info(f"Received request: {len(queries)} queries, {len(doc_ids_batch)} doc_id lists")
        
        # Generate request ID
        request_id = data.get('request_id', f"gen_{time.time()}_{id(data)}")
        
        # Check if already processed (deduplication)
        with results_lock:
            if request_id in results:
                result = results.pop(request_id)
                if result['success']:
                    return jsonify({'responses': result['responses']}), 200
                else:
                    return jsonify({'error': result.get('error', 'Unknown error')}), 500
        
        # Enqueue request (using validated/normalized data)
        try:
            request_queue.put({
                'request_id': request_id,
                'queries': queries,  # Already validated above
                'doc_ids': doc_ids_batch,  # Already validated above
                'timestamp': time.time()
            }, timeout=5)
        except:
            return jsonify({'error': 'Queue full, service overloaded'}), 503
        
        # Poll for result
        timeout = 300  # 5 minutes for generation
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            with results_lock:
                if request_id in results:
                    result = results.pop(request_id)
                    if result['success']:
                        return jsonify({'responses': result['responses']}), 200
                    else:
                        return jsonify({'error': result.get('error', 'Unknown error')}), 500
            
            time.sleep(0.02)  # Poll every 20ms
        
        return jsonify({'error': 'Generation timeout'}), 504

    except Exception as e:
        logger.error(f"Generation endpoint error: {e}")
        return jsonify({'error': str(e)}), 500


def main():
    load_models()
    
    # Start background worker thread
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()
    logger.info("Node 2 worker thread started")

    hostname = NODE_2_IP_RAW.split(':')[0]
    port = int(NODE_2_IP_RAW.split(':')[1]) if ':' in NODE_2_IP_RAW else 8002

    logger.info(f"Node 2 Inference Service starting on {hostname}:{port}")
    
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


if __name__ == "__main__":
    main()