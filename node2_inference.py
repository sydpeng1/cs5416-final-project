import gc
import logging
import os
import sqlite3
from typing import List, Dict
import time

import torch
from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from metrics import MetricsCollector, StepSampler, StepMetrics, NodeMonitor

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
# Even if we receive 32 requests, process them 4 at a time on GPU
GPU_MICRO_BATCH_SIZE = 8

# Device Setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Global Model State
reranker_tokenizer = None
reranker_model = None
llm_tokenizer = None
llm_model = None

# Metrics
metrics = MetricsCollector(
    enable_metrics=True,
    metrics_file_path=f"metrics_node{NODE_NUMBER}.jsonl",
    metrics_summary_file_path=f"metrics_summary_node{NODE_NUMBER}.jsonl",
    immediate_flush=True,
)
node_monitor = None


def load_models():
    """Loads heavy models into memory at startup."""
    global reranker_tokenizer, reranker_model, llm_tokenizer, llm_model

    logger.info(f"Node 2: Loading models on {device}...")

    # 1. Load Reranker
    logger.info(f"Loading Reranker: {RERANKER_MODEL_NAME}")
    t0 = time.perf_counter()
    with StepSampler() as s_rerank:
        reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME).to(device)
        reranker_model.eval()
    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="generate_responses.load_model",
            node_number=NODE_NUMBER,
            timestamp=time.time(),
            batch_size=0,
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s_rerank, 'rss_samples_mb', []),
            rss_aggregates=getattr(s_rerank, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass

    # 2. Load LLM
    logger.info(f"Loading LLM: {LLM_MODEL_NAME}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    # Use float16 if on GPU for memory savings, otherwise float32 for CPU compatibility
    if device.type == 'cuda' or device.type == 'mps':
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    t0 = time.perf_counter()
    with StepSampler() as s_llm:
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            dtype=model_dtype
        ).to(device)
        llm_model.eval()
    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="generate_responses.load_model",
            node_number=NODE_NUMBER,
            timestamp=time.time(),
            batch_size=0,
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s_llm, 'rss_samples_mb', []),
            rss_aggregates=getattr(s_llm, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass

    logger.info("All models loaded successfully.")


def fetch_documents(doc_ids_batch: List[List[int]]) -> List[List[Dict]]:
    """Fetch actual text content from SQLite using IDs."""
    db_path = os.path.join(DOCUMENTS_DIR, 'documents.db')

    if not os.path.exists(db_path):
        logger.error(f"Documents DB not found at {db_path}")
        return [[] for _ in doc_ids_batch]

    t0 = time.perf_counter()
    with StepSampler() as s:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        batch_docs = []
        for doc_ids in doc_ids_batch:
            docs = []
            for doc_id in doc_ids:
                cursor.execute(
                    'SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?',
                    (doc_id,)
                )
                result = cursor.fetchone()
                if result:
                    docs.append({
                        'doc_id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'category': result[3]
                    })
            batch_docs.append(docs)

        conn.close()
    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="fetch_documents",
            node_number=NODE_NUMBER,
            timestamp=time.time(),
            batch_size=len(doc_ids_batch),
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s, 'rss_samples_mb', []),
            rss_aggregates=getattr(s, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass
    return batch_docs


def rerank_documents(queries: List[str], documents_batch: List[List[Dict]]) -> List[List[Dict]]:
    """Rerank retrieved documents using the Cross-Encoder model."""
    reranked_batches = []

    t0 = time.perf_counter()
    with StepSampler() as s:
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue

            # Prepare pairs for cross-encoder
            pairs = [[query, doc['content']] for doc in documents]

            with torch.no_grad():
                inputs = reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=TRUNCATE_LENGTH
                ).to(device)

                # Forward pass
                scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()

            # Zip docs with scores and sort descending
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Return only the documents (sorted)
            reranked_batches.append([doc for doc, _ in doc_scores])

    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="rerank_documents",
            node_number=NODE_NUMBER,
            timestamp=time.time(),
            batch_size=len(queries),
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s, 'rss_samples_mb', []),
            rss_aggregates=getattr(s, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass
    return reranked_batches


def generate_responses_microbatched(queries: List[str], documents_batch: List[List[Dict]]) -> List[str]:
    """Generate final answers using the LLM."""
    total_responses = []
    total_items = len(queries)

    t0 = time.perf_counter()
    with StepSampler() as s:
        for i in range(0, total_items, GPU_MICRO_BATCH_SIZE):
            # Slice the input
            chunk_queries = queries[i: i + GPU_MICRO_BATCH_SIZE]
            chunk_docs = documents_batch[i: i + GPU_MICRO_BATCH_SIZE]

            logger.info(f"   Processing Micro-Batch {i // GPU_MICRO_BATCH_SIZE + 1} ({len(chunk_queries)} items)")

            chunk_responses = []
            for query, documents in zip(chunk_queries, chunk_docs):
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

                model_inputs = llm_tokenizer([text], return_tensors="pt").to(device)

                with torch.no_grad():
                    generated_ids = llm_model.generate(
                        **model_inputs,
                        max_new_tokens=MAX_TOKENS,
                        temperature=0.01,
                        pad_token_id=llm_tokenizer.eos_token_id
                    )

                # Decode only the new tokens
                generated_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                chunk_responses.append(response)

            total_responses.extend(chunk_responses)

    t1 = time.perf_counter()
    try:
        metrics.record_step(StepMetrics(
            step_name="generate_responses",
            node_number=NODE_NUMBER,
            timestamp=time.time(),
            batch_size=total_items,
            request_ids=[],
            duration_ms=(t1 - t0) * 1000.0,
            rss_samples_mb=getattr(s, 'rss_samples_mb', []),
            rss_aggregates=getattr(s, 'rss_aggregates', {})
        ))
        metrics.flush()
    except Exception:
        pass
    return total_responses


# Routes
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'node': NODE_NUMBER,
        'total_nodes': TOTAL_NODES
    }), 200


@app.route('/generate', methods=['POST'])
def generate():
    try:
        req_start = time.perf_counter()
        data = request.json
        queries = data.get('queries')
        doc_ids_batch = data.get('doc_ids')
        request_ids = data.get('request_ids', [])

        if not queries or not doc_ids_batch:
            return jsonify({'error': 'Missing queries or doc_ids'}), 400

        logger.info(f"Processing batch of {len(queries)} requests")

        # 1. Fetch Documents (I/O Bound)
        documents_batch = fetch_documents(doc_ids_batch)

        # 2. Rerank (Compute Bound)
        reranked_docs = rerank_documents(queries, documents_batch)

        # 3. Generate (Compute Bound)
        responses = generate_responses_microbatched(queries, reranked_docs)

        try:
            metrics.record_step(StepMetrics(
                step_name="node2_total",
                node_number=NODE_NUMBER,
                timestamp=time.time(),
                batch_size=len(queries) if queries else 0,
                request_ids=request_ids if isinstance(request_ids, list) else [],
                duration_ms=(time.perf_counter() - req_start) * 1000.0,
                rss_samples_mb=[],
                rss_aggregates={}
            ))
            metrics.flush()
        except Exception:
            pass

        return jsonify({'responses': responses}), 200

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({'error': str(e)}), 500


def main():
    global node_monitor
    load_models()
    try:
        node_monitor = NodeMonitor(node_number=NODE_NUMBER, metrics_file_path=f"metrics_node{NODE_NUMBER}.jsonl", sample_interval_s=0.02)
        node_monitor.start()
        metrics.start()
    except Exception:
        pass

    hostname = NODE_2_IP_RAW.split(':')[0]
    port = int(NODE_2_IP_RAW.split(':')[1]) if ':' in NODE_2_IP_RAW else 8002

    logger.info(f"Node 2 Inference Service starting on {hostname}:{port}")
    app.run(host=hostname, port=port, threaded=True)


if __name__ == "__main__":
    main()