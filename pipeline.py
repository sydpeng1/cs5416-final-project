import os
import gc
import json
import time
import numpy as np
import torch
import faiss
import sqlite3
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from queue import Queue
import threading
from collections import deque
import logging

from metrics import MetricsCollector, StepSampler, StepMetrics

# Read environment variables
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 0))
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8000')
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8000')
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faiss_index.bin')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')

# Configuration
CONFIG = {
    'faiss_index_path': FAISS_INDEX_PATH,
    'documents_path': DOCUMENTS_DIR,
    'faiss_dim': 768, #You must use this dimension
    'max_tokens': 128, #You must use this max token limit
    'retrieval_k': 10, #You must retrieve this many documents from the FAISS index
    'truncate_length': 512, # You must use this truncate length
    'enable_metrics': True,
    'metrics_file_path': 'metrics.jsonl',
    'metrics_summary_file_path': 'metrics_summary.jsonl',
    'metrics_flush_interval_s': 10.0,
    'metrics_sample_interval_s': 0.02,
    'metrics_flush_after_batch': True
}

# Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Request queue and results storage
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

class MonolithicPipeline:

    """
    Deliberately inefficient monolithic pipeline
    """
    
    def __init__(self):
        self.device = torch.device('cpu')
        print(f"Initializing pipeline on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
        print(f"FAISS index path: {CONFIG['faiss_index_path']}")
        print(f"Documents path: {CONFIG['documents_path']}")
        
        # Model names
        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
        self.reranker_model_name = 'BAAI/bge-reranker-base'
        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        self.safety_model_name = 'unitary/toxic-bert'
        # Metrics
        self.metrics = MetricsCollector(
            enable_metrics=CONFIG.get('enable_metrics', True),
            metrics_file_path=CONFIG.get('metrics_file_path', 'metrics.jsonl'),
            metrics_summary_file_path=CONFIG.get('metrics_summary_file_path', 'metrics_summary.jsonl'),
            metrics_flush_interval_s=CONFIG.get('metrics_flush_interval_s', 10.0),
            metrics_sample_interval_s=CONFIG.get('metrics_sample_interval_s', 0.02),
            immediate_flush=CONFIG.get('metrics_flush_after_batch', False),
        )
        self.metrics.start()
        # Track completed request timestamps for throughput (rpm)
        self._completion_times = deque()
        self._throughput_file = 'throughput.jsonl'
    
    def process_request(self, request: PipelineRequest) -> PipelineResponse:
        """
        Backwards-compatible single-request entry point that delegates
        to the batch processor with a batch size of 1.
        """
        responses = self.process_batch([request])
        return responses[0]

    def process_batch(self, requests: List[PipelineRequest]) -> List[PipelineResponse]:
        """
        Main pipeline execution for a batch of requests.
        """
        if not requests:
            return []

        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        queries = [req.query for req in requests]

        print("\n" + "="*60)
        print(f"Processing batch of {batch_size} requests")
        print("="*60)
        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")
        
        # Step 1: Generate embeddings
        print("\n[Step 1/7] Generating embeddings for batch...")
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            query_embeddings = self._generate_embeddings_batch(queries)
        self._record_step_metrics(
            step_name='generate_embeddings',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=[req.request_id for req in requests]
        )

        # Step 2: FAISS ANN search
        print("\n[Step 2/7] Performing FAISS ANN search for batch...")
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            doc_id_batches = self._faiss_search_batch(
                query_embeddings,
                request_ids=[req.request_id for req in requests],
                batch_size=batch_size,
            )
        self._record_step_metrics(
            step_name='faiss_search',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=[req.request_id for req in requests]
        )

        # Step 3: Fetch documents from disk
        print("\n[Step 3/7] Fetching documents for batch...")
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            documents_batch = self._fetch_documents_batch(doc_id_batches)
        self._record_step_metrics(
            step_name='fetch_documents',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=[req.request_id for req in requests]
        )

        # Step 4: Rerank documents
        print("\n[Step 4/7] Reranking documents for batch...")
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            reranked_docs_batch = self._rerank_documents_batch(
                queries,
                documents_batch
            )
        self._record_step_metrics(
            step_name='rerank_documents',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=[req.request_id for req in requests]
        )

        # Step 5: Generate LLM responses
        print("\n[Step 5/7] Generating LLM responses for batch...")
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            responses_text = self._generate_responses_batch(
                queries,
                reranked_docs_batch,
                request_ids=[req.request_id for req in requests],
                batch_size=batch_size,
            )
        self._record_step_metrics(
            step_name='generate_responses',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=[req.request_id for req in requests]
        )

        # Step 6: Sentiment analysis
        print("\n[Step 6/7] Analyzing sentiment for batch...")
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            sentiments = self._analyze_sentiment_batch(responses_text)
        self._record_step_metrics(
            step_name='analyze_sentiment',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=[req.request_id for req in requests]
        )

        # Step 7: Safety filter on responses
        print("\n[Step 7/7] Applying safety filter to batch...")
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            toxicity_flags = self._filter_response_safety_batch(responses_text)
        self._record_step_metrics(
            step_name='safety_filter',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=[req.request_id for req in requests]
        )
        
        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(f"\n✓ Request {request.request_id} processed in {processing_time:.2f} seconds")
            sensitivity_result = "true" if toxicity_flags[idx] else "false"
            responses.append(PipelineResponse(
                request_id=request.request_id,
                generated_response=responses_text[idx],
                sentiment=sentiments[idx],
                is_toxic=sensitivity_result,
                processing_time=processing_time
            ))
            # Update throughput (requests per minute)
            try:
                now = time.time()
                self._completion_times.append(now)
                # prune entries older than 60s
                cutoff = now - 60.0
                while self._completion_times and self._completion_times[0] < cutoff:
                    self._completion_times.popleft()
                rpm = len(self._completion_times)
                # Append to throughput.jsonl
                with open(self._throughput_file, 'a') as tf:
                    tf.write(json.dumps({'timestamp': now, 'rpm': rpm}) + "\n")
                logging.info(f"[metrics] throughput rpm={rpm}")
            except Exception as e:
                logging.warning(f"throughput update failed: {e}")
        # Immediate flush of metrics after batch completes (if enabled)
        try:
            if CONFIG.get('metrics_flush_after_batch', False):
                self.metrics.flush()
        except Exception:
            pass

        return responses

    def _record_step_metrics(self, step_name: str, sampler: StepSampler, batch_size: int, request_ids: List[str]):
        try:
            m = StepMetrics(
                step_name=step_name,
                node_number=NODE_NUMBER,
                timestamp=time.time(),
                batch_size=batch_size,
                request_ids=request_ids,
                duration_ms=sampler.time_samples_ms[-1] if sampler.time_samples_ms else 0.0,
                rss_samples_mb=sampler.rss_samples_mb,
                rss_aggregates=sampler.rss_aggregates,
            )
            self.metrics.record_step(m)
            logging.info(f"[metrics] {step_name} duration_ms={m.duration_ms:.2f} rss_avg_mb={m.rss_aggregates['avg']:.2f}")
        except Exception as e:
            logging.warning(f"metrics record failed for {step_name}: {e}")
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Step 2: Generate embeddings for a batch of queries"""
        model = SentenceTransformer(self.embedding_model_name).to(self.device)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        del model
        gc.collect()
        return embeddings
    
    def _faiss_search_batch(self, query_embeddings: np.ndarray, request_ids: List[str], batch_size: int) -> List[List[int]]:
        """Step 3: Perform FAISS ANN search for a batch of embeddings"""
        if not os.path.exists(CONFIG['faiss_index_path']):
            raise FileNotFoundError("FAISS index not found. Please create the index before running the pipeline.")
        
        print("Loading FAISS index")
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            index = faiss.read_index(CONFIG['faiss_index_path'])
        self._record_step_metrics(
            step_name='faiss_search.load_index',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=request_ids,
        )

        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            query_embeddings = query_embeddings.astype('float32')
            _, indices = index.search(query_embeddings, CONFIG['retrieval_k'])
        self._record_step_metrics(
            step_name='faiss_search.search',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=request_ids,
        )

        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            del index
            gc.collect()
        self._record_step_metrics(
            step_name='faiss_search.cleanup',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=request_ids,
        )
        return [row.tolist() for row in indices]
    
    def _fetch_documents_batch(self, doc_id_batches: List[List[int]]) -> List[List[Dict]]:
        """Step 4: Fetch documents for each query in the batch using SQLite"""
        db_path = f"{CONFIG['documents_path']}/documents.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        documents_batch = []
        for doc_ids in doc_id_batches:
            documents = []
            for doc_id in doc_ids:
                cursor.execute(
                    'SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?',
                    (doc_id,)
                )
                result = cursor.fetchone()
                if result:
                    documents.append({
                        'doc_id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'category': result[3]
                    })
            documents_batch.append(documents)
        conn.close()
        return documents_batch
    
    def _rerank_documents_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[List[Dict]]:
        """Step 5: Rerank retrieved documents for each query in the batch"""
        tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name).to(self.device)
        model.eval()
        reranked_batches = []
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue
            pairs = [[query, doc['content']] for doc in documents]
            with torch.no_grad():
                inputs = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=CONFIG['truncate_length']
                ).to(self.device)
                scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        del model, tokenizer
        gc.collect()
        return reranked_batches
    
    def _generate_responses_batch(self, queries: List[str], documents_batch: List[List[Dict]], request_ids: List[str], batch_size: int) -> List[str]:
        """Step 6: Generate LLM responses for each query in the batch"""
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                dtype=torch.float16,
            ).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self._record_step_metrics(
            step_name='generate_responses.load_model',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=request_ids,
        )
        responses = []
        # Prepare prompts
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            texts = []
            for query, documents in zip(queries, documents_batch):
                context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc in documents[:3]])
                messages = [
                    {"role": "system",
                     "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
                    {"role": "user",
                     "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)
        self._record_step_metrics(
            step_name='generate_responses.prepare_prompts',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=request_ids,
        )

        # Tokenize
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        self._record_step_metrics(
            step_name='generate_responses.tokenize',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=request_ids,
        )

        # Generate
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=CONFIG['max_tokens'],
                temperature=0.01,
                pad_token_id=tokenizer.eos_token_id
            )
        self._record_step_metrics(
            step_name='generate_responses.generate',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=request_ids,
        )

        # Slice and decode
        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            sliced_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            decoded = tokenizer.batch_decode(sliced_ids, skip_special_tokens=True)
            responses.extend(decoded)
        self._record_step_metrics(
            step_name='generate_responses.decode',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=request_ids,
        )

        with StepSampler(CONFIG['metrics_sample_interval_s']) as sampler:
            del model, tokenizer
            gc.collect()
        self._record_step_metrics(
            step_name='generate_responses.cleanup',
            sampler=sampler,
            batch_size=batch_size,
            request_ids=request_ids,
        )
        return responses
    
    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Step 7: Analyze sentiment for each generated response"""
        classifier = hf_pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=self.device
        )
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = classifier(truncated_texts)
        sentiment_map = {
            '1 star': 'very negative',
            '2 stars': 'negative',
            '3 stars': 'neutral',
            '4 stars': 'positive',
            '5 stars': 'very positive'
        }
        sentiments = []
        for result in raw_results:
            sentiments.append(sentiment_map.get(result['label'], 'neutral'))
        del classifier
        gc.collect()
        return sentiments
    
    def _filter_response_safety_batch(self, texts: List[str]) -> List[bool]:
        """Step 8: Filter responses for safety for each entry in the batch"""
        classifier = hf_pipeline(
            "text-classification",
            model=self.safety_model_name,
            device=self.device
        )
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = classifier(truncated_texts)
        toxicity_flags = []
        for result in raw_results:
            toxicity_flags.append(result['score'] > 0.5)
        del classifier
        gc.collect()
        return toxicity_flags


# Global pipeline instance
pipeline = None

def process_requests_worker():
    """Worker thread that processes requests from the queue"""
    global pipeline
    while True:
        try:
            request_data = request_queue.get()
            if request_data is None:  # Shutdown signal
                break
            
            # Create request object
            start_processing_ts = time.time()
            req = PipelineRequest(
                request_id=request_data['request_id'],
                query=request_data['query'],
                timestamp=start_processing_ts
            )
            try:
                enqueue_ts = request_data.get('enqueue_ts')
                if enqueue_ts is not None:
                    queue_wait_s = max(0.0, start_processing_ts - float(enqueue_ts))
                    m_wait = StepMetrics(
                        step_name='request_queue_wait',
                        node_number=NODE_NUMBER,
                        timestamp=time.time(),
                        batch_size=1,
                        request_ids=[req.request_id],
                        duration_ms=queue_wait_s * 1000.0,
                        rss_samples_mb=[],
                        rss_aggregates={'min': 0.0, 'max': 0.0, 'avg': 0.0, 'std': 0.0},
                    )
                    pipeline.metrics.record_step(m_wait)
            except Exception as e:
                logging.warning(f"metrics record failed before processing (queue wait): {e}")

            # Process request
            response = pipeline.process_request(req)
            
            # Store result
            with results_lock:
                results[request_data['request_id']] = {
                    'request_id': response.request_id,
                    'generated_response': response.generated_response,
                    'sentiment': response.sentiment,
                    'is_toxic': response.is_toxic
                }
            # Metrics: total latency (queue wait + processing) recorded AFTER processing
            try:
                enqueue_ts = request_data.get('enqueue_ts')
                if enqueue_ts is not None:
                    queue_wait_s = max(0.0, start_processing_ts - float(enqueue_ts))
                    total_latency_ms = queue_wait_s * 1000.0 + response.processing_time * 1000.0
                    m_total = StepMetrics(
                        step_name='request_total',
                        node_number=NODE_NUMBER,
                        timestamp=time.time(),
                        batch_size=1,
                        request_ids=[response.request_id],
                        duration_ms=total_latency_ms,
                        rss_samples_mb=[],
                        rss_aggregates={'min': 0.0, 'max': 0.0, 'avg': 0.0, 'std': 0.0},
                    )
                    pipeline.metrics.record_step(m_total)
            except Exception as e:
                logging.warning(f"metrics record failed after processing (total latency): {e}")
            
            request_queue.task_done()
        except Exception as e:
            print(f"Error processing request: {e}")
            request_queue.task_done()


@app.route('/query', methods=['POST'])
def handle_query():
    """Handle incoming query requests"""
    try:
        data = request.json
        request_id = data.get('request_id')
        query = data.get('query')
        
        if not request_id or not query:
            return jsonify({'error': 'Missing request_id or query'}), 400
        
        # Check if result already exists (request already processed)
        with results_lock:
            if request_id in results:
                return jsonify(results[request_id]), 200
        
        print(f"queueing request {request_id}")
        # Add to queue
        request_queue.put({
            'request_id': request_id,
            'query': query,
            'enqueue_ts': time.time()
        })

        # Wait for processing (with timeout). Very inefficient - would suggest using a more efficient waiting and timeout mechanism.
        timeout = 300  # 5 minutes
        start_wait = time.time()
        while True:
            with results_lock:
                if request_id in results:
                    result = results.pop(request_id)
                    return jsonify(result), 200
            
            if time.time() - start_wait > timeout:
                return jsonify({'error': 'Request timeout'}), 504
            
            time.sleep(0.1)
        
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
    """
    Main execution function
    """
    global pipeline
    
    print("="*60)
    print("MONOLITHIC CUSTOMER SUPPORT PIPELINE")
    print("="*60)
    print(f"\nRunning on Node {NODE_NUMBER} of {TOTAL_NODES} nodes")
    print(f"Node IPs: 0={NODE_0_IP}, 1={NODE_1_IP}, 2={NODE_2_IP}")
    print("\nNOTE: This implementation is deliberately inefficient.")
    print("Your task is to optimize this for a 3-node cluster.\n")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = MonolithicPipeline()
    print("Pipeline initialized!")
    
    # Start worker thread
    worker_thread = threading.Thread(target=process_requests_worker, daemon=True)
    worker_thread.start()
    print("Worker thread started!")
    
    # Start Flask server
    print(f"\nStarting Flask server")
    hostname = NODE_0_IP.split(':')[0]
    port = int(NODE_0_IP.split(':')[1]) if ':' in NODE_0_IP else 8000
    app.run(host=hostname, port=port, threaded=True)
    # On server stop, flush metrics
    try:
        pipeline.metrics.flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
