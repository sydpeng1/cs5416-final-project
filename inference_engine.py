import os
import sqlite3
import logging
import torch
from typing import List, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline as hf_pipeline
)

logger = logging.getLogger(__name__)

# Constants
RERANKER_MODEL = 'BAAI/bge-reranker-base'
LLM_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
MAX_TOKENS = 128
TRUNCATE_LENGTH = 512
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')


class InferenceWorker:
    def __init__(self, micro_batch_size=4):
        self.micro_batch_size = micro_batch_size

        # Hardware Detection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.dtype = torch.float16
        else:
            self.device = torch.device('cpu')
            self.dtype = torch.float32

        logger.info(f"Initializing InferenceWorker on {self.device}")
        self._load_models()

    def _load_models(self):
        logger.info("Loading Reranker & LLM...")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL).to(self.device)
        self.rerank_model.eval()

        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        # Required for batched generation to work correctly.
        # Otherwise, the model sees padding as input and outputs garbage.
        self.llm_tokenizer.padding_side = 'left'
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, dtype=self.dtype).to(self.device)
        self.llm_model.eval()

        logger.info("Loading Analysis Pipelines...")
        # device id: 0 for CUDA, -1 for CPU. Note: MPS requires passing object, handled below.
        pipe_device = self.device if self.device.type == 'mps' else (0 if self.device.type == 'cuda' else -1)

        self.sentiment_pipe = hf_pipeline("sentiment-analysis",
                                          model='nlptown/bert-base-multilingual-uncased-sentiment',
                                          device=pipe_device)
        self.safety_pipe = hf_pipeline("text-classification", model='unitary/toxic-bert', device=pipe_device)
        logger.info("Models Loaded.")

    def fetch_documents(self, doc_ids_batch: List[List[int]]) -> List[List[Dict]]:
        db_path = os.path.join(DOCUMENTS_DIR, 'documents.db')
        if not os.path.exists(db_path): return [[] for _ in doc_ids_batch]

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        batch_docs = []
        for doc_ids in doc_ids_batch:
            docs = []
            for doc_id in doc_ids:
                cursor.execute('SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?', (doc_id,))
                res = cursor.fetchone()
                if res: docs.append({'title': res[1], 'content': res[2]})
            batch_docs.append(docs)
        conn.close()
        return batch_docs

    def rerank(self, queries: List[str], docs_batch: List[List[Dict]]) -> List[List[Dict]]:
        reranked = []
        for q, docs in zip(queries, docs_batch):
            if not docs:
                reranked.append([])
                continue
            pairs = [[q, d['content']] for d in docs]
            with torch.no_grad():
                inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt',
                                               max_length=512).to(self.device)
                scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
            doc_scores = list(zip(docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked.append([d for d, _ in doc_scores])
        return reranked

    def generate(self, queries: List[str], docs_batch: List[List[Dict]]) -> List[str]:
        total_responses = []

        for i in range(0, len(queries), self.micro_batch_size):
            chunk_q = queries[i: i + self.micro_batch_size]
            chunk_d = docs_batch[i: i + self.micro_batch_size]

            batch_prompts = []
            for q, docs in zip(chunk_q, chunk_d):
                context = "\n".join([f"- {d['title']}: {d['content'][:200]}" for d in docs[:3]])
                msg = [
                    {"role": "system",
                     "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"}
                ]
                text = self.llm_tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                batch_prompts.append(text)

            # Padding is now handled correctly on the LEFT
            inputs = self.llm_tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    temperature=0.01,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )

            # Correctly slice output based on input length
            input_len = inputs.input_ids.shape[1]
            new_tokens = generated_ids[:, input_len:]
            responses = self.llm_tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            total_responses.extend(responses)

        return total_responses

    def analyze(self, texts: List[str]) -> List[Dict]:
        truncated = [t[:TRUNCATE_LENGTH] for t in texts]
        raw_sent = self.sentiment_pipe(truncated)
        raw_safe = self.safety_pipe(truncated)
        s_map = {'1 star': 'very negative', '2 stars': 'negative', '3 stars': 'neutral', '4 stars': 'positive',
                 '5 stars': 'very positive'}

        results = []
        for i in range(len(texts)):
            results.append({
                'text': texts[i],
                'sentiment': s_map.get(raw_sent[i]['label'], 'neutral'),
                'is_toxic': "true" if raw_safe[i]['score'] > 0.5 else "false"
            })
        return results

    def run_pipeline(self, queries: List[str], doc_ids_batch: List[List[int]]) -> List[Dict]:
        """Runs the full inference chain: Fetch -> Rerank -> Generate -> Analyze"""
        docs = self.fetch_documents(doc_ids_batch)
        ranked = self.rerank(queries, docs)
        texts = self.generate(queries, ranked)
        return self.analyze(texts)
