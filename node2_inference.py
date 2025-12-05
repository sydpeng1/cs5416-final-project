#!/usr/bin/env python3
import os
import logging
import time

from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============== ENV FIX (macOS OpenMP) ==============
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ====================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Node2")

app = Flask(__name__)

NODE_2_IP = os.environ.get("NODE_2_IP", "localhost:8002")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_TOKENS = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None


def load_llm():
    global tokenizer, model
    logger.info(f"[Node2] Loading LLM: {LLM_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    # CPU 上用 float32，GPU 才用 float16
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=dtype,
    ).to(device)

    model.eval()
    logger.info("[Node2] LLM ready.")


def generate_answer(query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "Answer in concise form."},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.01,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][len(inputs["input_ids"][0]) :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@app.route("/generate", methods=["POST"])
def generate_route():
    try:
        data = request.json or {}
        queries = data.get("queries", [])
        contexts = data.get("contexts", [])

        if not queries or not contexts or len(queries) != len(contexts):
            return jsonify({"error": "invalid queries or contexts"}), 400

        batch_start = time.time()
        outputs = []
        for idx, (q, ctx) in enumerate(zip(queries, contexts), start=1):
            t0 = time.time()
            outputs.append(generate_answer(q, ctx))
            logger.info(
                "[Node2] Generated response idx=%d time=%.3fs",
                idx,
                time.time() - t0,
            )

        total_time = time.time() - batch_start
        avg = total_time / len(outputs) if outputs else 0.0
        logger.info(
            "[Node2] Batch complete size=%d total_time=%.3fs avg=%.3fs",
            len(outputs),
            total_time,
            avg,
        )

        return jsonify({"responses": outputs}), 200

    except Exception as e:
        logger.error(f"[Node2] Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


def main():
    load_llm()
    host, port = NODE_2_IP.split(":")
    logger.info(f"[Node2] Running on {host}:{port}")
    app.run(host=host, port=int(port), threaded=True)


if __name__ == "__main__":
    main()
