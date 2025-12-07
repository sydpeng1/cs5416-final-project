#!/usr/bin/env python3
"""
Client script for testing the ML inference pipeline.
Maintains at most N concurrent in-flight requests (sliding window).
Total requests = 1000. When one finishes, send another, until all done.
"""

import os
import time
import requests
import json
import threading
from datetime import datetime
from typing import Dict, Optional

NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
SERVER_URL = f"http://{NODE_0_IP}/query"

TEST_QUERIES = [
    "How do I return a defective product?",
    "What is your refund policy?",
    "My order hasn't arrived yet, tracking number is ABC123",
    "How do I update my billing information?",
    "Is there a warranty on electronic items?",
    "Can I change my shipping address after placing an order?",
    "What payment methods do you accept?",
    "How long does shipping typically take?"
]


TOTAL_REQUESTS = 1000
MAX_INFLIGHT = 100 

results = {}
results_lock = threading.Lock()
requests_sent = []
requests_lock = threading.Lock()
print_lock = threading.Lock()

inflight_sema = threading.Semaphore(MAX_INFLIGHT)


def safe_print(msg):
    with print_lock:
        print(msg)


def send_request_async(request_id: str, query: str, send_time: float):
    """Send a single request to the server asynchronously"""
    try:
        safe_print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sending request {request_id}\nQuery: {query}")

        payload = {
            'request_id': request_id,
            'query': query
        }

        start_time = time.time()
        response = requests.post(SERVER_URL, json=payload, timeout=300)
        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()

            with results_lock:
                results[request_id] = {
                    'result': result,
                    'elapsed_time': elapsed_time,
                    'send_time': send_time,
                    'success': True
                }

            msg = (
                f"\n[{datetime.now().strftime('%H:%M:%S')}] Response received for {request_id} in {elapsed_time:.2f}s\n"
                f"  Generated Response: {result.get('generated_response', '')[:100]}...\n"
                f"  Sentiment: {result.get('sentiment')}\n"
                f"  Is Toxic: {result.get('is_toxic')}"
            )
            safe_print(msg)
        else:
            with results_lock:
                results[request_id] = {
                    'error': f"HTTP {response.status_code}",
                    'elapsed_time': elapsed_time,
                    'send_time': send_time,
                    'success': False
                }

            msg = (
                f"\n[{datetime.now().strftime('%H:%M:%S')}] Error for {request_id}: HTTP {response.status_code}\n"
                f"  Response: {response.text}"
            )
            safe_print(msg)

    except requests.exceptions.Timeout:
        safe_print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} timed out after 300s")
        with results_lock:
            results[request_id] = {
                'error': 'Timeout',
                'send_time': send_time,
                'success': False
            }
    except requests.exceptions.ConnectionError:
        safe_print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Failed to connect to server for {request_id}")
        with results_lock:
            results[request_id] = {
                'error': 'Connection error',
                'send_time': send_time,
                'success': False
            }
    except Exception as e:
        safe_print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error for {request_id}: {str(e)}")
        with results_lock:
            results[request_id] = {
                'error': str(e),
                'send_time': send_time,
                'success': False
            }
    finally:
        inflight_sema.release()


def main():
    """
    Main function: keep at most MAX_INFLIGHT in-flight requests.
    Total requests = TOTAL_REQUESTS.
    """
    print("=" * 70)
    print("ML INFERENCE PIPELINE CLIENT")
    print("=" * 70)
    print(f"Server URL: {SERVER_URL}")
    print(f"Total Requests: {TOTAL_REQUESTS}")
    print(f"Max In-Flight: {MAX_INFLIGHT}")
    print("=" * 70)

    try:
        health_response = requests.get(f"http://{NODE_0_IP}/health", timeout=5)
        if health_response.status_code == 200:
            print(f"Server is healthy: {health_response.json()}")
        else:
            print(f"Server health check returned status {health_response.status_code}")
    except Exception:
        print("Could not reach server health endpoint")

    start_time = time.time()
    threads = []

    for i in range(TOTAL_REQUESTS):
        inflight_sema.acquire()

        request_id = f"req_{int(time.time())}_{i}"
        query = TEST_QUERIES[i % len(TEST_QUERIES)]

        with requests_lock:
            requests_sent.append({
                'request_id': request_id,
                'query': query,
                'send_time': time.time()
            })

        thread = threading.Thread(
            target=send_request_async,
            args=(request_id, query, time.time())
        )
        thread.start()
        threads.append(thread)

    print("\nAll requests dispatched, waiting for responses...")
    for thread in threads:
        thread.join()

    total_time = time.time() - start_time

    with results_lock:
        successful = sum(1 for r in results.values() if r.get('success', False))
        failed = TOTAL_REQUESTS - successful

        latencies = [r['elapsed_time'] for r in results.values() if r.get('success')]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        throughput = successful / total_time if total_time > 0 else 0.0

    msg = [
        "BENCHMARK COMPLETE",
        f"Total Requests: {TOTAL_REQUESTS}",
        f"Successful:     {successful}",
        f"Failed:         {failed}",
        f"Throughput:     {throughput:.2f} req/sec",
        f"Avg Latency:    {avg_latency:.2f} sec",
        f"Total Time:     {total_time:.2f} sec"
    ]

    width = 40
    print("\n")
    print("╔" + "═" * (width - 2) + "╗")
    for line in msg:
        print(f"║ {line:<{width - 4}} ║")
    print("╚" + "═" * (width - 2) + "╝")
    print("\n")


if __name__ == "__main__":
    main()
