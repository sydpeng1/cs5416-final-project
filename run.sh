#!/bin/bash

# Run script for ML Inference Pipeline
# This script will be executed on each node

# =================================================================
#  CENTRALIZED CLUSTER CONFIGURATION
#  Tuned for: 3 Nodes, each with 6 vCPUs (Intel Xeon Gold)
# =================================================================

# --- CASE 1: LOCAL MAC (M2 Max) CONFIG ---
# Resource Strategy: Conservative concurrency, High vectorization (MPS loves batches)

#export BATCH_SIZE=16           # Moderate batch size for responsiveness
#export BATCH_TIMEOUT=0.1
#export MAX_WORKERS=2           # Limit concurrency: 1 Local Task + 1 Remote Task.
#                               # Prevents local resource thrashing.
#
#export OMP_NUM_THREADS=4       # Not strictly used by MPS, but good fallback
#export GPU_MICRO_BATCH_SIZE=4  # M2 GPU is powerful; process 4 items at once for speed.


# --- CASE 2: REMOTE LINUX (CPU ONLY) CONFIG ---
# Resource Strategy: High concurrency, Serial processing (CPU hates batches)

## 1. Pipeline Depth
## Keep pipeline full (2 Active Compute + 2 Network Wait)
#export MAX_WORKERS=4
#
## 2. CPU Threading (The Critical Fix)
## Limit PyTorch to 3 cores per task.
## Since Node 0 runs ~2 tasks locally at once, 2 * 3 = 6 Cores. Perfect fit.
#export OMP_NUM_THREADS=3
#
## 3. Batching
#export BATCH_SIZE=16
#export BATCH_TIMEOUT=0.1
## CPU vectorization is poor. Serial processing (1 at a time) is often faster/safer.
## If you had a T4 GPU, you would set this to 4 or 8.
#export GPU_MICRO_BATCH_SIZE=1


# --- CASE 3: REMOTE LINUX (TESLA T4 GPU) CONFIG ---
# Resource Strategy: High concurrency, Parallel processing (GPU loves batches)

#export BATCH_SIZE=32           # Large batches to minimize HTTP overhead
#export BATCH_TIMEOUT=0.1
#export MAX_WORKERS=4           # Maximize pipeline depth
#
## CPU is just a manager here (Kernel launch + Network).
## 4 threads is plenty safe.
#export OMP_NUM_THREADS=4
#
#export GPU_MICRO_BATCH_SIZE=4  # T4 optimization: Process 4 items in parallel on VRAM
# =================================================================

if [ "$NODE_NUMBER" -eq 0 ]; then
    echo "Starting Node $NODE_NUMBER..."
    exec python3 pipeline.py

elif [ "$NODE_NUMBER" -eq 1 ]; then
    echo "Starting Node $NODE_NUMBER..."
    exec python3 node1_retrieval.py

elif [ "$NODE_NUMBER" -eq 2 ]; then
    echo "Starting Node $NODE_NUMBER..."
    exec python3 node2_inference.py

else
    echo "Error: Invalid NODE_NUMBER ($NODE_NUMBER). Must be 0, 1, or 2."
    exit 1
fi