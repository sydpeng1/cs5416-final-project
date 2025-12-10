#!/bin/bash

# Run script for ML Inference Pipeline
# This script will be executed on each node

# =================================================================
#  CENTRALIZED CLUSTER CONFIGURATION
#  Tuned for: 3 Nodes, each with 6 vCPUs (Intel Xeon Gold)
# =================================================================

# Default to GPU mode unless explicitly told otherwise
if [ "$ONLY_CPU" = "true" ]; then
    MODE="CPU"
else
    MODE="GPU"
fi

echo "=================================================="
echo "Initializing Node $NODE_NUMBER in [$MODE MODE]"
echo "Hardware: $(nproc) Cores detected."
echo "=================================================="

# --- SHARED SETTINGS ---
# 4 workers ensure the pipeline stays full (Local + Remote + IO Wait)
# This works well for both CPU (latency masking) and GPU (throughput).
export MAX_WORKERS=4
export BATCH_SIZE=32
export BATCH_TIMEOUT=0.1

# Threading: Limit to 50% capacity per worker.
# Since Node 0/2 run ~2 tasks at once, 2 * 50% = 100% Load.
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))


if [ "$MODE" = "CPU" ]; then
    export LLM_GEN_MICROBATCH_SIZE=8

else
    export LLM_GEN_MICROBATCH_SIZE=32

fi


if [ "$NODE_NUMBER" -eq 0 ]; then
    echo "Starting Node $NODE_NUMBER..."
    exec python3 pipeline.py

elif [ "$NODE_NUMBER" -eq 1 ]; then
    echo "Starting Node $NODE_NUMBER..."
    export OMP_NUM_THREADS=$(nproc)
    exec python3 node1_retrieval.py

elif [ "$NODE_NUMBER" -eq 2 ]; then
    echo "Starting Node $NODE_NUMBER..."
    exec python3 node2_inference.py

else
    echo "Error: Invalid NODE_NUMBER ($NODE_NUMBER). Must be 0, 1, or 2."
    exit 1
fi
