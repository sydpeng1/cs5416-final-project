#!/bin/bash

### ---- 1. Enable conda inside scripts ----
# THIS is the correct way on macOS (miniforge)
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

### ---- 2. Activate your environment ----
conda activate final5416

### ---- 3. Debug: print python path ----
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "PYTHONPATH=$PYTHONPATH"
echo "NODE_NUMBER=$NODE_NUMBER"

### ---- 4. Start correct node ----
if [ "$NODE_NUMBER" -eq 0 ]; then
    echo "Starting Node 0..."
    exec python pipeline.py
elif [ "$NODE_NUMBER" -eq 1 ]; then
    echo "Starting Node 1..."
    exec python node1_retrieval.py
elif [ "$NODE_NUMBER" -eq 2 ]; then
    echo "Starting Node 2..."
    export KMP_DUPLICATE_LIB_OK=TRUE      # prevent OpenMP crash
    exec python node2_inference.py
else
    echo "Invalid NODE_NUMBER: $NODE_NUMBER"
    exit 1
fi