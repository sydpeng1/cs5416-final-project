#!/bin/bash

# Run script for ML Inference Pipeline
# This script will be executed on each node

echo "Starting pipeline on Node $NODE_NUMBER..."
if [ "$NODE_NUMBER" -eq 0 ]; then
    echo "Starting Node $NODE_NUMBER..."
    exec python endpoint_node.py

elif [ "$NODE_NUMBER" -eq 1 ]; then
    echo "Starting Node $NODE_NUMBER..."
    exec python rag_node1.py

elif [ "$NODE_NUMBER" -eq 2 ]; then
    echo "Starting Node $NODE_NUMBER..."
    exec python rag_node2.py

else
    echo "Error: Invalid NODE_NUMBER ($NODE_NUMBER). Must be 0, 1, or 2."
    exit 1
fi