# Plotting Metrics

This guide shows how to generate plots from the metrics produced by the pipeline and microservices.

The repo contains two plotting scripts:
- `plot_rss.py`: Per-request RSS (MB) evolution across pipeline steps and per-step durations (ms).
- `plot_node.py`: Per-node CPU% and RSS (MB) timelines sampled continuously by each node.

## Prerequisites

- Python 3.9+ (any recent Python 3 is fine)
- Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Make sure your services are running with metrics enabled. Each service writes to a `metrics.jsonl` file in its working directory. The file contains two kinds of records:
  - `type = occurrence`: per-step metrics (used by `plot_rss.py`)
  - `type = node_sample`: periodic CPU%/RSS samples (used by `plot_node.py`)

> Note: If `type` is missing on some older lines, `plot_rss.py` treats them as `occurrence` for backward compatibility.

Each node writes its own `metrics_node<N>.jsonl` (and `metrics_summary_node<N>.jsonl`). To create cluster-wide plots, you can aggregate multiple files into one.

Basic usage:

```bash
# plot all metrics_node*.jsonl files under the directory
python3 plot_rss.py --metrics-dir ${directory}
```

## Plotting per-node CPU% and RSS timelines

`plot_node.py` reads `node_sample` lines and outputs timelines for each node; with `--combined`, it overlays all nodes.

Basic usage:

```bash
# Per-node plots from a single metrics file
python3 plot_node.py --metrics-file metrics_node0.jsonl --output-dir metrics_plots

# Also produce combined overlays across nodes
python3 plot_node.py --metrics-file metrics_all.jsonl --combined --output-dir metrics_plots
```

Notes:
- Time is normalized per node (relative to each node’s first sample). Overlays are for comparison, not strict cross-node alignment.
- The script ignores non-`node_sample` lines if present in the same file.

## Quick recap

- Use `plot_rss.py` to understand per-request memory evolution and step durations.
- Use `plot_node.py` to see CPU%/memory trends per node and across nodes.
- For distributed runs, concatenate metrics files and use `--combined` for overlays.
