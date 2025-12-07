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

## Where metrics come from

- Node 0 (orchestrator): per-request totals and selected local steps; also node-level sampling.
- Node 1 (retrieval): FAISS search steps; also node-level sampling.
- Node 2 (inference): document fetch/rerank/generate steps; also node-level sampling.

Each node writes its own `metrics_node<N>.jsonl` (and `metrics_summary_node<N>.jsonl`). To create cluster-wide plots, you can aggregate multiple files into one.

## Aggregating metrics across nodes (optional)

If all three nodes write to different files/machines, collect them into one file before plotting combined overlays.

- Copy remote files locally (example):

```bash
# From your laptop (macOS zsh)
scp user@node0:/path/to/metrics_node0.jsonl node0_metrics.jsonl
scp user@node1:/path/to/metrics_node1.jsonl node1_metrics.jsonl
scp user@node2:/path/to/metrics_node2.jsonl node2_metrics.jsonl
```

- Concatenate into a single file:

```bash
cat node0_metrics.jsonl node1_metrics.jsonl node2_metrics.jsonl > metrics_all.jsonl
```

This unified file can be used by both plotting scripts. They will internally ignore unrelated record types.

Optional: If you want to filter only specific record types, use `jq`:

```bash
# Only node samples
jq -c 'select(.type=="node_sample")' metrics_all.jsonl > nodes_only.jsonl

# Only occurrences
jq -c 'select(.type=="occurrence" or .type==null)' metrics_all.jsonl > occurrences_only.jsonl
```

## Plotting per-request RSS and durations

`plot_rss.py` reads `occurrence` lines and outputs per-request charts by default.

Basic usage:

```bash
# Per-request plots from a single metrics file
python3 plot_rss.py --metrics-file metrics_node0.jsonl --output-dir metrics_plots

# Per-request plots from aggregated metrics
python3 plot_rss.py --metrics-file metrics_all.jsonl --output-dir metrics_plots
```

Combined overlays (no individual request charts):

```bash
# Overlay RSS evolution across up to N requests
python3 plot_rss.py --metrics-file metrics_all.jsonl --combined --max-combined 15 --output-dir metrics_plots
```

Outputs include:
- `rss_request_<REQUEST_ID>.png`: RSS evolution across steps (all samples).
- `durations_request_<REQUEST_ID>.png`: Bar chart of per-step durations (ms).
- With `--combined`: `rss_combined.png` and `durations_combined.png` overlays.

Notes:
- The script understands fine-grained steps such as `faiss_search.load_index`, `faiss_search.search`, `generate_responses.tokenize`, etc.
- It only uses `occurrence` records; `node_sample` lines are ignored if present in the same file.

## Plotting per-node CPU% and RSS timelines

`plot_node.py` reads `node_sample` lines and outputs timelines for each node; with `--combined`, it overlays all nodes.

Basic usage:

```bash
# Per-node plots from a single metrics file
python3 plot_node.py --metrics-file metrics_node0.jsonl --output-dir metrics_plots

# From aggregated file (multiple nodes)
python3 plot_node.py --metrics-file metrics_all.jsonl --output-dir metrics_plots

# Also produce combined overlays across nodes
python3 plot_node.py --metrics-file metrics_all.jsonl --combined --output-dir metrics_plots
```

Outputs include:
- `node_<N>_timeline.png`: Two-panel plot (CPU% and RSS MB) vs. time since that node's first sample.
- With `--combined`: `nodes_cpu_overlay.png` and `nodes_rss_overlay.png` overlaying all nodes.

Notes:
- Time is normalized per node (relative to each node’s first sample). Overlays are for comparison, not strict cross-node alignment.
- The script ignores non-`node_sample` lines if present in the same file.

## Troubleshooting

- Empty plots or "No records found": ensure the relevant record types exist in your input file (`occurrence` for `plot_rss.py`, `node_sample` for `plot_node.py`).
- Missing `matplotlib`: run `pip install -r requirements.txt` inside your virtualenv.
- Very short steps may be skipped in RSS plots to reduce clutter; durations still appear in the bar chart.

## Quick recap

- Use `plot_rss.py` to understand per-request memory evolution and step durations.
- Use `plot_node.py` to see CPU%/memory trends per node and across nodes.
- For distributed runs, concatenate metrics files and use `--combined` for overlays.
