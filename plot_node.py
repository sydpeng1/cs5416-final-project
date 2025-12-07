#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_node_samples(metrics_path: str) -> Dict[int, List[Tuple[float, float, float]]]:
    """Read node-level samples from metrics.jsonl.

    Returns:
        dict: node_number -> list of tuples (timestamp, cpu_percent, rss_mb)
    """
    per_node: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "node_sample":
                continue
            node = int(obj.get("node_number", -1))
            ts = float(obj.get("timestamp", 0.0))
            cpu = float(obj.get("cpu_percent", 0.0))
            rss = float(obj.get("rss_mb", 0.0))
            per_node[node].append((ts, cpu, rss))
    # sort by timestamp
    for node in per_node:
        per_node[node].sort(key=lambda x: x[0])
    return per_node


def normalize_time(per_node: Dict[int, List[Tuple[float, float, float]]]) -> Dict[int, List[Tuple[float, float, float]]]:
    """Convert timestamps to seconds since the first sample of that node."""
    out: Dict[int, List[Tuple[float, float, float]]] = {}
    for node, samples in per_node.items():
        if not samples:
            out[node] = []
            continue
        t0 = samples[0][0]
        out[node] = [(s[0] - t0, s[1], s[2]) for s in samples]
    return out


def palette(n: int) -> List[str]:
    colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#000000', '#ff00aa'
    ]
    if n <= len(colors):
        return colors[:n]
    # repeat if more nodes
    return [colors[i % len(colors)] for i in range(n)]


def plot_per_node(per_node_rel: Dict[int, List[Tuple[float, float, float]]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for node, samples in per_node_rel.items():
        if not samples:
            continue
        t = [s[0] for s in samples]
        cpu = [s[1] for s in samples]
        rss = [s[2] for s in samples]

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(t, cpu, color='#1f77b4', linewidth=1.8)
        axes[0].set_ylabel('CPU (%)')
        axes[0].grid(True, linestyle='--', alpha=0.4)
        axes[0].set_title(f'Node {node} CPU and RSS over time')

        axes[1].plot(t, rss, color='#d62728', linewidth=1.8)
        axes[1].set_xlabel('Time since start (s)')
        axes[1].set_ylabel('RSS (MB)')
        axes[1].grid(True, linestyle='--', alpha=0.4)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'node_{node}_timeline.png'))
        plt.close(fig)


def plot_combined(per_node_rel: Dict[int, List[Tuple[float, float, float]]], out_dir: str):
    nodes = sorted(per_node_rel.keys())
    if not nodes:
        return
    cols = palette(len(nodes))

    # CPU overlay
    plt.figure(figsize=(12, 4))
    for i, node in enumerate(nodes):
        samples = per_node_rel[node]
        if not samples:
            continue
        t = [s[0] for s in samples]
        cpu = [s[1] for s in samples]
        plt.plot(t, cpu, color=cols[i], linewidth=1.6, label=f'Node {node}')
    plt.xlabel('Time since start (s)')
    plt.ylabel('CPU (%)')
    plt.title('CPU% overlay by node')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'nodes_cpu_overlay.png'))
    plt.close()

    # RSS overlay
    plt.figure(figsize=(12, 4))
    for i, node in enumerate(nodes):
        samples = per_node_rel[node]
        if not samples:
            continue
        t = [s[0] for s in samples]
        rss = [s[2] for s in samples]
        plt.plot(t, rss, color=cols[i], linewidth=1.6, label=f'Node {node}')
    plt.xlabel('Time since start (s)')
    plt.ylabel('RSS (MB)')
    plt.title('RSS overlay by node')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'nodes_rss_overlay.png'))
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Plot node CPU% and RSS timelines from metrics.jsonl node_sample lines.')
    ap.add_argument('--metrics-file', default='metrics.jsonl', help='Path to metrics.jsonl')
    ap.add_argument('--output-dir', default='metrics_plots', help='Output directory for plots')
    ap.add_argument('--combined', action='store_true', help='Also produce combined overlays across nodes')
    args = ap.parse_args()

    per_node = read_node_samples(args.metrics_file)
    if not per_node:
        print(f'No node_sample records found in {args.metrics_file}')
        return
    per_node_rel = normalize_time(per_node)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_per_node(per_node_rel, args.output_dir)
    if args.combined:
        plot_combined(per_node_rel, args.output_dir)

    print(f'Wrote node plots to {args.output_dir}')


if __name__ == '__main__':
    main()
