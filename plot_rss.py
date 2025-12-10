#!/usr/bin/env python3
import argparse
import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Expected pipeline step order
STEP_ORDER = [
    # "request_queue_wait",
    "generate_embeddings",
    # "faiss_search.load_index",
    "faiss_search.search",
    "faiss_search.cleanup",
    "faiss_search",
    "fetch_documents",
    "rerank_documents",
    # Fine-grained generate steps if present; otherwise we record aggregate below
    # "generate_responses.load_model",
    "generate_responses.prepare_prompts",
    "generate_responses.tokenize",
    "generate_responses.generate",
    "generate_responses.decode",
    "generate_responses.cleanup",
    # Aggregate generate step (used by current instrumentation)
    "generate_responses",
    "analyze_sentiment",
    "safety_filter",
    # "request_total",
]
STEP_INDEX = {name: i for i, name in enumerate(STEP_ORDER)}


def read_occurrences_from_file(metrics_path: str) -> List[Dict[str, Any]]:
    occurrences = []
    try:
        with open(metrics_path, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # Support both typed records and raw occurrence lines
                if obj.get("type") and obj.get("type") != "occurrence":
                    continue
                rec = obj if obj.get("type") == "occurrence" else {**obj, "type": "occurrence"}
                # Attach a fallback unique id for grouping when request_ids are missing
                rec.setdefault("occurrence_index", idx)
                occurrences.append(rec)
    except Exception:
        pass
    return occurrences


def read_occurrences(metrics_paths: List[str]) -> List[Dict[str, Any]]:
    """Read and merge occurrences from multiple metrics files."""
    all_occ: List[Dict[str, Any]] = []
    for p in metrics_paths:
        all_occ.extend(read_occurrences_from_file(p))
    # Sort by timestamp ascending
    all_occ.sort(key=lambda x: x.get("timestamp", 0))
    return all_occ


def prepare_series_all_samples(occurrences: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, List[float]]]]:
    """Build per-request series with all RSS samples per step.

    Returns:
        dict: request_id -> list of tuples (step_name, rss_samples_mb_list) ordered by STEP_ORDER
    """
    per_req_steps: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for occ in occurrences:
        if not {"step_name", "request_ids", "rss_samples_mb"}.issubset(occ.keys()):
            continue
        step = occ["step_name"]
        if step not in STEP_INDEX:
            continue
        rss_samples = occ.get("rss_samples_mb") or []
        req_ids = occ.get("request_ids") or []
        if not req_ids:
            # Synthesize a request id when missing
            ts = occ.get("timestamp", 0)
            rid = f"req_{int(ts)}_{occ.get('occurrence_index', 0)}"
            req_ids = [rid]
        for rid in req_ids:
            # extend in case multiple fragments recorded for the same step
            per_req_steps[rid][step].extend([float(x) for x in rss_samples])

    # order steps per request by STEP_ORDER
    per_req_ordered: Dict[str, List[Tuple[str, List[float]]]] = {}
    for rid, step_map in per_req_steps.items():
        ordered = [(step, step_map.get(step, [])) for step in STEP_ORDER if step in step_map]
        per_req_ordered[rid] = ordered

    return per_req_ordered


def prepare_durations(occurrences: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, float]]]:
    """Build per-request list of (step_name, duration_ms) ordered by STEP_ORDER.

    If multiple occurrences exist for the same step/request (shouldn't in current pipeline), keep the last.
    """
    per_req_steps: Dict[str, Dict[str, float]] = defaultdict(dict)
    for occ in occurrences:
        if not {"step_name", "request_ids", "duration_ms"}.issubset(occ.keys()):
            continue
        step = occ["step_name"]
        if step not in STEP_INDEX:
            continue
        duration = float(occ.get("duration_ms", 0.0))
        req_ids = occ.get("request_ids") or []
        if not req_ids:
            ts = occ.get("timestamp", 0)
            rid = f"req_{int(ts)}_{occ.get('occurrence_index', 0)}"
            req_ids = [rid]
        for rid in req_ids:
            per_req_steps[rid][step] = duration

    per_req_ordered: Dict[str, List[Tuple[str, float]]] = {}
    for rid, step_map in per_req_steps.items():
        ordered = [(step, step_map.get(step, 0.0)) for step in STEP_ORDER if step in step_map]
        per_req_ordered[rid] = ordered
    return per_req_ordered


def prepare_duration_distribution(occurrences: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Collect duration lists per step across all occurrences, independent of request ids."""
    per_step: Dict[str, List[float]] = defaultdict(list)
    for occ in occurrences:
        if not {"step_name", "duration_ms"}.issubset(occ.keys()):
            continue
        step = occ["step_name"]
        if step not in STEP_INDEX:
            continue
        try:
            per_step[step].append(float(occ.get("duration_ms", 0.0)))
        except Exception:
            continue
    return per_step


def plot_duration_scatter(per_step_durations: Dict[str, List[float]], out_path: str):
    """Scatter plot showing duration distribution per step.

    Each step is plotted at a distinct x-index, with all its occurrence durations scattered vertically.
    """
    steps_present = [s for s in STEP_ORDER if s in per_step_durations and per_step_durations[s]]
    if not steps_present:
        return
    xs = list(range(len(steps_present)))
    plt.figure(figsize=(12, 5))
    # jitter points slightly around integer x for visibility
    for i, step in enumerate(steps_present):
        durations = per_step_durations[step]
        if not durations:
            continue
        jitter = [i + (0.05 * ((j % 5) - 2)) for j in range(len(durations))]
        plt.scatter(jitter, durations, alpha=0.7, s=20)
    abbrev = {
        'request_queue_wait': 'queue_wait',
        'generate_embeddings': 'embeddings',
        'faiss_search.load_index': 'faiss.load',
        'faiss_search.search': 'faiss.search',
        'faiss_search.cleanup': 'faiss.clean',
        'fetch_documents': 'fetch_docs',
        'rerank_documents': 'rerank',
        'generate_responses.load_model': 'resp.load',
        'generate_responses.prepare_prompts': 'resp.prompts',
        'generate_responses.tokenize': 'resp.tokenize',
        'generate_responses.generate': 'resp.generate',
        'generate_responses.decode': 'resp.decode',
        'generate_responses.cleanup': 'resp.clean',
        'generate_responses': 'generate_responses',
        'analyze_sentiment': 'sentiment and safety',
        'safety_filter': 'safety',
    }
    plt.xticks(xs, [abbrev.get(s, s) for s in steps_present], rotation=35, ha='right', fontsize=9)
    plt.ylabel('Duration (ms)')
    plt.title('Step Duration Distribution (scatter)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def prepare_rss_distribution(occurrences: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Collect RSS samples per step across all occurrences, independent of request ids.

    Flattens rss_samples_mb arrays into per-step lists.
    """
    per_step: Dict[str, List[float]] = defaultdict(list)
    for occ in occurrences:
        if not {"step_name", "rss_samples_mb"}.issubset(occ.keys()):
            continue
        step = occ["step_name"]
        if step not in STEP_INDEX:
            continue
        try:
            samples = occ.get("rss_samples_mb") or []
            for s in samples:
                per_step[step].append(float(s))
        except Exception:
            continue
    return per_step


def plot_rss_scatter(per_step_rss: Dict[str, List[float]], out_path: str):
    """Scatter plot showing RSS distribution per step.

    Each step is plotted at a distinct x-index, with all its RSS samples scattered vertically.
    """
    steps_present = [s for s in STEP_ORDER if s in per_step_rss and per_step_rss[s]]
    if not steps_present:
        return
    xs = list(range(len(steps_present)))
    plt.figure(figsize=(12, 5))
    for i, step in enumerate(steps_present):
        rss_vals = per_step_rss[step]
        if not rss_vals:
            continue
        jitter = [i + (0.05 * ((j % 5) - 2)) for j in range(len(rss_vals))]
        plt.scatter(jitter, rss_vals, alpha=0.7, s=20)
    abbrev = {
        'request_queue_wait': 'queue_wait',
        'generate_embeddings': 'embeddings',
        'faiss_search.load_index': 'faiss.load',
        'faiss_search.search': 'faiss.search',
        'faiss_search.cleanup': 'faiss.clean',
        'fetch_documents': 'fetch_docs',
        'rerank_documents': 'rerank',
        'generate_responses.load_model': 'resp.load',
        'generate_responses.prepare_prompts': 'resp.prompts',
        'generate_responses.tokenize': 'resp.tokenize',
        'generate_responses.generate': 'resp.generate',
        'generate_responses.decode': 'resp.decode',
        'generate_responses.cleanup': 'resp.clean',
        'generate_responses': 'generate_responses',
        'analyze_sentiment': 'sentiment',
        'safety_filter': 'safety',
    }
    plt.xticks(xs, [abbrev.get(s, s) for s in steps_present], rotation=35, ha='right', fontsize=9)
    plt.ylabel('RSS (MB)')
    plt.title('Step RSS Distribution (scatter)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_request_all_samples(series: List[Tuple[str, List[float]]], out_path: str, title: str):
    # Assign distinct, high-contrast colors per step
    distinct_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#000000', '#ff0000',
        '#00aa00', '#aa00ff', '#ffaa00', '#00ccff', '#ff00aa', '#964B00',
        '#4B0082', '#228B22'
    ]
    step_colors = {step: distinct_colors[i % len(distinct_colors)] for i, step in enumerate(STEP_ORDER)}

    plt.figure(figsize=(12, 5))

    x_cursor = 0
    tick_positions = []
    tick_labels = []
    min_samples = 3  # skip very short segments to avoid ugly tiny gaps
    # Abbreviations to reduce label crowding
    abbrev = {
        'request_queue_wait': 'queue_wait',
        'generate_embeddings': 'embeddings',
        'faiss_search.load_index': 'faiss.load',
        'faiss_search.search': 'faiss.search',
        'faiss_search.cleanup': 'faiss.clean',
        'faiss_search': 'faiss',
        'fetch_documents': 'fetch_docs',
        'rerank_documents': 'rerank',
        'generate_responses.load_model': 'resp.load',
        'generate_responses.prepare_prompts': 'resp.prompts',
        'generate_responses.tokenize': 'resp.tokenize',
        'generate_responses.generate': 'resp.generate',
        'generate_responses.decode': 'resp.decode',
        'generate_responses.cleanup': 'resp.clean',
        'generate_responses': 'generate_responses',
        'analyze_sentiment': 'sentiment',
        'safety_filter': 'safety',
        'request_total': 'total'
    }

    for step_name, samples in series:
        if not samples or len(samples) < min_samples:
            # Do not plot or advance cursor for very short segments
            continue
        xs = list(range(x_cursor, x_cursor + len(samples)))
        ys = samples
        plt.plot(xs, ys, color=step_colors.get(step_name, 'k'), label=step_name, linewidth=2)
        # Place tick at the boundary and add a gap
        tick_positions.append(x_cursor)
        tick_labels.append(abbrev.get(step_name, step_name))
        x_cursor += len(samples)

    if x_cursor == 0:
        plt.close()
        return

    # De-duplicate legend entries (one per step)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    # Place legend outside to avoid overlap
    plt.legend(unique.values(), [abbrev.get(lbl, lbl) for lbl in unique.keys()],
               bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, borderaxespad=0.)

    plt.xticks(tick_positions, tick_labels, rotation=35, ha='right', fontsize=9)
    plt.xlabel('Step progression (sample index across steps)')
    plt.ylabel('RSS (MB)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    # Add bottom margin for rotated ticks
    plt.gcf().subplots_adjust(bottom=0.25, right=0.80)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_request_durations(series: List[Tuple[str, float]], out_path: str, title: str):
    steps = [s for s, _ in series]
    durations = [d for _, d in series]
    xs = list(range(len(steps)))
    # Use same distinct color palette
    distinct_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#000000', '#ff0000',
        '#00aa00', '#aa00ff', '#ffaa00', '#00ccff', '#ff00aa', '#964B00',
        '#4B0082', '#228B22'
    ]
    colors = [distinct_colors[STEP_INDEX.get(s, 0) % len(distinct_colors)] for s in steps]

    plt.figure(figsize=(10, 4))
    plt.bar(xs, durations, color=colors)
    # Abbreviate labels for clarity
    abbrev_steps = [
        {
            'request_queue_wait': 'queue_wait',
            'generate_embeddings': 'embeddings',
            'faiss_search.load_index': 'faiss.load',
            'faiss_search.search': 'faiss.search',
            'faiss_search.cleanup': 'faiss.clean',
            'faiss_search': 'faiss',
            'fetch_documents': 'fetch_docs',
            'rerank_documents': 'rerank',
            'generate_responses.load_model': 'resp.load',
            'generate_responses.prepare_prompts': 'resp.prompts',
            'generate_responses.tokenize': 'resp.tokenize',
            'generate_responses.generate': 'resp.generate',
            'generate_responses.decode': 'resp.decode',
            'generate_responses.cleanup': 'resp.clean',
            'generate_responses': 'generate_responses',
            'analyze_sentiment': 'sentiment',
            'safety_filter': 'safety',
            'request_total': 'total'
        }.get(s, s) for s in steps
    ]
    plt.xticks(xs, abbrev_steps, rotation=35, ha='right', fontsize=9)
    plt.ylabel('Duration (ms)')
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_combined_overlay(per_req: Dict[str, List[Tuple[str, List[float]]]], out_path: str, max_requests: int = 5):
    # Overlay many colored segments is visually busy; use alpha blending
    # High-contrast colors for overlay
    distinct_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#000000', '#ff0000',
        '#00aa00', '#aa00ff', '#ffaa00', '#00ccff', '#ff00aa', '#964B00',
        '#4B0082', '#228B22'
    ]
    step_colors = {step: distinct_colors[i % len(distinct_colors)] for i, step in enumerate(STEP_ORDER)}
    plt.figure(figsize=(12, 6))
    for i, (rid, series) in enumerate(per_req.items()):
        if i >= max_requests:
            break
        x_cursor = 0
        for step_name, samples in series:
            if not samples:
                continue
            xs = list(range(x_cursor, x_cursor + len(samples)))
            ys = samples
            plt.plot(xs, ys, color=step_colors.get(step_name, 'k'), alpha=0.6)
            x_cursor += len(samples)
    plt.xlabel('Step progression (sample index across steps)')
    plt.ylabel('RSS (MB)')
    plt.title(f'RSS Evolution Overlay (up to {max_requests} requests)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Plot per-request RSS evolution across pipeline steps.")
    ap.add_argument('--metrics-file', help='Path to a single metrics file (optional). If omitted, auto-detect metrics_node*.jsonl')
    ap.add_argument('--metrics-dir', help='Directory to search for metrics files (metrics_*.jsonl). If provided, plots will also be written here.')
    ap.add_argument('--output-dir', default='metrics_plots', help='Directory to write plots')
    ap.add_argument('--combined', action='store_true', help='Produce combined overlay plots (defaults to on unless --per-request is set)')
    ap.add_argument('--per-request', action='store_true', help='Produce per-request plots (disabled by default)')
    ap.add_argument('--max-combined', type=int, default=10, help='Max requests to overlay in combined plot')
    args = ap.parse_args()

    metrics_paths: List[str] = []
    if args.metrics_file:
        metrics_paths = [args.metrics_file]
    else:
        search_dir = args.metrics_dir if args.metrics_dir else os.getcwd()
        pattern = os.path.join(search_dir, 'metrics_node*.jsonl')
        # Auto-detect all node metrics files in provided directory (or CWD)
        metrics_paths = sorted(glob.glob(pattern))
        # Fallback to legacy name if no node files
        legacy = os.path.join(search_dir, 'metrics.jsonl')
        if not metrics_paths and os.path.exists(legacy):
            metrics_paths = [legacy]

    if not metrics_paths:
        print("No metrics files found. Looked for metrics_node*.jsonl and metrics.jsonl.")
        return

    occurrences = read_occurrences(metrics_paths)
    if not occurrences:
        print("No occurrence records found in provided metrics files.")
        return

    per_req = prepare_series_all_samples(occurrences)
    per_req_durations = prepare_durations(occurrences)
    per_step_duration_dist = prepare_duration_distribution(occurrences)
    per_step_rss_dist = prepare_rss_distribution(occurrences)
    if not per_req:
        print("No per-request RSS data available to plot.")
        return

    # If metrics-dir is provided, write plots into that directory
    output_dir = args.metrics_dir if args.metrics_dir else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Default behavior: combined plots unless per-request explicitly requested
    if args.combined or not args.per_request:
        # Combined overlay only
        out_path = os.path.join(output_dir, "rss_combined.png")
        plot_combined_overlay(per_req, out_path, max_requests=args.max_combined)
        # And duration scatter distribution per step
        sout = os.path.join(output_dir, "durations_scatter.png")
        plot_duration_scatter(per_step_duration_dist, sout)
        # And RSS scatter distribution per step
        rsout = os.path.join(output_dir, "rss_scatter.png")
        plot_rss_scatter(per_step_rss_dist, rsout)
    if args.per_request:
        # Per-request plots (always include durations)
        for rid, series in per_req.items():
            if not series:
                continue
            out_path = os.path.join(output_dir, f"rss_request_{rid}.png")
            plot_request_all_samples(series, out_path, title=f"RSS Evolution (all samples) for {rid}")
            if rid in per_req_durations:
                dseries = per_req_durations[rid]
                dout = os.path.join(output_dir, f"durations_request_{rid}.png")
                plot_request_durations(dseries, dout, title=f"Step Durations for {rid}")
        # Also write global scatter distributions
        sout = os.path.join(output_dir, "durations_scatter.png")
        plot_duration_scatter(per_step_duration_dist, sout)
        rsout = os.path.join(output_dir, "rss_scatter.png")
        plot_rss_scatter(per_step_rss_dist, rsout)

    print(f"Wrote plots to {output_dir}")


if __name__ == '__main__':
    main()
