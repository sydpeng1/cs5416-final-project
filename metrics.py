import time
import json
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
import os
try:
    import resource  # Unix-specific; provides ru_maxrss
except Exception:
    resource = None
try:
    import psutil
    _psutil_proc = psutil.Process(os.getpid())
except Exception:
    psutil = None
    _psutil_proc = None

try:
    from memory_profiler import memory_usage
except Exception:
    memory_usage = None


def _compute_stats(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    n = len(samples)
    s = sum(samples)
    avg = s / n
    mn = min(samples)
    mx = max(samples)
    # population std
    var = (sum((x - avg) ** 2 for x in samples) / n) if n > 0 else 0.0
    std = var ** 0.5
    return {"avg": avg, "min": mn, "max": mx, "std": std}


@dataclass
class StepMetrics:
    step_name: str
    node_number: int
    timestamp: float
    batch_size: int
    request_ids: List[str]
    duration_ms: float
    rss_samples_mb: List[float]
    rss_aggregates: Dict[str, float]


class MetricsCollector:
    def __init__(self,
                 enable_metrics: bool = True,
                 metrics_file_path: str = "metrics.jsonl",
                 metrics_summary_file_path: str = "metrics_summary.jsonl",
                 metrics_flush_interval_s: float = 10.0,
                 metrics_sample_interval_s: float = 0.1,
                 immediate_flush: bool = False):
        self.enable_metrics = enable_metrics
        self.metrics_file_path = metrics_file_path
        self.metrics_summary_file_path = metrics_summary_file_path
        self.metrics_flush_interval_s = metrics_flush_interval_s
        self.metrics_sample_interval_s = metrics_sample_interval_s
        self.immediate_flush = immediate_flush
        self._buffer: List[StepMetrics] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Rolling aggregates per step
        self._rolling: Dict[str, Dict[str, Dict[str, float]]] = {}

    def start(self):
        if not self.enable_metrics:
            return
        # Skip background writer if immediate flush mode
        if self.immediate_flush:
            return
        if self._thread is None and self.metrics_flush_interval_s and self.metrics_flush_interval_s > 0:
            self._thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None

    def record_step(self, m: StepMetrics):
        if not self.enable_metrics:
            return
        with self._lock:
            self._buffer.append(m)
            # Update rolling aggregates
            if m.step_name not in self._rolling:
                self._rolling[m.step_name] = {
                    'duration_ms': {'count': 0, 'sum': 0.0, 'sumsq': 0.0, 'min': float('inf'), 'max': float('-inf')},
                    'rss_mb': {'count': 0, 'sum': 0.0, 'sumsq': 0.0, 'min': float('inf'), 'max': float('-inf')},
                }
            r = self._rolling[m.step_name]
            # Duration
            r['duration_ms']['count'] += 1
            r['duration_ms']['sum'] += m.duration_ms
            r['duration_ms']['sumsq'] += m.duration_ms * m.duration_ms
            r['duration_ms']['min'] = min(r['duration_ms']['min'], m.duration_ms)
            r['duration_ms']['max'] = max(r['duration_ms']['max'], m.duration_ms)
            # Memory (use max sample per step)
            if m.rss_samples_mb:
                mem_val = max(m.rss_samples_mb)
                r['rss_mb']['count'] += 1
                r['rss_mb']['sum'] += mem_val
                r['rss_mb']['sumsq'] += mem_val * mem_val
                r['rss_mb']['min'] = min(r['rss_mb']['min'], mem_val)
                r['rss_mb']['max'] = max(r['rss_mb']['max'], mem_val)

    def _writer_loop(self):
        while not self._stop.is_set():
            time.sleep(self.metrics_flush_interval_s)
            self.flush()

    def flush(self):
        if not self.enable_metrics:
            return
        records: List[StepMetrics] = []
        with self._lock:
            if not self._buffer:
                # Write summary if available
                summary = self._serialize_summary()
                if summary:
                    try:
                        with open(self.metrics_summary_file_path, "a") as f:
                            f.write(json.dumps(summary) + "\n")
                    except Exception:
                        pass
                return
            records = self._buffer
            self._buffer = []
        try:
            with open(self.metrics_file_path, "a") as f:
                for r in records:
                    # Serialize dataclass to JSON lines
                    obj = {
                        "step_name": r.step_name,
                        "node_number": r.node_number,
                        "timestamp": r.timestamp,
                        "batch_size": r.batch_size,
                        "request_ids": r.request_ids,
                        "duration_ms": r.duration_ms,
                        "rss_samples_mb": r.rss_samples_mb,
                        "rss_aggregates": r.rss_aggregates,
                    }
                    f.write(json.dumps({"type": "occurrence", **obj}) + "\n")
            # Write summary to separate file
            summary = self._serialize_summary()
            if summary:
                with open(self.metrics_summary_file_path, "a") as sf:
                    sf.write(json.dumps(summary) + "\n")
        except Exception:
            # Best-effort; avoid crashing pipeline for metrics I/O issues
            pass

    def _serialize_summary(self) -> Optional[Dict[str, Any]]:
        if not self._rolling:
            return None
        steps = {}
        for step, r in self._rolling.items():
            # Duration stats
            dur = r['duration_ms']
            if dur['count'] > 0:
                avg = dur['sum'] / dur['count']
                var = (dur['sumsq'] / dur['count']) - (avg * avg)
                dur_stats = {"avg": avg, "min": dur['min'], "max": dur['max'], "std": (var ** 0.5 if var > 0 else 0.0)}
            else:
                dur_stats = {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
            # Memory stats
            mem = r['rss_mb']
            if mem['count'] > 0:
                mavg = mem['sum'] / mem['count']
                mvar = (mem['sumsq'] / mem['count']) - (mavg * mavg)
                mem_stats = {"avg": mavg, "min": mem['min'], "max": mem['max'], "std": (mvar ** 0.5 if mvar > 0 else 0.0)}
            else:
                mem_stats = {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
            steps[step] = {
                "duration_ms": dur_stats,
                "rss_mb": mem_stats,
            }
        return {"steps": steps}


class StepSampler:
    """
    Context manager to sample wall time, CPU time, and RSS memory during a callable execution.
    Returns raw samples arrays and aggregates.
    """

    def __init__(self, sample_interval_s: float = 0.1):
        self.sample_interval_s = sample_interval_s
        self.time_samples_ms: List[float] = []
        self.rss_samples_mb: List[float] = []
        self._sampler_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def __enter__(self):
        self._start_time = time.perf_counter()
        self._stop.clear()
        self._sampler_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._sampler_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._sampler_thread:
            self._sampler_thread.join(timeout=1.0)
        # Ensure at least one sample capturing end
        end_perf = time.perf_counter()
        elapsed_ms = (end_perf - self._start_time) * 1000.0
        self.time_samples_ms.append(elapsed_ms)
        # Compute aggregates (memory only)
        self.rss_aggregates = _compute_stats(self.rss_samples_mb)
        return False  # do not suppress exceptions

    def _sampling_loop(self):
        # Continuously sample until stop
        start_perf = time.perf_counter()
        while not self._stop.is_set():
            now_perf = time.perf_counter()
            elapsed_ms = (now_perf - start_perf) * 1000.0
            self.time_samples_ms.append(elapsed_ms)
            # Sample RSS with robust fallbacks
            sampled = False
            # Try memory_profiler
            if memory_usage is not None:
                try:
                    rss_list = memory_usage(-1, interval=0.0, timeout=1.0)
                    if rss_list and rss_list[0] is not None:
                        self.rss_samples_mb.append(float(rss_list[0]))
                        sampled = True
                except Exception:
                    sampled = False
            # Fallback to psutil
            if not sampled and _psutil_proc is not None:
                try:
                    rss_mb = _psutil_proc.memory_info().rss / (1024 * 1024)
                    self.rss_samples_mb.append(float(rss_mb))
                    sampled = True
                except Exception:
                    sampled = False
            # Fallback to resource.ru_maxrss
            if not sampled and resource is not None:
                try:
                    ru = resource.getrusage(resource.RUSAGE_SELF)
                    rss = ru.ru_maxrss
                    # ru_maxrss units differ per platform:
                    # - Linux: kilobytes
                    # - macOS: bytes
                    if rss > 0:
                        # Heuristic: treat values > 1e9 as bytes (macOS), else KB (Linux)
                        if rss > 1_000_000_000:
                            rss_mb = rss / (1024 * 1024)
                        else:
                            rss_mb = rss / 1024.0
                        self.rss_samples_mb.append(float(rss_mb))
                except Exception:
                    pass
            time.sleep(self.sample_interval_s)


class NodeMonitor:
    """
    Background sampler to record node-level CPU% and RSS MB periodically.
    Writes JSONL records with type="node_sample" to the given metrics file.
    """

    def __init__(self, node_number: int, metrics_file_path: str = "metrics.jsonl", sample_interval_s: float = 0.02):
        self.node_number = node_number
        self.metrics_file_path = metrics_file_path
        self.sample_interval_s = sample_interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._buffer: List[Dict[str, Any]] = []
        # Prepare process handle
        self._proc = _psutil_proc
        # Prime cpu_percent for differential readings (Linux may require a first interval)
        try:
            if self._proc is not None:
                self._proc.cpu_percent(interval=0.0)
        except Exception:
            pass

    def start(self):
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        try:
            with open(self.metrics_file_path, "a") as f:
                for rec in self._buffer:
                    f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        finally:
            self._buffer = []

    def _loop(self):
        last_flush = time.time()
        while not self._stop.is_set():
            ts = time.time()
            cpu_pct = None
            rss_mb = None
            # CPU percent via psutil (use a short interval on Linux to avoid 0%)
            try:
                if self._proc is not None:
                    cpu_pct = float(self._proc.cpu_percent(interval=max(self.sample_interval_s, 0.02)))
            except Exception:
                cpu_pct = None
            # RSS via psutil/resource fallback
            sampled = False
            try:
                if self._proc is not None:
                    rss_mb = float(self._proc.memory_info().rss) / (1024 * 1024)
                    sampled = True
            except Exception:
                sampled = False
            if not sampled and resource is not None:
                try:
                    ru = resource.getrusage(resource.RUSAGE_SELF)
                    rss = ru.ru_maxrss
                    if rss > 0:
                        if rss > 1_000_000_000:
                            rss_mb = rss / (1024 * 1024)
                        else:
                            rss_mb = rss / 1024.0
                except Exception:
                    pass

            self._buffer.append({
                "type": "node_sample",
                "node_number": self.node_number,
                "timestamp": ts,
                "cpu_percent": cpu_pct if cpu_pct is not None else 0.0,
                "rss_mb": rss_mb if rss_mb is not None else 0.0,
            })

            # Flush every ~1s to reduce I/O overhead
            if time.time() - last_flush >= 1.0 or len(self._buffer) >= 200:
                self._flush()
                last_flush = time.time()

            time.sleep(self.sample_interval_s)
