# benchmarks.py
# ------------------------------------------------------------
# Sorting benchmark script for my dissertation.
# Compares Bubble, Merge, Quick, and TimSort across a few
# input distributions, and records runtime + CPU% + memory.
# Exports raw runs and grouped summaries; optional plots/XLSX.
# ------------------------------------------------------------

import os, sys, csv, time, math, argparse, platform, threading, json
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List

# third-party libs
try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("This script needs numpy and pandas. Install with:\n  pip install numpy pandas")
    raise

# psutil is optional — if missing, CPU%/RSS sampling gracefully disables
try:
    import psutil
except ImportError:
    psutil = None

import tracemalloc


# =========================
# Sorting algorithms
# =========================
def bubble_sort(arr: List[int]) -> List[int]:
    """
    Bubble Sort (O(n^2)). Copies input and early-stops if already sorted.
    Keeping it here for completeness / baseline.
    """
    a = arr.copy()
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


def merge_sort(arr: List[int]) -> List[int]:
    """
    Standard recursive Merge Sort (O(n log n)), stable.
    Returns a new sorted list (doesn’t mutate input).
    """
    n = len(arr)
    if n <= 1:
        return arr.copy()
    mid = n // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left: List[int], right: List[int]) -> List[int]:
    """
    Merge step used by merge_sort. Also stable.
    """
    i = j = 0
    out: List[int] = []
    nl, nr = len(left), len(right)
    while i < nl and j < nr:
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    if i < nl: out.extend(left[i:])
    if j < nr: out.extend(right[j:])
    return out


def quick_sort(arr: List[int]) -> List[int]:
    """
    Simple recursive Quick Sort (middle element pivot).
    Returns a new list (not in-place).
    """
    n = len(arr)
    if n <= 1:
        return arr.copy()
    pivot = arr[n // 2]
    left  = [x for x in arr if x < pivot]
    mid   = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)


def tim_sort(arr: List[int]) -> List[int]:
    """
    Uses Python’s built-in sort (TimSort). Fast and stable.
    """
    out = arr.copy()
    out.sort()
    return out


# mapping used by the CLI
ALGORITHMS: Dict[str, Callable[[List[int]], List[int]]] = {
    "Bubble": bubble_sort,
    "Merge":  merge_sort,
    "Quick":  quick_sort,
    "TimSort": tim_sort,
}


# =========================
# Dataset generation
# =========================
def generate_dataset(n: int, dist: str, rng: np.random.Generator) -> List[int]:
    """
    Builds test arrays of length n with different structures:
    - random: uniform ints
    - reversed: strictly decreasing
    - disorder10/disorder50: partially shuffled indices
    - duplicates: small alphabet => many ties
    """
    if dist == "random":
        return rng.integers(0, max(10, n * 10), size=n, dtype=np.int64).tolist()
    if dist == "reversed":
        return np.arange(n - 1, -1, -1, dtype=np.int64).tolist()
    if dist == "disorder10":
        base = np.arange(n, dtype=np.int64)
        k = max(1, int(0.10 * n))
        idx = rng.choice(n, size=k, replace=False)
        shuffled = base[idx].copy()
        rng.shuffle(shuffled)
        base[idx] = shuffled
        return base.tolist()
    if dist == "disorder50":
        base = np.arange(n, dtype=np.int64)
        k = max(1, int(0.50 * n))
        idx = rng.choice(n, size=k, replace=False)
        shuffled = base[idx].copy()
        rng.shuffle(shuffled)
        base[idx] = shuffled
        return base.tolist()
    if dist == "duplicates":
        # small alphabet grows slowly with n so we keep duplicates common
        alphabet = min(8, max(3, int(math.log2(n + 2))))
        return rng.integers(0, alphabet, size=n, dtype=np.int64).tolist()
    raise ValueError(f"Unknown distribution: {dist}")


# =========================
# Measurement (runtime, CPU%, memory)
# =========================
@dataclass
class RunRecord:
    """
    One row in the raw CSV (one trial).
    """
    algorithm: str
    size: int
    distribution: str
    trial: int
    runtime_sec: float
    cpu_avg_percent: float
    cpu_peak_percent: float
    rss_peak_delta_bytes: int
    py_heap_peak_bytes: int
    correct: bool


def _is_sorted(a: List[int]) -> bool:
    """quick correctness check (non-decreasing)."""
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


def run_once_with_cpu(func: Callable[[List[int]], List[int]],
                      data: List[int],
                      sample_interval: float = 0.05) -> RunRecord:
    """
    Runs a single sort call on a background thread while (optionally) sampling
    CPU% and RSS via psutil. Also tracks Python heap peak via tracemalloc.
    """
    proc = psutil.Process(os.getpid()) if psutil else None
    rss_start = proc.memory_info().rss if proc else 0

    result_container = {"out": None}
    def worker():
        result_container["out"] = func(data)

    tracemalloc.start()
    if proc: proc.cpu_percent(interval=None)  # reset window

    t0 = time.perf_counter()
    th = threading.Thread(target=worker, daemon=True)
    th.start()

    cpu_samples: List[float] = []
    rss_peak = rss_start

    # sample while the sort runs
    while th.is_alive():
        if proc:
            cpu = proc.cpu_percent(interval=sample_interval)  # blocks for interval
            cpu_samples.append(cpu)
            rss_now = proc.memory_info().rss
            if rss_now > rss_peak: rss_peak = rss_now
        else:
            time.sleep(sample_interval)

    th.join()
    dt = time.perf_counter() - t0

    current, py_peak = tracemalloc.get_traced_memory()  # current not used; keeping for completeness
    tracemalloc.stop()

    cpu_avg = float(sum(cpu_samples) / len(cpu_samples)) if cpu_samples else 0.0
    cpu_peak = float(max(cpu_samples)) if cpu_samples else 0.0
    rss_peak_delta = int(max(0, rss_peak - rss_start))

    out = result_container["out"]
    ok = _is_sorted(out) if out is not None else False

    return RunRecord(
        algorithm="", size=0, distribution="", trial=0,
        runtime_sec=float(dt),
        cpu_avg_percent=cpu_avg,
        cpu_peak_percent=cpu_peak,
        rss_peak_delta_bytes=rss_peak_delta,
        py_heap_peak_bytes=int(py_peak),
        correct=bool(ok)
    )


# =========================
# Summaries
# =========================
def summarize_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by (algorithm, size, distribution) and computes summary stats.
    NOTE: column names and sort order are kept exactly as original.
    """
    g = df.groupby(["algorithm","size","distribution"], as_index=False).agg(
        MedianTimeSec=("runtime_sec","median"),
        MeanTimeSec=("runtime_sec","mean"),
        StdTimeSec=("runtime_sec","std"),
        P95TimeSec=("runtime_sec", lambda x: float(np.percentile(x, 95))),
        CPU_Avg_Percent=("cpu_avg_percent","mean"),
        CPU_Peak_Percent=("cpu_peak_percent","mean"),
        RSS_Peak_Delta_Bytes=("rss_peak_delta_bytes","mean"),
        PyHeap_Peak_Bytes=("py_heap_peak_bytes","mean"),
        AllTrialsCorrect=("correct","all"),
    )
    g["StdTimeSec"] = g["StdTimeSec"].fillna(0.0)
    return g.sort_values(["distribution","size","algorithm"])


# =========================
# Env capture
# =========================
def capture_environment(outdir: str) -> dict:
    """
    Saves basic machine info (useful in the appendix to show setup).
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2) if psutil else None,
        "logical_cores": psutil.cpu_count(logical=True) if psutil else None,
        "physical_cores": psutil.cpu_count(logical=False) if psutil else None,
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "environment_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


# =========================
# CLI / Main
# =========================
def parse_args() -> argparse.Namespace:
    """
    CLI options kept exactly the same as the original script.
    """
    p = argparse.ArgumentParser(description="Sorting benchmarks with CPU% & memory sampling (Methodology-ready).")
    p.add_argument("--sizes", type=str, default="1000,10000,100000,1000000",
                   help="Comma-separated sizes, e.g., 1000,10000,100000,1000000")
    p.add_argument("--dists", type=str,
                   default="random,disorder50,reversed,disorder10,duplicates",
                   help="Distributions: random, disorder50, reversed, disorder10, duplicates")
    p.add_argument("--algos", type=str, default="Bubble,Merge,Quick,TimSort",
                   help="Subset of algorithms to run")
    p.add_argument("--repeats", type=int, default=5,
                   help="Trials per (algo,size,dist)")
    p.add_argument("--bubble_max_n", type=int, default=10_000,
                   help="Skip Bubble for sizes > this value (protect laptops)")
    p.add_argument("--outdir", type=str, default="results",
                   help="Output directory for CSVs")
    p.add_argument("--plots", action="store_true",
                   help="Generate simple PNG plots (requires matplotlib)")
    p.add_argument("--excel", action="store_true",
                   help="Write an Excel workbook with raw+summary")
    return p.parse_args()


def maybe_plot(summary_df: pd.DataFrame, outdir: str) -> None:
    """
    Quick line plots of median runtime vs n, per distribution.
    (Note: expects columns named with capitals — left as-is on purpose.)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    for dist in sorted(summary_df["Distribution"].unique()):
        sub = summary_df[summary_df["Distribution"] == dist]
        plt.figure()
        for algo in sorted(sub["Algorithm"].unique()):
            s = sub[sub["Algorithm"] == algo].sort_values("Size")
            plt.plot(s["Size"], s["MedianTimeSec"], marker="o", label=algo)
        plt.xlabel("n"); plt.ylabel("Median runtime (s)")
        plt.title(f"Median runtime vs n — {dist}")
        plt.legend(); plt.grid(True, alpha=0.3)
        path = os.path.join(outdir, "plots", f"median_runtime_{dist}.png")
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def main():
    """
    Main flow:
      1) parse args
      2) capture environment
      3) generate datasets
      4) run benchmarks and collect raw rows
      5) write raw + summary CSVs (and optional XLSX/plots)
    """
    args = parse_args()
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    dists = [d.strip() for d in args.dists.split(",") if d.strip()]
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    unknown = [a for a in algos if a not in ALGORITHMS]
    if unknown:
        print(f"Unknown algorithms: {unknown}. Available: {list(ALGORITHMS.keys())}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    raw_csv = os.path.join(args.outdir, "benchmark_raw_runs.csv")
    summary_csv = os.path.join(args.outdir, "benchmark_summary.csv")

    env = capture_environment(args.outdir)

    print("=== Sorting Benchmarks (Methodology-aligned) ===")
    print(f"Machine : {env.get('platform','?')} | Python {env.get('python_version','?')}")
    if "logical_cores" in env and env["logical_cores"] is not None:
        print(f"Cores   : {env['logical_cores']} logical ({env.get('physical_cores','?')} physical)")
    if "total_ram_gb" in env and env["total_ram_gb"] is not None:
        print(f"RAM     : ~{env['total_ram_gb']} GB")
    print(f"Algos   : {algos}")
    print(f"Sizes   : {sizes}")
    print(f"Dists   : {dists}")
    print(f"Repeats : {args.repeats}")
    print(f"Bubble  : skip when n > {args.bubble_max_n}")
    print("Running...\n")

    raw_rows: List[RunRecord] = []
    total_cells = len(sizes) * len(dists) * len(algos)
    cell = 0

    for n in sizes:
        # fixed seed style preserved (42 + n), so results stay identical
        rng = np.random.default_rng(42 + n)
        datasets = {dist: generate_dataset(n, dist, rng) for dist in dists}

        for dist in dists:
            data = datasets[dist]
            for algo in algos:
                cell += 1
                if algo == "Bubble" and n > args.bubble_max_n:
                    print(f"[{cell}/{total_cells}] ⏭️  Bubble | n={n:,} | {dist} (skipped)")
                    continue

                func = ALGORITHMS[algo]
                print(f"[{cell}/{total_cells}] ▶️  {algo:7s} | n={n:10,d} | {dist:10s} | ", end="", flush=True)

                times: List[float] = []
                oks: List[bool] = []
                cpu_avgs: List[float] = []
                cpu_peaks: List[float] = []
                rss_peaks: List[int] = []
                heap_peaks: List[int] = []

                for trial in range(1, args.repeats + 1):
                    rec = run_once_with_cpu(func, data)
                    rec.algorithm = algo
                    rec.size = n
                    rec.distribution = dist
                    rec.trial = trial

                    raw_rows.append(rec)
                    times.append(rec.runtime_sec)
                    oks.append(rec.correct)
                    cpu_avgs.append(rec.cpu_avg_percent)
                    cpu_peaks.append(rec.cpu_peak_percent)
                    rss_peaks.append(rec.rss_peak_delta_bytes)
                    heap_peaks.append(rec.py_heap_peak_bytes)

                med = float(np.median(times))
                std = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
                ok_all = all(oks)
                print(f"median={med:.4f}s  std={std:.4f}s  ok={ok_all}")

    # Save raw (env line as a comment on top — same as before)
    raw_df = pd.DataFrame([asdict(r) for r in raw_rows])
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        f.write("# environment: " + json.dumps(env) + "\n")
        raw_df.to_csv(f, index=False)
    print(f"\nSaved raw runs -> {raw_csv}")

    # Build and save summary
    if not raw_df.empty:
        summary_df = summarize_group(raw_df)
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            f.write("# environment: " + json.dumps(env) + "\n")
            summary_df.to_csv(f, index=False)
        print(f"Saved summary  -> {summary_csv}")

        if args.excel:
            xlsx = os.path.join(args.outdir, "benchmarks.xlsx")
            with pd.ExcelWriter(xlsx, engine="xlsxwriter") as w:
                raw_df.to_excel(w, sheet_name="raw_runs", index=False)
                summary_df.to_excel(w, sheet_name="summary", index=False)
            print(f"Saved Excel    -> {xlsx}")

        if args.plots:
            maybe_plot(summary_df, args.outdir)
            print(f"Saved plots    -> {os.path.join(args.outdir, 'plots')}")

        bad = summary_df[~summary_df["AllTrialsCorrect"]]
        if bad.empty:
            print("\nAll algorithm runs produced sorted output ✅")
        else:
            print("\nSome runs failed correctness ❌")
            with pd.option_context("display.max_rows", None):
                print(bad)
    else:
        print("No results produced — check sizes/dists/algos.")


if __name__ == "__main__":
    main()
