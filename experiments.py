"""
experiments.py - Batch Experiment Runner for Memory-Bounded Search Algorithms.

Runs systematic comparisons across multiple puzzles and memory limits,
computing:
- Success Rate (%)
- Average Solution Length (for solved puzzles)
- Average Optimality Gap (vs. shortest known solution)
- Average Execution Time (seconds)
- Average Nodes Expanded
- Average Peak Memory (states stored simultaneously)
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import pandas as pd

from puzzle import scramble, GOAL_STATE
from algorithms import astar, idastar, rbfs, sma_star

ALGORITHM_MAP: Dict[str, Callable] = {
    "A*": astar,
    "IDA*": idastar,
    "RBFS": rbfs,
    "SMA*": sma_star,
}


def generate_benchmark_puzzles(
    num_puzzles: int,
    scramble_moves: int,
    base_seed: int = 42
) -> List[Tuple[Tuple[int, ...], Optional[int]]]:
    """
    Generates a reproducible set of benchmark puzzles.
    Also computes the ground-truth optimal solution length for each puzzle
    using A* / IDA* with a generous memory limit.
    """
    benchmarks: List[Tuple[Tuple[int, ...], Optional[int]]] = []

    for i in range(num_puzzles):
        puzzle_seed = base_seed + (i * 7919)
        start_state = scramble(scramble_moves, seed=puzzle_seed)

        # Compute optimal reference length using A* (or fallback to IDA*)
        ref = astar(start_state, GOAL_STATE, mem_limit=200000, time_limit=5.0)
        opt_len = ref["solution_length"] if ref["solved"] else None

        if opt_len is None:
            ref_ida = idastar(start_state, GOAL_STATE, mem_limit=200000, time_limit=5.0)
            opt_len = ref_ida["solution_length"] if ref_ida["solved"] else None

        benchmarks.append((start_state, opt_len))

    return benchmarks


def run_batch_experiments(
    puzzles_with_opt: List[Tuple[Tuple[int, ...], Optional[int]]],
    algorithms: List[str],
    memory_limits: List[int],
    time_limit_per_run: float = 3.0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs each selected algorithm on each puzzle under each specified memory cap.

    Returns:
    - raw_df: DataFrame containing every individual trial run.
    - agg_df: Aggregated summary statistics grouped by Algorithm and Memory Limit.
    """
    raw_records: List[Dict[str, Any]] = []
    total_steps = len(algorithms) * len(memory_limits) * len(puzzles_with_opt)
    step_count = 0

    for mem_lim in memory_limits:
        for algo_name in algorithms:
            algo_func = ALGORITHM_MAP[algo_name]

            for p_idx, (start_state, opt_len) in enumerate(puzzles_with_opt):
                step_count += 1
                if progress_callback:
                    msg = f"Testing {algo_name} | Memory Cap: {mem_lim} | Puzzle #{p_idx + 1}"
                    progress_callback(step_count, total_steps, msg)

                res = algo_func(
                    start_state,
                    GOAL_STATE,
                    mem_limit=mem_lim,
                    time_limit=time_limit_per_run
                )

                # Calculate optimality gap if solved and optimal length is known
                opt_gap: Optional[int] = None
                if res["solved"] and res["solution_length"] is not None and opt_len is not None:
                    opt_gap = max(0, res["solution_length"] - opt_len)

                raw_records.append({
                    "Puzzle Index": p_idx + 1,
                    "Algorithm": algo_name,
                    "Memory Limit": mem_lim,
                    "Solved": res["solved"],
                    "Reason": res["reason"],
                    "Solution Length": res["solution_length"],
                    "Optimal Length": opt_len,
                    "Optimality Gap": opt_gap,
                    "Nodes Expanded": res["nodes_expanded"],
                    "Peak Memory": res["peak_memory"],
                    "Time (s)": round(res["time_sec"], 5),
                    "Iterations": res["iterations"]
                })

    raw_df = pd.DataFrame(raw_records)

    # Compute aggregated metrics grouped by Algorithm and Memory Limit
    summary_rows: List[Dict[str, Any]] = []

    for (algo, mem), group in raw_df.groupby(["Algorithm", "Memory Limit"], sort=False):
        total_runs = len(group)
        solved_group = group[group["Solved"] == True]
        num_solved = len(solved_group)
        success_rate = (num_solved / total_runs) * 100.0 if total_runs > 0 else 0.0

        avg_len = solved_group["Solution Length"].mean() if num_solved > 0 else None
        avg_gap = solved_group["Optimality Gap"].dropna().mean() if num_solved > 0 else None

        avg_time = group["Time (s)"].mean()
        avg_expanded = group["Nodes Expanded"].mean()
        avg_peak_mem = group["Peak Memory"].mean()

        summary_rows.append({
            "Algorithm": algo,
            "Memory Limit": mem,
            "Success Rate (%)": round(success_rate, 1),
            "Avg Solution Length": round(avg_len, 2) if avg_len is not None else None,
            "Avg Optimality Gap": round(avg_gap, 2) if avg_gap is not None else None,
            "Avg Time (s)": round(avg_time, 5),
            "Avg Nodes Expanded": round(avg_expanded, 1),
            "Avg Peak Memory": round(avg_peak_mem, 1),
            "Solved Count": f"{num_solved} / {total_runs}"
        })

    agg_df = pd.DataFrame(summary_rows)

    return raw_df, agg_df