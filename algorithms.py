"""
algorithms.py - Implementation of Heuristic Search Algorithms for the 8-Puzzle.

Implements:
1. Conventional A* (Graph Search)
2. IDA* (Iterative Deepening A*)
3. RBFS (Recursive Best-First Search)
4. SMA* (Simplified Memory-Bounded A*)

Memory Model (Nodes stored simultaneously):
- A*: len(open_heap) + len(closed_set). Fails with 'out of memory' if > mem_limit.
- IDA*: len(current_path) (depth of recursion stack). Fails if > mem_limit.
- RBFS: current recursion path nodes + their stored sibling entries. Fails if > mem_limit.
- SMA*: strictly bounded by mem_limit. Deletes highest-f leaf when full, backs up f-value.

All algorithms return a standardized dictionary format:
{
    "algorithm": str,
    "solved": bool,
    "reason": str,            # "solved", "out of memory", "timeout", "unsolvable"
    "path": list of states,   # start to goal, empty if unsolved
    "moves": list of str,     # move names: "Up", "Down", "Left", "Right"
    "solution_length": int or None,
    "nodes_expanded": int,
    "peak_memory": int,
    "time_sec": float,
    "iterations": int         # for IDA*: number of f-bound rounds, else 0
}
"""

import sys
import time
import heapq
import itertools
from typing import Dict, List, Tuple, Optional, Any, Set

from puzzle import get_neighbors, manhattan, is_goal, is_solvable, GOAL_STATE

# Set recursion depth safely for deep branch exploration in RBFS and IDA*
sys.setrecursionlimit(50000)


# =====================================================================
# 1. CONVENTIONAL A* SEARCH
# =====================================================================
def astar(
    start: Tuple[int, ...],
    goal: Tuple[int, ...],
    mem_limit: int,
    time_limit: float = 10.0,
    node_limit: int = 500000
) -> Dict[str, Any]:
    """
    Conventional Graph-Search A*.
    Maintains a min-heap (open set) and a closed set of explored states.
    Memory is calculated as len(open_heap) + len(closed_set).
    """
    start_time = time.perf_counter()

    # Edge Case: Start is already Goal
    if start == goal:
        return {
            "algorithm": "A*", "solved": True, "reason": "solved",
            "path": [start], "moves": [], "solution_length": 0,
            "nodes_expanded": 0, "peak_memory": 1,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }

    # Edge Case: Unsolvable state check
    if not is_solvable(start):
        return {
            "algorithm": "A*", "solved": False, "reason": "unsolvable",
            "path": [], "moves": [], "solution_length": None,
            "nodes_expanded": 0, "peak_memory": 1,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }

    # Edge Case: Memory limit too small
    if mem_limit < 1:
        return {
            "algorithm": "A*", "solved": False, "reason": "out of memory",
            "path": [], "moves": [], "solution_length": None,
            "nodes_expanded": 0, "peak_memory": 0,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }

    tie_counter = itertools.count()
    open_heap: List[Tuple[int, int, int, Tuple[int, ...]]] = []
    # Heap entry: (f_score, tie_breaker, g_score, state)
    heapq.heappush(open_heap, (manhattan(start), next(tie_counter), 0, start))

    closed_set: Set[Tuple[int, ...]] = set()
    g_score: Dict[Tuple[int, ...], int] = {start: 0}
    came_from: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], str]] = {}

    nodes_expanded = 0
    peak_memory = 1

    while open_heap:
        now = time.perf_counter()
        if now - start_time > time_limit or nodes_expanded >= node_limit:
            return {
                "algorithm": "A*", "solved": False, "reason": "timeout",
                "path": [], "moves": [], "solution_length": None,
                "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
                "time_sec": now - start_time, "iterations": 0
            }

        f, _, g, current = heapq.heappop(open_heap)

        # Skip stale heap entries
        if g > g_score.get(current, float('inf')):
            continue

        if current == goal:
            # Reconstruct solution path and move sequence
            path = [current]
            moves = []
            curr = current
            while curr in came_from:
                prev, mv = came_from[curr]
                path.append(prev)
                moves.append(mv)
                curr = prev
            path.reverse()
            moves.reverse()
            return {
                "algorithm": "A*", "solved": True, "reason": "solved",
                "path": path, "moves": moves, "solution_length": len(moves),
                "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
                "time_sec": time.perf_counter() - start_time, "iterations": 0
            }

        closed_set.add(current)
        nodes_expanded += 1

        for move, neighbor in get_neighbors(current):
            tentative_g = g + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = (current, move)
                h = manhattan(neighbor)
                f_val = tentative_g + h
                heapq.heappush(open_heap, (f_val, next(tie_counter), tentative_g, neighbor))

                # Track and guard memory
                curr_mem = len(open_heap) + len(closed_set)
                if curr_mem > peak_memory:
                    peak_memory = curr_mem
                if curr_mem > mem_limit:
                    return {
                        "algorithm": "A*", "solved": False, "reason": "out of memory",
                        "path": [], "moves": [], "solution_length": None,
                        "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
                        "time_sec": time.perf_counter() - start_time, "iterations": 0
                    }

    return {
        "algorithm": "A*", "solved": False, "reason": "unsolvable",
        "path": [], "moves": [], "solution_length": None,
        "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
        "time_sec": time.perf_counter() - start_time, "iterations": 0
    }


# =====================================================================
# 2. ITERATIVE DEEPENING A* (IDA*)
# =====================================================================
def idastar(
    start: Tuple[int, ...],
    goal: Tuple[int, ...],
    mem_limit: int,
    time_limit: float = 10.0,
    node_limit: int = 500000
) -> Dict[str, Any]:
    """
    Iterative Deepening A* (IDA*).
    Performs successive DFS searches bounded by f-cost thresholds.
    Memory is the current search path length (depth of the recursion stack).
    """
    start_time = time.perf_counter()

    if start == goal:
        return {
            "algorithm": "IDA*", "solved": True, "reason": "solved",
            "path": [start], "moves": [], "solution_length": 0,
            "nodes_expanded": 0, "peak_memory": 1,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }

    if not is_solvable(start):
        return {
            "algorithm": "IDA*", "solved": False, "reason": "unsolvable",
            "path": [], "moves": [], "solution_length": None,
            "nodes_expanded": 0, "peak_memory": 1,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }

    if mem_limit < 1:
        return {
            "algorithm": "IDA*", "solved": False, "reason": "out of memory",
            "path": [], "moves": [], "solution_length": None,
            "nodes_expanded": 0, "peak_memory": 0,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }

    bound = manhattan(start)
    path: List[Tuple[int, ...]] = [start]
    path_set: Set[Tuple[int, ...]] = {start}
    moves: List[str] = []

    nodes_expanded = 0
    peak_memory = 1
    iterations = 0

    class IDATimeout(Exception):
        pass

    class IDAOOM(Exception):
        pass

    def dfs(g: int, current_bound: int) -> Tuple[bool, int]:
        nonlocal nodes_expanded, peak_memory

        now = time.perf_counter()
        if now - start_time > time_limit or nodes_expanded >= node_limit:
            raise IDATimeout()

        current = path[-1]
        f = g + manhattan(current)

        if f > current_bound:
            return False, f

        if current == goal:
            return True, f

        # Memory check: depth of active branch
        curr_mem = len(path)
        if curr_mem > peak_memory:
            peak_memory = curr_mem
        if curr_mem > mem_limit:
            raise IDAOOM()

        nodes_expanded += 1
        min_exceeded = float('inf')

        for move, neighbor in get_neighbors(current):
            # Avoid cycles on the active path
            if neighbor in path_set:
                continue

            path.append(neighbor)
            path_set.add(neighbor)
            moves.append(move)

            found, next_bound = dfs(g + 1, current_bound)
            if found:
                return True, next_bound

            if next_bound < min_exceeded:
                min_exceeded = next_bound

            path.pop()
            path_set.remove(neighbor)
            moves.pop()

        return False, min_exceeded

    try:
        while True:
            iterations += 1
            found, next_bound = dfs(0, bound)
            if found:
                return {
                    "algorithm": "IDA*", "solved": True, "reason": "solved",
                    "path": list(path), "moves": list(moves),
                    "solution_length": len(moves),
                    "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
                    "time_sec": time.perf_counter() - start_time, "iterations": iterations
                }
            if next_bound == float('inf'):
                return {
                    "algorithm": "IDA*", "solved": False, "reason": "unsolvable",
                    "path": [], "moves": [], "solution_length": None,
                    "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
                    "time_sec": time.perf_counter() - start_time, "iterations": iterations
                }
            bound = next_bound
    except IDATimeout:
        return {
            "algorithm": "IDA*", "solved": False, "reason": "timeout",
            "path": [], "moves": [], "solution_length": None,
            "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
            "time_sec": time.perf_counter() - start_time, "iterations": iterations
        }
    except IDAOOM:
        return {
            "algorithm": "IDA*", "solved": False, "reason": "out of memory",
            "path": [], "moves": [], "solution_length": None,
            "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
            "time_sec": time.perf_counter() - start_time, "iterations": iterations
        }


# =====================================================================
# 3. RECURSIVE BEST-FIRST SEARCH (RBFS)
# =====================================================================
class _RBFSNode:
    __slots__ = ('state', 'parent', 'move', 'g', 'h', 'f')
    def __init__(self, state: Tuple[int, ...], parent: Optional['_RBFSNode'], move: Optional[str], g: int, h: int, f: int):
        self.state = state
        self.parent = parent
        self.move = move
        self.g = g
        self.h = h
        self.f = f


def rbfs(
    start: Tuple[int, ...],
    goal: Tuple[int, ...],
    mem_limit: int,
    time_limit: float = 10.0,
    node_limit: int = 500000
) -> Dict[str, Any]:
    """
    Recursive Best-First Search (RBFS, Russell & Norvig AIMA).
    Maintains best and second-best alternative f-limits while recursing.
    Memory = active path nodes + their stored sibling entries at each level.
    """
    start_time = time.perf_counter()

    if start == goal:
        return {
            "algorithm": "RBFS", "solved": True, "reason": "solved",
            "path": [start], "moves": [], "solution_length": 0,
            "nodes_expanded": 0, "peak_memory": 1,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }

    if not is_solvable(start):
        return {
            "algorithm": "RBFS", "solved": False, "reason": "unsolvable",
            "path": [], "moves": [], "solution_length": None,
            "nodes_expanded": 0, "peak_memory": 1,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }

    if mem_limit < 1:
        return {
            "algorithm": "RBFS", "solved": False, "reason": "out of memory",
            "path": [], "moves": [], "solution_length": None,
            "nodes_expanded": 0, "peak_memory": 0,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }

    nodes_expanded = 0
    current_memory = 1  # Root node
    peak_memory = 1

    class RBFSTimeout(Exception):
        pass

    class RBFSOOM(Exception):
        pass

    def rbfs_rec(node: _RBFSNode, f_limit: float) -> Tuple[Optional[_RBFSNode], float]:
        nonlocal nodes_expanded, current_memory, peak_memory

        now = time.perf_counter()
        if now - start_time > time_limit or nodes_expanded >= node_limit:
            raise RBFSTimeout()

        if node.state == goal:
            return node, node.f

        nodes_expanded += 1

        # Collect ancestors on path to avoid cycles
        ancestors: Set[Tuple[int, ...]] = set()
        curr: Optional[_RBFSNode] = node
        while curr is not None:
            ancestors.add(curr.state)
            curr = curr.parent

        raw_neighbors = get_neighbors(node.state)
        successors: List[_RBFSNode] = []
        for mv, nxt in raw_neighbors:
            if nxt in ancestors:
                continue
            g = node.g + 1
            h = manhattan(nxt)
            # Pathmax property: f is non-decreasing along the branch
            f = max(g + h, node.f)
            child = _RBFSNode(nxt, node, mv, g, h, f)
            successors.append(child)

        if not successors:
            return None, float('inf')

        # RBFS Memory: Path nodes + stored sibling entries at each recursion level
        current_memory += len(successors)
        if current_memory > peak_memory:
            peak_memory = current_memory
        if current_memory > mem_limit:
            current_memory -= len(successors)
            raise RBFSOOM()

        try:
            while True:
                now_inner = time.perf_counter()
                if now_inner - start_time > time_limit or nodes_expanded >= node_limit:
                    raise RBFSTimeout()

                # Sort by lowest f, breaking ties by deeper g
                successors.sort(key=lambda x: (x.f, -x.g))
                best = successors[0]

                if best.f > f_limit:
                    return None, best.f

                alt = successors[1].f if len(successors) > 1 else float('inf')

                result, best.f = rbfs_rec(best, min(f_limit, alt))
                if result is not None:
                    return result, best.f
        finally:
            current_memory -= len(successors)

    root = _RBFSNode(start, None, None, 0, manhattan(start), manhattan(start))

    try:
        sol_node, _ = rbfs_rec(root, float('inf'))
        if sol_node is not None:
            path: List[Tuple[int, ...]] = []
            moves: List[str] = []
            cur: Optional[_RBFSNode] = sol_node
            while cur is not None:
                path.append(cur.state)
                if cur.move is not None:
                    moves.append(cur.move)
                cur = cur.parent
            path.reverse()
            moves.reverse()
            return {
                "algorithm": "RBFS", "solved": True, "reason": "solved",
                "path": path, "moves": moves, "solution_length": len(moves),
                "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
                "time_sec": time.perf_counter() - start_time, "iterations": 0
            }
        else:
            return {
                "algorithm": "RBFS", "solved": False, "reason": "unsolvable",
                "path": [], "moves": [], "solution_length": None,
                "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
                "time_sec": time.perf_counter() - start_time, "iterations": 0
            }
    except RBFSTimeout:
        return {
            "algorithm": "RBFS", "solved": False, "reason": "timeout",
            "path": [], "moves": [], "solution_length": None,
            "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }
    except RBFSOOM:
        return {
            "algorithm": "RBFS", "solved": False, "reason": "out of memory",
            "path": [], "moves": [], "solution_length": None,
            "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }


# =====================================================================
# 4. SIMPLIFIED MEMORY-BOUNDED A* (SMA*)
# =====================================================================
class _SMANode:
    __slots__ = ('nid', 'state', 'parent_id', 'move', 'g', 'h', 'f', 'children', 'forgotten_children', 'is_expanded')
    def __init__(self, nid: int, state: Tuple[int, ...], parent_id: Optional[int], move: Optional[str], g: int, h: int, f: int):
        self.nid = nid
        self.state = state
        self.parent_id = parent_id
        self.move = move
        self.g = g
        self.h = h
        self.f = f
        self.children: Set[int] = set()                   # Active child nids in memory
        self.forgotten_children: Dict[Tuple[str, Tuple[int, ...]], int] = {}  # Dropped successors and their f
        self.is_expanded: bool = False


def sma_star(
    start: Tuple[int, ...],
    goal: Tuple[int, ...],
    mem_limit: int,
    time_limit: float = 10.0,
    node_limit: int = 500000
) -> Dict[str, Any]:
    """
    Simplified Memory-Bounded A* (SMA*).
    
    Memory Simplification Documentation:
    1. Node Cap (M): Exactly at most M nodes exist simultaneously in memory (nodes dictionary).
    2. Leaf Selection: Selects the best leaf node (lowest f, breaking ties with deepest g).
    3. Pruning: When len(nodes) >= M, the worst leaf (highest f, shallowest g) that is not on
       the active expansion path is pruned from memory.
    4. Backup: Pruned leaf f-values are stored in parent.forgotten_children, and parent f-value
       is updated as min(active_children.f, forgotten_children.f), propagating up ancestors.
    5. Regeneration: If all other alternatives become worse than the parent's backed-up f-value,
       the parent is re-expanded and forgotten successors are regenerated.
    6. Failure: If M is too small to even hold the active path nodes, it terminates with 'out of memory'.
    """
    start_time = time.perf_counter()

    if start == goal:
        return {
            "algorithm": "SMA*", "solved": True, "reason": "solved",
            "path": [start], "moves": [], "solution_length": 0,
            "nodes_expanded": 0, "peak_memory": 1,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }

    if not is_solvable(start):
        return {
            "algorithm": "SMA*", "solved": False, "reason": "unsolvable",
            "path": [], "moves": [], "solution_length": None,
            "nodes_expanded": 0, "peak_memory": 1,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }

    if mem_limit < 2:
        return {
            "algorithm": "SMA*", "solved": False, "reason": "out of memory",
            "path": [], "moves": [], "solution_length": None,
            "nodes_expanded": 0, "peak_memory": 0,
            "time_sec": time.perf_counter() - start_time, "iterations": 0
        }

    next_nid = 0
    root = _SMANode(next_nid, start, None, None, 0, manhattan(start), manhattan(start))
    next_nid += 1

    nodes: Dict[int, _SMANode] = {root.nid: root}
    nodes_expanded = 0
    peak_memory = 1

    def backup(node_id: Optional[int]) -> None:
        """Propagates updated f-values upwards to ancestor nodes."""
        curr_id = node_id
        while curr_id is not None:
            n = nodes.get(curr_id)
            if not n:
                break
            vals = [nodes[cid].f for cid in n.children if cid in nodes]
            vals.extend(n.forgotten_children.values())
            if vals:
                new_f = min(vals)
                if new_f != n.f:
                    n.f = new_f
                    curr_id = n.parent_id
                else:
                    break
            else:
                break

    while nodes:
        now = time.perf_counter()
        if now - start_time > time_limit or nodes_expanded >= node_limit:
            return {
                "algorithm": "SMA*", "solved": False, "reason": "timeout",
                "path": [], "moves": [], "solution_length": None,
                "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
                "time_sec": now - start_time, "iterations": 0
            }

        # Leaves in memory are nodes currently having no children in memory
        leaves = [n for n in nodes.values() if len(n.children) == 0]
        if not leaves:
            break

        # Best leaf: lowest f, break ties by deepest g
        leaves.sort(key=lambda n: (n.f, -n.g))
        best = leaves[0]

        if best.f == float('inf'):
            break

        if best.state == goal:
            path: List[Tuple[int, ...]] = []
            moves: List[str] = []
            cur: Optional[_SMANode] = best
            while cur is not None:
                path.append(cur.state)
                if cur.move is not None:
                    moves.append(cur.move)
                cur = nodes.get(cur.parent_id) if cur.parent_id is not None else None
            path.reverse()
            moves.reverse()
            return {
                "algorithm": "SMA*", "solved": True, "reason": "solved",
                "path": path, "moves": moves, "solution_length": len(moves),
                "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
                "time_sec": time.perf_counter() - start_time, "iterations": 0
            }

        # Expand best leaf
        nodes_expanded += 1
        best.is_expanded = True

        # Trace ancestors along active branch
        ancestors: Set[Tuple[int, ...]] = set()
        anc_nids: Set[int] = set()
        curr_anc: Optional[_SMANode] = best
        while curr_anc is not None:
            ancestors.add(curr_anc.state)
            anc_nids.add(curr_anc.nid)
            curr_anc = nodes.get(curr_anc.parent_id) if curr_anc.parent_id is not None else None

        all_neighbors = get_neighbors(best.state)
        best.forgotten_children.clear()

        valid_successors = [(m, nxt) for m, nxt in all_neighbors if nxt not in ancestors]

        if not valid_successors:
            best.f = float('inf')
            backup(best.parent_id)
            continue

        for m, nxt in valid_successors:
            # Maintain memory cap M: prune worst leaf if memory is full
            while len(nodes) >= mem_limit:
                candidate_prunes = [
                    n for n in nodes.values()
                    if len(n.children) == 0 and n.nid != best.nid and n.nid not in anc_nids
                ]
                if not candidate_prunes:
                    break
                # Worst leaf: highest f, lowest g (shallowest)
                candidate_prunes.sort(key=lambda n: (-n.f, n.g))
                worst = candidate_prunes[0]

                # Back up f-value to worst leaf's parent
                p = nodes.get(worst.parent_id) if worst.parent_id is not None else None
                if p is not None:
                    p.children.discard(worst.nid)
                    p.forgotten_children[(worst.move, worst.state)] = worst.f
                    backup(p.nid)
                del nodes[worst.nid]

            if len(nodes) >= mem_limit:
                # Memory cap too tight to expand this path further
                best.f = float('inf')
                backup(best.parent_id)
                break

            g = best.g + 1
            h = manhattan(nxt)
            f = max(g + h, best.f)
            child = _SMANode(next_nid, nxt, best.nid, m, g, h, f)
            next_nid += 1
            nodes[child.nid] = child
            best.children.add(child.nid)
            if len(nodes) > peak_memory:
                peak_memory = len(nodes)

        backup(best.nid)

    return {
        "algorithm": "SMA*", "solved": False,
        "reason": "out of memory" if len(nodes) >= mem_limit else "unsolvable",
        "path": [], "moves": [], "solution_length": None,
        "nodes_expanded": nodes_expanded, "peak_memory": peak_memory,
        "time_sec": time.perf_counter() - start_time, "iterations": 0
    }


# =====================================================================
# VERIFICATION SUITE & SMOKE TESTS
# =====================================================================
if __name__ == "__main__":
    from puzzle import scramble

    print("Running 8-puzzle verification tests across all 4 algorithms...")
    test_scrambles = [
        (8, 101),
        (12, 202),
        (14, 303)
    ]

    algos = [("A*", astar), ("IDA*", idastar), ("RBFS", rbfs), ("SMA*", sma_star)]
    all_passed = True

    for moves_count, seed in test_scrambles:
        start_state = scramble(moves_count, seed=seed)
        print(f"\n--- Testing Scramble: {moves_count} moves (seed={seed}) ---")
        print(f"Start State: {start_state}, Manhattan: {manhattan(start_state)}")

        results = {}
        for name, func in algos:
            res = func(start_state, GOAL_STATE, mem_limit=100000, time_limit=5.0)
            results[name] = res
            print(f"  [{name:5s}] Solved: {res['solved']} | Len: {res['solution_length']} | "
                  f"Exp: {res['nodes_expanded']:4d} | PeakMem: {res['peak_memory']:4d} | "
                  f"Time: {res['time_sec']:.4f}s")

        # Verify all solved
        solved_all = all(r["solved"] for r in results.values())
        if not solved_all:
            print("  FAIL: Some algorithms failed to solve!")
            all_passed = False
            continue

        # Verify all returned identical optimal solution length
        lengths = {name: r["solution_length"] for name, r in results.items()}
        if len(set(lengths.values())) != 1:
            print(f"  FAIL: Inconsistent solution lengths: {lengths}")
            all_passed = False
        else:
            opt_len = next(iter(lengths.values()))
            print(f"  PASS: All 4 algorithms returned identical optimal length: {opt_len}")

    print("\n" + ("=" * 50))
    if all_passed:
        print("ALL ALGORITHM VERIFICATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED.")
    print("=" * 50)