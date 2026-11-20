# Memory-Bounded Heuristic Search Benchmark: 8-Puzzle

An interactive Python + Streamlit web application that implements, visualizes, and compares memory-bounded heuristic search algorithms on the 8-puzzle against conventional A*.

---

## Project Overview

Conventional A* search guarantees completeness and optimality, but its memory requirements scale exponentially ($O(b^d)$) with solution depth. In many practical settings, A* exhausts all available memory before it runs out of CPU time.

This project implements and evaluates three memory-bounded heuristic search techniques alongside conventional A*:
1. **Conventional A\*** (Graph Search with Open Heap and Closed Set)
2. **IDA\*** (Iterative Deepening A*)
3. **RBFS** (Recursive Best-First Search)
4. **SMA\*** (Simplified Memory-Bounded A*)

By placing an artificial cap on available memory ($M$ nodes), this application investigates how memory limits impact **success rates**, **solution optimality**, **execution time**, and **search tree re-expansions**.

---

## File Structure

```text
├── puzzle.py        # 8-puzzle state representation, moves, Manhattan heuristic, solvability, and scrambler
├── algorithms.py    # Search algorithms (A*, IDA*, RBFS, SMA*) and verification suite
├── experiments.py   # Batch experiment runner, statistical aggregations, and benchmark generation
├── app.py           # Streamlit web interface (Single Run & Playback, Memory Experiments)
├── requirements.txt # Python package dependencies (streamlit, pandas, plotly)
└── README.md        # Setup guide, algorithm documentation, and memory model details
```

---

## Installation and Setup

### 1. Prerequisites
- Python 3.9 or higher

### 2. Create and Activate a Virtual Environment (Recommended)

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Verification Tests
To confirm that all four algorithms return identical optimal path lengths on sample puzzles:
```bash
python algorithms.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## Memory Model and Accounting

"Memory" is defined as the **peak number of board states stored simultaneously** during search:

- **Conventional A\*:**
  $$\text{Memory} = |\text{Open Heap}| + |\text{Closed Set}|$$
  If this sum exceeds the node limit $M$, the search halts immediately and returns `out of memory`.

- **IDA\*:**
  $$\text{Memory} = \text{Current Path Length (Stack Depth)}$$
  IDA* stores only states along the active DFS recursion branch ($O(d)$ space). It fails with `out of memory` only if the path length exceeds $M$.

- **RBFS:**
  $$\text{Memory} = \text{Current Path Nodes} + \text{Stored Sibling Entries}$$
  RBFS keeps the current recursion branch plus generated sibling nodes at each depth level ($O(b \cdot d)$ space). If this total exceeds $M$, it returns `out of memory`.

- **SMA\*:**
  $$\text{Memory} \le M$$
  SMA* keeps a strictly bounded tree of at most $M$ nodes in memory. When full, it drops the worst leaf (highest $f$-value, shallowest $g$) and backs up that leaf's $f$-value into its parent. If that branch becomes promising later, the parent regenerates it. SMA* fails with `out of memory` only if $M$ cannot hold the active path.

---

## Algorithm Comparison Summary

| Algorithm | Space Complexity | Time Complexity | Optimality Guarantee | Memory Failure Trigger |
| :--- | :--- | :--- | :--- | :--- |
| **A\*** | $O(b^d)$ | $O(b^d)$ | Optimal (with consistent $h$) | $|\text{Open}| + |\text{Closed}| > M$ |
| **IDA\*** | $O(d)$ | $O(b^d)$ | Optimal | $\text{Depth} > M$ |
| **RBFS** | $O(b \cdot d)$ | $O(b^d)$ | Optimal | $\text{Path} + \text{Siblings} > M$ |
| **SMA\*** | $O(M)$ | $O(b^d)$ | Optimal if $M \ge d^*$; else sub-optimal / fails | Active path nodes $> M$ |

---

## Problem Formulation & Heuristic

- **State Representation:** 9-tuple representing the 3×3 grid in row-major order, with `0` denoting the blank tile.
  - Goal state: `(1, 2, 3, 4, 5, 6, 7, 8, 0)`
- **Solvability Check:** An 8-puzzle configuration is solvable if and only if the number of inversions among non-blank tiles is **even**. Horizontal moves leave tile order unchanged; vertical moves skip two tiles, changing inversions by $\pm 2$ or $0$ (preserving parity).
- **Heuristic Function:** Sum of Manhattan distances of all non-blank tiles (1 through 8) to their goal coordinates:
  $$h(n) = \sum_{i=1}^{8} \left( |\text{row}(i) - \text{goal\_row}(i)| + |\text{col}(i) - \text{goal\_col}(i)| \right)$$
  This heuristic is **admissible** ($h(n) \le h^*(n)$) and **consistent** ($|h(n) - h(n')| \le 1$), guaranteeing optimal solutions for A*, IDA*, and RBFS.

---

## Application Features

### Tab 1: Single Run & Playback
- **Shuffle Moves:** Control how many random non-undoing moves are applied from the goal state to create the initial puzzle.
- **Shuffle New Puzzle Button:** Generates a new random puzzle without requiring manual seed entry.
- **Memory Cap Slider:** Select limits ranging from 50 to 100,000 nodes.
- **Algorithm Selector:** Choose any combination of A*, IDA*, RBFS, and SMA*.
- **Visual Board Displays:** Start and Goal board states rendered side-by-side with Manhattan distance and solvability status.
- **Performance Table & Bar Charts:** Compare status, solution length, nodes expanded, peak memory, and execution time.
- **Interactive Solution Playback:** Step-by-step slider allowing you to replay tile movements on the board for any solved algorithm.

### Tab 2: Memory Experiments
- **Benchmark Suite:** Runs selected algorithms across reproducible benchmark puzzles with a fixed seed (`base_seed=42`).
- **Memory Caps Tested:** Quick, informative limits (`50`, `200`, `1000`, `5000`).
- **Aggregated Summary Table:** Displays success rate (%), average solution length, average optimality gap, execution time, nodes expanded, and peak memory.
- **Scaling Curves (Plotly):** Log-scale visualization across memory limits:
  1. Success Rate (%) vs. Memory Cap
  2. Solution Quality / Optimality Gap vs. Memory Cap
  3. Avg Execution Time (s) vs. Memory Cap
  4. Avg Nodes Expanded vs. Memory Cap
  5. Avg Peak Memory vs. Memory Cap

---

## Key Experimental Observations

1. **A\* Memory Wall:** Conventional A* is fast when memory is plentiful, but experiences complete failure (0% success rate) once the memory limit drops below the required frontier and closed set size.
2. **IDA\* and RBFS Space Economy:** Both algorithms maintain a minimal memory footprint ($\le 35$ nodes for typical 8-puzzle instances) and easily operate within tiny memory budgets ($M=50$). However, they re-explore nodes frequently across iterations, increasing runtimes on deeper puzzles.
3. **SMA\* Graceful Degradation:** When $M$ is large, SMA* acts like standard A*. As $M$ decreases, SMA* utilizes all available memory up to $M$ and dynamically drops and regenerates leaves, providing a smooth trade-off between memory and computation time.
