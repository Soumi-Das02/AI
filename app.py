"""
app.py - Streamlit Web Application: Memory-Bounded Heuristic Search on the 8-Puzzle.

Compares Conventional A*, IDA*, RBFS, and SMA* under artificial memory bounds.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import plotly.express as px
import streamlit as st

from puzzle import (
    scramble,
    is_solvable,
    manhattan,
    GOAL_STATE,
)
from algorithms import astar, idastar, rbfs, sma_star
from experiments import generate_benchmark_puzzles, run_batch_experiments

# Set wide page configuration without icons
st.set_page_config(
    page_title="8-Puzzle: Memory-Bounded Search Benchmark",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling board tiles, badges, and headers
st.markdown("""
<style>
    .tile-grid {
        display: inline-grid;
        grid-template-columns: repeat(3, 68px);
        grid-gap: 8px;
        background-color: #1e222d;
        padding: 12px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .tile {
        width: 68px;
        height: 68px;
        background: linear-gradient(135deg, #1976d2, #0d47a1);
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.25);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .tile-blank {
        width: 68px;
        height: 68px;
        background-color: #2b303c;
        border: 2px dashed #4f5869;
        border-radius: 8px;
    }
    .board-card {
        background-color: #0e1117;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


def render_board_html(state: Tuple[int, ...], label: str = "") -> str:
    """Generates an HTML snippet rendering the 3x3 board."""
    tiles_html = []
    for tile in state:
        if tile == 0:
            tiles_html.append('<div class="tile-blank"></div>')
        else:
            tiles_html.append(f'<div class="tile">{tile}</div>')

    grid_content = "".join(tiles_html)
    return f"""
    <div class="board-card">
        <h5 style="margin-bottom: 10px; color: #e0e0e0;">{label}</h5>
        <div class="tile-grid">{grid_content}</div>
    </div>
    """


# Initialize session state for persistent interactive data
if "current_state" not in st.session_state:
    st.session_state["current_state"] = scramble(14, seed=42)
if "single_results" not in st.session_state:
    st.session_state["single_results"] = None
if "playback_step" not in st.session_state:
    st.session_state["playback_step"] = 0
if "exp_agg_df" not in st.session_state:
    st.session_state["exp_agg_df"] = None


# App Header
st.title("Memory-Bounded Heuristic Search Benchmark: 8-Puzzle")
st.markdown(
    r"Comparative analysis of **Conventional A\***, **IDA\***, **Recursive Best-First Search (RBFS)**, "
    r"and **Simplified Memory-Bounded A\* (SMA\*)** under artificial memory limits."
)

tab1, tab2 = st.tabs(["Single Run & Playback", "Memory Experiments"])


# =====================================================================
# TAB 1: SINGLE RUN
# =====================================================================
with tab1:
    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.subheader("Puzzle Setup")

        scramble_moves = st.slider(
            "Shuffle Moves",
            min_value=5,
            max_value=40,
            value=14,
            step=1,
            help="Number of random moves made from the solved board. Higher values usually give harder puzzles."
        )

        if st.button("Shuffle New Puzzle", use_container_width=True):
            st.session_state["current_state"] = scramble(scramble_moves)
            st.session_state["single_results"] = None
            st.session_state["playback_step"] = 0
            st.rerun()

        st.divider()
        st.subheader("Execution Settings")

        # Memory limit slider with log-spaced steps
        memory_choices = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000]
        mem_limit = st.select_slider(
            "Memory Cap (Max Nodes)",
            options=memory_choices,
            value=10000
        )

        selected_algos = st.multiselect(
            "Select Algorithms",
            options=["A*", "IDA*", "RBFS", "SMA*"],
            default=["A*", "IDA*", "RBFS", "SMA*"]
        )

        time_limit = st.slider("Timeout per Algorithm (s)", min_value=1, max_value=20, value=5, step=1)

        run_btn = st.button("Run Search", type="primary", use_container_width=True)

    with col_main:
        # Display Start and Goal Boards side-by-side
        cur_state = st.session_state["current_state"]
        man_dist = manhattan(cur_state)
        solv = is_solvable(cur_state)

        b_col1, b_col2, b_col3 = st.columns([1, 1, 1])
        with b_col1:
            st.markdown(render_board_html(cur_state, "Start Board (Initial)"), unsafe_allow_html=True)
        with b_col2:
            st.markdown(render_board_html(GOAL_STATE, "Target Board (Goal)"), unsafe_allow_html=True)
        with b_col3:
            st.markdown(f"""
            <div class="board-card" style="text-align: left;">
                <h5 style="color: #e0e0e0;">Board Info</h5>
                <p><b>Initial Manhattan:</b> {man_dist}</p>
                <p><b>Solvable:</b> {'Yes' if solv else 'No'}</p>
                <p><b>Memory Cap:</b> {mem_limit:,} nodes</p>
                <p><b>Blank (0) Pos:</b> row {cur_state.index(0)//3}, col {cur_state.index(0)%3}</p>
            </div>
            """, unsafe_allow_html=True)

        if run_btn:
            if not selected_algos:
                st.warning("Please select at least one algorithm.")
            else:
                algo_funcs = {
                    "A*": astar,
                    "IDA*": idastar,
                    "RBFS": rbfs,
                    "SMA*": sma_star
                }

                results = {}
                with st.spinner("Executing algorithms..."):
                    for algo_name in selected_algos:
                        fn = algo_funcs[algo_name]
                        res = fn(cur_state, GOAL_STATE, mem_limit=mem_limit, time_limit=time_limit)
                        results[algo_name] = res

                st.session_state["single_results"] = results
                st.session_state["playback_step"] = 0

        # Display Single Run Results if available
        if st.session_state["single_results"]:
            results = st.session_state["single_results"]
            st.subheader("Performance Comparison")

            rows = []
            for name, r in results.items():
                rows.append({
                    "Algorithm": name,
                    "Solved": "Yes" if r["solved"] else "No",
                    "Reason": r["reason"].capitalize(),
                    "Solution Length": r["solution_length"] if r["solution_length"] is not None else "—",
                    "Nodes Expanded": f"{r['nodes_expanded']:,}",
                    "Peak Memory": f"{r['peak_memory']:,}",
                    "Time (s)": f"{r['time_sec']:.5f}",
                    "_time_raw": r["time_sec"],
                    "_solved": r["solved"]
                })

            df_res = pd.DataFrame(rows)

            display_cols = ["Algorithm", "Solved", "Reason", "Solution Length", "Nodes Expanded", "Peak Memory", "Time (s)"]
            st.dataframe(
                df_res[display_cols],
                use_container_width=True,
                hide_index=True
            )

            # Bar Charts for Metrics
            chart_col1, chart_col2, chart_col3 = st.columns(3)

            chart_data = [
                {
                    "Algorithm": name,
                    "Nodes Expanded": r["nodes_expanded"],
                    "Peak Memory": r["peak_memory"],
                    "Time (s)": r["time_sec"],
                    "Solved": r["solved"]
                }
                for name, r in results.items()
            ]
            cdf = pd.DataFrame(chart_data)

            with chart_col1:
                fig1 = px.bar(
                    cdf, x="Algorithm", y="Nodes Expanded",
                    color="Algorithm", title="Nodes Expanded",
                    text_auto=True, color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig1.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig1, use_container_width=True)

            with chart_col2:
                fig2 = px.bar(
                    cdf, x="Algorithm", y="Peak Memory",
                    color="Algorithm", title="Peak Memory (Nodes Held)",
                    text_auto=True, color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig2.add_hline(y=mem_limit, line_dash="dash", line_color="red", annotation_text="Cap")
                fig2.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig2, use_container_width=True)

            with chart_col3:
                fig3 = px.bar(
                    cdf, x="Algorithm", y="Time (s)",
                    color="Algorithm", title="Execution Time (s)",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig3.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig3, use_container_width=True)

            # Solution Playback Section
            solved_algos = [name for name, r in results.items() if r["solved"] and len(r["path"]) > 0]
            if solved_algos:
                st.divider()
                st.subheader("Interactive Solution Playback")

                play_algo = st.selectbox("Select Solution to Replay", options=solved_algos)
                sol_path = results[play_algo]["path"]
                sol_moves = results[play_algo]["moves"]
                max_step = len(sol_path) - 1

                step = st.slider("Step Index", min_value=0, max_value=max_step, value=0)

                step_state = sol_path[step]
                move_taken = sol_moves[step - 1] if step > 0 else "Initial State"

                p_col1, p_col2 = st.columns([1, 2])
                with p_col1:
                    st.markdown(
                        render_board_html(step_state, f"Step {step}/{max_step} ({move_taken})"),
                        unsafe_allow_html=True
                    )
                with p_col2:
                    st.markdown("#### Move Details")
                    st.write(f"- **Current Step:** `{step}` of `{max_step}`")
                    st.write(f"- **Move Applied to Blank:** `{move_taken}`")
                    st.write(f"- **Manhattan to Goal:** `{manhattan(step_state)}`")
                    st.write(f"- **Full Moves Sequence:** `{' -> '.join(sol_moves)}`")
            else:
                st.info("No algorithm found a solution within the given memory cap and timeout limits.")


# =====================================================================
# TAB 2: BATCH MEMORY EXPERIMENT
# =====================================================================
with tab2:
    st.subheader("Batch Memory Scaling Experiment")
    st.markdown(
        "Investigate how artificial node capacity limits impact **success rates**, "
        "**optimality gaps**, **execution times**, and **search tree re-expansions**."
    )

    # Balanced 2-column parameter layout
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        num_puzzles = st.slider(
            "Number of Benchmark Puzzles",
            min_value=3,
            max_value=15,
            value=10,
            step=1
        )
        exp_scramble = st.slider(
            "Shuffle Moves (Difficulty)",
            min_value=8,
            max_value=35,
            value=20,
            step=1,
            help="Number of random moves used to mix each benchmark puzzle."
        )
        timeout_per_run = st.number_input(
            "Timeout per Run (s)",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.5
        )

    with exp_col2:
        available_caps = [50, 200, 1000, 5000]
        selected_caps = st.multiselect(
            "Memory Caps to Test (Nodes)",
            options=available_caps,
            default=[50, 200, 1000, 5000]
        )
        exp_algos = st.multiselect(
            "Algorithms to Compare",
            options=["A*", "IDA*", "RBFS", "SMA*"],
            default=["A*", "IDA*", "RBFS", "SMA*"]
        )
        st.write("")  # Spacing
        run_exp_btn = st.button("Run Full Experiment", type="primary", use_container_width=True)

    if run_exp_btn:
        if not exp_algos or not selected_caps:
            st.error("Please select at least one algorithm and one memory limit.")
        else:
            sorted_caps = sorted(selected_caps)
            prog_bar = st.progress(0.0)
            status_text = st.empty()

            status_text.text("Generating benchmark puzzles and computing optimal solution baselines...")
            benchmarks = generate_benchmark_puzzles(num_puzzles, exp_scramble, base_seed=42)

            def update_progress(step: int, total: int, msg: str):
                prog_bar.progress(step / total)
                status_text.text(f"[{step}/{total}] {msg}")

            _, agg_df = run_batch_experiments(
                puzzles_with_opt=benchmarks,
                algorithms=exp_algos,
                memory_limits=sorted_caps,
                time_limit_per_run=timeout_per_run,
                progress_callback=update_progress
            )

            prog_bar.progress(1.0)
            status_text.text("Experiment completed successfully!")

            st.session_state["exp_agg_df"] = agg_df

    # Render Batch Results if available in session_state
    if st.session_state["exp_agg_df"] is not None:
        agg_df = st.session_state["exp_agg_df"]

        st.divider()
        st.subheader("Aggregated Summary Table")
        st.dataframe(agg_df, use_container_width=True, hide_index=True)

        st.subheader("Comparative Scaling Curves")
        c1, c2 = st.columns(2)

        with c1:
            # (a) Success Rate vs Memory Limit
            fig_a = px.line(
                agg_df,
                x="Memory Limit",
                y="Success Rate (%)",
                color="Algorithm",
                markers=True,
                log_x=True,
                title="(a) Success Rate (%) vs. Memory Cap",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_a.update_layout(yaxis_range=[-5, 105])
            st.plotly_chart(fig_a, use_container_width=True)

        with c2:
            # (b) Solution Quality / Optimality Gap vs Memory Limit
            # Fall back to Avg Solution Length if Optimality Gap is all null
            has_gap = agg_df["Avg Optimality Gap"].notna().any()
            y_metric = "Avg Optimality Gap" if has_gap else "Avg Solution Length"
            fig_b = px.line(
                agg_df,
                x="Memory Limit",
                y=y_metric,
                color="Algorithm",
                markers=True,
                log_x=True,
                title=f"(b) {y_metric} vs. Memory Cap",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_b, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            # (c) Execution Time vs Memory Limit
            fig_c = px.line(
                agg_df,
                x="Memory Limit",
                y="Avg Time (s)",
                color="Algorithm",
                markers=True,
                log_x=True,
                title="(c) Avg Execution Time (s) vs. Memory Cap",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_c, use_container_width=True)

        with c4:
            # (d) Nodes Expanded vs Memory Limit
            fig_d = px.line(
                agg_df,
                x="Memory Limit",
                y="Avg Nodes Expanded",
                color="Algorithm",
                markers=True,
                log_x=True,
                title="(d) Avg Nodes Expanded vs. Memory Cap",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_d, use_container_width=True)

        # (e) Peak Memory vs Memory Limit
        c5, _ = st.columns([1, 1])
        with c5:
            fig_e = px.line(
                agg_df,
                x="Memory Limit",
                y="Avg Peak Memory",
                color="Algorithm",
                markers=True,
                log_x=True,
                title="(e) Avg Peak Memory vs. Memory Cap",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_e, use_container_width=True)
