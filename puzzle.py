"""
puzzle.py - 8-Puzzle Game Logic, Manhattan Heuristic, and Utilities.

State Representation:
A 9-element tuple of integers (0-8) representing the 3x3 board in row-major order:
    0 1 2
    3 4 5
    6 7 8
0 represents the blank tile.
Standard Goal State: (1, 2, 3, 4, 5, 6, 7, 8, 0).
"""

from typing import List, Tuple, Optional
import random

# Standard goal configuration for the 8-puzzle
GOAL_STATE: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Precompute target coordinates (row, col) for each tile value 1..8
# Note: tile 0 (blank) is excluded from Manhattan distance calculations.
GOAL_POSITIONS = {
    tile: ((tile - 1) // 3, (tile - 1) % 3) for tile in range(1, 9)
}

# Precomputed Manhattan distance table: MANHATTAN_LOOKUP[val][pos_idx]
# gives the exact Manhattan distance for placing `val` at board index `pos_idx`.
MANHATTAN_LOOKUP: List[List[int]] = []
for val in range(9):
    row_table: List[int] = []
    if val == 0:
        row_table = [0] * 9
    else:
        target_r, target_c = GOAL_POSITIONS[val]
        for idx in range(9):
            cur_r, cur_c = idx // 3, idx % 3
            row_table.append(abs(cur_r - target_r) + abs(cur_c - target_c))
    MANHATTAN_LOOKUP.append(row_table)

# Opposite moves map to prevent immediate backtracking during scramble
OPPOSITE_MOVES = {
    "Up": "Down",
    "Down": "Up",
    "Left": "Right",
    "Right": "Left"
}


def is_goal(state: Tuple[int, ...]) -> bool:
    """Returns True if the given state matches the standard goal state."""
    return state == GOAL_STATE


def manhattan(state: Tuple[int, ...]) -> int:
    """
    Computes the sum of Manhattan distances of all non-blank tiles from their goal positions.
    This heuristic is both admissible (h(n) <= h*(n)) and consistent (monotonic).
    """
    return sum(MANHATTAN_LOOKUP[val][idx] for idx, val in enumerate(state))


def get_neighbors(state: Tuple[int, ...]) -> List[Tuple[str, Tuple[int, ...]]]:
    """
    Returns valid neighbors of the given state as a list of (move_name, new_state).
    'move_name' indicates the direction the blank space (0) moves ('Up', 'Down', 'Left', 'Right').
    """
    blank_idx = state.index(0)
    row, col = blank_idx // 3, blank_idx % 3
    neighbors: List[Tuple[str, Tuple[int, ...]]] = []

    # Blank moves Up (tile above slides down into blank)
    if row > 0:
        new_idx = blank_idx - 3
        st = list(state)
        st[blank_idx], st[new_idx] = st[new_idx], st[blank_idx]
        neighbors.append(("Up", tuple(st)))

    # Blank moves Down (tile below slides up into blank)
    if row < 2:
        new_idx = blank_idx + 3
        st = list(state)
        st[blank_idx], st[new_idx] = st[new_idx], st[blank_idx]
        neighbors.append(("Down", tuple(st)))

    # Blank moves Left (tile on left slides right into blank)
    if col > 0:
        new_idx = blank_idx - 1
        st = list(state)
        st[blank_idx], st[new_idx] = st[new_idx], st[blank_idx]
        neighbors.append(("Left", tuple(st)))

    # Blank moves Right (tile on right slides left into blank)
    if col < 2:
        new_idx = blank_idx + 1
        st = list(state)
        st[blank_idx], st[new_idx] = st[new_idx], st[blank_idx]
        neighbors.append(("Right", tuple(st)))

    return neighbors


def is_solvable(state: Tuple[int, ...]) -> bool:
    """
    Determines if an 8-puzzle board configuration is solvable to the goal (1..8, 0).
    For a 3x3 puzzle (odd grid width), an arrangement is solvable if and only if
    the number of inversions is even. The blank tile is excluded from inversion counts.
    """
    non_blank = [x for x in state if x != 0]
    inversions = 0
    for i in range(len(non_blank)):
        for j in range(i + 1, len(non_blank)):
            if non_blank[i] > non_blank[j]:
                inversions += 1
    return (inversions % 2) == 0


def scramble(n_moves: int, seed: Optional[int] = None) -> Tuple[int, ...]:
    """
    Generates a solvable board by taking a random walk of `n_moves` from the goal state.
    Avoids immediately undoing the previous move to ensure a meaningful scramble.
    """
    rng = random.Random(seed)
    current_state = GOAL_STATE
    last_move: Optional[str] = None

    for _ in range(n_moves):
        neighbors = get_neighbors(current_state)
        if last_move is not None:
            undo_move = OPPOSITE_MOVES.get(last_move)
            filtered = [n for n in neighbors if n[0] != undo_move]
            choices = filtered if filtered else neighbors
        else:
            choices = neighbors

        move_name, next_state = rng.choice(choices)
        current_state = next_state
        last_move = move_name

    return current_state


def random_solvable_state(seed: Optional[int] = None) -> Tuple[int, ...]:
    """
    Generates a uniformly random solvable state.
    Shuffles [0..8], checks solvability; if odd inversions, swaps two non-blank tiles.
    """
    rng = random.Random(seed)
    tiles = list(range(9))
    rng.shuffle(tiles)

    if not is_solvable(tuple(tiles)):
        idx1 = next(i for i, v in enumerate(tiles) if v != 0)
        idx2 = next(i for i, v in enumerate(tiles) if v != 0 and i != idx1)
        tiles[idx1], tiles[idx2] = tiles[idx2], tiles[idx1]

    return tuple(tiles)