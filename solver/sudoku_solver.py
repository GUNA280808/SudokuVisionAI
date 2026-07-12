"""
solver/sudoku_solver.py

Classic backtracking Sudoku solver, instrumented to count the number of
recursive steps taken (used later for difficulty estimation) and to time
the solve.
"""

import copy
import time


class SudokuSolver:
    def __init__(self, grid):
        self.grid = copy.deepcopy(grid)
        self.steps = 0

    def _find_empty(self):
        for r in range(9):
            for c in range(9):
                if self.grid[r][c] == 0:
                    return r, c
        return None

    def _is_safe(self, row, col, num):
        if num in self.grid[row]:
            return False
        if num in [self.grid[r][col] for r in range(9)]:
            return False

        br, bc = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if self.grid[br + i][bc + j] == num:
                    return False
        return True

    def _solve(self):
        self.steps += 1
        empty = self._find_empty()
        if not empty:
            return True

        row, col = empty
        for num in range(1, 10):
            if self._is_safe(row, col, num):
                self.grid[row][col] = num
                if self._solve():
                    return True
                self.grid[row][col] = 0

        return False

    def solve(self, timeout_seconds=10):
        start = time.time()
        success = self._solve()
        elapsed = time.time() - start
        if elapsed > timeout_seconds and not success:
            raise TimeoutError("Solving exceeded the allotted time budget.")
        return success, self.grid, self.steps, elapsed


def solve_sudoku(grid):
    """
    Convenience wrapper.
    Returns dict: {solved: bool, grid: 9x9, steps: int, time_seconds: float}
    """
    solver = SudokuSolver(grid)
    success, solved_grid, steps, elapsed = solver.solve()
    return {
        "solved": success,
        "grid": solved_grid,
        "steps": steps,
        "time_seconds": round(elapsed, 4),
    }


if __name__ == "__main__":
    # Quick manual sanity check
    demo_puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    result = solve_sudoku(demo_puzzle)
    for row in result["grid"]:
        print(row)
    print("Solved:", result["solved"], "Steps:", result["steps"], "Time:", result["time_seconds"])
