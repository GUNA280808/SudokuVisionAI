"""
solver/validator.py

Validates that a recognized 9x9 grid is a legal (possibly incomplete)
Sudoku puzzle before attempting to solve it.
"""


def is_valid_group(group):
    """A group (row, column, or box) is valid if it has no duplicate non-zero digits."""
    nums = [n for n in group if n != 0]
    return len(nums) == len(set(nums))


def get_row(grid, r):
    return grid[r]


def get_col(grid, c):
    return [grid[r][c] for r in range(9)]


def get_box(grid, r, c):
    br, bc = 3 * (r // 3), 3 * (c // 3)
    return [grid[br + i][bc + j] for i in range(3) for j in range(3)]


def is_valid_grid(grid):
    """Checks all rows, columns, and 3x3 boxes for duplicate conflicts."""
    for i in range(9):
        if not is_valid_group(get_row(grid, i)):
            return False, f"Row {i + 1} has a duplicate digit."
        if not is_valid_group(get_col(grid, i)):
            return False, f"Column {i + 1} has a duplicate digit."

    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = [grid[br + i][bc + j] for i in range(3) for j in range(3)]
            if not is_valid_group(box):
                return False, f"3x3 box at row {br + 1}, col {bc + 1} has a duplicate digit."

    return True, "Grid is valid."


def count_filled_cells(grid):
    return sum(1 for row in grid for val in row if val != 0)
