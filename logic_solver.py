"""Pure-Python match-three move search and scoring with AlphaGo-level heuristics."""

from __future__ import annotations

from typing import Sequence

import numpy as np

Board = Sequence[Sequence[int]]
Move = tuple[tuple[int, int], tuple[int, int]]


class LogicSolver:
    """Find the highest-scoring adjacent swap using advanced heuristics.

    Scoring system:
    - 5-match (straight line): +1000 points
    - T-shape / L-shape (intersecting bomb): +500 points
    - 4-match (straight line special): +300 points
    - Normal 3-match: +100 points
    - Bottom-heavy weight: row_index * 10 (encourages bottom matches for cascades)
    """

    @staticmethod
    def _validate_board(board: Board) -> tuple[int, int]:
        if isinstance(board, np.ndarray):
            if board.size == 0:
                return 0, 0
            rows, cols = board.shape
        else:
            if not board:
                return 0, 0
            rows = len(board)
            if rows == 0:
                return 0, 0
            cols = len(board[0])
            if any(len(row) != cols for row in board):
                raise ValueError("Board must be rectangular")
        return rows, cols

    @classmethod
    def _find_matches(cls, board: np.ndarray) -> list[dict]:
        """Find all matches on the board.

        Returns a list of match dictionaries with:
        - 'cells': set of (row, col) tuples
        - 'type': '5', '4', '3', 'T', 'L'
        - 'score': base score for this match
        """
        rows, cols = board.shape
        matches = []
        visited = set()

        # Check horizontal matches
        for r in range(rows):
            c = 0
            while c < cols:
                if board[r, c] == 0:
                    c += 1
                    continue
                value = board[r, c]
                start_c = c
                while c < cols and board[r, c] == value:
                    c += 1
                length = c - start_c
                if length >= 3:
                    cells = {(r, x) for x in range(start_c, c)}
                    # Check if this forms a T or L shape with vertical
                    match_type = cls._classify_match(board, cells, 'h')
                    score = cls._match_base_score(match_type, length)
                    matches.append({
                        'cells': cells,
                        'type': match_type,
                        'length': length,
                        'score': score,
                        'orientation': 'h'
                    })
                    visited.update(cells)

        # Check vertical matches
        for c in range(cols):
            r = 0
            while r < rows:
                if board[r, c] == 0:
                    r += 1
                    continue
                value = board[r, c]
                start_r = r
                while r < rows and board[r, c] == value:
                    r += 1
                length = r - start_r
                if length >= 3:
                    cells = {(y, c) for y in range(start_r, r)}
                    # Check if this forms a T or L shape with horizontal
                    match_type = cls._classify_match(board, cells, 'v')
                    score = cls._match_base_score(match_type, length)
                    matches.append({
                        'cells': cells,
                        'type': match_type,
                        'length': length,
                        'score': score,
                        'orientation': 'v'
                    })
                    visited.update(cells)

        return matches

    @classmethod
    def _classify_match(cls, board: np.ndarray, cells: set[tuple[int, int]], orientation: str) -> str:
        """Classify match type: '5', '4', '3', 'T', 'L'."""
        length = len(cells)
        if length >= 5:
            return '5'
        if length == 4:
            return '4'

        # For 3-match, check if it forms T or L with perpendicular matches
        if length == 3:
            rows, cols = board.shape
            for r, c in cells:
                value = board[r, c]
                if value == 0:
                    continue

                if orientation == 'h':
                    # Check vertical at each cell of horizontal match
                    # Count consecutive same-color cells vertically
                    up = r
                    while up > 0 and board[up - 1, c] == value:
                        up -= 1
                    down = r + 1
                    while down < rows and board[down, c] == value:
                        down += 1
                    v_length = down - up
                    if v_length >= 3:
                        # Check if it's T or L shape
                        # T-shape: vertical extends on both sides of intersection
                        # L-shape: vertical extends only on one side
                        has_up = up < r
                        has_down = down > r + 1
                        if has_up and has_down:
                            return 'T'
                        else:
                            return 'L'
                else:
                    # Check horizontal at each cell of vertical match
                    left = c
                    while left > 0 and board[r, left - 1] == value:
                        left -= 1
                    right = c + 1
                    while right < cols and board[r, right] == value:
                        right += 1
                    h_length = right - left
                    if h_length >= 3:
                        has_left = left < c
                        has_right = right > c + 1
                        if has_left and has_right:
                            return 'T'
                        else:
                            return 'L'

        return '3'

    @staticmethod
    def _match_base_score(match_type: str, length: int) -> int:
        """Get base score for a match type."""
        if match_type == '5':
            return 1000
        if match_type == 'T' or match_type == 'L':
            return 500
        if match_type == '4':
            return 300
        if match_type == '3':
            return 100
        return 0

    @classmethod
    def _calculate_total_score(cls, matches: list[dict], rows: int) -> int:
        """Calculate total score including bottom-heavy weight."""
        total_score = 0
        all_matched_cells = set()

        for match in matches:
            base_score = match['score']
            # Add base score once per match
            total_score += base_score
            # Add bottom-heavy weight: lower rows (higher row index) get more points per cell
            for r, c in match['cells']:
                bottom_weight = r * 10  # row_index * 10
                total_score += bottom_weight
                all_matched_cells.add((r, c))

        return total_score

    @classmethod
    def _simulate_and_score(cls, board: np.ndarray, move: Move) -> int:
        """Simulate a swap and return the total score.

        Args:
            board: Current board state as numpy array
            move: ((r1, c1), (r2, c2)) swap to simulate

        Returns:
            Total heuristic score after the swap, or 0 if no matches
        """
        (r1, c1), (r2, c2) = move

        # Perform swap
        board[r1, c1], board[r2, c2] = board[r2, c2], board[r1, c1]

        # Find all matches after swap
        matches = cls._find_matches(board)

        # Calculate total score
        rows = board.shape[0]
        score = cls._calculate_total_score(matches, rows)

        # Undo swap
        board[r1, c1], board[r2, c2] = board[r2, c2], board[r1, c1]

        return score

    def find_best_move(self, board_matrix: Board, banned_moves: set[Move] | None = None) -> Move | None:
        """Return the highest-scoring adjacent swap, or None if no valid move exists.

        Args:
            board_matrix: 2D sequence representing the game board
            banned_moves: Set of moves to skip (for avoiding loops)

        Returns:
            Best move as ((r1, c1), (r2, c2)) or None
        """
        rows, cols = self._validate_board(board_matrix)
        if rows == 0 or cols == 0:
            return None

        # Convert to numpy array for efficient manipulation
        board = np.array(board_matrix, dtype=int)
        best_move: Move | None = None
        best_score = 0
        best_priority = (-1, -1)

        banned = banned_moves or set()

        for r in range(rows):
            for c in range(cols):
                # Check right neighbor (horizontal swap)
                if c + 1 < cols:
                    move = ((r, c), (r, c + 1))
                    if move not in banned:
                        # Skip swaps where both cells are zero (no effect)
                        if board[r, c] == 0 and board[r, c + 1] == 0:
                            pass
                        else:
                            score = self._simulate_and_score(board, move)
                            if score > 0:
                                priority = (max(r, r), max(c, c + 1))
                                if score > best_score or (score == best_score and priority > best_priority):
                                    best_score = score
                                    best_move = move
                                    best_priority = priority

                # Check down neighbor (vertical swap)
                if r + 1 < rows:
                    move = ((r, c), (r + 1, c))
                    if move not in banned:
                        # Skip swaps where both cells are zero (no effect)
                        if board[r, c] == 0 and board[r + 1, c] == 0:
                            pass
                        else:
                            score = self._simulate_and_score(board, move)
                            if score > 0:
                                priority = (max(r, r + 1), max(c, c))
                                if score > best_score or (score == best_score and priority > best_priority):
                                    best_score = score
                                    best_move = move
                                    best_priority = priority

        return best_move


if __name__ == "__main__":
    pass
