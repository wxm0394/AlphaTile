"""Phase 3: Logic solver unit tests (no device, no video)."""

from __future__ import annotations

from logic_solver import LogicSolver


def print_board(board: list[list[int]]) -> None:
    for row in board:
        print("  " + " ".join(f"{v:2d}" for v in row))


def run_test(name: str, board: list[list[int]], expected: tuple[tuple[int, int], tuple[int, int]] | None) -> bool:
    print(f"\n=== {name} ===")
    print("Input board:")
    print_board(board)
    solver = LogicSolver()
    move = solver.find_best_move(board)
    print(f"Best move: {move}")
    if move == expected:
        print("✓ PASS")
        return True
    else:
        print(f"✗ FAIL (expected {expected})")
        return False


def main() -> int:
    all_pass = True

    # Test 1: Horizontal 3-match by swapping right
    # Row 0: [1, 1, 0, 1] -> swap (0,2) with (0,3) makes [1, 1, 1, 0]
    board1 = [
        [1, 1, 0, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
    ]
    all_pass &= run_test("Horizontal 3-match (swap right)", board1, ((0, 2), (0, 3)))

    # Test 2: Vertical 4-match by swapping down
    # Col 0: [1, 1, 0, 1] -> swap (2,0) with (3,0) makes [1, 1, 1, 1] (four in a column)
    board2 = [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [0, 2, 3, 4],
        [1, 2, 3, 4],
    ]
    all_pass &= run_test("Vertical 4-match (swap down)", board2, ((2, 0), (3, 0)))

    # Test 3: Board with a valid move (not dead)
    # Swap (2,2)=1 with (3,2)=2 creates horizontal 3-matches in rows 2 and 3
    board3 = [
        [1, 2, 1, 2],
        [2, 1, 2, 1],
        [1, 2, 1, 2],
        [2, 1, 2, 1],
    ]
    all_pass &= run_test("Board with valid move", board3, ((2, 2), (3, 2)))

    # Test 4: All zeros
    board4 = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    all_pass &= run_test("All zeros", board4, None)

    # Test 5: T-shape intersection bonus
    # After swap, row 1 has three 1s, col 1 has three 1s, intersecting at (1,1)
    board5 = [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    # Both ((1,1),(1,2)) and ((1,1),(2,1)) create T-shape with same score.
    # Tie-break prefers larger max row -> ((1,1),(2,1)) wins (max row 2 > 1)
    all_pass &= run_test("T-shape intersection", board5, ((1, 1), (2, 1)))

    # Test 6: Multiple equal-score moves -> prefer lower row
    # Two horizontal 3-matches possible: row 0 and row 2. Should pick row 2.
    board6 = [
        [1, 1, 0, 1],
        [2, 2, 2, 2],
        [3, 3, 0, 3],
        [4, 4, 4, 4],
    ]
    all_pass &= run_test("Tie-break: prefer lower row", board6, ((2, 2), (2, 3)))

    # Test 7: Swap with empty cell still valid if it creates a match
    # (0,0)=1, (0,1)=0 -> swapping creates column 0 match [1,2,3,4] no...
    # Actually swap (0,0)<->(0,1): row0=[0,1,1,1] makes 3-match at cols 1-3
    board7 = [
        [1, 0, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
    ]
    all_pass &= run_test("Swap with empty cell", board7, ((0, 0), (0, 1)))

    print("\n" + "=" * 40)
    if all_pass:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())