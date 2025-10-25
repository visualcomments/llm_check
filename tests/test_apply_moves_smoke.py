
from typing import List, Tuple

def is_cyclic_neighbor(i: int, j: int, n: int) -> bool:
    return j == (i + 1) % n

def apply_moves(vec: List[int], moves: List[Tuple[int, int]]) -> List[int]:
    a = list(vec)
    n = len(a)
    for (i, j) in moves:
        assert 0 <= i < n and 0 <= j < n
        assert is_cyclic_neighbor(i, j, n)
        a[i], a[j] = a[j], a[i]
    return a

def test_apply_moves_simple():
    vec = [3,2,1]
    moves = [(1,2), (0,1), (1,2)]
    assert apply_moves(vec, moves) == [1,2,3]

def test_cyclic_wrap():
    vec = [2,1,0,3]
    moves = [(3,0), (2,3), (1,2)]
    out = apply_moves(vec, moves)
    assert len(out) == 4  # sanity
