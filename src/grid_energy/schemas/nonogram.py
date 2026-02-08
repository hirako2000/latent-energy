from typing import List

import msgspec


class NonogramHints(msgspec.Struct):
    row_hints: List[List[int]]
    col_hints: List[List[int]]

class NonogramPuzzle(msgspec.Struct, rename="lower"):
    size: int
    initialization: List[List[str]]
    solution: List[List[str]]
    hints: NonogramHints
    id: str = "unknown"
