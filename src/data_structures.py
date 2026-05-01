from dataclasses import dataclass
from typing import List


@dataclass
class Clause:
    positive_vars: List[int]
    negative_vars: List[int]
    weight: float = 1.0


@dataclass
class MaxSATInstance:
    num_vars: int
    clauses: List[Clause]