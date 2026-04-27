from __future__ import annotations

import random
from typing import Optional, Tuple

from src.data_structures import Clause, MaxSATInstance


def generate_random_maxsat_instance(
    num_vars: int,
    num_clauses: int,
    clause_len_range: Tuple[int, int] = (1, 3),
    weight_range: Tuple[int, int] = (1, 10),
    seed: Optional[int] = None,
) -> MaxSATInstance:
    """
    Generate a random weighted MAX SAT instance.

    Args:
        num_vars: number of Boolean variables
        num_clauses: number of clauses
        clause_len_range: inclusive range for clause length, e.g. (1, 3)
        weight_range: inclusive range for clause weights, e.g. (1, 10)
        seed: optional random seed for reproducibility

    Returns:
        MaxSATInstance
    """
    rng = random.Random(seed)

    min_len, max_len = clause_len_range
    if num_vars <= 0:
        raise ValueError("num_vars must be positive")
    if num_clauses <= 0:
        raise ValueError("num_clauses must be positive")
    if min_len < 1 or max_len < min_len:
        raise ValueError("invalid clause_len_range")
    if weight_range[0] > weight_range[1]:
        raise ValueError("invalid weight_range")

    clauses = []

    for _ in range(num_clauses):
        k = rng.randint(min_len, min(max_len, num_vars))
        chosen_vars = rng.sample(range(num_vars), k)

        positive_vars = []
        negative_vars = []

        for var in chosen_vars:
            if rng.random() < 0.5:
                positive_vars.append(var)
            else:
                negative_vars.append(var)

        weight = float(rng.randint(weight_range[0], weight_range[1]))
        clauses.append(
            Clause(
                positive_vars=positive_vars,
                negative_vars=negative_vars,
                weight=weight,
            )
        )

    return MaxSATInstance(num_vars=num_vars, clauses=clauses)