from typing import List
from src.data_structures import Clause, MaxSATInstance

def is_clause_satisfied(clause: Clause, assignment: List[bool]) -> bool:
    for var in clause.positive_vars:
        if assignment[var]:
            return True

    for var in clause.negative_vars:
        if not assignment[var]:
            return True

    return False


def evaluate_assignment(instance: MaxSATInstance, assignment: List[bool]) -> float:
    total = 0.0
    for clause in instance.clauses:
        if is_clause_satisfied(clause, assignment):
            total += clause.weight
    return total