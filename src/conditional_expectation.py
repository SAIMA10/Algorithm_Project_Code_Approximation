from typing import List, Optional
from src.data_structures import Clause, MaxSATInstance


def clause_satisfaction_probability(
    clause: Clause,
    partial_assignment: List[Optional[bool]],
    probabilities: List[float],
) -> float:
    """
    Return the probability that this clause will be satisfied,
    given a partial assignment and probabilities for unassigned variables.
    Calculates the Pr[clause satisfied] = 1 - Pr[all literals fail]
    """
    fail_prob = 1.0

    # Non-negated literals fail when their variable is False
    for var in clause.positive_vars:
        value = partial_assignment[var]
        if value is True:
            return 1.0
        elif value is False:
            continue
        else:
            fail_prob *= 1.0 - probabilities[var]

    # Negated literals fail when their variable is True
    for var in clause.negative_vars:
        value = partial_assignment[var]
        if value is False:
            return 1.0
        elif value is True:
            continue
        else:
            fail_prob *= probabilities[var]

    return 1.0 - fail_prob


def conditional_expected_value(
    instance: MaxSATInstance,
    partial_assignment: List[Optional[bool]],
    probabilities: List[float],
) -> float:
    """
    For each clause it computes the Pr of it being satisfied under the current partial assignment and remaining random choices
    then it adds: wj. Pr(Cj satisfied) over all clauses
    """
    total = 0.0
    for clause in instance.clauses:
        prob_sat = clause_satisfaction_probability(
            clause, partial_assignment, probabilities
        )
        total += clause.weight * prob_sat
    return total
