from typing import List, Optional
from src.data_structures import MaxSATInstance
from src.conditional_expectation import conditional_expected_value
from src.lp_solver import solve_lp_relaxation


def lp_rounding_assignment(
    instance: MaxSATInstance,
    y_star: List[float] | None = None,
) -> List[bool]:
    """
    The LP relaxation is solved first to obtain y*.
    Then variables are fixed one by one, choosing the value
    that maximizes the conditional expected total satisfied weight.
    """
    if y_star is None:
        y_star, _, _ = solve_lp_relaxation(instance)

    assignment: List[Optional[bool]] = [None] * instance.num_vars

    for i in range(instance.num_vars):
        assignment[i] = True
        exp_true = conditional_expected_value(instance, assignment, y_star)

        assignment[i] = False
        exp_false = conditional_expected_value(instance, assignment, y_star)

        if exp_true >= exp_false:
            assignment[i] = True
        else:
            assignment[i] = False

    return [bool(x) for x in assignment]
