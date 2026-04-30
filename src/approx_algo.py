from typing import List, Tuple

from .data_structures import MaxSATInstance
from .evaluation import evaluate_assignment
from .johnson import johnson_assignment
from .lp_rounding import lp_rounding_assignment
from src.lp_solver import solve_lp_relaxation


def three_quarter_approximation(instance: MaxSATInstance) -> Tuple[List[bool], float, str]:
    """
    Run Johnson's algorithm and LP-rounding algorithm,
    evaluate both assignments, and return the better one.
    """
    y_star, z_star, lp_value = solve_lp_relaxation(instance)

    johnson_assign = johnson_assignment(instance)
    johnson_value = evaluate_assignment(instance, johnson_assign)

    lp_assign = lp_rounding_assignment(instance, y_star=y_star)
    lp_round_value = evaluate_assignment(instance, lp_assign)

    if johnson_value >= lp_round_value:
        return johnson_assign, johnson_value, "johnson", lp_value
    else:
        return lp_assign, lp_round_value, "lp_rounding", lp_value