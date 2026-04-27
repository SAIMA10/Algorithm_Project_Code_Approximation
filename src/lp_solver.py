'''
LP construction
LP solving
returning y*, z*
'''
from typing import List, Tuple
import pulp
from src.data_structures import MaxSATInstance

def solve_lp_relaxation(instance: MaxSATInstance) -> Tuple[List[float], List[float], float]:
    """
    Solve the LP relaxation of MAX SAT.

    Returns:
        y_star: fractional values for variable truth assignments
        z_star: fractional values for clause satisfaction variables
        lp_value: optimal LP objective value
    """
    n = instance.num_vars
    m = len(instance.clauses)

    # Create maximization LP
    prob = pulp.LpProblem("MAX_SAT_LP_Relaxation", pulp.LpMaximize)

    # y_i in [0,1]
    y = [
        pulp.LpVariable(f"y_{i}", lowBound=0.0, upBound=1.0, cat="Continuous")
        for i in range(n)
    ]

    # z_j in [0,1]
    z = [
        pulp.LpVariable(f"z_{j}", lowBound=0.0, upBound=1.0, cat="Continuous")
        for j in range(m)
    ]

    # Objective: maximize sum_j w_j z_j
    prob += pulp.lpSum(instance.clauses[j].weight * z[j] for j in range(m))

    # Clause constraints:
    # sum_{i in I_j^+} y_i + sum_{i in I_j^-} (1 - y_i) >= z_j
    for j, clause in enumerate(instance.clauses):
        lhs = (
            pulp.lpSum(y[i] for i in clause.positive_vars)
            + pulp.lpSum(1 - y[i] for i in clause.negative_vars)
        )
        prob += lhs >= z[j], f"clause_constraint_{j}"

    # Solve using CBC - elaborate in report, msg = True, to show the solver output
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP solve failed with status: {pulp.LpStatus[status]}")

    # y* will give list of LP variable values, same with z* will give list of clause values, lp_value = LP optimum
    y_star = [pulp.value(var) for var in y]
    z_star = [pulp.value(var) for var in z]
    lp_value = float(pulp.value(prob.objective))

    return y_star, z_star, lp_value
    