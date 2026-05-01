from src.data_structures import MaxSATInstance, Clause
from src.evaluation import evaluate_assignment
# from src.conditional_expectation import clause_satisfaction_probability, conditional_expected_value
from src.johnson import johnson_assignment
from src.lp_solver import solve_lp_relaxation
from src.lp_rounding import lp_rounding_assignment
from src.approx_algo import three_quarter_approximation
# from data.generate_random_examples import generate_random_maxsat_instance

# testing with specific instances, it will give 3/4 ratio
'''
x1, x2
!x1, x2
x1, !x2
!x1, !x2
'''

instance = MaxSATInstance(
    num_vars=3,
    clauses=[
        Clause(positive_vars=[1,2], negative_vars=[], weight=1.0),
        Clause(positive_vars=[2], negative_vars=[1], weight=1.0),
        Clause(positive_vars=[1], negative_vars=[2], weight=1.0),
        Clause(positive_vars=[], negative_vars=[1,2], weight=1.0),
    ]
)

y_star, z_star, lp_value = solve_lp_relaxation(instance)

johnson_assign = johnson_assignment(instance)
johnson_value = evaluate_assignment(instance, johnson_assign)

lp_assign = lp_rounding_assignment(instance, y_star=y_star)
lp_round_value = evaluate_assignment(instance, lp_assign)

best_assignment, best_value, chosen_method, _ = three_quarter_approximation(instance)

if __name__ == "__main__":    
    y_star, z_star, lp_value = solve_lp_relaxation(instance)
    print(f"LP value: {lp_value:.10f}")
    print(f"Johnson value: {johnson_value:.10f}")
    print(f"LP-rounding value: {lp_round_value:.10f}")
    print(f"Best value: {best_value:.10f}")
    print(f"Chosen method: {chosen_method}")
    print(f"Ratio best/LP: {best_value / lp_value:.10f}")
    print("First 10 y*:", y_star[:10])
    print("First 10 z*:", z_star[:10])
    print("Best assignment:", best_assignment)   



# TESTING CODE BELOW, IGNORE
# Testing #1
# instanceOne = MaxSATInstance(
#     num_vars=3,
#     clauses=[
#         Clause(positive_vars=[0], negative_vars=[], weight=2.0),      # x0
#         Clause(positive_vars=[], negative_vars=[1], weight=3.0),      # not x1
#         Clause(positive_vars=[1], negative_vars=[2], weight=4.0),     # x1 or not x2
#     ]
# )
# assignment = [True, False, True]
# print(evaluate_assignment(instance, assignment))

# Testing #1
# clause = Clause(positive_vars=[0], negative_vars=[1], weight=1.0)  # x0 or not x1
# probabilities = [0.7, 0.4]
# print(clause_satisfaction_probability(clause, partial_assignment, probabilities))

# instance = MaxSATInstance(
#     num_vars=2,
#     clauses=[
#         Clause(positive_vars=[0], negative_vars=[], weight=2.0),      # x0
#         Clause(positive_vars=[], negative_vars=[1], weight=3.0),      # not x1
#         Clause(positive_vars=[0], negative_vars=[1], weight=5.0),     # x0 or not x1
#     ]
# )
# partial_assignment = [None, None]
# probabilities = [0.5, 0.5]
# value = conditional_expected_value(instance, partial_assignment, probabilities)
# print(value)

# Testing johnson
# instance = MaxSATInstance(
#     num_vars=3,
#     clauses=[
#         Clause(positive_vars=[0], negative_vars=[], weight=2.0),      # x0
#         Clause(positive_vars=[], negative_vars=[1], weight=3.0),      # not x1
#         Clause(positive_vars=[0], negative_vars=[2], weight=5.0),     # x0 or not x2
#         Clause(positive_vars=[1, 2], negative_vars=[], weight=4.0),   # x1 or x2
#     ]
# )

# assignment = johnson_assignment(instance)
# value = evaluate_assignment(instance, assignment)

# Testing LP Relaxation
# instance = MaxSATInstance(
#     num_vars=2,
#     clauses=[
#         Clause(positive_vars=[0], negative_vars=[], weight=2.0),      # x0
#         Clause(positive_vars=[], negative_vars=[1], weight=3.0),      # not x1
#         Clause(positive_vars=[0], negative_vars=[1], weight=5.0),     # x0 or not x1
#     ]
# )
# y_star, z_star, lp_value = solve_lp_relaxation(instance)


# Testing and Comparison LP Relaxation + LP Rounding 
# instance = MaxSATInstance(
#     num_vars=3,
#     clauses=[
#         Clause(positive_vars=[0], negative_vars=[], weight=2.0),
#         Clause(positive_vars=[], negative_vars=[1], weight=3.0),
#         Clause(positive_vars=[0], negative_vars=[2], weight=5.0),
#         Clause(positive_vars=[1, 2], negative_vars=[], weight=4.0),
#     ]
# )

# y_star, z_star, lp_value = solve_lp_relaxation(instance)
# assignment = lp_rounding_assignment(instance)
# value = evaluate_assignment(instance, assignment)
# print("y* = ", y_star)
# print("z* = ", z_star)
# print("LP Value = ", lp_value)
# print("LP-rounding assignment:", assignment)
# print("LP-rounding value:", value)

# testing 3/4 approx algiorithm
# instance = MaxSATInstance(
#     num_vars=3,
#     clauses=[
#         Clause(positive_vars=[0], negative_vars=[], weight=2.0),
#         Clause(positive_vars=[], negative_vars=[1], weight=3.0),
#         Clause(positive_vars=[0], negative_vars=[2], weight=5.0),
#         Clause(positive_vars=[1, 2], negative_vars=[], weight=4.0),
#     ]
# )

# best_assignment, best_value, chosen_method, lp_value = three_quarter_approximation(instance)
# print("LP value:", lp_value)
# print("Best assignment:", best_assignment)
# print("Best value:", best_value)
# print("Chosen method:", chosen_method)
# print("Ratio best/LP:", best_value / lp_value if lp_value > 0 else 0.0)


# Testing with randomly generated examples
# instance = generate_random_maxsat_instance(
#     num_vars=200,
#     num_clauses=500,
#     clause_len_range=(1, 3),
#     weight_range=(1, 10),
#     seed=42,
# )
     
