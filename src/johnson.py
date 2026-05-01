from typing import List, Optional
from src.data_structures import MaxSATInstance
from src.conditional_expectation import conditional_expected_value

def johnson_assignment(instance: MaxSATInstance) -> List[bool]:
    """
    Each variable starts with probability 1/2 of being True.
    Variables are then fixed one by one by choosing the value that
    maximizes the conditional expected total satisfied weight.
    """
    probabilities = [0.5] * instance.num_vars
    assignment: List[Optional[bool]] = [None] * instance.num_vars
    # starting with variables unassigned

    for i in range(instance.num_vars):
        assignment[i] = True
        # temporary fix to True, later unassigned variables still behave randomly with Pr 1/2 
        exp_true = conditional_expected_value(instance, assignment, probabilities)

        assignment[i] = False
        #  temporary with False
        exp_false = conditional_expected_value(instance, assignment, probabilities)

        # compare and then choose the better choice
        if exp_true >= exp_false:
            assignment[i] = True
        else:
            assignment[i] = False
    #  by the end no entries will be None, so this converts the list into List[bool]
    return [bool(x) for x in assignment]