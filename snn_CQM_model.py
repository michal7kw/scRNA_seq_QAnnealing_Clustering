import os
import itertools
import click
import pandas as pd
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel, QuadraticModel

def build_cqm(costs, weights, max_weight):
    """Construct a CQM for the knapsack problem.
    Args:
        costs (array-like):
            Array of costs for the items.
        weights (array-like):
            Array of weights for the items.
        max_weight (int):
            Maximum allowable weight for the knapsack.
    Returns:
        Constrained quadratic model instance that represents the knapsack problem.
    """
    num_items = len(costs) #number of nodes

    cqm = ConstrainedQuadraticModel()
    obj = BinaryQuadraticModel(vartype='BINARY')
    constraint = QuadraticModel()

    for i in range(num_items):
        # Objective is to maximize the total costs
        obj.add_variable(i)
        obj.set_linear(i, -costs[i])
        # Constraint is to keep the sum of items' weights under or equal capacity
        constraint.add_variable('BINARY', i)
        constraint.set_linear(i, weights[i])

    cqm.set_objective(obj)
    cqm.add_constraint(constraint, sense="<=", rhs=max_weight, label='capacity')

    return cqm