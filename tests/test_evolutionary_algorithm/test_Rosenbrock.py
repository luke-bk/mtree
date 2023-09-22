import numpy as np
import pytest

# Import the rosenbrock function
from evolutionary_algorithm.evaluation.fitness_function.rosenbrock import rosenbrock


# Test case 1: Minimum value at (x=1, y=1)
def test_rosenbrock_minimum():
    input1 = [1, 1]
    assert rosenbrock(input1) == 0


def test_rosenbrock_minimum_30d():
    input1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert rosenbrock(input1) == 0


# Test case 2: A point away from the minimum (x=0, y=0)
def test_rosenbrock_away_from_minimum():
    input2 = [0, 0]
    # input2 = np.array([0, 0])
    assert rosenbrock(input2) == 101


# Test case 3: A point farther from the minimum (x=2, y=3)
def test_rosenbrock_far_from_minimum():
    input3 = [2, 3]
    assert rosenbrock(input3) == 401


# Test case 4: A point with negative values (x=-1, y=2)
def test_rosenbrock_negative_values():
    input4 = [-1, 2]
    assert rosenbrock(input4) == 104
