import pickle
from numpy.core.numeric import isscalar
import pytest
from copy import deepcopy
import sys
from pathlib import Path
import numpy as np
import os
from dataclasses import is_dataclass, astuple
from collections.abc import Iterable

assignment_name = "eskf"

this_file = Path(__file__)
tests_folder = this_file.parent
test_data_file = tests_folder.joinpath("test_data.pickle")
project_folder = tests_folder.parent
code_folder = project_folder.joinpath(assignment_name)

sys.path.insert(0, str(code_folder))

import solution  # nopep8
import cross_matrix, eskf, nis_nees, quaternion  # nopep8


@pytest.fixture
def test_data():
    with open(test_data_file, "rb") as file:
        test_data = pickle.load(file)
    return test_data


def compare(a, b):
    if isinstance(b, np.ndarray) or np.isscalar(b):
        return np.allclose(a, b, atol=1e-6)

    elif is_dataclass(b):
        if type(a).__name__ != type(b).__name__:
            return False
        a_tup, b_tup = astuple(a), astuple(b)
        return all([compare(i, j) for i, j in zip(a_tup, b_tup)])

    elif isinstance(b, Iterable):
        return all([compare(i, j) for i, j in zip(a, b)])

    else:
        return a == b


class Test_get_cross_matrix:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["cross_matrix.get_cross_matrix"]:
            params = tuple(finput.values())

            vec_1, = deepcopy(params)

            vec_2, = deepcopy(params)

            S_1 = cross_matrix.get_cross_matrix(vec_1,)

            S_2 = solution.cross_matrix.get_cross_matrix(vec_2,)
            
            assert compare(S_1, S_2)
            
            assert compare(vec_1, vec_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["cross_matrix.get_cross_matrix"][:1]:
            params = finput

            solution.used["cross_matrix.get_cross_matrix"] = False

            cross_matrix.get_cross_matrix(**params)

            assert not solution.used["cross_matrix.get_cross_matrix"], "The function uses the solution"


if __name__ == "__main__":
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
