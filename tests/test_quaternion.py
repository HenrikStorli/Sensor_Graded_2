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


class Test_RotationQuaterion_multiply:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["quaternion.RotationQuaterion.multiply"]:
            params = tuple(finput.values())

            self_1, other_1 = deepcopy(params)

            self_2, other_2 = deepcopy(params)

            quaternion_product_1 = quaternion.RotationQuaterion.multiply(self_1, other_1)

            quaternion_product_2 = solution.quaternion.RotationQuaterion.multiply(self_2, other_2)
            
            assert compare(quaternion_product_1, quaternion_product_2)
            
            assert compare(self_1, self_2)
            assert compare(other_1, other_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["quaternion.RotationQuaterion.multiply"][:1]:
            params = finput

            solution.used["quaternion.RotationQuaterion.multiply"] = False

            quaternion.RotationQuaterion.multiply(**params)

            assert not solution.used["quaternion.RotationQuaterion.multiply"], "The function uses the solution"


class Test_RotationQuaterion_conjugate:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["quaternion.RotationQuaterion.conjugate"]:
            params = tuple(finput.values())

            self_1, = deepcopy(params)

            self_2, = deepcopy(params)

            conj_1 = quaternion.RotationQuaterion.conjugate(self_1,)

            conj_2 = solution.quaternion.RotationQuaterion.conjugate(self_2,)
            
            assert compare(conj_1, conj_2)
            
            assert compare(self_1, self_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["quaternion.RotationQuaterion.conjugate"][:1]:
            params = finput

            solution.used["quaternion.RotationQuaterion.conjugate"] = False

            quaternion.RotationQuaterion.conjugate(**params)

            assert not solution.used["quaternion.RotationQuaterion.conjugate"], "The function uses the solution"


class Test_RotationQuaterion_as_rotmat:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["quaternion.RotationQuaterion.as_rotmat"]:
            params = tuple(finput.values())

            self_1, = deepcopy(params)

            self_2, = deepcopy(params)

            R_1 = quaternion.RotationQuaterion.as_rotmat(self_1,)

            R_2 = solution.quaternion.RotationQuaterion.as_rotmat(self_2,)
            
            assert compare(R_1, R_2)
            
            assert compare(self_1, self_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["quaternion.RotationQuaterion.as_rotmat"][:1]:
            params = finput

            solution.used["quaternion.RotationQuaterion.as_rotmat"] = False

            quaternion.RotationQuaterion.as_rotmat(**params)

            assert not solution.used["quaternion.RotationQuaterion.as_rotmat"], "The function uses the solution"


class Test_RotationQuaterion_as_euler:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["quaternion.RotationQuaterion.as_euler"]:
            params = tuple(finput.values())

            self_1, = deepcopy(params)

            self_2, = deepcopy(params)

            euler_1 = quaternion.RotationQuaterion.as_euler(self_1,)

            euler_2 = solution.quaternion.RotationQuaterion.as_euler(self_2,)
            
            assert compare(euler_1, euler_2)
            
            assert compare(self_1, self_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["quaternion.RotationQuaterion.as_euler"][:1]:
            params = finput

            solution.used["quaternion.RotationQuaterion.as_euler"] = False

            quaternion.RotationQuaterion.as_euler(**params)

            assert not solution.used["quaternion.RotationQuaterion.as_euler"], "The function uses the solution"


class Test_RotationQuaterion_as_avec:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["quaternion.RotationQuaterion.as_avec"]:
            params = tuple(finput.values())

            self_1, = deepcopy(params)

            self_2, = deepcopy(params)

            avec_1 = quaternion.RotationQuaterion.as_avec(self_1,)

            avec_2 = solution.quaternion.RotationQuaterion.as_avec(self_2,)
            
            assert compare(avec_1, avec_2)
            
            assert compare(self_1, self_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["quaternion.RotationQuaterion.as_avec"][:1]:
            params = finput

            solution.used["quaternion.RotationQuaterion.as_avec"] = False

            quaternion.RotationQuaterion.as_avec(**params)

            assert not solution.used["quaternion.RotationQuaterion.as_avec"], "The function uses the solution"


if __name__ == "__main__":
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
