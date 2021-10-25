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


class Test_get_NIS:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["nis_nees.get_NIS"]:
            params = tuple(finput.values())

            z_gnss_1, z_gnss_pred_gauss_1, marginal_idxs_1 = deepcopy(params)

            z_gnss_2, z_gnss_pred_gauss_2, marginal_idxs_2 = deepcopy(params)

            NIS_1 = nis_nees.get_NIS(z_gnss_1, z_gnss_pred_gauss_1, marginal_idxs_1)

            NIS_2 = solution.nis_nees.get_NIS(z_gnss_2, z_gnss_pred_gauss_2, marginal_idxs_2)
            
            assert compare(NIS_1, NIS_2)
            
            assert compare(z_gnss_1, z_gnss_2)
            assert compare(z_gnss_pred_gauss_1, z_gnss_pred_gauss_2)
            assert compare(marginal_idxs_1, marginal_idxs_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["nis_nees.get_NIS"][:1]:
            params = finput

            solution.used["nis_nees.get_NIS"] = False

            nis_nees.get_NIS(**params)

            assert not solution.used["nis_nees.get_NIS"], "The function uses the solution"


class Test_get_error:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["nis_nees.get_error"]:
            params = tuple(finput.values())

            x_true_1, x_nom_1 = deepcopy(params)

            x_true_2, x_nom_2 = deepcopy(params)

            error_1 = nis_nees.get_error(x_true_1, x_nom_1)

            error_2 = solution.nis_nees.get_error(x_true_2, x_nom_2)
            
            assert compare(error_1, error_2)
            
            assert compare(x_true_1, x_true_2)
            assert compare(x_nom_1, x_nom_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["nis_nees.get_error"][:1]:
            params = finput

            solution.used["nis_nees.get_error"] = False

            nis_nees.get_error(**params)

            assert not solution.used["nis_nees.get_error"], "The function uses the solution"


class Test_get_NEES:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["nis_nees.get_NEES"]:
            params = tuple(finput.values())

            error_1, x_err_1, marginal_idxs_1 = deepcopy(params)

            error_2, x_err_2, marginal_idxs_2 = deepcopy(params)

            NEES_1 = nis_nees.get_NEES(error_1, x_err_1, marginal_idxs_1)

            NEES_2 = solution.nis_nees.get_NEES(error_2, x_err_2, marginal_idxs_2)
            
            assert compare(NEES_1, NEES_2)
            
            assert compare(error_1, error_2)
            assert compare(x_err_1, x_err_2)
            assert compare(marginal_idxs_1, marginal_idxs_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["nis_nees.get_NEES"][:1]:
            params = finput

            solution.used["nis_nees.get_NEES"] = False

            nis_nees.get_NEES(**params)

            assert not solution.used["nis_nees.get_NEES"], "The function uses the solution"


if __name__ == "__main__":
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
