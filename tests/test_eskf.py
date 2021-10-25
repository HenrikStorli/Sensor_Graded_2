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


class Test_ESKF_correct_z_imu:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["eskf.ESKF.correct_z_imu"]:
            params = tuple(finput.values())

            self_1, x_nom_prev_1, z_imu_1 = deepcopy(params)

            self_2, x_nom_prev_2, z_imu_2 = deepcopy(params)

            z_corr_1 = eskf.ESKF.correct_z_imu(self_1, x_nom_prev_1, z_imu_1)

            z_corr_2 = solution.eskf.ESKF.correct_z_imu(self_2, x_nom_prev_2, z_imu_2)
            
            assert compare(z_corr_1, z_corr_2)
            
            assert compare(self_1, self_2)
            assert compare(x_nom_prev_1, x_nom_prev_2)
            assert compare(z_imu_1, z_imu_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["eskf.ESKF.correct_z_imu"][:1]:
            params = finput

            solution.used["eskf.ESKF.correct_z_imu"] = False

            eskf.ESKF.correct_z_imu(**params)

            assert not solution.used["eskf.ESKF.correct_z_imu"], "The function uses the solution"


class Test_ESKF_predict_nominal:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["eskf.ESKF.predict_nominal"]:
            params = tuple(finput.values())

            self_1, x_nom_prev_1, z_corr_1 = deepcopy(params)

            self_2, x_nom_prev_2, z_corr_2 = deepcopy(params)

            x_nom_pred_1 = eskf.ESKF.predict_nominal(self_1, x_nom_prev_1, z_corr_1)

            x_nom_pred_2 = solution.eskf.ESKF.predict_nominal(self_2, x_nom_prev_2, z_corr_2)
            
            assert compare(x_nom_pred_1, x_nom_pred_2)
            
            assert compare(self_1, self_2)
            assert compare(x_nom_prev_1, x_nom_prev_2)
            assert compare(z_corr_1, z_corr_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["eskf.ESKF.predict_nominal"][:1]:
            params = finput

            solution.used["eskf.ESKF.predict_nominal"] = False

            eskf.ESKF.predict_nominal(**params)

            assert not solution.used["eskf.ESKF.predict_nominal"], "The function uses the solution"


class Test_ESKF_get_error_A_continous:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["eskf.ESKF.get_error_A_continous"]:
            params = tuple(finput.values())

            self_1, x_nom_prev_1, z_corr_1 = deepcopy(params)

            self_2, x_nom_prev_2, z_corr_2 = deepcopy(params)

            A_1 = eskf.ESKF.get_error_A_continous(self_1, x_nom_prev_1, z_corr_1)

            A_2 = solution.eskf.ESKF.get_error_A_continous(self_2, x_nom_prev_2, z_corr_2)
            
            assert compare(A_1, A_2)
            
            assert compare(self_1, self_2)
            assert compare(x_nom_prev_1, x_nom_prev_2)
            assert compare(z_corr_1, z_corr_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["eskf.ESKF.get_error_A_continous"][:1]:
            params = finput

            solution.used["eskf.ESKF.get_error_A_continous"] = False

            eskf.ESKF.get_error_A_continous(**params)

            assert not solution.used["eskf.ESKF.get_error_A_continous"], "The function uses the solution"


class Test_ESKF_get_error_GQGT_continous:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["eskf.ESKF.get_error_GQGT_continous"]:
            params = tuple(finput.values())

            self_1, x_nom_prev_1 = deepcopy(params)

            self_2, x_nom_prev_2 = deepcopy(params)

            GQGT_1 = eskf.ESKF.get_error_GQGT_continous(self_1, x_nom_prev_1)

            GQGT_2 = solution.eskf.ESKF.get_error_GQGT_continous(self_2, x_nom_prev_2)
            
            assert compare(GQGT_1, GQGT_2)
            
            assert compare(self_1, self_2)
            assert compare(x_nom_prev_1, x_nom_prev_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["eskf.ESKF.get_error_GQGT_continous"][:1]:
            params = finput

            solution.used["eskf.ESKF.get_error_GQGT_continous"] = False

            eskf.ESKF.get_error_GQGT_continous(**params)

            assert not solution.used["eskf.ESKF.get_error_GQGT_continous"], "The function uses the solution"


class Test_ESKF_get_discrete_error_diff:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["eskf.ESKF.get_discrete_error_diff"]:
            params = tuple(finput.values())

            self_1, x_nom_prev_1, z_corr_1 = deepcopy(params)

            self_2, x_nom_prev_2, z_corr_2 = deepcopy(params)

            Ad_1, GQGTd_1 = eskf.ESKF.get_discrete_error_diff(self_1, x_nom_prev_1, z_corr_1)

            Ad_2, GQGTd_2 = solution.eskf.ESKF.get_discrete_error_diff(self_2, x_nom_prev_2, z_corr_2)
            
            assert compare(Ad_1, Ad_2)
            assert compare(GQGTd_1, GQGTd_2)
            
            assert compare(self_1, self_2)
            assert compare(x_nom_prev_1, x_nom_prev_2)
            assert compare(z_corr_1, z_corr_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["eskf.ESKF.get_discrete_error_diff"][:1]:
            params = finput

            solution.used["eskf.ESKF.get_discrete_error_diff"] = False

            eskf.ESKF.get_discrete_error_diff(**params)

            assert not solution.used["eskf.ESKF.get_discrete_error_diff"], "The function uses the solution"


class Test_ESKF_predict_x_err:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["eskf.ESKF.predict_x_err"]:
            params = tuple(finput.values())

            self_1, x_nom_prev_1, x_err_prev_gauss_1, z_corr_1 = deepcopy(params)

            self_2, x_nom_prev_2, x_err_prev_gauss_2, z_corr_2 = deepcopy(params)

            x_err_pred_1 = eskf.ESKF.predict_x_err(self_1, x_nom_prev_1, x_err_prev_gauss_1, z_corr_1)

            x_err_pred_2 = solution.eskf.ESKF.predict_x_err(self_2, x_nom_prev_2, x_err_prev_gauss_2, z_corr_2)
            
            assert compare(x_err_pred_1, x_err_pred_2)
            
            assert compare(self_1, self_2)
            assert compare(x_nom_prev_1, x_nom_prev_2)
            assert compare(x_err_prev_gauss_1, x_err_prev_gauss_2)
            assert compare(z_corr_1, z_corr_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["eskf.ESKF.predict_x_err"][:1]:
            params = finput

            solution.used["eskf.ESKF.predict_x_err"] = False

            eskf.ESKF.predict_x_err(**params)

            assert not solution.used["eskf.ESKF.predict_x_err"], "The function uses the solution"


class Test_ESKF_predict_from_imu:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["eskf.ESKF.predict_from_imu"]:
            params = tuple(finput.values())

            self_1, x_nom_prev_1, x_err_gauss_1, z_imu_1 = deepcopy(params)

            self_2, x_nom_prev_2, x_err_gauss_2, z_imu_2 = deepcopy(params)

            x_nom_pred_1, x_err_pred_1 = eskf.ESKF.predict_from_imu(self_1, x_nom_prev_1, x_err_gauss_1, z_imu_1)

            x_nom_pred_2, x_err_pred_2 = solution.eskf.ESKF.predict_from_imu(self_2, x_nom_prev_2, x_err_gauss_2, z_imu_2)
            
            assert compare(x_nom_pred_1, x_nom_pred_2)
            assert compare(x_err_pred_1, x_err_pred_2)
            
            assert compare(self_1, self_2)
            assert compare(x_nom_prev_1, x_nom_prev_2)
            assert compare(x_err_gauss_1, x_err_gauss_2)
            assert compare(z_imu_1, z_imu_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["eskf.ESKF.predict_from_imu"][:1]:
            params = finput

            solution.used["eskf.ESKF.predict_from_imu"] = False

            eskf.ESKF.predict_from_imu(**params)

            assert not solution.used["eskf.ESKF.predict_from_imu"], "The function uses the solution"


class Test_ESKF_get_gnss_measurment_jac:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["eskf.ESKF.get_gnss_measurment_jac"]:
            params = tuple(finput.values())

            self_1, x_nom_1 = deepcopy(params)

            self_2, x_nom_2 = deepcopy(params)

            H_1 = eskf.ESKF.get_gnss_measurment_jac(self_1, x_nom_1)

            H_2 = solution.eskf.ESKF.get_gnss_measurment_jac(self_2, x_nom_2)
            
            assert compare(H_1, H_2)
            
            assert compare(self_1, self_2)
            assert compare(x_nom_1, x_nom_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["eskf.ESKF.get_gnss_measurment_jac"][:1]:
            params = finput

            solution.used["eskf.ESKF.get_gnss_measurment_jac"] = False

            eskf.ESKF.get_gnss_measurment_jac(**params)

            assert not solution.used["eskf.ESKF.get_gnss_measurment_jac"], "The function uses the solution"


class Test_ESKF_predict_gnss_measurement:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["eskf.ESKF.predict_gnss_measurement"]:
            params = tuple(finput.values())

            self_1, x_nom_1, x_err_1, z_gnss_1 = deepcopy(params)

            self_2, x_nom_2, x_err_2, z_gnss_2 = deepcopy(params)

            z_gnss_pred_gauss_1 = eskf.ESKF.predict_gnss_measurement(self_1, x_nom_1, x_err_1, z_gnss_1)

            z_gnss_pred_gauss_2 = solution.eskf.ESKF.predict_gnss_measurement(self_2, x_nom_2, x_err_2, z_gnss_2)
            
            assert compare(z_gnss_pred_gauss_1, z_gnss_pred_gauss_2)
            
            assert compare(self_1, self_2)
            assert compare(x_nom_1, x_nom_2)
            assert compare(x_err_1, x_err_2)
            assert compare(z_gnss_1, z_gnss_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["eskf.ESKF.predict_gnss_measurement"][:1]:
            params = finput

            solution.used["eskf.ESKF.predict_gnss_measurement"] = False

            eskf.ESKF.predict_gnss_measurement(**params)

            assert not solution.used["eskf.ESKF.predict_gnss_measurement"], "The function uses the solution"


class Test_ESKF_get_x_err_upd:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["eskf.ESKF.get_x_err_upd"]:
            params = tuple(finput.values())

            self_1, x_nom_1, x_err_1, z_gnss_pred_gauss_1, z_gnss_1 = deepcopy(params)

            self_2, x_nom_2, x_err_2, z_gnss_pred_gauss_2, z_gnss_2 = deepcopy(params)

            x_err_upd_gauss_1 = eskf.ESKF.get_x_err_upd(self_1, x_nom_1, x_err_1, z_gnss_pred_gauss_1, z_gnss_1)

            x_err_upd_gauss_2 = solution.eskf.ESKF.get_x_err_upd(self_2, x_nom_2, x_err_2, z_gnss_pred_gauss_2, z_gnss_2)
            
            assert compare(x_err_upd_gauss_1, x_err_upd_gauss_2)
            
            assert compare(self_1, self_2)
            assert compare(x_nom_1, x_nom_2)
            assert compare(x_err_1, x_err_2)
            assert compare(z_gnss_pred_gauss_1, z_gnss_pred_gauss_2)
            assert compare(z_gnss_1, z_gnss_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["eskf.ESKF.get_x_err_upd"][:1]:
            params = finput

            solution.used["eskf.ESKF.get_x_err_upd"] = False

            eskf.ESKF.get_x_err_upd(**params)

            assert not solution.used["eskf.ESKF.get_x_err_upd"], "The function uses the solution"


class Test_ESKF_inject:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["eskf.ESKF.inject"]:
            params = tuple(finput.values())

            self_1, x_nom_prev_1, x_err_upd_1 = deepcopy(params)

            self_2, x_nom_prev_2, x_err_upd_2 = deepcopy(params)

            x_nom_inj_1, x_err_inj_1 = eskf.ESKF.inject(self_1, x_nom_prev_1, x_err_upd_1)

            x_nom_inj_2, x_err_inj_2 = solution.eskf.ESKF.inject(self_2, x_nom_prev_2, x_err_upd_2)
            
            assert compare(x_nom_inj_1, x_nom_inj_2)
            assert compare(x_err_inj_1, x_err_inj_2)
            
            assert compare(self_1, self_2)
            assert compare(x_nom_prev_1, x_nom_prev_2)
            assert compare(x_err_upd_1, x_err_upd_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["eskf.ESKF.inject"][:1]:
            params = finput

            solution.used["eskf.ESKF.inject"] = False

            eskf.ESKF.inject(**params)

            assert not solution.used["eskf.ESKF.inject"], "The function uses the solution"


class Test_ESKF_update_from_gnss:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["eskf.ESKF.update_from_gnss"]:
            params = tuple(finput.values())

            self_1, x_nom_prev_1, x_err_prev_1, z_gnss_1 = deepcopy(params)

            self_2, x_nom_prev_2, x_err_prev_2, z_gnss_2 = deepcopy(params)

            x_nom_inj_1, x_err_inj_1, z_gnss_pred_gauss_1 = eskf.ESKF.update_from_gnss(self_1, x_nom_prev_1, x_err_prev_1, z_gnss_1)

            x_nom_inj_2, x_err_inj_2, z_gnss_pred_gauss_2 = solution.eskf.ESKF.update_from_gnss(self_2, x_nom_prev_2, x_err_prev_2, z_gnss_2)
            
            assert compare(x_nom_inj_1, x_nom_inj_2)
            assert compare(x_err_inj_1, x_err_inj_2)
            assert compare(z_gnss_pred_gauss_1, z_gnss_pred_gauss_2)
            
            assert compare(self_1, self_2)
            assert compare(x_nom_prev_1, x_nom_prev_2)
            assert compare(x_err_prev_1, x_err_prev_2)
            assert compare(z_gnss_1, z_gnss_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["eskf.ESKF.update_from_gnss"][:1]:
            params = finput

            solution.used["eskf.ESKF.update_from_gnss"] = False

            eskf.ESKF.update_from_gnss(**params)

            assert not solution.used["eskf.ESKF.update_from_gnss"], "The function uses the solution"


if __name__ == "__main__":
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
