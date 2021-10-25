import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_sim = ESKFTuningParams(
    accm_std=1.,
    accm_bias_std=1.,
    accm_bias_p=1.,

    gyro_std=1.,
    gyro_bias_std=1.,
    gyro_bias_p=1.,

    gnss_std_ne=1.,
    gnss_std_d=1.)

x_nom_init_sim = NominalState(
    np.array([0., 0., 0.]),  # position
    np.array([0., 0., 0.]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0.]),  # orientation
    np.zeros(3),  # accelerometer bias
    np.zeros(3),  # gyro bias
    ts=0.)

init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
                         a=[1.,  # position
                            1.,  # velocity
                            np.deg2rad(1),  # angle vector
                            1.,  # accelerometer bias
                            1.])  # gyro bias
x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)
