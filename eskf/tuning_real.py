import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_real = ESKFTuningParams(
    accm_std=7.336e-2,
    accm_bias_std=5.529e-4,
    accm_bias_p=0.,

    gyro_std=3.703e-2,
    gyro_bias_std=4.288e-4,# 3.0e-5,
    gyro_bias_p=0.,

    gnss_std_ne= 0.15,# 0.05, with accuracy = True
    gnss_std_d= 0.66336,# 2., with accuracy = True

    use_gnss_accuracy=True)

x_nom_init_real = NominalState(
    np.array([0., 0., 0.]),  # position np.array([1., -0.3, -1.]),
    np.array([0., 0., 0.]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0.]),  # orientation
    np.zeros(3),  # accelerometer bias
    np.zeros(3),  # gyro bias
    ts=0.)

init_std_real = np.repeat(repeats=3,  # repeat each element 3 times
                          a=[1.1,  # position
                             1.1,  # velocity
                             np.deg2rad(90),  # angle vector
                             0.05,  # accelerometer bias
                             0.0002])  # gyro bias

x_err_init_real = ErrorStateGauss(np.zeros(15), np.diag(init_std_real**2), 0.)
