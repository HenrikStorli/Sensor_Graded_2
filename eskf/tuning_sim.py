import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_sim = ESKFTuningParams(
    accm_std=0.5,
    accm_bias_std=1.,
    accm_bias_p=0.000001,

    gyro_std=4.46e-4,
    gyro_bias_std=0.01,
    gyro_bias_p=0.000001,

    gnss_std_ne=1.,
    gnss_std_d=3.)

x_nom_init_sim = NominalState(
   np.array([ 1.99990169e-01,  1.37733179e-04, -4.99943339e+00]),  # position
    np.array([19.99803385,  0.02754664,  0.11332131]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0.]),  # orientation RotationQuaterion(0.9999988368091641, np.array([0.00071232, 0.00124198, 0.00052581])), 
    np.array([-0.08751116, -0.08758619,  0.06912216]), #np.zeros(3),  # accelerometer bias
    np.array([-0.00095494, -0.00073245,  0.00037308]),# np.zeros(3),  # gyro bias
    ts=0.)

init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
                         a=[0.1,  # position
                            0.1,  # velocity
                            np.deg2rad(10.),  # angle vector
                            500,  # accelerometer bias
                            500])  # gyro bias
x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)
