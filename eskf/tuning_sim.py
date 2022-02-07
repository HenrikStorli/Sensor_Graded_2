import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_sim = ESKFTuningParams(
    accm_std=0.0116,
    accm_bias_std=0.001, 
    accm_bias_p=0, 

    gyro_std=0.000236,
    gyro_bias_std=0.0000136,
    gyro_bias_p=0,

    gnss_std_ne=0.3,
    gnss_std_d=0.66336)

x_nom_init_sim = NominalState(
    np.array([ 2e-01,  1.4e-04, -5e+00]),  # position
    np.array([20,  0.03,  0.1]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0.]),  # orientation  
    np.array([0, 0,  0]),   # accelerometer bias
    np.array([0, 0,  0]), # gyro bias
    ts=0.)

init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
                         a=[0.0006,  # position
                            0.009,  # velocity
                            np.deg2rad(0.1),  # angle vector
                            0.09,  # accelerometer bias
                            0.002])  # gyro bias
                            
x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)
