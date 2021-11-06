import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss


tuning_params_sim = ESKFTuningParams(
    accm_std=1.57e-3,
    accm_bias_std=3.0e-2, # Good value 
    accm_bias_p=1.0e-5, # good value. 1 is also a good value

    gyro_std=5.0e-5,
    gyro_bias_std=1.0e-2, #0.0002,0.000000002
    gyro_bias_p=1.0e-5,#1/(10*3600), # 1/(h*3600)

    gnss_std_ne=0.3,
    gnss_std_d=2)



x_nom_init_sim = NominalState(
    np.array([ 2e-01,  1e-04, -5]),  # position
    np.array([20,  0.03,  0.1]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0.]),  # orientation RotationQuaterion(0.9999988368091641, np.array([0.00071232, 0.00124198, 0.00052581])), 
    np.array([-0.09, -0.09,  0.07]), #np.zeros(3),  # accelerometer bias
    np.array([0, 0, 0]),#np.array([-0.001, -0.0007,  0.0004]),# np.zeros(3),  # gyro bias
    ts=0.)

init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
                         a=[0.0005,  # position
                            0.06,  # velocity
                            np.deg2rad(0.1),  # angle vector
                            0.05,  # accelerometer bias
                            0.005])  # gyro bias
x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)


# x_nom_init_sim = NominalState(
#     np.array([ 1,  1, -5]),  # position
#     np.array([20,  0.03,  0.1]),  # velocity
#     RotationQuaterion.from_euler([0., 0., 0.]),  # orientation RotationQuaterion(0.9999988368091641, np.array([0.00071232, 0.00124198, 0.00052581])), 
#     np.array([-0.1, -0.1,  0.1]), #np.zeros(3),  # accelerometer bias
#     np.array([-0.01, -0.001,  0.001]),# np.zeros(3),  # gyro bias
#     ts=0.)

# init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
#                          a=[5.0e-1,  # position
#                             0.03,  # velocity
#                             np.deg2rad(0.1),  # angle vector
#                             0.05,  # accelerometer bias
#                             0.05])  # gyro bias 0.00005 rad/s 0.0017
# x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)






# x_nom_init_sim = NominalState(
#     np.array([ 1.99990169e-01,  1.37733179e-04, -4.99943339e+00]),  # position
#     np.array([19.99803385,  0.02754664,  0.11332131]),  # velocity
#     RotationQuaterion.from_euler([0., 0., 0.]),  # orientation RotationQuaterion(0.9999988368091641, np.array([0.00071232, 0.00124198, 0.00052581])), 
#     np.array([-0.08751116, -0.08758619,  0.06912216]), #np.zeros(3),  # accelerometer bias
#     np.array([-0.00095494, -0.00073245,  0.00037308]),# np.zeros(3),  # gyro bias
#     ts=0.)

# init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
#                          a=[0.000000005,  # position
#                             0.000000006,  # velocity
#                             np.deg2rad(0.1),  # angle vector
#                             0.000000001,  # accelerometer bias
#                             0.000000005])  # gyro bias
# x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)












# tuning_params_sim = ESKFTuningParams(
#     accm_std=0.004,
#     accm_bias_std=0.25, # Good value 
#     accm_bias_p=0, # good value. 1 is also a good value

#     gyro_std=0.005,
#     gyro_bias_std=0.0005,
#     gyro_bias_p=0,

#     gnss_std_ne=0.25,
#     gnss_std_d=0.38)


# x_nom_init_sim = NominalState(
#     np.array([ 2e-01,  1e-04, -5]),  # position
#     np.array([20,  0.03,  0.1]),  # velocity
#     RotationQuaterion.from_euler([0., 0., 0.]),  # orientation RotationQuaterion(0.9999988368091641, np.array([0.00071232, 0.00124198, 0.00052581])), 
#     np.array([-0.09, -0.09,  0.07]), #np.zeros(3),  # accelerometer bias
#     np.array([-0.001, -0.0007,  0.0004]),# np.zeros(3),  # gyro bias
#     ts=0.)

# init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
#                          a=[0.0005,  # position
#                             0.06,  # velocity
#                             np.deg2rad(0.1),  # angle vector
#                             0.005,  # accelerometer bias
#                             0.000009])  # gyro bias
# x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)