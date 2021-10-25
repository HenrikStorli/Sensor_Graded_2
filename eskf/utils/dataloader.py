from pathlib import Path
import numpy as np
from scipy.io import loadmat

from datatypes.measurements import ImuMeasurement, GnssMeasurement
from datatypes.eskf_states import NominalState
from datatypes.eskf_params import ESKFStaticParams

from quaternion import RotationQuaterion

data_dir = Path(__file__).parents[2].joinpath('data')
simulated_data_file = data_dir.joinpath('task_simulation.mat')
real_data_file = data_dir.joinpath('task_real.mat')


def load_sim_data(max_time=np.inf):
    loaded_data = loadmat(simulated_data_file)

    x_true = loaded_data["xtrue"].T

    timeGNSS = loaded_data["timeGNSS"].ravel()
    z_GNSS = loaded_data["zGNSS"].T

    timeIMU = loaded_data["timeIMU"].ravel()
    z_acceleration = loaded_data["zAcc"].T
    z_gyroscope = loaded_data["zGyro"].T

    lever_arm = loaded_data["leverarm"].ravel()
    S_a = loaded_data["S_a"]
    S_g = loaded_data["S_g"]

    x_nom_true_data = [NominalState(x[:3], x[3:6],
                                    RotationQuaterion(x[6], x[7:10]),
                                    x[10:13], x[13:16],
                                    ts)
                       for x, ts in zip(x_true, timeIMU)
                       if ts <= max_time]

    imu_measurements = [ImuMeasurement(ts, acc, gyro) for ts, acc, gyro
                        in zip(timeIMU, z_acceleration, z_gyroscope)
                        if ts <= max_time]
    gnss_measurements = [GnssMeasurement(ts, pos) for ts, pos
                         in zip(timeGNSS, z_GNSS)
                         if ts <= max_time]
    drone_params = ESKFStaticParams(S_a, S_g, lever_arm)

    return x_nom_true_data, imu_measurements, gnss_measurements, drone_params


def load_real_data(max_time=np.inf):
    loaded_data = loadmat(real_data_file)

    timeGNSS = loaded_data["timeGNSS"].ravel()
    z_GNSS = loaded_data["zGNSS"].T
    accuracy_GNSS = loaded_data["GNSSaccuracy"].ravel()

    timeIMU = loaded_data["timeIMU"].ravel()
    z_acceleration = loaded_data["zAcc"].T
    z_gyroscope = loaded_data["zGyro"].T

    lever_arm = loaded_data["leverarm"].ravel()
    S_a = loaded_data["S_a"]
    S_g = loaded_data["S_g"]

    start_time = 302850
    imu_measurements = [ImuMeasurement(ts-start_time, acc, gyro)
                        for ts, acc, gyro
                        in zip(timeIMU, z_acceleration, z_gyroscope)
                        if start_time <= ts < max_time+start_time]

    gnss_measurements = [GnssMeasurement(ts-start_time, pos, precision)
                         for ts, pos, precision
                         in zip(timeGNSS, z_GNSS, accuracy_GNSS)
                         if start_time <= ts < max_time+start_time]

    drone_params = ESKFStaticParams(S_a, S_g, lever_arm)

    return imu_measurements, gnss_measurements, drone_params
