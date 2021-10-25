from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from dataclasses import asdict

from utils.dataloader import load_sim_data, load_real_data
from datatypes.eskf_params import ESKFTuningParams, ESKFStaticParams
from datatypes.eskf_states import ErrorStateGauss, NominalState
from datatypes.multivargaussian import MultiVarGaussStamped
from datatypes.measurements import ImuMeasurement, GnssMeasurement

from plotting import (plot_state, plot_position_path_3d,
                      plot_nis, plot_errors, plot_nees)

from eskf import ESKF
from nis_nees import get_NIS, get_NEES, get_error, get_time_pairs
import config
import tuning_sim
import tuning_real


def run_eskf(eskf_tuning_params: ESKFTuningParams,
             eskf_static_params: ESKFStaticParams,
             imu_measurements: List[ImuMeasurement],
             gnss_measurements: List[GnssMeasurement],
             x_nom_init: NominalState,
             x_err_gauss_init: ErrorStateGauss
             ) -> Tuple[List[NominalState],
                        List[ErrorStateGauss],
                        List[MultiVarGaussStamped]]:

    eskf = ESKF(**asdict(eskf_tuning_params),
                **asdict(eskf_static_params),
                do_approximations=config.DO_APPROXIMATIONS)

    x_nom = x_nom_init
    x_err_gauss = x_err_gauss_init

    x_nom_seq = []
    x_err_gauss_seq = []
    z_gnss_pred_gauss_seq = []
    gnss_measurements_copy = gnss_measurements.copy()
    next_logging_time = 0
    LOGGING_DELTA = 0.1
    for z_imu in tqdm(imu_measurements):
        x_nom, x_err_gauss = eskf.predict_from_imu(
            x_nom, x_err_gauss, z_imu)

        if (len(gnss_measurements_copy) > 0
                and z_imu.ts >= gnss_measurements_copy[0].ts):
            z_gnss = gnss_measurements_copy.pop(0)

            # we pretend z_gnss arrived at the same time as the last z_imu
            # this is not ideal, but works fine as the IMU intervals are small
            z_gnss.ts = z_imu.ts

            x_nom, x_err_gauss, z_gnss_pred_gauss = eskf.update_from_gnss(
                x_nom, x_err_gauss, z_gnss)
            z_gnss_pred_gauss_seq.append(z_gnss_pred_gauss)
            next_logging_time = -np.inf

        if z_imu.ts >= next_logging_time:
            x_nom_seq.append(x_nom)
            x_err_gauss_seq.append(x_err_gauss)
            next_logging_time = z_imu.ts + LOGGING_DELTA
    return x_nom_seq, x_err_gauss_seq, z_gnss_pred_gauss_seq


def main():
    if config.RUN == 'sim':
        print(f"Running {config.MAX_TIME} seconds of simulated data set")
        (x_true_data, z_imu_data, z_gnss_data, drone_params
         ) = load_sim_data(config.MAX_TIME)
        tuning_params = tuning_sim.tuning_params_sim
        x_nom_init = tuning_sim.x_nom_init_sim
        x_err_init = tuning_sim.x_err_init_sim

    elif config.RUN == 'real':
        print(f"Running {config.MAX_TIME} seconds of real data set")
        x_true_data = None
        (z_imu_data, z_gnss_data, drone_params
         ) = load_real_data(config.MAX_TIME)
        tuning_params = tuning_real.tuning_params_real
        x_nom_init = tuning_real.x_nom_init_real
        x_err_init = tuning_real.x_err_init_real
    else:
        raise IndexError("config.RUN must be 'sim' or 'real'")

    x_nom_seq, x_err_gauss_seq, z_gnss_pred_gauss_seq = run_eskf(
        tuning_params, drone_params,
        z_imu_data, z_gnss_data,
        x_nom_init, x_err_init)

    NIS_times, z_true_pred_pairs = get_time_pairs(z_gnss_data,
                                                  z_gnss_pred_gauss_seq)

    NISxyz_seq = [(get_NIS(z, pred)) for z, pred in z_true_pred_pairs]
    NISxy_seq = [(get_NIS(z, pred, [0, 1])) for z, pred in z_true_pred_pairs]
    NISz_seq = [(get_NIS(z, pred, [2])) for z, pred in z_true_pred_pairs]
    plot_nis(NIS_times, NISxyz_seq, NISxy_seq, NISz_seq)

    if x_true_data:
        x_times, x_true_nom_pairs = get_time_pairs(x_true_data,
                                                   x_nom_seq)
        errors = np.array([get_error(x_true, x_nom)
                           for x_true, x_nom in x_true_nom_pairs])
        err_gt_est_pairs = list(zip(errors, x_err_gauss_seq))
        NEES_pos_seq = [(get_NEES(gt, est, [0, 1, 2]))
                        for gt, est in err_gt_est_pairs]
        NEES_vel_seq = [(get_NEES(gt, est, [3, 4, 5]))
                        for gt, est in err_gt_est_pairs]
        NEES_avec_seq = [(get_NEES(gt, est, [6, 7, 8]))
                         for gt, est in err_gt_est_pairs]
        NEES_accm_seq = [(get_NEES(gt, est, [9, 10, 11]))
                         for gt, est in err_gt_est_pairs]
        NEES_gyro_seq = [(get_NEES(gt, est, [12, 13, 14]))
                         for gt, est in err_gt_est_pairs]

        plot_errors(x_times, errors)
        plot_nees(x_times, NEES_pos_seq, NEES_vel_seq,
                  NEES_avec_seq, NEES_accm_seq, NEES_gyro_seq)

    plot_state(x_nom_seq)
    plot_position_path_3d(x_nom_seq, x_true_data)

    plt.show(block=True)


if __name__ == '__main__':
    main()
