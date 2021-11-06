import numpy as np
from numpy import ndarray
from typing import Sequence, Optional

from datatypes.measurements import GnssMeasurement
from datatypes.eskf_states import NominalState, ErrorStateGauss
from datatypes.multivargaussian import MultiVarGaussStamped

import solution


def get_NIS(z_gnss: GnssMeasurement,
            z_gnss_pred_gauss: MultiVarGaussStamped,
            marginal_idxs: Optional[Sequence[int]] = None
            ) -> float:
    """Calculate NIS

    Args:
        z_gnss (GnssMeasurement): gnss measurement
        z_gnss_pred_gauss (MultiVarGaussStamped): predicted gnss measurement
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NIS in only xy direction.  

    Returns:
        NIS (float): NIS value
    """
    if marginal_idxs != None:
        z_gnss_pred_gauss_marginalized = z_gnss_pred_gauss.marginalize(marginal_idxs)
        NIS_guess = z_gnss_pred_gauss_marginalized.mahalanobis_distance_sq(z_gnss.pos[marginal_idxs])
    else:
        NIS_guess = z_gnss_pred_gauss.mahalanobis_distance_sq(z_gnss.pos)

    
    # TODO replace this with your own code
    # NIS = solution.nis_nees.get_NIS(z_gnss, z_gnss_pred_gauss, marginal_idxs)

    return NIS_guess


def get_error(x_true: NominalState,
              x_nom: NominalState,
              ) -> 'ndarray[15]':
    """Finds the error (difference) between True state and 
    nominal state. See (Table 10.1).


    Returns:
        error (ndarray[15]): difference between x_true and x_nom. 
    """
    error = np.zeros((15,))
    error[0:3] = x_true.pos - x_nom.pos
    error[3:6] = x_true.vel - x_nom.vel
    quaternion_error = x_nom.ori.conjugate()@x_true.ori
    error[6:9] = quaternion_error.as_avec()
    error[9:12] = x_true.accm_bias - x_nom.accm_bias
    error[12:15] = x_true.gyro_bias - x_nom.gyro_bias
    # TODO replace this with your own code
    # error = solution.nis_nees.get_error(x_true, x_nom)

    return error


def get_NEES(error: 'ndarray[15]',
             x_err: ErrorStateGauss,
             marginal_idxs: Optional[Sequence[int]] = None
             ) -> float:
    """Calculate NEES

    Args:
        error (ndarray[15]): errors between x_true and x_nom (from get_error)
        x_err (ErrorStateGauss): estimated error
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NEES for only the position. 

    Returns:
        NEES (float): NEES value
    """
    # if marginal_idxs != None:
    #     x_err_marginalized = x_err.marginalize(marginal_idxs)
    #     NEES_guess = x_err_marginalized.mahalanobis_distance_sq(x_err.mean[marginal_idxs])
    # else:
    #     NIS_guess = z_gnss_pred_gauss.mahalanobis_distance_sq(z_gnss.pos)

    if marginal_idxs != None:
        x_err_marginalized = x_err.marginalize(marginal_idxs)
        error_marginalized = error[marginal_idxs]
        v = x_err_marginalized.mean - error_marginalized
        P = x_err_marginalized.cov
    else:
        v = x_err.mean - error
        P = x_err.cov
    
    NEES = v.T@np.linalg.inv(P)@v

    if marginal_idxs != None:
        x_err_marginalized = x_err.marginalize(marginal_idxs)
        error_marginalized = error[marginal_idxs]
        v = x_err_marginalized.mean - error_marginalized
        P = x_err_marginalized.cov
    else:
        v = x_err.mean - error
        P = x_err.cov
    
    NEES = v.T@np.linalg.inv(P)@v

    # TODO replace this with your own code
    # NEES = solution.nis_nees.get_NEES(error, x_err, marginal_idxs)

    # print("NEES:", NEES)
    # print("marginal_idxs", marginal_idxs)
    return NEES


def get_time_pairs(unique_data, data):
    """match data from two different time series based on timestamps"""
    gt_dict = dict(([x.ts, x] for x in unique_data))
    pairs = [(gt_dict[x.ts], x) for x in data if x.ts in gt_dict]
    times = [pair[0].ts for pair in pairs]
    return times, pairs

def get_average(list):
    number_of_samples = len(list)
    sum = 0
    for value in list:
        sum = sum + value
    average = sum/number_of_samples

    return average

def get_RMSE(error):

    N = len(error)
    sum_error_squared = 0
    for i in range(N):
        error_squared = error[i]**2
        sum_error_squared = sum_error_squared + error_squared
    
    mean_error_squared = sum_error_squared/N
    RMSE = np.sqrt(mean_error_squared)

    return RMSE

def print_RMSE(errors):

    names = ["pos x", "pos y", "pos z", "vel u", "vel v", "vel w", "phi", "theta", "psi", "accm bias x", "accm bias y", "accm bias z", "gyro bias phi", "gyro bias theta", "gyro bias psi"]
    for i in range(len(names)):
        RMSE = get_RMSE(errors[i,:])
        print("RMSE of", names[i], " is ", round(RMSE,4),"\n" )
 
