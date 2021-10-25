from numpy import ndarray
from dataclasses import dataclass


@dataclass
class ESKFStaticParams():
    """Dataclass containing static parameter for the drone and IMU

    lever_arm (ndarray[3]): GPS position relative to imu (in body frame)
    accm_correction (ndarray[3,3]): accelerometer correction matrix
    gyro_correction (ndarray[3,3]): gyro correction matrix
    """
    accm_correction: 'ndarray[3]'
    gyro_correction: 'ndarray[3]'
    lever_arm: 'ndarray[3]'


@dataclass
class ESKFTuningParams():
    """Dataclass containing tunable parameter for the eskf

    acc_std (float): accelerometer standard deviation
    acc_bias_std (float): accelerometer bias standard deviation (see 10.50)
    acc_bias_p (float): accelerometer bias random walk gain (see 10.50)

    gyro_std (float): gyro standard deviation
    gyro_bias_std (float): gyro bias standard deviation (see 10.50)
    gyro_bias_p (float): gyro bias random walk gain (see 10.50)

    gnss_std_ne (float): gnss standard deviation in north and east dir (xy)
    gnss_std_d (float): gnss standard deviation in down dir (z)

    use_gnss_accuracy (bool): to use the gnss measurements estimated accuracy
    """
    accm_std: float
    accm_bias_std: float
    accm_bias_p: float

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_p: float

    gnss_std_ne: float
    gnss_std_d: float

    use_gnss_accuracy: bool = False
