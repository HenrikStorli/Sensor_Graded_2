from numpy import ndarray
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImuMeasurement():
    """Represents raw data received from the imu

    Args:
        ts (float): IMU measurement timestamp
        acc (ndarray[3]): accelerometer measurement
        avel (ndarray[3]): gyro measurement
    """
    ts: float
    acc: 'ndarray[3]'
    avel: 'ndarray[3]'

    def __post_init__(self):
        assert self.acc.shape == (3,)
        assert self.avel.shape == (3,)


@dataclass
class CorrectedImuMeasurement(ImuMeasurement):
    """Represents processed data from the IMU.
    Corrected for axis alignmentand scale scale, and bias. 

    Not 'corrected' for gravity.

    Implementation is exaclty the same as ImuMeasurement

    Args:
        ts (float): IMU measurement timestamp
        acc (ndarray[3]): corrected accelerometer measurement
        avel (ndarray[3]): corrected gyro measurement
    """


@ dataclass
class GnssMeasurement():
    """Represents data received from gnss
    Args:
        ts(ndarray[:]): IMU measurement timestamp
        position(ndarray[:, 3]): GPS position measurement
        accuracy (Optional[float]): the reported accuracy from the gnss
    """
    ts: float
    pos: 'ndarray[3]'
    accuracy: Optional[float] = None

    def __post_init__(self):
        assert self.pos.shape == (3,)
