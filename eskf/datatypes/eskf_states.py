from numpy import ndarray
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # used to avoid circular imports with solution
    from quaternion import RotationQuaterion

from datatypes.multivargaussian import MultiVarGaussStamped

from config import DEBUG


@dataclass
class NominalState:
    """Class representing a nominal state. See (Table 10.1) in the book.

    Args:
        pos (ndarray[3]): position in NED
        vel (ndarray[3]): velocity in NED
        ori (RotationQuaterion): orientation as a quaternion in NED
        accm_bias (ndarray[3]): accelerometer bias
        gyro_bias (ndarray[3]): gyro bias
    """
    pos: 'ndarray[3]'
    vel: 'ndarray[3]'
    ori: 'RotationQuaterion'
    accm_bias: 'ndarray[3]'
    gyro_bias: 'ndarray[3]'

    ts: Optional[float] = None

    def __post_init__(self):
        if DEBUG:
            assert self.pos.shape == (3,)
            assert self.vel.shape == (3,)
            # hack to avoid circular imports with solution
            assert type(self.ori).__name__ == 'RotationQuaterion'
            assert self.accm_bias.shape == (3,)
            assert self.gyro_bias.shape == (3,)


@dataclass
class ErrorStateGauss(MultiVarGaussStamped):
    """A multivariate gaussian representing the error state.
    Has some properties to fetch out useful indexes"""

    def __post_init__(self):
        super().__post_init__()
        assert self.mean.shape == (15,)

    @property
    def pos(self):
        """position"""
        return self.mean[0:3]

    @property
    def vel(self):
        """velocity"""
        return self.mean[3:6]

    @property
    def avec(self):
        """angles vector
        this is often called a rotation vector
        """
        return self.mean[6:9]

    @property
    def accm_bias(self):
        """accelerometer bias"""
        return self.mean[9:12]

    @property
    def gyro_bias(self):
        """gyro bias"""
        return self.mean[12:15]
