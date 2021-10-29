import numpy as np
from numpy import ndarray
import scipy
from dataclasses import dataclass, field
from typing import Tuple
from functools import cache

from datatypes.multivargaussian import MultiVarGaussStamped
from datatypes.measurements import (ImuMeasurement,
                                    CorrectedImuMeasurement,
                                    GnssMeasurement)
from datatypes.eskf_states import NominalState, ErrorStateGauss
from utils.indexing import block_3x3

from quaternion import RotationQuaterion
from cross_matrix import get_cross_matrix

import solution


@dataclass
class ESKF():

    accm_std: float
    accm_bias_std: float
    accm_bias_p: float

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_p: float

    gnss_std_ne: float
    gnss_std_d: float

    accm_correction: 'ndarray[3,3]'
    gyro_correction: 'ndarray[3,3]'
    lever_arm: 'ndarray[3]'

    do_approximations: bool
    use_gnss_accuracy: bool = False

    Q_err: 'ndarray[12,12]' = field(init=False, repr=False)
    g: 'ndarray[3]' = np.array([0, 0, 9.82])

    def __post_init__(self):

        self.Q_err = scipy.linalg.block_diag(
            self.accm_std ** 2 * self.accm_correction @ self.accm_correction.T,
            self.gyro_std ** 2 * self.gyro_correction @ self.gyro_correction.T,
            self.accm_bias_std ** 2 * np.eye(3),
            self.gyro_bias_std ** 2 * np.eye(3),
        )
        self.gnss_cov = np.diag([self.gnss_std_ne]*2 + [self.gnss_std_d])**2

    def correct_z_imu(self,
                      x_nom_prev: NominalState,
                      z_imu: ImuMeasurement,
                      ) -> CorrectedImuMeasurement:
        """Correct IMU measurement so it gives a measurmenet of acceleration 
        and angular velocity in body.

        Hint: self.accm_correction and self.gyro_correction translates 
        measurements from IMU frame (probably not correct name) to body frame

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_imu (ImuMeasurement): raw IMU measurement

        Returns:
            CorrectedImuMeasurement: corrected IMU measurement
        """
        accelerometer_body =  self.accm_correction@(z_imu.acc - x_nom_prev.accm_bias)
        gyroscope_body = self.gyro_correction@(z_imu.avel - x_nom_prev.gyro_bias)

        z_corr = CorrectedImuMeasurement(z_imu.ts,accelerometer_body,gyroscope_body)
        # print("z_corr guess", z_corr)

        # TODO replace this with your own code
        # z_corr = solution.eskf.ESKF.correct_z_imu(self, x_nom_prev, z_imu)
        # print("z_corr true", z_corr)

        return z_corr

    def predict_nominal(self,
                        x_nom_prev: NominalState,
                        z_corr: CorrectedImuMeasurement,
                        ) -> NominalState:
        """Predict the nominal state, given a corrected IMU measurement

        Hint: Discrete time prediction of equation (10.58)
        See the assignment description for more hints 

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measuremnt

        Returns:
            x_nom_pred (NominalState): predicted nominal state
        """
        Ts = z_corr.ts - x_nom_prev.ts

        if Ts == 0:
            return x_nom_prev

        a = x_nom_prev.ori.as_rotmat()@z_corr.acc + self.g
        vel = x_nom_prev.vel + Ts*a
        pos = x_nom_prev.pos + Ts*x_nom_prev.vel + Ts**2/2*a

        kappa = Ts*(z_corr.avel)# - x_nom_prev.gyro_bias)
        kappa_norm = np.linalg.norm(kappa)

        real_part = np.cos(kappa_norm/2)
        vec_part = np.sin(kappa_norm/2)/kappa_norm*kappa

        rot = x_nom_prev.ori@RotationQuaterion(real_part,vec_part)

        acc_bias = np.exp(-self.accm_bias_p*Ts) * x_nom_prev.accm_bias
        gyro_bias = np.exp(-self.gyro_bias_p*Ts) * x_nom_prev.gyro_bias

        x_nom_pred_guess = NominalState(pos,vel,rot,acc_bias,gyro_bias, z_corr.ts)

        # TODO replace this with your own code
        # x_nom_pred = solution.eskf.ESKF.predict_nominal(
        #     self, x_nom_prev, z_corr)

        # print("pos:", pos - x_nom_pred.pos)
        # print("vel", vel - x_nom_pred.vel)
        # print("Ori real part:", rot.real_part - x_nom_pred.ori.real_part)
        # print("Ori vec part:", rot.vec_part - x_nom_pred.ori.vec_part)
        # print("accm bias:", gyro_bias -x_nom_pred.gyro_bias )
        # print("gyro bias:", gyro_bias -x_nom_pred.gyro_bias)

        return x_nom_pred_guess

    def get_error_A_continous(self,
                              x_nom_prev: NominalState,
                              z_corr: CorrectedImuMeasurement,
                              ) -> 'ndarray[15,15]':
        """Get the transition matrix, A, in (10.68)

        Hint: The S matrices can be created using get_cross_matrix. In the book
        a perfect IMU is expected (thus many I matrices). Here we have 
        to use the correction matrices, self.accm_correction and 
        self.gyro_correction, instead of som of the I matrices.  

        You can use block_3x3 to simplify indexing if you want to.
        The first I element in A can be set as A[block_3x3(0, 1)] = np.eye(3)

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measurement
        Returns:
            A (ndarray[15,15]): A
        """
        A = np.zeros((15,15))
        A[block_3x3(0,1)] = np.eye(3)
        A[block_3x3(1,2)] = -1*x_nom_prev.ori.as_rotmat()@get_cross_matrix(z_corr.acc)
        A[block_3x3(1,3)] = -1*x_nom_prev.ori.as_rotmat()@self.accm_correction
        A[block_3x3(2,2)] = -1*get_cross_matrix(z_corr.avel)
        A[block_3x3(2,4)] = -1*self.gyro_correction
        A[block_3x3(3,3)] = -1*self.accm_bias_p*np.eye(3)
        A[block_3x3(4,4)] = -1*self.gyro_bias_p*np.eye(3)

        # TODO replace this with your own code
        # A = solution.eskf.ESKF.get_error_A_continous(self, x_nom_prev, z_corr)

        return A

    def get_error_GQGT_continous(self,
                                 x_nom_prev: NominalState
                                 ) -> 'ndarray[15, 15]':
        """The noise covariance matrix, GQGT, in (10.68)

        From (Theorem 3.2.2) we can see that (10.68) can be written as 
        d/dt x_err = A@x_err + G@n == A@x_err + m
        where m is gaussian with mean 0 and covariance G @ Q @ G.T. Thats why
        we need GQGT.

        Hint: you can use block_3x3 to simplify indexing if you want to.
        The first I element in G can be set as G[block_3x3(2, 1)] = -np.eye(3)

        Args:
            x_nom_prev (NominalState): previous nominal state
        Returns:
            GQGT (ndarray[15, 15]): G @ Q @ G.T
        """
        G = np.zeros((15,12))
        G[block_3x3(1,0)] = -1*x_nom_prev.ori.as_rotmat()
        G[block_3x3(2,1)] = -1*np.eye(3)
        G[block_3x3(3,2)] = np.eye(3)
        G[block_3x3(4,3)] = np.eye(3)

        Q = self.Q_err

        GQGT = G@Q@G.T

        # TODO replace this with your own code
        #GQGT = solution.eskf.ESKF.get_error_GQGT_continous(self, x_nom_prev)

        return GQGT

    def get_van_loan_matrix(self, V: 'ndarray[30, 30]'):
        """Use this funciton in get_discrete_error_diff to get the van loan 
        matrix. See (4.63)

        All the tests are ran with do_approximations=False

        Args:
            V (ndarray[30, 30]): [description]

        Returns:
            VanLoanMatrix (ndarray[30, 30]): VanLoanMatrix
        """
        if self.do_approximations:
            # second order approcimation of matrix exponential which is faster
            VanLoanMatrix = np.eye(*V.shape) + V + (V@V) / 2
        else:
            VanLoanMatrix = scipy.linalg.expm(V)
        return VanLoanMatrix

    def get_discrete_error_diff(self,
                                x_nom_prev: NominalState,
                                z_corr: CorrectedImuMeasurement,
                                ) -> Tuple['ndarray[15, 15]',
                                           'ndarray[15, 15]']:
        """Get the discrete equivalents of A and GQGT in (4.63)

        Hint: you should use get_van_loan_matrix to get the van loan matrix

        See (4.5 Discretization) and (4.63) for more information. 
        Or see "Discretization of process noise" in 
        https://en.wikipedia.org/wiki/Discretization

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measurement

        Returns:
            Ad (ndarray[15, 15]): discrede transition matrix
            GQGTd (ndarray[15, 15]): discrete noise covariance matrix
        """
        
        Ts = z_corr.ts - x_nom_prev.ts # Timestep

        error_A_continous = self.get_error_A_continous(x_nom_prev, z_corr) # ndarray[15, 15]

        error_GQGT_continous = self.get_error_GQGT_continous(x_nom_prev) # ndarray[15, 15]

        V = np.zeros((30,30))
        V[0:15,0:15] = -error_A_continous
        V[0:15,15:30] = error_GQGT_continous
        V[15:30,15:30] = error_A_continous.T
        V = V*Ts

        VanLoanMatrix = self.get_van_loan_matrix(V)

        Ad = VanLoanMatrix[15:30,15:30].T

        GQGTd = VanLoanMatrix[15:30,15:30].T@VanLoanMatrix[0:15,15:30]

        # TODO replace this with your own code
        # Ad, GQGTd = solution.eskf.ESKF.get_discrete_error_diff(
        #     self, x_nom_prev, z_corr)
        return Ad, GQGTd

    def predict_x_err(self,
                      x_nom_prev: NominalState,
                      x_err_prev_gauss: ErrorStateGauss,
                      z_corr: CorrectedImuMeasurement,
                      ) -> ErrorStateGauss:
        """Predict the error state

        Hint: This is doing a discrete step of (10.68) where x_err 
        is a multivariate gaussian.

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_prev_gauss (ErrorStateGauss): previous error state gaussian
            z_corr (CorrectedImuMeasurement): corrected IMU measuremnt

        Returns:
            x_err_pred (ErrorStateGauss): predicted error state
        """
        Ad, GQGTd = self.get_discrete_error_diff(x_nom_prev, z_corr) #-> Tuple['ndarray[15, 15]','ndarray[15, 15]']:

        x_err_pred_mean = Ad@x_err_prev_gauss.mean
        x_err_pred_cov = Ad@x_err_prev_gauss.cov@Ad.T + GQGTd

        x_err_pred = ErrorStateGauss(x_err_pred_mean, x_err_pred_cov,z_corr.ts)


        # # TODO replace this with your own code
        # x_err_pred = solution.eskf.ESKF.predict_x_err(
        #     self, x_nom_prev, x_err_prev_gauss, z_corr)

        return x_err_pred

    def predict_from_imu(self,
                         x_nom_prev: NominalState,
                         x_err_gauss: ErrorStateGauss,
                         z_imu: ImuMeasurement,
                         ) -> Tuple[NominalState, ErrorStateGauss]:
        """Method called every time an IMU measurement is received

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_gauss (ErrorStateGauss): previous error state gaussian
            z_imu (ImuMeasurement): raw IMU measurement

        Returns:
            x_nom_pred (NominalState): predicted nominal state
            x_err_pred (ErrorStateGauss): predicted error state
        """

        # Nominalstate          
        z_corr = self.correct_z_imu(x_nom_prev, z_imu)
        x_nom_pred = self.predict_nominal(x_nom_prev, z_corr)

        # Errorstate
        x_err_pred = self.predict_x_err(x_nom_prev, x_err_gauss, z_corr)


        # TODO replace this with your own code
        # x_nom_pred, x_err_pred = solution.eskf.ESKF.predict_from_imu(
        #     self, x_nom_prev, x_err_gauss, z_imu)

        return x_nom_pred, x_err_pred

    def get_gnss_measurment_jac(self, x_nom: NominalState) -> 'ndarray[3,15]':
        """Get the measurement jacobian, H.

        Hint: the gnss antenna has a relative position to the center given by
        self.lever_arm. How will the gnss measurement change if the drone is 
        rotated differently? Use get_cross_matrix and some other stuff :) 

        Returns:
            H (ndarray[3, 15]): [description]
        """
        arm = self.lever_arm
        as_euler = x_nom.ori.as_euler()
        rotmat =  x_nom.ori.as_rotmat()

        H_guess = np.zeros((3,15))
        H_guess[0:3,0:3] = np.eye(3)
        H_guess[0:3,6:9] = -rotmat@get_cross_matrix(arm)


        # TODO replace this with your own code
        # H = solution.eskf.ESKF.get_gnss_measurment_jac(self, x_nom)
        # print("H_guess - H", H_guess[0:3,6:9] - H[0:3,6:9])

        return H_guess

    def get_gnss_cov(self, z_gnss: GnssMeasurement) -> 'ndarray[3,3]':
        """Use this function in predict_gnss_measurement to get R. 
        Get gnss covariance estimate based on gnss estimated accuracy. 

        All the test data has self.use_gnss_accuracy=False, so this does not 
        affect the tests.

        There is no given solution to this function, feel free to play around!

        Returns:
            gnss_cov (ndarray[3,3]): the estimated gnss covariance
        """
        if self.use_gnss_accuracy and z_gnss.accuracy is not None:
            # play around with this part, the suggested way is not optimal
            gnss_cov = (z_gnss.accuracy/3)**2 * self.gnss_cov

        else:
            # dont change this part
            gnss_cov = self.gnss_cov
        return gnss_cov

    def predict_gnss_measurement(self,
                                 x_nom: NominalState,
                                 x_err: ErrorStateGauss,
                                 z_gnss: GnssMeasurement,
                                 ) -> MultiVarGaussStamped:
        """Predict the gnss measurement

        Hint: z_gnss is only used in get_gnss_cov and to get timestamp for 
        the predicted measurement

        Args:
            x_nom (NominalState): previous nominal state
            x_err (ErrorStateGauss): previous error state gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            z_gnss_pred_gauss (MultiVarGaussStamped): gnss prediction gaussian
        """
        R = self.get_gnss_cov(z_gnss) # 'ndarray[3,3]':

        H = self.get_gnss_measurment_jac(x_nom) # 'ndarray[3,15]':


        # True state:


        z_gnss_pred_gauss_cov = H@x_err.cov@H.T + R

        # nom_state_as_euler = np.zeros((15,))
        # nom_state_as_euler[0:3] = x_nom.pos
        # nom_state_as_euler[3:6] = x_nom.vel
        # nom_state_as_euler[6:9] = x_nom.ori.as_avec()
        # nom_state_as_euler[9:12] = x_nom.accm_bias
        # nom_state_as_euler[12:15] = x_nom.gyro_bias

        # z_gnss_pred_gauss_mean = H@x_err.mean + x_nom.pos - self.lever_arm
        # z_gnss_pred_gauss_mean = x_nom.pos + x_err.pos - self.lever_arm

        z_gnss_pred_gauss_mean = H @ (x_err.mean) +x_nom.pos + x_nom.ori.as_rotmat()@self.lever_arm #H@x_err.mean + x_nom.pos + self.lever_arm



        # predicted_x_err_gauss = self.predict_x_err(x_nom,x_err,z_corr) # ErrorStateGauss:

        z_gnss_pred_gauss_guess = MultiVarGaussStamped(z_gnss_pred_gauss_mean,z_gnss_pred_gauss_cov,z_gnss.ts)


        # TODO replace this with your own code
        # z_gnss_pred_gauss = solution.eskf.ESKF.predict_gnss_measurement(
        #     self, x_nom, x_err, z_gnss)

        # print("z_gnss_pred_gauss_guess - z_gnss_pred_gauss MEAN:",z_gnss_pred_gauss_mean - z_gnss_pred_gauss.mean)
        # print("z_gnss_pred_gauss_guess - z_gnss_pred_gauss COV:", z_gnss_pred_gauss_cov - z_gnss_pred_gauss.cov)

        return z_gnss_pred_gauss_guess

    def get_x_err_upd(self,
                      x_nom: NominalState,
                      x_err: ErrorStateGauss,
                      z_gnss_pred_gauss: MultiVarGaussStamped,
                      z_gnss: GnssMeasurement
                      ) -> ErrorStateGauss:
        """Update the error state from a gnss measurement

        Hint: see (10.75)
        Due to numerical error its recomended use the robust calculation of 
        posterior covariance.

        I_WH = np.eye(*P.shape) - W @ H
        P_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)

        Args:
            x_nom (NominalState): previous nominal state
            x_err (ErrorStateGauss): previous error state gaussian
            z_gnss_pred_gauss (MultiVarGaussStamped): gnss prediction gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            x_err_upd_gauss (ErrorStateGauss): updated error state gaussian
        """
        H = self.get_gnss_measurment_jac(x_nom) # 'ndarray[3,15]':
        P = x_err.cov
        R = self.get_gnss_cov(z_gnss)
        W = P@H.T@np.linalg.inv(H@P@H.T + R)

        x_err_mean = W@(z_gnss.pos - z_gnss_pred_gauss.mean)

        I_WH = np.eye(*P.shape) - W @ H
        P_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)

        x_err_upd_gauss = ErrorStateGauss(x_err_mean,P_upd,z_gnss.ts)

        # TODO replace this with your own code
        # x_err_upd_gauss = solution.eskf.ESKF.get_x_err_upd(
        #     self, x_nom, x_err, z_gnss_pred_gauss, z_gnss)

        return x_err_upd_gauss

    def inject(self,
               x_nom_prev: NominalState,
               x_err_upd: ErrorStateGauss
               ) -> Tuple[NominalState, ErrorStateGauss]:
        """Perform the injection step

        Hint: see (10.85) and (10.72) on how to inject into nominal state.
        See (10.86) on how to find error state after injection

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_upd (ErrorStateGauss): updated error state gaussian

        Returns:
            x_nom_inj (NominalState): nominal state after injection
            x_err_inj (ErrorStateGauss): error state gaussian after injection
        """

        # x_true = np.zeros((15,))
        # x_true[0:3] = x_nom_prev.pos + x_err_upd.pos
        # x_true[3:6] = 
        # x_true[6:9] = 
        # x_true[9:12] =
        # x_true[12:15] =  


        # x_nom_inj = 
        # x_err_inj = 

        # TODO replace this with your own code
        x_nom_inj, x_err_inj = solution.eskf.ESKF.inject(
            self, x_nom_prev, x_err_upd)

        return x_nom_inj, x_err_inj

    def update_from_gnss(self,
                         x_nom_prev: NominalState,
                         x_err_prev: NominalState,
                         z_gnss: GnssMeasurement,
                         ) -> Tuple[NominalState,
                                    ErrorStateGauss,
                                    MultiVarGaussStamped]:
        """Method called every time an gnss measurement is received.


        Args:
            x_nom_prev (NominalState): [description]
            x_nom_prev (NominalState): [description]
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            x_nom_inj (NominalState): previous nominal state 
            x_err_inj (ErrorStateGauss): previous error state
            z_gnss_pred_gauss (MultiVarGaussStamped): predicted gnss 
                measurement, used for NIS calculations.
        """

        # TODO replace this with your own code
        x_nom_inj, x_err_inj, z_gnss_pred_gauss = solution.eskf.ESKF.update_from_gnss(
            self, x_nom_prev, x_err_prev, z_gnss)

        return x_nom_inj, x_err_inj, z_gnss_pred_gauss
