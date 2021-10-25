from typing import Sequence
import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from pathlib import Path

from datatypes.eskf_states import NominalState
import config
from scipy.stats import chi2

plot_folder = Path(__file__).parents[1].joinpath('plots')
plot_folder.mkdir(exist_ok=True)

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["legend.framealpha"] = 1


def plot_state(x_nom_seq: Sequence[NominalState]):
    fig, ax = plt.subplots(5, sharex=True, figsize=(6.4, 7))
    fig.canvas.manager.set_window_title("States")
    times = [x.ts for x in x_nom_seq]

    ax[0].plot(times, [x.pos for x in x_nom_seq],
               label=[f"${s}$" for s in "xyz"])
    ax[0].set_ylabel(r"$\mathbf{\rho}$ [$m$]")

    ax[1].plot(times, [x.vel for x in x_nom_seq],
               label=[f"${s}$" for s in "uvw"])
    ax[1].set_ylabel(r"$\mathbf{v}$ [$m/s$]")

    ax[2].plot(times, [np.rad2deg(x.ori.as_euler()) for x in x_nom_seq],
               label=[f"${s}$" for s in [r"\phi", r"\theta", r"\psi"]])
    ax[2].set_ylabel(r"$\mathbf{q}$ (as euler) [deg]")

    ax[3].plot(times, [x.accm_bias for x in x_nom_seq],
               label=[f"${s}$" for s in "xyz"])
    ax[3].set_ylabel(r"$\mathbf{a}_b$ [$m/s^2$]")

    ax[4].plot(times, [np.rad2deg(x.gyro_bias) for x in x_nom_seq],
               label=[f"${s}$" for s in [r"\phi", r"\theta", r"\psi"]])
    ax[4].set_ylabel(r"$\mathbf{\omega}_b$ [deg$/s$]")

    ax[-1].set_xlabel("$t$ [$s$]")

    for i in range(len(ax)):
        ax[i].legend(loc="upper right")

    fig.align_ylabels(ax)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.08, top=0.97,
                        hspace=0.1)
    fig.savefig(plot_folder.joinpath("States.pdf"))


def plot_errors(times: Sequence[float], errors: Sequence['ndarray[15]']):
    fig, ax = plt.subplots(5, sharex=True, figsize=(6.4, 7))
    fig.canvas.manager.set_window_title("Errors")

    ax[0].plot(times, errors[:, :3],
               label=[f"${s}$" for s in "xyz"])
    ax[0].set_ylabel(r"$\mathbf{\delta \rho}$ [$m$]")

    ax[1].plot(times, errors[:, 3:6],
               label=[f"${s}$" for s in "uvw"])
    ax[1].set_ylabel(r"$\mathbf{\delta v}$ [$m/s$]")

    ax[2].plot(times, np.rad2deg(errors[:, 6:9]),
               label=[f"${s}$" for s in [r"\phi", r"\theta", r"\psi"]])
    ax[2].set_ylabel(r"$\mathbf{\delta \Theta}$ [deg]")

    ax[3].plot(times, errors[:, 9:12],
               label=[f"${s}$" for s in "xyz"])
    ax[3].set_ylabel(r"$\mathbf{\delta a}_b$ [$m/s^2$]")

    ax[4].plot(times, np.rad2deg(errors[:, 12:15]),
               label=[f"${s}$" for s in [r"\phi", r"\theta", r"\psi"]])
    ax[4].set_ylabel(r"$\mathbf{ \delta\omega}_b$ [deg$/s$]")

    ax[-1].set_xlabel("$t$ [$s$]")

    for i in range(len(ax)):
        ax[i].legend(loc="upper right")

    fig.align_ylabels(ax)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.08, top=0.97,
                        hspace=0.1)
    fig.savefig(plot_folder.joinpath("Errors.pdf"))


def plot_position_path_3d(x_nom, x_true=None):

    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title("Position 3D")
    if x_true:
        ax.plot(*np.array([x.pos * np.array([1, 1, -1]) for x in x_true]).T,
                c='C1', label=r"$\mathbf{\rho}_t$")
    ax.plot(*np.array([x.pos * np.array([1, 1, -1]) for x in x_nom]).T,
            c='C0', label=r"$\mathbf{\rho}$")
    ax.legend(loc="upper right")
    ax.set_xlabel("north ($x$) [$m$]")
    ax.set_ylabel("east ($y$) [$m$]")
    ax.set_zlabel("up ($-z$) [$m$]")
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95)
    fig.savefig(plot_folder.joinpath("Position3D.pdf"))


def plot_nis(times, NIS_xyz, NIS_xy, NIS_z, confidence=0.90):
    confidence_intervals = [np.array(chi2.interval(confidence, ndof))
                            for ndof in range(1, 4)]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6.4, 5.2))
    fig.canvas.manager.set_window_title("NIS")

    for i, (NIS, name, nstates) in enumerate(zip([NIS_xyz, NIS_xy, NIS_z],
                                                 ['xyz', 'xy', 'z'],
                                                 [3, 2, 1])):
        ci_lower, ci_upper = confidence_intervals[nstates-1]
        n_total = len(NIS)
        n_below = len([None for value in NIS if value < ci_lower])
        n_above = len([None for value in NIS if value > ci_upper])
        frac_inside = (n_total - n_below - n_above)/n_total
        frac_below = n_below/n_total
        frac_above = n_above/n_total

        ax[i].plot(times, NIS, label=fr"$NIS_{{{name}}}$")
        ax[i].hlines([ci_lower, ci_upper], min(times), max(times), 'C3', ":",
                     label=f"{confidence:2.1%} conf")
        ax[i].set_title(
            f"NIS ${{{name}}}$ "
            f"({frac_inside:2.1%} inside, {frac_below:2.1%} below, "
            f"{frac_above:2.1%} above "
            f" [{confidence:2.1%} conf])")

        ax[i].set_yscale('log')

    ax[-1].set_xlabel('$t$ [$s$]')
    fig.align_ylabels(ax)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.1, top=0.93,
                        hspace=0.3)
    fig.savefig(plot_folder.joinpath('NIS.pdf'))


def plot_nees(times, pos, vel, avec, accm, gyro, confidence=0.90):
    ci_lower, ci_upper = np.array(chi2.interval(confidence, 4))
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(6.4, 9))
    fig.canvas.manager.set_window_title("NEES")

    enu = enumerate(zip(
        [pos, vel, avec, accm, gyro],
        [r"\mathbf{\rho}", r"\mathbf{v}", r"\mathbf{\Theta}",
         r"\mathbf{a}_b", r"\mathbf{\omega}_b"]))
    for i, (NEES, name) in enu:
        n_total = len(NEES)
        n_below = len([None for value in NEES if value < ci_lower])
        n_above = len([None for value in NEES if value > ci_upper])
        frac_inside = (n_total - n_below - n_above)/n_total
        frac_below = n_below/n_total
        frac_above = n_above/n_total

        ax[i].plot(times, NEES, label=fr"$NEES_{{{name}}}$")
        ax[i].hlines([ci_lower, ci_upper], min(times), max(times), 'C3', ":",
                     label=f"{confidence:2.1%} conf")
        ax[i].set_title(
            fr"NEES ${{{name}}}$ "
            fr"({frac_inside:2.1%} inside, "
            f" {frac_below:2.1%} below, {frac_above:2.1%} above "
            f"[{confidence:2.1%} conf])"
        )

        ax[i].set_yscale('log')

    ax[-1].set_xlabel('$t$ [$s$]')
    fig.align_ylabels(ax)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.06, top=0.94,
                        hspace=0.3)
    fig.savefig(plot_folder.joinpath('NEES.pdf'))
