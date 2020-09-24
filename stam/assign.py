import time
from tqdm import tqdm
import numpy as np
from stam.interp import triangulate_tracks, interpolate_tracks


def assign_param(color, color_error, mag, mag_error, tracks, n_realizations=10, param="mass"):
    """
    assign_param(color, color_error, mag, mag_error, tracks, n_realizations=10, param="mass")

    Assign a specific parameter (mass or metallicity) to a star, based on its color and magnitude.

    Parameters
    ----------
    color : array_like
        Color of the stars (usually Gaia's Gbp-Grp).
    color_error : array_like
        `color` uncertainty (same size as `color`).
    mag : array_like
        Absolute magnitude of the stars (usually Gaia's M_G; same size as `color`).
    mag_error : array_like
        `mag` uncertainty (same size as `color`).
    tracks : Table
        Stellar-track grid, as retrieved by `stam.tracks.get_isomasses` or `stam.tracks.get_combined_isomasses`.
    n_realizations : int, optional
        Number of realizations to draw for each star, from a 2D-Gaussian distribution around the color-magnitude
        position of the star (default: 10).
    param : str, optional
        Which parameter to estimate (options: "mass", "mh"; default: "mass").

    Returns
    -------
    param_mean : array_like
        Estimated parameter mean for each of stars (same size as `color`).
    param_error : array_like
        Estimated parameter standard deviation for each of stars (same size as `color`).
    """

    tri = triangulate_tracks(tracks)

    param_mean = np.zeros(len(color))
    param_error = np.zeros(len(color))

    t = time.time()
    for i in tqdm(range(len(color))):  # for each gaia source
        mean = [color[i], mag[i]]
        cov = [[color_error[i], 0], [0, mag_error[i]]]

        x = np.random.multivariate_normal(mean, cov, size=n_realizations)

        curr_param, nan_idx = interpolate_tracks(tri, x[:, 0], x[:, 1], tracks, param=param)

        param_mean[i] = np.nanmean(curr_param)
        param_error[i] = np.nanstd(curr_param)

    print(f"Calculating mean took {time.time() - t:.1} sec.")

    return param_mean, param_error
