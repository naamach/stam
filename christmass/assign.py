import time
from tqdm import tqdm
import numpy as np
from christmass.interp import triangulate_tracks, interpolate_tracks


def assign_param(color, color_error, mag, mag_error, tracks, n_realizations=10, param="mass"):

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
