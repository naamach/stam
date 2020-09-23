import numpy as np
import scipy.spatial.qhull as qhull

# based on
# https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids


def triangulate(xy):
    tri = qhull.Delaunay(xy)
    return tri


def interpolate(values, tri, uv, fill_value=np.nan, d=2):
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    ret = np.einsum('nj,nj->n', np.take(values, vertices), weights)
    ret[np.any(weights < 0, axis=1)] = fill_value

    return ret


def triangulate_tracks(tracks, color="bp_rp", mag="mg"):
    track_points = np.array([tracks[color], tracks[mag]]).T
    tri = triangulate(track_points)
    return tri


def interpolate_tracks(tri, color, mag, tracks, param="mass"):
    obs_points = np.array([color, mag]).T
    x = interpolate(tracks[param], tri, obs_points)
    nan_idx = np.isnan(x)

    return x, nan_idx
