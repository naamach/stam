from scipy.interpolate import Rbf


def rbfi_tracks(tracks, color="bp_rp", mag="mg", param="mass", **kwargs):
    """
    rbfi_tracks(tracks, color="bp_rp", mag="mg", param="mass")

    Compute the radial basis function interpolator instance of the stellar tracks.

    Parameters
    ----------
    tracks : Table
        Stellar-track grid, as retrieved by `stam.gentracks.get_isomasses` or `stam.gentracks.get_combined_isomasses`.
    color : array_like
        Color of the stars (usually Gaia's Gbp-Grp).
    mag : array_like
        Absolute magnitude of the stars (usually Gaia's M_G; same size as `color`).
    param : str, optional
        The parameter to evaluate (options: "mass", "age", "mh"; default: "mass").

    Returns
    -------
    rbfi : Rbf interpolator instance
        Radial basis function interpolator instance based on the stellar tracks.
    """

    rbfi = Rbf(tracks[color], tracks[mag], tracks[param], **kwargs)

    return rbfi


def interpolate_tracks(rbfi, color, mag):
    """
    interpolate_tracks(tri, color, mag, tracks, param="mass")

    Interpolate color-magnitude location over the stellar track grid.

    Parameters
    ----------
    rbfi : Rbf interpolator instance
        Radial basis function interpolator instance, as retrieved by `stam.rbf.rbfi_tracks`.
    color : array_like
        Color of the stars (usually Gaia's Gbp-Grp).
    mag : array_like
        Absolute magnitude of the stars (usually Gaia's M_G; same size as `color`).

    Returns
    -------
    x : array_like
        The evaluated parameter.
    """

    x = rbfi(color, mag)

    return x
