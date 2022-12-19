import numpy as np
from scipy.special import erf
from astropy.io import fits
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from .gentracks import get_isotrack, get_isochrone_polygon, get_isochrone_side
from .plot import plot_cmd, plot_track


def calc_bp_rp_uncertainty(gaia):
    """
    calc_bp_rp_uncertainty(gaia)

    Calculate the Gaia Gbp-Grp color uncertainty.

    Parameters
    ----------
    gaia : FITS table
        Gaia table.

    Returns
    -------
    bp_rp_error : array_like
        Gaia Gbp-Grp uncertainty.

    """
    bp_rp_error = 2.5 * np.log10(np.e) * np.sqrt(
        1 / gaia['phot_bp_mean_flux_over_error'] ** 2 + 1 / gaia['phot_rp_mean_flux_over_error'] ** 2)

    return bp_rp_error


def calc_mg_uncertainty(gaia):
    """
    calc_mg_uncertainty(gaia)

    Calculate the Gaia G-band absolute magnitude uncertainty.

    Parameters
    ----------
    gaia : FITS table
        Gaia table.

    Returns
    -------
    mg_error : array_like
        Gaia G-band absolute magnitude uncertainty.
    """

    mg_error = 2.5 * np.log10(np.e) * np.sqrt(
        1 / gaia['phot_g_mean_flux_over_error'] ** 2 + 4 / gaia['parallax_over_error'] ** 2)

    return mg_error


def calc_extinction(b, parallax, z_sun=1.4, sigma_dust=150, R_G=2.740, R_BP=3.374, R_RP=2.035):
    """
    calc_extinction(b, parallax, z_sun=1.4, sigma_dust=150, R_G=2.740, R_BP=3.374, R_RP=2.035)

    Calculate the extinction and reddening, based on Galactic latitude and distance, using eq. 2 of Sollima (2019):
    https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.2377S/abstract
    R coefficients from Casagrande & VandenBerg (2018), table 2:
    https://ui.adsabs.harvard.edu/abs/2018MNRAS.479L.102C/abstract

    Parameters
    ----------
    b : array_like
        Galactic latitude, in deg.
    parallax : array_like
        Parallax, in mas.
    z_sun : float, optional
        Sun's height above the Galactic plane, in pc (default: 1.4 pc).
    sigma_dust : float, optional
        The Gaussian distribution sigma of the Galactic dust, in pc (default: 150 pc).
    R_G : float, optional
        Gaia G-band extinction coefficient (default: 2.740).
    R_BP : float, optional
        Gaia Gbp-band extinction coefficient (default: 3.374).
    R_RP : float, optional
        Gaia Grp-band extinction coefficient (default: 2.035).

    Returns
    -------
    e_bprp : array_like
        Reddening in the Gaia bands, E(Gbp-Grp).
    A_G : array_like
        Gaia G-band extinction, A_G = R_G * E(Gbp-Grp).

    """

    sin_b = np.sin(np.deg2rad(b))
    e_bv = 0.03 / sin_b * (
            erf((sin_b * 1000 / parallax + z_sun) / np.sqrt(2) / sigma_dust) - erf(z_sun / np.sqrt(2) / sigma_dust))
    A_G = e_bv * R_G
    e_bprp = (R_BP - R_RP) * e_bv

    return e_bprp, A_G


def calc_gaia_extinction(gaia, **kwargs):
    """
    calc_gaia_extinction(gaia, **kwargs)

    Calculate the extinction and reddening for all the sources in the Gaia table.

    Parameters
    ----------
    gaia : FITS table
        Gaia table.
    kwargs : optional
        Any additional keyword arguments to be passed to `stam.gaia.calc_extinction`.

    Returns
    -------
    e_bprp : array_like
        Reddening in the Gaia bands, E(Gbp-Grp).
    A_G : array_like
        Gaia G-band extinction, A_G = R_G * E(Gbp-Grp).
    """

    e_bprp, A_G = calc_extinction(gaia['b'], gaia['parallax'], **kwargs)

    return e_bprp, A_G


def read_gaia_data(file):
    """
    read_gaia_data(file)

    Read Gaia table.

    Parameters
    ----------
    file : str
        Gaia file name (including path). Should be a `*.FITS` file.

    Returns
    -------
    gaia : FITS table
        Gaia table.
    """

    hdul = fits.open(file)
    gaia = hdul[1].data

    return gaia


def get_gaia_subsample(gaia, sample_settings):
    """
    get_gaia_subsample(gaia, sample_settings)

    Get a Gaia subsample based on tangential velocity and distance.

    Parameters
    ----------
    gaia : FITS table
        Gaia table.
    sample_settings : dict
        The subsample settings: a dictionary including keywords "vmin", "vmax", and "dist";
        specifying the transverse velocity limits (in km/s) and the maximal distance (in pc).

    Returns
    -------
    gaia_idx : array_like
        Row indices of Gaia sources included in the subsample
    """

    # tangential velocity
    gaia_idx = (sample_settings["vmin"] <= gaia["v_tan"]) & (gaia["v_tan"] < sample_settings["vmax"])

    # maximum distance
    if sample_settings["dist"] is not None:
        within_dist = 1000 / gaia["parallax"] <= sample_settings["dist"]
        gaia_idx = gaia_idx & within_dist

    gaia_idx = np.where(gaia_idx)[0]

    return gaia_idx


def get_gaia_isochrone_subsample(bp_rp, mg, models, age1, mh1, age2, mh2,
                                 age_res=0.001, mh_res=0.05, mass_res=0.007, mass_max=1.2,
                                 stage1=1, stage2=1, bp_rp_min=-np.inf, bp_rp_max=np.inf,
                                 bp_rp_shift1=0, bp_rp_shift2=0, mg_shift1=0, mg_shift2=0,
                                 is_plot=False, title=None, ax=None):
    """
    get_gaia_isochrone_subsample(bp_rp, mg, models, age1, mh1, age2, mh2,
                                 age_res=0.001, mh_res=0.05, mass_res=0.007, mass_max=1.2,
                                 stage1=1, stage2=1, bp_rp_min=-np.inf, bp_rp_max=np.inf,
                                 bp_rp_shift1=0, bp_rp_shift2=0, mg_shift1=0, mg_shift2=0,
                                 is_plot=False, title=None, ax=None)

    Get the polygon enclosed by two isochrones.

    Parameters
    ----------
    bp_rp : array_like
        Gaia Gbp-Grp color.
    mg : array_like
        Gaia G-band absolute magnitude (same size as `bp_rp`).
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    age1 : float, optional
        Stellar track age of the first track, in Gyr.
    mh1 : float
        Stellar track metallicity ([M/H]) of the first track, in dex.
    age2 : float, optional
        Stellar track age of the second track, in Gyr.
    mh2 : float
        Stellar track metallicity ([M/H]) of the second track, in dex.
    age_res : float, optional
        Age resolution, in Gyr (default: 0.001 Gyr).
    mh_res : float, optional
        Metallicity resolution, in dex (default: 0.05 dex).
    mass_res : float, optional
        Mass resolution, in Msun (default: 0.007 Msun).
    mass_max : float, optional
        Maximum mass to consider, in Msun (if no fixed mass was chosen; default: 1.2 Msun).
    stage1 : int, optional
        Stellar evolution stage label of the first track (0 = pre-MS, 1 = MS, etc.; default: 1).
    stage2 : int, optional
        Stellar evolution stage label of the second track (0 = pre-MS, 1 = MS, etc.; default: 1).
    bp_rp_min : float, optional
        Minimal Gaia Gbp-Grp color to consider (default: -np.inf).
    bp_rp_max : float, optional
        Maximal Gaia Gbp-Grp color to consider (default: np.inf).
    bp_rp_shift1 : float, optional
        How much to shift the Gaia Gbp-Grp color of the first track (default: 0).
    bp_rp_shift2 : float, optional
        How much to shift the Gaia Gbp-Grp color of the second track (default: 0).
    mg_shift1 : float, optional
        How much to shift the Gaia G-band of the first track (default: 0).
    mg_shift2 : float, optional
        How much to shift the Gaia G-band of the second track (default: 0).
    is_plot: bool, optional
        Plot Gaia CMD with highlighted region? (default: False).
    title: str, optional
        Plot title (default: None).
    ax : Axes object, optional
        Matplotlib Axes in which to plot (default: None).

    Returns
    -------
    idx : array_like
        Row indices of Gaia sources inside the subsample polygon.

    """
    polygon, BP_RP1, G1, mass1, BP_RP2, G2, mass2 = get_isochrone_polygon(models, age1, mh1, age2, mh2,
                                                                          age_res=age_res, mh_res=mh_res,
                                                                          mass_max=mass_max, mass_res=mass_res,
                                                                          stage1=stage1, stage2=stage2,
                                                                          bp_rp_min=bp_rp_min, bp_rp_max=bp_rp_max,
                                                                          bp_rp_shift1=bp_rp_shift1,
                                                                          bp_rp_shift2=bp_rp_shift2,
                                                                          mg_shift1=mg_shift1, mg_shift2=mg_shift2)

    idx = polygon.contains_points(np.array([bp_rp, mg]).T)
    print(
        f"{len(np.nonzero(idx)[0])} Gaia sources left out of {len(bp_rp)} ({len(np.nonzero(idx)[0]) / len(bp_rp) * 100:.1}%)")

    if is_plot:
        if ax is None:
            ax = plot_cmd(bp_rp, mg)
        else:
            plot_cmd(bp_rp, mg, ax=ax)

        patch = patches.PathPatch(polygon, facecolor='tab:red', linewidth=None, alpha=0.1)
        ax.add_patch(patch)

        plot_track(BP_RP1, G1, title=None, ax=ax, c="tab:red", label=f"[M/H]={mh1}")
        plot_track(BP_RP2, G2, title=None, ax=ax, c="tab:red", label=f"[M/H]={mh2}", linestyle="--")
        if title is not None:
            ax.set_title(title)
        ax.legend()
        plt.show()

    return idx


def get_gaia_isochrone_side_subsample(bp_rp, mg, models, age, mh, side="blue",
                                      age_res=0.001, mh_res=0.05, mass_res=0.007, mass_max=1.2,
                                      stage=1, stage_min=0, stage_max=np.inf,
                                      bp_rp_min=-np.inf, bp_rp_max=np.inf, bp_rp_shift=0, mg_shift=0,
                                      is_plot=False, title=None, ax=None):
    """
    get_gaia_isochrone_side_subsample(bp_rp, mg, models, age, mh, side="blue",
                                      age_res=0.001, mh_res=0.05, mass_res=0.007, mass_max=1.2,
                                      stage=1, stage_min=0, stage_max=np.inf,
                                      bp_rp_min=-np.inf, bp_rp_max=np.inf, bp_rp_shift=0, mg_shift=0,
                                      is_plot=False, title=None, ax=None)

    Get the subsample of one side of an evolutionary track.

    Parameters
    ----------
    bp_rp : array_like
        Gaia Gbp-Grp color.
    mg : array_like
        Gaia G-band absolute magnitude (same size as `bp_rp`).
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    age : float, optional
        Stellar track age of the track, in Gyr.
    mh : float
        Stellar track metallicity ([M/H]) of the track, in dex.
    side : str, optional
        Which side ("blue"/"red") of the track to include (default: "blue").
    age_res : float, optional
        Age resolution, in Gyr (default: 0.001 Gyr).
    mh_res : float, optional
        Metallicity resolution, in dex (default: 0.05 dex).
    mass_res : float, optional
        Mass resolution, in Msun (default: 0.007 Msun).
    mass_max : float, optional
        Maximum mass to consider, in Msun (if no fixed mass was chosen; default: 1.2 Msun).
    stage : int, optional
        Stellar evolution stage label of the track (0 = pre-MS, 1 = MS, etc.; default: 1).
    stage_min : int, optional
        Minimum stellar evolution stage label to consider (if no fixed stage was chosen; default: 0).
    stage_max : int, optional
        Maximum stellar evolution stage label to consider (if no fixed stage was chosen; default: `np.inf`).
    bp_rp_min : float, optional
        Minimal Gaia Gbp-Grp color to consider (default: -np.inf).
    bp_rp_max : float, optional
        Maximal Gaia Gbp-Grp color to consider (default: np.inf).
    bp_rp_shift : float, optional
        How much to shift the Gaia Gbp-Grp color of the track (default: 0).
    mg_shift : float, optional
        How much to shift the Gaia G-band of the track (default: 0).
    is_plot: bool, optional
        Plot Gaia CMD with highlighted region? (default: False).
    title: str, optional
        Plot title (default: None).
    ax : Axes object, optional
        Matplotlib Axes in which to plot (default: None).

    Returns
    -------
    idx : array_like
        Row indices of Gaia sources inside the subsample polygon.

    """
    polygon, BP_RP, G, mass = get_isochrone_side(models, age, mh, side=side, age_res=age_res, mh_res=mh_res,
                                                 mass_max=mass_max, mass_res=mass_res,
                                                 stage=stage, stage_min=stage_min, stage_max=stage_max,
                                                 bp_rp_min=bp_rp_min, bp_rp_max=bp_rp_max, bp_rp_shift=bp_rp_shift,
                                                 mg_shift=mg_shift)

    idx = polygon.contains_points(np.array([bp_rp, mg]).T)
    print(
        f"{len(np.nonzero(idx)[0])} Gaia sources left out of {len(bp_rp)} ({len(np.nonzero(idx)[0]) / len(bp_rp) * 100:.1}%)")

    if is_plot:
        if ax is None:
            ax = plot_cmd(bp_rp, mg)
        else:
            plot_cmd(bp_rp, mg, ax=ax)

        patch = patches.PathPatch(polygon, facecolor='tab:red', linewidth=None, alpha=0.1)
        ax.add_patch(patch)

        plot_track(BP_RP, G, title=None, ax=ax, c="tab:red", label=f"[M/H]={mh}")
        if title is not None:
            ax.set_title(title)
        ax.legend()
        plt.show()

    return idx
