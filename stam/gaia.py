import numpy as np
from scipy.special import erf
from astropy.io import fits


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
    bp_rp_error = 2.5*np.log10(np.e)*np.sqrt(1/gaia['phot_bp_mean_flux_over_error']**2 + 1/gaia['phot_rp_mean_flux_over_error']**2)

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

    mg_error = 2.5*np.log10(np.e)*np.sqrt(1/gaia['phot_g_mean_flux_over_error']**2 + 4/gaia['parallax_over_error']**2)

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
