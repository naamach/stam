import numpy as np
from scipy.special import erf
from astropy.io import fits


def calc_bp_rp_uncertainty(gaia):
    bp_rp_error = 2.5*np.log10(np.e)*np.sqrt(1/gaia['phot_bp_mean_flux_over_error']**2 + 1/gaia['phot_rp_mean_flux_over_error']**2)

    return bp_rp_error


def calc_mg_uncertainty(gaia):
    mg_error = 2.5*np.log10(np.e)*np.sqrt(1/gaia['phot_g_mean_flux_over_error']**2 + 4/gaia['parallax_over_error']**2)

    return mg_error


def calc_extinction(b, parallax, z_sun=1.4, sigma_dust=150, R_G=2.740, R_BP=3.374, R_RP=2.035):
    # Calculate extinction and reddening using Sollima (2019) eq. 2:
    # https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.2377S/abstract
    # R coefficients from Casagrande & VandenBerg (2018), table 2:
    # https://ui.adsabs.harvard.edu/abs/2018MNRAS.479L.102C/abstract

    sin_b = np.sin(np.deg2rad(b))
    e_bv = 0.03 / sin_b * (
                erf((sin_b * 1000 / parallax + z_sun) / np.sqrt(2) / sigma_dust) - erf(z_sun / np.sqrt(2) / sigma_dust))
    A_G = e_bv * R_G
    e_bprp = (R_BP - R_RP) * e_bv

    return e_bprp, A_G


def calc_gaia_extinction(gaia, **kwargs):
    e_bprp, A_G = calc_extinction(gaia['b'], gaia['parallax'], **kwargs)

    return e_bprp, A_G


def read_gaia_data(file):
    hdul = fits.open(file)
    gaia = hdul[1].data

    return gaia
