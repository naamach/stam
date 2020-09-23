import os
import time
import numpy as np
from stam.utils import get_config, init_log, close_log
from stam.gaia import read_gaia_data, calc_bp_rp_uncertainty, calc_mg_uncertainty, calc_gaia_extinction
from stam.models import read_parsec
from stam.tracks import get_combined_isomasses
from stam.assign import assign_param


def get_mass_and_metallicity(idx=None, suffix=None, config_file="config.ini"):
    config = get_config(config_file)
    log = init_log(time.strftime("%Y%m%d_%H%M%S", time.gmtime()), config_file)

    is_save = config.getboolean("GENERAL", "SAVE")
    output_type = config.get("GENERAL", "OUTPUT_TYPE").lower()
    path = config.get("GENERAL", "PATH")
    csv_format = "%" + config.get("GENERAL", "CSV_FORMAT")

    if suffix is not None:
        suffix = "_" + suffix
    else:
        suffix = ""

    gaia_file = os.path.join(config.get("GAIA", "PATH"), config.get("GAIA", "FILE"))
    log.info(f"Reading Gaia table from {gaia_file}...")
    gaia = read_gaia_data(gaia_file)
    log.info(f"Gaia table has {len(gaia)} sources.")

    if idx is not None:
        log.info(f"Using {np.count_nonzero(idx)} Gaia sources out of {len(gaia)} ({100*np.count_nonzero(idx)/len(gaia):.1f}%).")
        gaia = gaia[idx]

    bp_rp = gaia["bp_rp"]
    mg = gaia["mg"]
    bp_rp_error = calc_bp_rp_uncertainty(gaia)
    mg_error = calc_mg_uncertainty(gaia)

    if config.getboolean("GAIA", "CORRECT_EXTINCTION"):
        log.info("Applying extinction correction...")
        e_bprp, A_G = calc_gaia_extinction(gaia)
        bp_rp = bp_rp - e_bprp
        mg = mg - A_G

    # get tracks
    if config.get("MODELS", "SOURCE") == "PARSEC":
        log.info("Using PARSEC evolution tracks...")
        models = read_parsec(config_file=config_file)
    else:
        log.error(f"{config.get('MODELS', 'SOURCE')} models not implemented yet!")

    mass_bins = np.arange(config.getfloat("MODELS", "M_MIN"), config.getfloat("MODELS", "M_MAX"), config.getfloat("MODELS", "M_STEP"))
    age = config.getfloat("MODELS", "AGE")
    mh_pre_ms = config.getfloat("MODELS", "MH_PRE_MS")
    is_smooth = config.getboolean("MODELS", "SMOOTH")
    smooth_sigma = config.getint("MODELS", "SMOOTH_SIGMA")
    tracks = get_combined_isomasses(models, mass=mass_bins, age=age, mh_pre_ms=mh_pre_ms, is_smooth=is_smooth,
                                    smooth_sigma=smooth_sigma)

    # assign mass
    log.info("Assigning masses...")
    n_realizations = config.getint("MASS", "N_REALIZATIONS")
    m_mean, m_error = assign_param(bp_rp, bp_rp_error, mg, mg_error, tracks, n_realizations=n_realizations, param="mass")

    if is_save:
        log.info("Saving masses...")
        if output_type == "npy":
            np.save(os.path.join(path, f"Mmean{suffix}.npy"), m_mean, allow_pickle=True)
            np.save(os.path.join(path, f"Mstd{suffix}.npy"), m_error, allow_pickle=True)
        elif output_type == "csv":
            np.savetxt(os.path.join(path, f"Mmean{suffix}.csv"), m_mean, fmt=csv_format, delimiter=",")
            np.savetxt(os.path.join(path, f"Mstd{suffix}.csv"), m_error, fmt=csv_format, delimiter=",")

    # assign mh
    log.info("Assigning [M/H]...")
    n_realizations = config.getint("MH", "N_REALIZATIONS")
    mh_mean, mh_error = assign_param(bp_rp, bp_rp_error, mg, mg_error, tracks, n_realizations=n_realizations, param="mh")

    if is_save:
        log.info("Saving metallicities...")
        if output_type == "npy":
            np.save(os.path.join(path, f"MHmean{suffix}.npy"), mh_mean, allow_pickle=True)
            np.save(os.path.join(path, f"MHstd{suffix}.npy"), mh_error, allow_pickle=True)
        elif output_type == "csv":
            np.savetxt(os.path.join(path, f"MHmean{suffix}.csv"), mh_mean, fmt=csv_format, delimiter=",")
            np.savetxt(os.path.join(path, f"MHstd{suffix}.csv"), mh_error, fmt=csv_format, delimiter=",")

    close_log(log)

    return m_mean, m_error, mh_mean, mh_error
