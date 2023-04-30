import os
import time
import numpy as np
from .utils import get_config, init_log, close_log
from .gaia import read_gaia_data, calc_bp_rp_uncertainty, calc_mg_uncertainty, calc_gaia_extinction, get_gaia_subsample, \
    get_gaia_isochrone_subsample, get_extinction_in_gaia_band
from .getmodels import read_parsec
from .gentracks import get_combined_isomasses, get_isochrone_polygon, get_isotrack
from .assign import assign_param


def get_param(bp_rp, bp_rp_error, mg, mg_error, tracks, param="mass", suffix=None, is_save=True, log=None,
              output_type="csv", output_path="", csv_format="%.8f", n_realizations=10, interp_fun="rbf",
              binary_polygon=None, **kwargs):
    """
    get_param(bp_rp, bp_rp_error, mg, mg_error, tracks, param="mass", suffix=None, is_save=True, log=None,
              output_type="csv", output_path="", csv_format="%.8f", n_realizations=10, interp_fun="rbf",
              binary_polygon=None, **kwargs)

    Write config file.

    Parameters
    ----------
    bp_rp : array_like
        Gaia Gbp-Grp color.
    bp_rp_error : array_like
        Gaia Gbp-Grp color uncertainty (same size as `bp_rp`).
    mg : array_like
        Gaia G-band absolute magnitude (same size as `bp_rp`).
    mg_error : array_like
        Gaia G-band absolute magnitude uncertainty (same size as `bp_rp`).
    tracks : Table
        Stellar-track grid, as retrieved by `stam.gentracks.get_isomasses` or `stam.gentracks.get_combined_isomasses`.
    param : str, optional
        The parameter to evaluate (options: "mass", "age", "mh"; default: "mass").
    suffix : str, optional
        Output file name suffix (default: None).
    is_save : bool, optional
        Save results to file? (default: True).
    log : Logger, optional
        Logger object (default: None).
    output_type : str, optional
        Output type (npy, csv; default: "csv").
    output_path : str, optional
        Result file destination path (default: "").
    csv_format : str, optional
        CSV format (default: "%.8f").
    n_realizations : float, optional
        Number of parameter assignment realizations (default: 10).
    interp_fun : str, optional
        Interpolation method (rbf, griddata, nurbs; default: "rbf").
    binary_polygon : Path object, optional
        The polygon defining the equal-mass binary region on the HR-diagram (default: None).
    **kwargs :
        Additional arguments to pass to `stam.assign.assign_param

    Returns
    -------
    param_mean : array_like
        Estimated parameter mean for each of stars.
    param_error : array_like
        Estimated parameter standard deviation for each of stars.
    """

    is_local_log = False
    if log is None:
        # setup console log
        log = init_log(config_file=None)
        is_local_log = True

    if suffix is None:
        suffix = ""

    if param == "mass":
        prefix = "M"
    elif param == "mh":
        prefix = "MH"
    elif param == "age":
        prefix = "Age"

    if not os.path.exists(output_path):
        log.info(f"Creating output folder {output_path}...")
        os.makedirs(output_path)

    # assign parameter
    log.info(f"Assigning {param}...")
    if binary_polygon is None:
        log.info("Ignoring twin binary sequence...")
        param_mean, param_error = assign_param(bp_rp, bp_rp_error, mg, mg_error, tracks, n_realizations=n_realizations,
                                               param=param, interp_fun=interp_fun, binary_polygon=None, **kwargs)
    else:
        log.info("Taking twin binary sequence into account...")
        param_mean, param_error, binary_param_mean, binary_param_error, weight = \
            assign_param(bp_rp, bp_rp_error, mg, mg_error, tracks, n_realizations=n_realizations,
                         param=param, interp_fun=interp_fun, binary_polygon=binary_polygon, **kwargs)

    if is_save:
        log.info(f"Saving {param}...")
        if output_type == "npy":
            np.save(os.path.join(output_path, f"{prefix}_mean{suffix}.npy"), param_mean, allow_pickle=True)
            np.save(os.path.join(output_path, f"{prefix}_std{suffix}.npy"), param_error, allow_pickle=True)
            if binary_polygon is not None:
                np.save(os.path.join(output_path, f"{prefix}_mean_binary{suffix}.npy"), binary_param_mean,
                        allow_pickle=True)
                np.save(os.path.join(output_path, f"{prefix}_std_binary{suffix}.npy"), binary_param_error,
                        allow_pickle=True)
                np.save(os.path.join(output_path, f"{prefix}_weight_binary{suffix}.npy"), weight,
                        allow_pickle=True)
        elif output_type == "csv":
            np.savetxt(os.path.join(output_path, f"{prefix}_mean{suffix}.csv"), param_mean, fmt=csv_format,
                       delimiter=",")
            np.savetxt(os.path.join(output_path, f"{prefix}_std{suffix}.csv"), param_error, fmt=csv_format,
                       delimiter=",")
            if binary_polygon is not None:
                np.savetxt(os.path.join(output_path, f"{prefix}_mean_binary{suffix}.csv"), binary_param_mean,
                           fmt=csv_format, delimiter=",")
                np.savetxt(os.path.join(output_path, f"{prefix}_std_binary{suffix}.csv"), binary_param_error,
                           fmt=csv_format, delimiter=",")
                np.savetxt(os.path.join(output_path, f"{prefix}_weight_binary{suffix}.csv"), weight, fmt=csv_format,
                           delimiter=",")

    if is_local_log:
        close_log(log)

    if binary_polygon is None:
        return param_mean, param_error
    else:
        # take binary sequence into account
        return param_mean, param_error, binary_param_mean, binary_param_error, weight


def get_mass_and_metallicity(idx=None, suffix=None, config_file="config.ini", sample_settings=None):
    """
    get_mass_and_metallicity(idx=None, suffix=None, config_file="config.ini")

    Estimate mass and metallicity for all the stars in the Gaia table.

    Parameters
    ----------
    idx : None or array_like, optional
        Gaia table row indices to use, `None` to select all rows (default: None).
    suffix : None or str, optional
        Customized suffix to add to the output file names (default: None).
    config_file : str, optional
        The configuration file name, including path (default: "config.ini").
    sample_settings : dict, optional
        A dictionary including keywords "vmin", "vmax", and "dist" (default: None).
        If provided, only Gaia sources within the specific transverse velocities (in km/s) and distance (in pc),
        will be evaluated.

    Returns
    -------
    m_mean : array_like
        Estimated mass mean for each of stars.
    m_error : array_like
        Estimated mass standard deviation for each of stars.
    mh_mean : array_like
        Estimated metallicity mean for each of stars.
    mh_error : array_like
        Estimated metallicity standard deviation for each of stars.
    """

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
        log.info(
            f"Using {np.count_nonzero(idx)} Gaia sources out of {len(gaia)} ({100 * np.count_nonzero(idx) / len(gaia):.1f}%).")
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
        suffix += "_extinction"

    # get tracks
    if config.get("MODELS", "SOURCE") == "PARSEC":
        log.info("Using PARSEC evolutionary tracks...")
        models = read_parsec(config_file=config_file)
    else:
        log.error(f"{config.get('MODELS', 'SOURCE')} models not implemented yet!")

    mass_bins = np.arange(config.getfloat("MODELS", "M_MIN"),
                          config.getfloat("MODELS", "M_MAX") + config.getfloat("MODELS", "M_STEP"),
                          config.getfloat("MODELS", "M_STEP"))
    exclude_pre_ms_masses = config.get("MODELS", "EXCLUDE_PRE_MS_MASSES")
    if exclude_pre_ms_masses == "":
        exclude_pre_ms_masses = []
    else:
        exclude_pre_ms_masses = [float(x) for x in exclude_pre_ms_masses.split(",")]
    age = config.getfloat("MODELS", "AGE")
    mh_pre_ms = config.getfloat("MODELS", "MH_PRE_MS")
    is_smooth = config.getboolean("MODELS", "SMOOTH")
    smooth_sigma = config.getint("MODELS", "SMOOTH_SIGMA")
    tracks = get_combined_isomasses(models, mass=mass_bins, age=age, mh_pre_ms=mh_pre_ms, is_smooth=is_smooth,
                                    smooth_sigma=smooth_sigma, exclude_pre_ms_masses=exclude_pre_ms_masses)

    if sample_settings is not None:
        gaia_idx = get_gaia_subsample(gaia, sample_settings)
        log.info(
            f"Using {np.count_nonzero(gaia_idx)} Gaia sources out of {len(gaia)} ({100 * np.count_nonzero(gaia_idx) / len(gaia):.1f}%).")
        bp_rp = bp_rp[gaia_idx]
        mg = mg[gaia_idx]
        bp_rp_error = bp_rp_error[gaia_idx]
        mg_error = mg_error[gaia_idx]
        ms_idx = get_gaia_isochrone_subsample(bp_rp, mg, models, sample_settings["age1"], sample_settings["mh1"],
                                              sample_settings["age2"], sample_settings["mh2"],
                                              stage1=sample_settings["stage1"], stage2=sample_settings["stage2"])
        bp_rp = bp_rp[ms_idx]
        mg = mg[ms_idx]
        bp_rp_error = bp_rp_error[ms_idx]
        mg_error = mg_error[ms_idx]

    # interpolation parameters
    interp_fun = config.get("INTERP", "METHOD")
    if interp_fun == "rbf":
        kwargs = {"function": config.get("INTERP", "RBF_FUN")}
    else:
        kwargs = {}

    binary_polygon = None
    if config.getboolean("BINARY", "CONSIDER_TWINS"):
        if sample_settings is not None:
            log.info("Taking binary twin sequence into account.")
            binary_polygon = \
                get_isochrone_polygon(models, sample_settings["binary age1"], sample_settings["binary mh1"],
                                      sample_settings["binary age1"], sample_settings["binary mh1"],
                                      stage1=1, stage2=1, bp_rp_max=sample_settings["binary bp_rp max"],
                                      mg_shift1=-2.5 * np.log10(config.getfloat("BINARY", "FLUX_RATIO_MIN")),
                                      mg_shift2=-2.5 * np.log10(config.getfloat("BINARY", "FLUX_RATIO_MAX")))[0]
        else:
            log.warning("No sample settings provided, ignoring binary twin sequence.")

    # assign mass
    n_realizations = config.getint("MASS", "N_REALIZATIONS")
    log.info(f"Assigning masses using {n_realizations} realizations...")
    if binary_polygon is None:
        m_mean, m_error = get_param(bp_rp, bp_rp_error, mg, mg_error, tracks, param="mass", suffix=suffix,
                                    is_save=is_save,
                                    log=log, output_type=output_type, output_path=path, csv_format=csv_format,
                                    n_realizations=n_realizations, interp_fun=interp_fun,
                                    binary_polygon=binary_polygon, **kwargs)
    else:
        # take binary sequence into account
        m_mean, m_error, binary_m_mean, binary_m_error, m_weight = \
            get_param(bp_rp, bp_rp_error, mg, mg_error, tracks, param="mass", suffix=suffix, is_save=is_save,
                      log=log, output_type=output_type, output_path=path, csv_format=csv_format,
                      n_realizations=n_realizations, interp_fun=interp_fun,
                      binary_polygon=binary_polygon, **kwargs)

    # assign mh
    n_realizations = config.getint("MH", "N_REALIZATIONS")
    log.info(f"Assigning [M/H] using {n_realizations} realizations...")
    if binary_polygon is None:
        mh_mean, mh_error = get_param(bp_rp, bp_rp_error, mg, mg_error, tracks, param="mh", suffix=suffix,
                                      is_save=is_save,
                                      log=log, output_type=output_type, output_path=path, csv_format=csv_format,
                                      n_realizations=n_realizations, interp_fun=interp_fun,
                                      binary_polygon=binary_polygon, **kwargs)
    else:
        # take binary sequence into account
        mh_mean, mh_error, binary_mh_mean, binary_mh_error, mh_weight = \
            get_param(bp_rp, bp_rp_error, mg, mg_error, tracks, param="mh", suffix=suffix, is_save=is_save,
                      log=log, output_type=output_type, output_path=path, csv_format=csv_format,
                      n_realizations=n_realizations, interp_fun=interp_fun,
                      binary_polygon=binary_polygon, **kwargs)

    close_log(log)

    if binary_polygon is None:
        return m_mean, m_error, mh_mean, mh_error
    else:
        # take binary sequence into account
        return m_mean, m_error, binary_m_mean, binary_m_error, m_weight, mh_mean, mh_error, binary_mh_mean, binary_mh_error, mh_weight


def get_param_from_isotrack(sources, vals=[5, 0], params=("age", "mh"), suffix=None, is_save=True,
                            output_type="csv", output_path="", csv_format="%.8f", n_realizations=10, interp_fun="rbf",
                            path_models='./PARSEC/', correct_extinction=True, reddening_key="av",
                            color_excess_key="e_bv", use_reddening_key=True, **kwargs):
    """


    Parameters
    ----------


    Returns
    -------
    param_mean : array_like
        Estimated parameter mean for each of stars.
    param_error : array_like
        Estimated parameter standard deviation for each of stars.
    """

    log = init_log(time.strftime("%Y%m%d_%H%M%S", time.gmtime()), config_file=None)

    if ~("mass" in params):
        param = "mass"
    elif ~("age" in params):
        param = "age"
    elif ~("mh" in params):
        param = "mh"
    else:
        log.error(f"Nothing to assign!")

    # get tracks
    log.info("Using PARSEC evolutionary tracks...")
    models = read_parsec(path=path_models)

    tracks = get_isotrack(models, vals, params=params, **kwargs)

    # calculate color and magnitude uncertainties
    if ~("mg" in sources.colnames):
        # calculate the absolute magnitude in G band
        sources["mg"] = sources["phot_g_mean_mag"] + 5 * np.log10(sources["parallax"]) - 10
    bp_rp = sources["bp_rp"]
    mg = sources["mg"]
    bp_rp_error = calc_bp_rp_uncertainty(sources)
    mg_error = calc_mg_uncertainty(sources)

    if correct_extinction:
        log.info("Applying extinction correction...")
        if use_reddening_key:
            e_bv = 3.1 * sources[reddening_key]
        else:
            e_bv = sources[color_excess_key]
        e_bprp, A_G = calc_gaia_extinction(e_bv)
        bp_rp = bp_rp - e_bprp
        mg = mg - A_G
        suffix += "_extinction"

    # assign param
    log.info(f"Assigning {param} using {n_realizations} realizations...")
    param_mean, param_error = get_param(bp_rp, bp_rp_error, mg, mg_error, tracks, param=param, suffix=suffix,
                                        is_save=is_save,
                                        log=log, output_type=output_type, output_path=output_path,
                                        csv_format=csv_format,
                                        n_realizations=n_realizations, interp_fun=interp_fun,
                                        binary_polygon=None, **kwargs)

    close_log(log)

    return param_mean, param_error
