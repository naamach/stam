import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.table import Table
from stam.models import colname


def get_isotrack(models, vals, params=("mass", "mh"),
                 mass_res=0.007, age_res=0.1, mh_res=0.05, stage=1,
                 mass_min=0, mass_max=1, age_min=0, age_max=np.inf, mh_min=-np.inf, mh_max=np.inf,
                 stage_min=0, stage_max=np.inf):
    """
    get_isotrack(models, vals, params=("mass", "mh"),
                 mass_res=0.007, age_res=0.1, mh_res=0.05, stage=1,
                 mass_min=0, mass_max=1, age_min=0, age_max=np.inf, mh_min=-np.inf, mh_max=np.inf,
                 stage_min=0, stage_max=np.inf)

    Get a specific stellar evolution track, with two out of three parameters fixed (mass, age, or metallicity).

    Parameters
    ----------
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    vals : array_like (of length 2)
        Values of the fixed parameters (units: [mass] = Msun, [age] = Gyr, [mh] = dex).
    params : tuple (of length 2), optional
        Fixed parameters names (default: ("mass", "mh")).
    mass_res : float, optional
        Mass resolution, in Msun (default: 0.007 Msun)
    age_res : float, optional
        Age resolution, in Gyr (default: 0.1 Gyr)
    mh_res : float, optional
        Metallicity resolution, in dex (default: 0.05 dex)
    stage : int, optional
        Stellar evolution stage label (0 = pre-MS, 1 = MS, etc.; default: 1).
    mass_min : float, optional
        Minimum mass to consider, in Msun (if no fixed mass was chosen; default: 0 Msun).
    mass_max : float, optional
        Maximum mass to consider, in Msun (if no fixed mass was chosen; default: 1 Msun).
    age_min : float, optional
        Minimum age to consider, in Gyr (if no fixed age was chosen; default: 0 Gyr).
    age_max : float, optional
        Maximum age to consider, in Gyr (if no fixed age was chosen; default: `np.inf` Gyr).
    mh_min : float, optional
        Minimum [M/H] to consider, in dex (if no fixed metallicity was chosen; default: `-np.inf`).
    mh_max : float, optional
        Maximum [M/H] to consider, in dex (if no fixed metallicity was chosen; default: `np.inf`).
    stage_min : int, optional
        Minimum stellar evolution stage label to consider (if no fixed stage was chosen; default: 0).
    stage_max : int, optional
        Maximum stellar evolution stage label to consider (if no fixed stage was chosen; default: `np.inf`).

    Returns
    -------
    bp : array_like
        Chosen stellar evolution track Gaia Gbp magnitude
    rp : array_like
        Chosen stellar evolution track Gaia Grp magnitude
    g : array_like
        Chosen stellar evolution track Gaia G magnitude
    mass : array_like
        Chosen stellar evolution track mass, in Msun
    mh : array_like
        Chosen stellar evolution track metallicity ([M/H]), in dex
    age : array_like
        Chosen stellar evolution track age, in Gyr
    """

    if "mass" in params:
        mass = vals[params.index("mass")]
        mass_idx = ((mass - mass_res) <= models[colname("m0")]) & (models[colname("m0")] < (mass + mass_res))
    else:
        mass_idx = (mass_min - mass_res <= models[colname("m0")]) & (models[colname("m0")] <= mass_max + mass_res)

    if "age" in params:
        age = vals[params.index("age")]
        age_idx = (np.log10((np.maximum(age - age_res, 0)) * 1e9) <= models[colname("log_age")]) & \
                  (models[colname("log_age")] < np.log10((age + age_res) * 1e9))
    else:
        age_idx = (np.log10((np.maximum(age_min - age_res, 0)) * 1e9) <= models[colname("log_age")]) & \
                  (models[colname("log_age")] <= np.log10((age_max + age_res) * 1e9))

    if "mh" in params:
        mh = vals[params.index("mh")]
        mh_idx = ((mh - mh_res) <= models[colname("mh")]) & (models[colname("mh")] < (mh + mh_res))
    else:
        mh_idx = (mh_min - mh_res <= models[colname("m0")]) & (models[colname("m0")] <= mh_max + mh_res)

    if stage is not None:
        stage_idx = models[colname("phase")] == stage  # 1 = main sequence
    else:
        stage_idx = (stage_min <= models[colname("phase")]) & (models[colname("phase")] <= stage_max)

    idx = mass_idx & age_idx & mh_idx & stage_idx

    bp = models[idx][colname("G_BPmag")]
    rp = models[idx][colname("G_RPmag")]
    g = models[idx][colname("Gmag")]
    mass = models[idx][colname("m0")]
    mh = models[idx][colname("mh")]
    age = 10 ** models[idx][colname("log_age")] * 1e-9  # [Gyr]

    sort_idx = np.argsort(mass)
    bp = bp[sort_idx]
    rp = rp[sort_idx]
    g = g[sort_idx]
    mass = mass[sort_idx]
    mh = mh[sort_idx]
    age = age[sort_idx]

    return bp, rp, g, mass, mh, age


def get_pre_ms_isomass(models, mass, mh, is_smooth=True, smooth_sigma=3, **kwargs):
    """
    get_pre_ms_isomass(models, mass, mh, is_smooth=True, smooth_sigma=3, **kwargs)

    Get a specific stellar evolution pre-MS track, with fixed mass and metallicity.

    Parameters
    ----------
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    mass : float
        Stellar track mass, in Msun
    mh : float
        Stellar track metallicity ([M/H]), in dex
    is_smooth : bool, optional
        Smooth the stellar track? (default: True).
    smooth_sigma : float, optional
        Smoothing Gaussian sigma (default: 3).
    kwargs : optional
        Any additional keyword arguments to be passed to `stam.tracks.get_isotrack`.

    Returns
    -------
    bp_rp : array_like
        Chosen stellar evolution track Gaia Gbp-Grp color.
    mg : array_like
        Chosen stellar evolution track Gaia G magnitude.
    age : array_like
        Chosen stellar evolution track age, in Gyr.
    """

    bp, rp, g, _, _, age = get_isotrack(models, [mass, mh], params=("mass", "mh"), stage=0, **kwargs)

    if is_smooth:
        bp_rp = gaussian_filter1d(np.array(bp - rp), smooth_sigma)
        mg = gaussian_filter1d(np.array(g), smooth_sigma)

        # remove transition to MS part
        d = np.diff(gaussian_filter1d(np.diff(bp_rp), 10))
        idx = np.argmax(d)
        if np.abs(mass - 0.6) < 0.01:  # fix plot wobble
            idx -= 30
        bp_rp = bp_rp[:idx]
        mg = mg[:idx]
        age = age[:idx]
    else:
        bp_rp = np.array(bp - rp)
        mg = np.array(g)

    return bp_rp, mg, age


def get_combined_isomass(models, mass, age, mh_pre_ms=0.7, is_smooth=True, smooth_sigma=3, **kwargs):
    """
    get_combined_isomass(models, mass, age, mh_pre_ms=0.7, is_smooth=True, smooth_sigma=3, **kwargs)

    Get a specific stellar evolution track, combining a pre-MS track with fixed mass and metallicity, and an MS track
    with fixed mass and age.

    Parameters
    ----------
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    mass : float
        Stellar track mass, in Msun.
    age : float
        MS stellar track age, in Gyr.
    mh_pre_ms : float, optional
        Pre-MS stellar track metallicity ([M/H]), in dex (default: 0.7 dex).
    is_smooth : bool, optional
        Smooth the stellar track? (default: True).
    smooth_sigma : float, optional
        Smoothing Gaussian sigma (default: 3).
    kwargs : optional
        Any additional keyword arguments to be passed to `stam.tracks.get_isotrack`.

    Returns
    -------
    bp_rp : array_like
        Chosen stellar evolution track Gaia Gbp-Grp color.
    mg : array_like
        Chosen stellar evolution track Gaia G magnitude.
    mh : array_like
        Chosen stellar evolution track metallicity, in dex.
    age_vec : array_like
        Chosen stellar evolution track age, in Gyr.
    ms_idx : array_like (bool)
        Indices of MS points on track.
    """

    # get MS track (fixed mass and age)
    bp, rp, g, _, mh, _ = get_isotrack(models, [mass, age], params=("mass", "age"), stage=1, **kwargs)

    # sort by metallicity
    sort_idx = np.argsort(mh)
    bp = bp[sort_idx]
    rp = rp[sort_idx]
    g = g[sort_idx]
    mh = mh[sort_idx]

    # get pre-MS track (fixed mass and metallicity)
    bp_rp0, mg0, age0 = get_pre_ms_isomass(models, mass, mh_pre_ms, is_smooth=is_smooth, smooth_sigma=smooth_sigma,
                                           **kwargs)

    # combine MS and pre-MS tracks
    bp_rp = np.append(bp - rp, np.flipud(bp_rp0))
    mg = np.append(g, np.flipud(mg0))
    mh = np.append(mh, mh_pre_ms * np.ones(len(mg0)))
    age_vec = np.append(age * np.ones(len(bp)), np.flipud(age0))
    ms_idx = np.ones_like(bp_rp, dtype=bool)  # main sequence
    ms_idx[len(bp):] = False

    if is_smooth & (mass != 0.15):
        bp_rp = gaussian_filter1d(bp_rp, smooth_sigma)
        mg = gaussian_filter1d(mg, smooth_sigma)

    return bp_rp, mg, mh, age_vec, ms_idx


def get_isomasses(models, mass=np.arange(0.1, 1.2, 0.1), age=5, **kwargs):
    """
    get_isomasses(models, mass=np.arange(0.1, 1.2, 0.1), age=5, **kwargs)

    Get fixed-age stellar evolution tracks, for a range of masses.
    If the `stage` keyword is not specified (under `kwargs`), it assumes `stage=1` (MS phase).

    Parameters
    ----------
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    mass : array_like, optional
        Stellar track mass array, in Msun (default: `np.arange(0.1, 1.2, 0.1)` Msun).
    age : float, optional
        Stellar track age, in Gyr (default: 5 Gyr).
    kwargs : optional
        Any additional keyword arguments to be passed to `stam.tracks.get_isotrack`.

    Returns
    -------
    tracks : Table
        Stellar evolution tracks, for the given age and range of masses, with columns `mass`, `bp_rp`, `mg`, and `mh`.
    """

    m_vec = np.array([])
    bp_rp = np.array([])
    mg = np.array([])
    mh = np.array([])
    for m in mass:
        bp, rp, g, _, curr_mh, _ = get_isotrack(models, [m, age], params=("mass", "age"), **kwargs)
        bp_rp = np.append(bp_rp, np.array([bp - rp]))
        mg = np.append(mg, np.array(g))
        mh = np.append(mh, np.array(curr_mh))
        m_vec = np.append(m_vec, m * np.ones(len(g)))

    tracks = Table([m_vec, bp_rp, mg, mh], names=('mass', 'bp_rp', 'mg', 'mh'))

    return tracks


def get_combined_isomasses(models, mass=np.arange(0.1, 1.2, 0.1), age=5, mh_pre_ms=0.7, is_smooth=True, smooth_sigma=3,
                           **kwargs):
    """
    get_combined_isomasses(models, mass=np.arange(0.1, 1.2, 0.1), age=5, mh_pre_ms=0.7, is_smooth=True, smooth_sigma=3,
                           **kwargs)

    Get combined stellar evolution tracks, for a range of masses. Each track combines a pre-MS track with fixed mass
    and metallicity, and an MS track with fixed mass and age.

    Parameters
    ----------
    models : Table
        All stellar evolution models in a single astropy table, as retrieved by `stam.models.read_parsec`.
    mass : array_like, optional
        Stellar track mass array, in Msun (default: `np.arange(0.1, 1.2, 0.1)` Msun).
    age : float, optional
        Stellar track age, in Gyr (default: 5 Gyr).
    mh_pre_ms : float, optional
        Pre-MS stellar track metallicity ([M/H]), in dex (default: 0.7 dex).
    is_smooth : bool, optional
        Smooth the stellar track? (default: True).
    smooth_sigma : float, optional
        Smoothing Gaussian sigma (default: 3).
    kwargs : optional
        Any additional keyword arguments to be passed to `stam.tracks.get_isotrack`.

    Returns
    -------
    tracks : Table
        Stellar evolution tracks, for the given age and range of masses, with columns `mass`, `bp_rp`, `mg`, and `mh`.

    """

    m_vec = np.array([])
    bp_rp = np.array([])
    mg = np.array([])
    mh = np.array([])
    for m in mass:
        curr_bp_rp, curr_mg, curr_mh, _, _ = get_combined_isomass(models, m, age, mh_pre_ms=mh_pre_ms,
                                                                  is_smooth=is_smooth, smooth_sigma=smooth_sigma,
                                                                  **kwargs)
        bp_rp = np.append(bp_rp, np.array([curr_bp_rp]))
        mg = np.append(mg, np.array(curr_mg))
        mh = np.append(mh, np.array(curr_mh))
        m_vec = np.append(m_vec, m * np.ones(len(curr_mg)))

    tracks = Table([m_vec, bp_rp, mg, mh], names=('mass', 'bp_rp', 'mg', 'mh'))

    return tracks
