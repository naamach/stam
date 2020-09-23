import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.table import Table
from stam.models import colname


def get_isotrack(models, vals, params=("mass", "mh"),
                 mass_res=0.007, age_res=0.1, mh_res=0.05, stage=1,
                 mass_min=0, mass_max=1, age_min=0, age_max=np.inf, mh_min=-np.inf, mh_max=np.inf,
                 stage_min=0, stage_max=np.inf):
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
