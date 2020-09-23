import glob
from astropy.io import ascii
from astropy.table import vstack
from christmass.utils import get_config


def read_parsec(files=None, path=None, config_file="config.ini"):
    """
    Read PARSEC isochrone *.dat files, as downloaded from http://stev.oapd.inaf.it/cgi-bin/cmd

    :param files: PARSEC *.dat file names. If None, read all  *.dat files in <path> (default: None)
    :param path: path to PARSEC *.dat files. If None, get from config.ini (default: None)
    :param config_file: the config file (default: "config.ini")
    :return: All PARSEC models in a single astropy table
    """
    config = get_config(config_file)
    if path is None:
        path = config.get('PARSEC', 'PATH')  # PARSEC models path

    if files is None:
        # read all .dat files in the path
        files = glob.glob(path + "*.dat")
    else:
        # attach path to filename
        files = [path + file for file in files]

    for i, file in enumerate(files):
        iso = ascii.read(file, header_start=12)
        if i == 0:
            models = iso
        else:
            models = vstack([models, iso])

    return models


def colname(param, model_name="parsec"):
    if model_name.lower() == "parsec":
        # Zini     MH   logAge Mini  int_IMF         Mass   logL    logTe  logg  label   McoreTP C_O  period0 period1
        # pmode  Mloss  tau1m   X   Y   Xc  Xn  Xo  Cexcess  Z 	 mbolmag  Gmag    G_BPmag  G_RPmag
        cols = {
            "Z0": "Zini",
            "mh": "MH",
            "log_age": "logAge",
            "m0": "Mini",
            "m": "Mass",
            "phase": "label",
            "Gmag": "Gmag",
            "G_BPmag": "G_BPmag",
            "G_RPmag": "G_RPmag"
        }
    else:
        raise Exception(f"Unknown model {model_name}!")

    return cols[param]
