# STAM

**Stellar-Track-based Assignment of Mass (STAM)** is a `python` package that assigns mass, metallicity, or age, to stars based on their location on the Hertzsprung-Russell diagram (color-magnitude diagram), using stellar evolutionary tracks (not provided by this package).

The package has been tested for _Gaia_ DR2 sub-solar main-sequence stars, using [PARSEC evolutionary tracks](http://stev.oapd.inaf.it/cgi-bin/cmd).
See [Hallakoun & Maoz 2021](https://ui.adsabs.harvard.edu/abs/2020arXiv200905047H/abstract) for details ([please cite this paper](#citing-and-attributing) if you use `STAM` in your publication). 

## Getting started

### Prerequisites

* `python 3.6` or above
* `astropy`
* `configparser`
* `numpy`
* `scipy`
* `tdqm`
* `geomdl`

### Installing

Create and activate a `conda` environment with the necessary modules:
```
$ conda create -n stam astropy configparser numpy scipy python=3.7.1
$ source activate stam
```
Install the `stam` package:
```
$ pip install git+https://github.com/naamach/stam.git
```

### Upgrading
To upgrade `stam` run:
```
$ pip install git+https://github.com/naamach/stam.git --upgrade
```

## Using `stam`

### Download the PARSEC models
First, you need to download some stellar evolution tracks, that will be used to estimate the stellar parameters.
You can download the PARSEC isochrones from [here](http://stev.oapd.inaf.it/cgi-bin/cmd).
You might need to download the table iteratively, in case your query exceeds 400 rows.
Save all the resulting `*.dat` files into a single folder.
`stam` will concatenate all the `*.dat` files in that folder into a single table.

### Download the Gaia data
You can query the Gaia DR2 database directly from the [Gaia Archive](https://gea.esac.esa.int/archive/),
or any other method of your preference. Save the resulting table in `*.FITS` format (it works faster for large tables).

### Prepare the configuration file

You will have to provide `stam` with a `config.ini` file in the working directory (the directory from which you run the script).
The file should look like that (see `config.ini.example` in the main directory):

```
; config.ini
[LOG]
PATH = /path/to/log/file/
CONSOLE_LEVEL = INFO ; DEBUG, INFO, WARNING, ERROR, CRITICAL
FILE_LEVEL = DEBUG ; DEBUG, INFO, WARNING, ERROR, CRITICAL

[GENERAL]
SAVE = TRUE ; save results to file?
PATH = /path/to/working/directory/ ; result file destination path
OUTPUT_TYPE = csv ; npy, csv
CSV_FORMAT = .8f ; CSV format (without "%")

[GAIA]
PATH = /path/to/gaia/files/ ; path to Gaia data folder
FILE = GaiaFileName.fits ; Gaia data file name (only works for FITS files)
CORRECT_EXTINCTION = TRUE ; correct Gaia data for extinction?

[BINARY]
CONSIDER_TWINS = TRUE ; consider equal-mass binary sequence when assigning mass/metallicity/age?
FLUX_RATIO_MIN = 1.9 ; minimal binary twin flux ratio
FLUX_RATIO_MAX = 2 ; maximal binary twin flux ratio

[MODELS]
SOURCE = PARSEC ; which isochrone models to use?
M_MIN = 0.15 ; [Msun] minimum track mass
M_MAX = 1 ; [Msun] maximum track mass
M_STEP = 0.05 ; [Msun] track mass step
AGE = 5 ; [Gyr] age of the MS tracks
MH_PRE_MS = 0.7 ; pre-MS track [M/H]
SMOOTH = True ; smooth track?
SMOOTH_SIGMA = 3 ; Gaussian smoothing sigma
EXCLUDE_PRE_MS_MASSES = ; [Msun] exclude the pre-MS tracks of these masses (to avoid crossing other tracks),
                                                       ; separate values by a comma, leave empty to include all

[PARSEC]
PATH = /path/to/PARSEC/files/ ; path to the PARSEC *.dat tables (concatenates all *.dat files in the folder to a single table)

[INTERP]
METHOD = rbf ; rbf, griddata, nurbs
RBF_FUN = linear

[MASS]
N_REALIZATIONS = 10 ; number of realizations

[MH]
N_REALIZATIONS = 10 ; number of realizations
```

### Running `stam`

To assign mass and metallicities to all the Gaia sources in your Gaia data file, run in `python`:

```
from stam.run import  get_mass_and_metallicity

m_mean, m_error, mh_mean, mh_error = get_mass_and_metallicity()
```

This will also save the results to files, according to the specifications in the `config.ini` file.
A log file will be saved to the same folder.

Optional input keywords for `get_mass_and_metallicity`:
* `idx`: select only specific rows in the Gaia table (default: `None`)
* `suffix`: add a customized suffix to the output file names (default: `None`)
* `config_file`: specify which configuration file to use (default: `config.ini`)
* `sample_settings`: a dictionary including keywords "vmin", "vmax", and "dist" (default: None).
If provided, only Gaia sources within the specific transverse velocities (in km/s) and distance (in pc), will be evaluated.

## Acknowledgements
The multicolor plot functions defined in `colorline.py` are taken from [David P. Sanders' `colorline` Jupyter Notebook](https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb).

## Citing and attributing
If you use `STAM` in your work, please provide a link to [this webpage](https://github.com/naamach/stam), and cite [Hallakoun & Maoz 2021](https://ui.adsabs.harvard.edu/abs/2020arXiv200905047H/abstract):
```
@ARTICLE{2020arXiv200905047H,
       author = {{Hallakoun}, Na'ama and {Maoz}, Dan},
        title = "{A bottom-heavy initial mass function for the accreted blue-halo stars of the Milky Way}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Solar and Stellar Astrophysics},
         year = 2020,
        month = sep,
          eid = {arXiv:2009.05047},
        pages = {arXiv:2009.05047},
archivePrefix = {arXiv},
       eprint = {2009.05047},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200905047H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
