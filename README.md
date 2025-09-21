# STAM

[![Documentation Status](https://readthedocs.org/projects/stam/badge/?version=latest)](https://stam.readthedocs.io/en/latest/)

**Stellar-Track-based Assignment of Mass (STAM)** is a `python` package that assigns mass, metallicity, or age, to stars based on their location on the Hertzsprung-Russell diagram (color-magnitude diagram), using stellar evolutionary tracks (not provided by this package).

The package has been tested for _Gaia_ DR2 sub-solar main-sequence stars, using [PARSEC evolutionary tracks](http://stev.oapd.inaf.it/cgi-bin/cmd).
See [Hallakoun & Maoz 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.507..398H/abstract) for details ([please cite this paper](#citing-and-attributing) if you use `STAM` in your publication). 

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
Make sure to download isochrones covering the full range of parameters you wish to probe (i.e. spanning the full relevant range of age, mass, and metallicity, in small enough steps).
For the analysis in [Hallakoun & Maoz 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.507..398H/abstract) we downloaded the following PARSEC tracks:
* Ages 1-400Myr in 1Myr steps, [M/H]=0.7 (for the pre-main-sequence tracks)
* Ages 0.5-15Gyr in 0.5Gyr steps, [M/H]=-2 to +0.7 in 0.1dex steps (for the main-sequence tracks)

You might need to download the isochrones iteratively, in case your query exceeds 400 rows.
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

Alternatively, you can use `stam.utils.write_config` to help you write the configuration file (consult the function help for details).

### Running `stam`

#### Without applying special treatment to equal-mass binaries
To assign mass and metallicities to all the Gaia sources in your Gaia data file, run in `python` (if the config keyword `CONSIDER_TWINS` is set to `FALSE`):

```
from stam.run import  get_mass_and_metallicity

m_mean, m_error, mh_mean, mh_error = get_mass_and_metallicity()
```

This returns, for each star, the mean values of the mass  (`m_mean`) and metallicity (`mh_mean`), and their corresponding standard deviations (`*_error`).

This will also save the results to files, according to the specifications in the `config.ini` file.
A log file will be saved to the same folder.

Optional input keywords for `get_mass_and_metallicity`:
* `idx`: select only specific rows in the Gaia table (default: `None`)
* `suffix`: add a customized suffix to the output file names (default: `None`)
* `config_file`: specify which configuration file to use (default: `config.ini`)
* `sample_settings`: a dictionary including keywords "vmin", "vmax", and "dist" (default: None).
If provided, only Gaia sources within the specific transverse velocities (in km/s) and distance (in pc), will be evaluated.
  
#### When applying special treatment to equal-mass binaries

If you want to treat the equal-mass binary sequence above the main sequence separately, set the config keyword `CONSIDER_TWINS` to `TRUE`.
To assign mass and metallicities to all the Gaia sources in your Gaia data file, run in `python`:

```
from stam.run import  get_mass_and_metallicity

m_mean, m_error, binary_m_mean, binary_m_error, m_weight,\
 mh_mean, mh_error, binary_mh_mean, binary_mh_error, mh_weight = get_mass_and_metallicity()
```

This time you will get some extra output variables:
* `*_mean`: the mass/metallicity mean value assuming a single star
* `*_error`: the mass/metallicity standard deviation assuming a single star
* `binary_*_mean`: the mass/metallicity mean value assuming an equal-mass binary
* `binary_*_error`: the mass/metallicity standard deviation assuming an equal-mass binary
* `*_weight`: the single-star probability = the fraction of realizations in which the star was **outside** of the defined equal-mass binary region

### Choosing the interpolation method

`STAM` includes three different interpolation methods, that can be used to interpolate the evolutionary track grid: `rbf`, `griddata`, and `nurbs`.
It is recommended to plot the resulting grid before running the full mass assignment procedure, to check for problems in the grid evaluation.
For example, to check the `rbf` `linear` interpolation:

```
import numpy as np
import scipy
import matplotlib.pyplot as plt
import stam

isochrone = stam.getmodels.read_parsec()
mass = np.arange(0.15, 1.1, 0.05)
tracks = stam.gentracks.get_combined_isomasses(isochrone, mass=mass, age=5, mh_pre_ms=0.7, age_min=0.005,
                                                is_smooth=True, smooth_sigma=3, exclude_pre_ms_masses=[0.15])

x = np.array(tracks["bp_rp"])
y = np.array(tracks["mg"])
z = np.array(tracks["mass"])

xstep=0.05
ystep=0.05

def tracks2grid(tracks, xparam="bp_rp", yparam="mg", xstep=0.05, ystep=0.05):

    xmin = np.min(np.around(tracks[xparam], -int(np.round(np.log10(xstep)))))
    xmax = np.max(np.around(tracks[xparam], -int(np.round(np.log10(xstep)))))
    ymin = np.min(np.around(tracks[yparam], -int(np.round(np.log10(ystep)))))
    ymax = np.max(np.around(tracks[yparam], -int(np.round(np.log10(ystep)))))
    x, y = np.meshgrid(np.arange(xmin, xmax, xstep), np.arange(ymin, ymax, ystep))
            
    return x, y, xmin, xmax, ymin, ymax

grid_x, grid_y, xmin, xmax, ymin, ymax = tracks2grid(tracks, xstep=xstep, ystep=ystep)

fun_type = "linear"

interp = scipy.interpolate.Rbf(x, y, z, function=fun_type)
grid_z = interp(grid_x, grid_y)

fig = plt.figure()
plt.plot(x, y, 'ko', markersize=1)
plt.imshow(grid_z, origin="lower", vmin=0.1, vmax=1.2, extent=[xmin, xmax, ymin, ymax], cmap='viridis')
plt.colorbar()
plt.title(fun_type)
plt.gca().invert_yaxis()

plt.show()
```

#### rbf
This is `scipy`'s radial basis function (RBF) interpolation.
See the [`scipy.interpolate.Rbf` reference page](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html) for details.
When using the `rbf` method, you should also provide the `RBF_FUN` keyword, which is the `function` argument of `scipy.interpolate.Rbf`.
We found the `rbf` method with `linear` function to work best in our case ([Hallakoun & Maoz 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.507..398H/abstract)).

#### griddata
This is based on `scipy`'s `griddata` linear interpolation: [`scipy.interpolate.griddata`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html),
but uses a faster implementation, based on [this answer](https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids).
In our case, it resulted in some artifacts when using combined pre-main-sequence+main-sequence evolutionary tracks.

#### nurbs
This is [`geomdl`'s NURBS library](https://nurbs-python.readthedocs.io/en/5.x/) (Non-Uniform Rational Basis Spline).


## Acknowledgements
The multicolor plot functions defined in `colorline.py` are taken from [David P. Sanders' `colorline` Jupyter Notebook](https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb).

## Citing and attributing
If you use `STAM` in your work, please provide a link to [this webpage](https://github.com/naamach/stam), and cite [Hallakoun & Maoz 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.507..398H/abstract):
```
@ARTICLE{2021MNRAS.507..398H,
       author = {{Hallakoun}, Na'ama and {Maoz}, Dan},
        title = "{A bottom-heavy initial mass function for the likely-accreted blue-halo stars of the Milky Way}",
      journal = {\mnras},
     keywords = {methods: statistical, Hertzsprung-Russell and colour-magnitude diagrams, stars: luminosity function, mass function, stars: statistics, solar neighbourhood, Galaxy: stellar content, Astrophysics - Astrophysics of Galaxies, Astrophysics - Solar and Stellar Astrophysics},
         year = 2021,
        month = oct,
       volume = {507},
       number = {1},
        pages = {398-413},
          doi = {10.1093/mnras/stab2145},
archivePrefix = {arXiv},
       eprint = {2009.05047},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.507..398H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
