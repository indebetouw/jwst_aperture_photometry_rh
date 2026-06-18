# PHANGS-JWST Aperture Photometry
 
A Python pipeline for performing aperture photometry on JWST NIRCam and MIRI imaging data from the [PHANGS-JWST survey](https://sites.google.com/view/phangs/home). The pipeline handles the full workflow from loading mosaics to producing calibrated photometric catalogs, including background subtraction, source detection, optimal aperture selection, and aperture corrections.

IMPORTANT: THIS MODULE IS STILL IN DEVELOPMENT. Please expect bugs. It is being updated regularly and currently does not have the QA infrastructure to know that the output catalogs are reliable. 
 
---
 
## Overview
 
This module takes JWST stage-3 mosaic products (typically anchored mosaics in FITS format) and produces source catalogs with aperture photometry in AB magnitudes. The pipeline is controlled via TOML configuration files, making it straightforward to apply consistently across many galaxies and filters.
 
**Key capabilities:**
 
- 2D background estimation and subtraction using `photutils.Background2D`
- Source detection with `IRAFStarFinder`, `DAOStarFinder` [*in development*], or `find_peaks`
- Optimal aperture radius selection via curve-of-growth analysis [*in development*]
- Circular aperture photometry with local sky annulus background subtraction
- Aperture corrections using CRDS calibration files or the empirical factors from Rodriguez et al. (2025)
- Flux conversion from Jy/sr (or MJy/sr) to AB magnitudes
- Catalog output in FITS or CSV format
- Quality-assurance (QA) plots [*in development*]

## Repository Structure
 
```
jwst_aperture_photometry/
├── phangs_phot.py          # Main pipeline: orchestrates all steps for each galaxy/filter
├── ap_phot.py              # Core photometry utilities (background subtraction, curve of growth)
├── source_find.py          # Source detection wrappers
├── io.py                   # I/O helpers
├── qa_plots.py             # Quality assurance plots
├── apcorr_rodriguez.ecsv   # Empirical aperture corrections (Rodriguez et al. 2025)
├── config/
│   ├── config.toml         # Photometry parameters (targets, filters, pipeline steps)
│   └── mpcdf.toml          # Local paths (data directory, output directory, CRDS directory)
├── catalogs/               # Output photometric catalogs
├── tests/                  # Unit tests
└── aperture_phot_example.ipynb  # End-to-end worked example
```

---
 
## Installation
 
### Prerequisites
 
Python 3.9+ is recommended. Install the required packages with pip:
 
```bash
pip install numpy scipy matplotlib astropy photutils astroquery
```
 
Or with conda:
 
```bash
conda install numpy scipy matplotlib astropy photutils
conda install -c conda-forge astroquery
```


### Clone the repository
 
```bash
git clone https://github.com/rebeccahoughton/jwst_aperture_photometry.git
cd jwst_aperture_photometry
```

---
 
## Configuration
 
All pipeline settings are controlled by two TOML files in the `config/` directory.
 
### `config/config.toml` — photometry parameters
 
This file specifies which galaxies and filters to process, which pipeline steps to run, and the parameters for each step.

### `config/mpcdf.toml` — local paths
 
```toml
jwst_dir  = "/path/to/jwst/data/"
out_dir   = "/path/to/output/catalogs/"
crds_dir  = "/path/to/crds/apcorr/files/"
```

---
 
## Usage
 
### Running the full pipeline
 
Edit the two config files to point at your data and set your parameters, then run:
 
```python
import phangs_phot as pp
 
pp.do_photometry(
    steps=pp.steps,
    targets=pp.targets,
    use_filter_fwhm=True,
    conf=pp.conf,
)
```
 
This loops over all galaxies and filters defined in `config.toml`, running whichever steps are listed in `steps`.


## Aperture Corrections
 
Two methods are supported, set via `apcorr_method` in the config:
 
- **`crds`** — Uses the official JWST Calibration Reference Data System (CRDS) `apcorr` files for NIRCam. Derives the aperture radius, sky annulus radii, and correction factor for a specified encircled energy fraction (default: 80%).
- **`cluster`** — Uses empirical correction factors for star clusters from Rodriguez et al. (2025), based on Deger et al. (2022). Valid for a fixed aperture radius of 4 pixels. Corrections are provided in Vega magnitudes and converted to AB internally via SVO filter zero points.
- **Note that aperture corrections for custom radii are not currently available**.


---
 
## Output Catalogs
 
Catalogs are written to `out_dir` as `{galaxy}_jwst_{band}_cat_cluster_apcorr.fits` (or `.csv`). Key columns will depend on the source finder method used. 

---

## Dependencies
 
| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `scipy` | Spatial matching (KD-tree) |
| `matplotlib` | Plotting and QA figures |
| `astropy` | FITS I/O, WCS, tables, units |
| `photutils` | Background estimation, source detection, aperture photometry |
| `astroquery` | SVO filter zero points for aperture corrections |
 
---
 