#!/usr/bin/env python3
"""Read a PSF FITS file, convolve with a Gaussian, and write a new FITS file."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter


# ---- Set these values before running ----
psf_dir = "/Users/ri3e/cv/jwst_clusters/instrument/psf"
in_path = Path(f"{psf_dir}/PSF_NIRCam_in_flight_opd_filter_F360M.fits")
sigma = 2.0 # quarter-pixels
out_path = Path(f"{psf_dir}/PSF_F360M_g{sigma}.fits")
hdu = 0


data, header = fits.getdata(in_path, hdu=hdu, header=True)
data = np.asarray(data, dtype=float)

conv = gaussian_filter(data, sigma=sigma)

# Match the behavior used in phangs_phot.py where broadened PSFs are normalized.
total = np.nansum(conv)
if np.isfinite(total) and total != 0:
    conv = conv / total

fits.PrimaryHDU(data=conv, header=header).writeto(out_path, overwrite=True)
