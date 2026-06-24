# This notebook was modified from https://github.com/JaysonAstro/prototype_HST_catalog_photometry/blob/main/HST_cats_with_IRAFStarFinder.ipynb
# which is based on https://qosmicqi.github.io/XRBID/chapters/photometry.html#sec-runphots
# and https://www.astropy.org/ccd-reduction-and-photometry-guide/v/pdev/notebooks/photometry/00.00-Preface.html


import fnmatch
import glob
import numpy as np
import matplotlib.pyplot as plt
import tomllib
import os
from sys import exit
from scipy.spatial import cKDTree

import astropy.units as u
from astropy import wcs
from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table, join, hstack
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.visualization import ImageNormalize, LogStretch

# Photutils imports
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from photutils.detection import IRAFStarFinder, DAOStarFinder, find_peaks
from photutils.centroids import centroid_quadratic
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from photutils.aperture import aperture_photometry

# SVO for aperture correction
from astroquery.svo_fps import SvoFps

# ------------------------------------------------
# Configs
# ------------------------------------------------

config_file = 'config/config.toml'     # Photometry parameters
local_file = 'config/mpcdf.toml'       # Paths to directories

def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)

# Unpack the parameters from the config file
conf = load_config(config_file)
local = load_config(local_file)

# Get top level parameters
steps   = conf['steps']
targets = conf['targets']
bands   = conf['bands']
projects = conf['projects']
product = conf['product']
version = conf['version']
ptype = conf['ptype']

# Number of targets to process
num_targets = len(targets)

finder_params = conf['parameters']['source_find']
phot_params = conf['parameters']['photometry']


# ------------------------------------------------
# Conversions and file management
# ------------------------------------------------
def get_path_to_file(wdir, version, project, galaxy, ptype, filter):
     """Get the path to the data file based on the version, project, galaxy, product type, and filter.
     Args:
          version: version of the data (e.g., v4p1)
          project: JWST PID (e.g., 4793)
          galaxy: galaxy name 
          ptype: product type (e.g., images (for anchored), features, psfmatch, etc.)
          filter: filter name."""
     # TODO: Add functionality for files not in the release directory
     path = f"{wdir}{version}/{project}/release/{galaxy}/{ptype}/"

     # Check that the path exists
     if os.path.exists(path):
          print(f"Found directory for {galaxy} {filter} in {path}")
     else:
          # look for plausible files in the directory and print a warning if we find any, but raise an error if we don't find any
          plausible_files = []
          for root, dirnames, filenames in os.walk(wdir + version + "/"):
               for f in fnmatch.filter(filenames, '*.fits'):
                    if galaxy.lower() in f.lower() and ptype.lower() in f.lower() and filter.lower() in f.lower():
                        plausible_files.append(os.path.join(root, f))

          if len(plausible_files) > 0:
               print(f"Warning: No file found for {galaxy} {filter} in release {path}, so using this plausible file: {plausible_files[0]}.")
               path = os.path.dirname(plausible_files[0])+"/"
          else:
               raise FileNotFoundError(f"No file found for {galaxy} {filter} in {path}. Please check the path and file naming conventions.")
     return path



def convert_aperture_sum_Jy_per_sr_to_abmag(aperture_sum_jy_sr, header):
     """Convert aperture sum in Jy/sr to AB magnitudes.
     Args:
          aperture_sum_jy_sr: aperture sum in Jy/sr or MJy/sr
          header: FITS header containing WCS information to get pixel area in steradians
                  and BUNIT for checking units of the input aperture sum.
     Returns:
          AB magnitudes"""
     
     # Check that the input is in Jy/sr
     if header.get('BUNIT', '').lower() in ['mjysr', 'mjy/sr', 'mj/steradian']:
          # If header is in MJy/sr, then convert to Jy/sr before calculating magnitude
          print(f"Warning: BUNIT in header is {header.get('BUNIT', 'unknown')}, but expected Jy/sr. Applying conversion to MJy/sr.")
          aperture_sum_jy_sr = np.array(aperture_sum_jy_sr) * 1e6
     elif not header.get('BUNIT', '').lower() in ['jy/sr', 'jy/steradian']:
          raise ValueError("Input aperture sum must be in Jy/sr or MJy/sr for conversion to AB magnitudes.")
    
     # Get pixel area in steradians from header
     pix_area_sr = get_pixarea_in_sr(header)
     fnu_jy = np.array(aperture_sum_jy_sr) * pix_area_sr
     fnu_jy = np.where(fnu_jy > 0, fnu_jy, np.nan)
     # Convert to magnitudes
     abmag = -2.5 * np.log10(fnu_jy / 3631.0)
     return abmag



def convert_abmag_to_Jy_per_sr(abmag, header, unit='MJy/sr'):
     """Convert AB magnitudes to Jy/sr.
     Args:
          abmag: AB magnitudes
          header: FITS header containing WCS information to get pixel area in steradians
                  and BUNIT for checking units of the output aperture sum.
     Returns:
          aperture sum in Jy/sr (numpy array)"""
     # Get pixel area in steradians from header
     pix_area_sr = get_pixarea_in_sr(header)
     fnu_jy = 3631.0 * 10**(-0.4 * abmag)
     aperture_sum_jy_sr = fnu_jy / pix_area_sr
     if unit == 'MJy/sr' or unit == 'MJ/sr' or unit == 'mjy/sr':
          aperture_sum_jy_sr = aperture_sum_jy_sr * 1e-6
     return aperture_sum_jy_sr



def get_pixarea_in_sr(header):
    """Get pixel area in steradians from FITS header.
    Args:
        header: FITS header containing WCS information
    Returns:
        pixel area in steradians (float)"""
    
    # JWST data should have a PIXAR_SR keyword
    if 'PIXAR_SR' in header:
        return float(header['PIXAR_SR'])
    
    # If keyword is not found, then we can try to compute it from CDELT or CD matrix
    elif ('CDELT1' in header) and ('CDELT2' in header):
        print("Warning: PIXAR_SR keyword not found in header. Computing pixel area from WCS information.")
        area_deg2 = np.abs(float(header['CDELT1']) * float(header['CDELT2']))
        if np.isfinite(area_deg2) and (area_deg2 > 0):
            return float((area_deg2 * u.deg**2).to(u.sr).value)
    elif 'CD1_1' in header:
        print("Warning: PIXAR_SR keyword not found in header. Computing pixel area from WCS information.")
        cd = np.array([[float(header['CD1_1']), float(header['CD1_2'])],
                        [float(header['CD2_1']), float(header['CD2_2'])]])
        area_deg2 = np.abs(np.linalg.det(cd))
        return float((area_deg2 * u.deg**2).to(u.sr).value)
    # And if we can't do either of those...
    else:
        raise ValueError("could not get pixel area in steradians from header/WCS")
     


def open_jwst(path, gal, dir, band, mosaic_ext="*anchor*.fits", get_coverage=True):
     """
     Open JWST data (from either MIRI/NIRCam) and return image, error, header.
     Using the stage 3 aligned data products, and it defaults to the anchored mosaic (which is the most aligned product).

     Args:
          path: path to the data directory
          gal: galaxy name
          dir: directory name
          band: filter name (e.g., F770W, F1000W, etc.)
          level: useful if data is hidden in a subdirectory (typical for pjpipe outputs)
          mosaic_ext: extension to search for (default is the anchored mosaic)
          get_coverage: whether to return a coverage mask (default True)
     Returns:
          img: 2D array of the image data
          err: 2D array of the error data
          snr_map: 2D array of the signal-to-noise ratio (img/err)
          coverage_mask: 2D boolean array where True indicates no coverage (NaN or zero in img or err)
          header: FITS header of the image data
     """
     # Load the files
     # TODO implement better fallbacks starting with mosaic_ext (default anchored) and then falling back to i2d or types
     print(f"Searching in {path} for {band} data, with extension: {mosaic_ext}")
     #     files = glob.glob(f"{path}/{gal.lower()}*{band.lower()}*{mosaic_ext}")
     files = glob.glob(path + f"*{band.lower()}*{mosaic_ext}*")
     print(f"Files found: {files}")

     # Sanity check that we are getting only one aligned mosaic
     if len(files) == 0:
          raise FileNotFoundError(f"No files found for {band} in {dir}{gal}")
     elif len(files) > 1:
          print(f"Warning: Multiple files found for {band} in {dir}{gal}. Using the first one: {files[0]}")

     # Initialize variables
     img_file = None
     err_file = None

     # Open the file and use extensions to assign data and header
     with fits.open(files[0]) as hdul:
          img_file = hdul['SCI']
          img = img_file.data
          header = img_file.header
          # Error
          err_file = hdul['ERR']
          err = err_file.data
          err_header = err_file.header
     # Check the names of the image and error extensions 
     print(f"Image file: {img_file}")
     print(f"Error file: {err_file}")

     # Handle NaNs and zeros
     snr_map = np.full_like(img, np.nan)
     valid = (np.isfinite(img)) & (np.isfinite(err)) & (err > 0)
     snr_map[valid] = img[valid] / err[valid]

     # Coverage mask
     if get_coverage:
          coverage_mask = (~np.isfinite(img)) | (img == 0) | (err == 0)
     else:
          coverage_mask = None

     return img, err, snr_map, coverage_mask, header


def match(
    catalog1, 
    catalog2, 
    npix=2, 
    keys=['catalog1', 'catalog2']):
    
    coords2 = np.array([catalog2['xcenter'], catalog2['ycenter']]).T
    coords1 = np.array([catalog1['xcenter'], catalog1['ycenter']]).T
    # Build a KD-tree for the first catalog
    tree = cKDTree(coords2)

    # Find matches within npix pixels
    # pixel_scale = 1#0.031
    max_distance = npix  # or e.g. 1.0 pixel if you want 1 pixel tolerance

    distances, indices = tree.query(coords1, k=1, distance_upper_bound=max_distance)

    # Create mask for valid matches (finite distance = match found)
    match_mask = np.isfinite(distances)

    # Build matched catalog
    matched1 = catalog1[match_mask]
    matched2 = catalog2[indices[match_mask]]

    # Optionally combine columns from both catalogs
    matched_cat = hstack([matched1, matched2], table_names=[keys[0],keys[1]])
    return matched_cat


# ------------------------------------------------
# Background subtraction
# ------------------------------------------------
def subtract_bkg(img, 
          box_size_pix=50, 
          filter_size_pix=3, 
          bkg_estimator=MedianBackground(), 
          coverage_mask=False,
          plot=False,
          sigma_to_clip_bkg=3.0,
          maxiters_for_bkg_clip=5,
          **kwargs):
     """Estimate and subtract background from image using Background2D.
     Args:
          img: 2D array of image data
          box_size_pix: size of boxes for background estimation (in pixels)
          filter_size_pix: size of median filter to apply to background (in pixels)
          bkg_estimator: background estimator to use (default is MedianBackground())
          coverage_mask: boolean array where True indicates pixels to exclude from background estimation (e.g., low coverage or bad data).
     Returns:
          img_sub: background-subtracted image
          bkg_mean: mean background level (in same units as img)
          bkg_rms: background RMS (in same units as img)"""
     
     # estimate background
     # TODO: need to include valid mask based on weight image or other metric
     sigma_clip = SigmaClip(sigma=sigma_to_clip_bkg, maxiters=maxiters_for_bkg_clip)

     # Check if box size is even. If it is, add one to each of the values
     if type(box_size_pix) != type([]):
          box_size_pix = (box_size_pix, box_size_pix)
     if box_size_pix[0] % 2 == 0:
          box_size_pix = (box_size_pix[0] + 1, box_size_pix[1] + 1)

     if type(filter_size_pix) != type([]):
          filter_size_pix = (filter_size_pix + 1, filter_size_pix + 1)
     if filter_size_pix[0] % 2 == 0:
          filter_size_pix = filter_size_pix + 1

     bkg_estimator = eval(bkg_estimator) if isinstance(bkg_estimator, str) else bkg_estimator

    
     if coverage_mask is False:
          print(f"Creating coverage mask")
          coverage_mask = (~np.isfinite(img)) | (img == 0)

     # note: photutils<3.0 used edge_method='pad' by default, but photutils>=3.0 does not have this option and instead pads with fill_value=0.0 by default.
     bkg = Background2D(
          img,
          box_size=box_size_pix,
          filter_size=filter_size_pix,
          sigma_clip=sigma_clip,
          bkg_estimator=bkg_estimator,
          coverage_mask=coverage_mask,
          )

     rms_map = np.array(bkg.background_rms, dtype=float)
     valid_rms = (~coverage_mask) & np.isfinite(rms_map) & (rms_map > 0)

     #print(f"bkg array {bkg.background}")
     bkg_rms = np.nanmedian(rms_map[valid_rms]) if np.any(valid_rms) else np.nan
     bkg_mean = np.nanmean(np.asarray(bkg.background, dtype=float)[~coverage_mask])
     print(f"Mean background: {bkg_mean}")
     print(f"Background rms: {bkg_rms}")
     print(f"Subtracting background...")
     

     # threshold_img = snr_threshold * bkg.background_rms
     img_sub = img - bkg.background
     print(f"Background subtraction complete.")

     if plot:
          # Plot the image, background, and background-subtracted image
          fig, ax = plt.subplots(1, 3, figsize=(18, 6))
          norm = ImageNormalize(vmin=np.nanpercentile(img, 25.00), 
                                vmax=np.nanpercentile(img, 99.99), 
                                stretch=LogStretch())
          ax[0].imshow(img, origin='lower', cmap='inferno', norm=norm)
          ax[0].set_title(f"{gal.upper()} {band.upper()} mosaic")
          # TODO: gal and band as global properties
          ax[1].imshow(bkg.background, origin='lower', cmap='inferno')
          ax[1].set_title(f"Estimated background")
          norm_sub = ImageNormalize(vmin=np.nanpercentile(img_sub, 25.00), 
                                    vmax=np.nanpercentile(img_sub, 99.99), 
                                    stretch=LogStretch())
          ax[2].imshow(img_sub, origin='lower', cmap='inferno', norm=norm_sub)
          ax[2].set_title(f"Background-subtracted image")
          # Add colourbars
          for a in ax:
               im = a.images[0]
               plt.colorbar(im, ax=a, pad=0.01, fraction=0.05)

     return img_sub, bkg_mean, bkg_rms, bkg


# ------------------------------------------------
# Source finding (using IRAF, DAO in progress)
# ------------------------------------------------
def run_source_finder(img, 
          header, 
          bkg,
          finder='iraf', 
          snr_threshold=3.0, 
          fwhm=2.0, 
          box_size_pix=(5,5),  # TODO reconcile RH's value of 50 with JR's value of 3 here
          roundlo=-0.5, 
          roundhi=0.5, 
          sharplo=0.2, 
          sharphi=1.0, 
          nsources=10000,
          cat_path='./',
          cat_filename=None,
     ):
     """Find sources in the image using IRAFStarFinder.
     Args:
          img: 2D array of background-subtracted image data
          header: FITS header of the image (used for WCS and pixel scale)
          finder: source finder to use (currently only 'iraf' supported)
          snr_threshold: signal-to-noise ratio threshold for source detection
          fwhm: FWHM of the PSF in pixels (used for source detection)
          roundlo, roundhi: roundness limits for source selection
          sharplo, sharphi: sharpness limits for source selection
          nsources: if not None, only return this many brightest sources in the catalog
     Returns:
          sources: Table of detected sources with columns x_centroid, y_centroid, flux, sharpness, roundness, mag, peak, etc."""
     # Run the source finder
     print(f"Running source finder: {finder}")
     
     # Get the threshold image from the background calculation
     ths = snr_threshold * bkg.background_rms

     # Add option to import an external source catalog
     # instead of running a source finder. Useful for testing 
     # without regen and matching cats with HST/MUSE

     # IRAFStarFinder 
     if finder == 'iraf':
          source_finder = IRAFStarFinder(threshold=ths,
               fwhm=fwhm,
               roundness_range=(roundlo, roundhi),
               sharpness_range=(sharplo, sharphi),
               n_brightest=nsources,
          )
          # Run the source finder
          sources = source_finder(img)

     # DAOStarFinder (can use elliptical apertures)
     elif finder == 'dao':
          source_finder = DAOStarFinder(threshold=ths,
               fwhm=fwhm,
               roundness_range=(roundlo, roundhi),
               sharpness_range=(sharplo, sharphi),
               n_brightest=nsources,
          )
          # Run the source finder
          sources = source_finder(img)

     elif finder == 'peaks':
          # find_peaks looks for local maxima above a specified threshold.
          # Requires a bit of extra work to get results in the same format as IRAFStarFinder/DAOStarFinder, 
          # and it doesn't calculate sharpness or roundness.
          # TODO: Add function converting find_peaks output to a table with xcentroid, ycentroid, flux, etc.

          # JR estimated a threshold from photutils with detect_threshold, which adds the background to a 
          # threshold map.  Instead, here we assume the background has already been subtracted and use a flat ths.

          # TODO: reconcile JR box_size_pix=3 with RH 50 

          # Check if box size is even. If it is, add one to each of the values
          if box_size_pix[0] % 2 == 0:
               box_size_pix = (box_size_pix[0] + 1, box_size_pix[1] + 1)
          sources = find_peaks(img, 
               threshold=ths, 
               box_size=box_size_pix,
               centroid_func=centroid_quadratic,
          )
          # For sources where the centroid could not be determined,
          # use the position of the peak instead.
          #----------------------------------------
          z=np.where(np.isnan(sources['x_centroid']))[0]
          if len(z)>0:
               sources['x_centroid'][z]=sources['x_peak'][z]        
               sources['y_centroid'][z]=sources['y_peak'][z]

          # TODO not sure if the difference betwen x_centroid and xcentroid is used outside of this function
          sources['xcentroid']=sources['x_centroid']        
          sources['ycentroid']=sources['y_centroid']

     elif finder != 'iraf' and finder != 'dao' and finder != 'peaks':
          raise ValueError(f"Starfinder {finder} not recognized. Currently only 'iraf' and 'peaks' are supported.")

     #convert from x,y in the image to  sky coordinates   
     #----------------------------------------
     sk = wcs.utils.pixel_to_skycoord(sources['xcentroid'], sources['ycentroid'], wcs=WCS(header)) 
     sources['ra']=sk.ra
     sources['dec']=sk.dec

     #convert from x,y in the image to  sky coordinates   
     #----------------------------------------
     sk = wcs.utils.pixel_to_skycoord(sources['xcentroid'], sources['ycentroid'], wcs=WCS(header)) 
     sources['ra']=sk.ra
     sources['dec']=sk.dec

     print(f"Found {len(sources)} sources")
     print(sources.colnames)
     return sources


# TODO: these things
def load_source_catalog():
     print("Load an external source catalog to use for photometry.")


def filter_catalog():
     print("Filtering catalog based on morphology and other criteria.")


# ------------------------------------------------
# Optimal aperture and photometry
# ------------------------------------------------
def get_optimal_aperture(data, sources, max_r=32, brightest=50, frac=0.95, plot=True):
     """Find the optimal aperture radius to use for the photometry from the 
        curve of growth of the brightest n sources. 
     Args:
          data: 2D array of image data (background-subtracted)
          sources: Table of sources from source finder 
                   (must contain x_centroid, y_centroid, flux)
          max_r: maximum aperture radius to test (in pixels)
          brightest: if not None, only use this many brightest sources to compute curve of growth
          frac: fraction of total flux to use as criterion for optimal radius 
          (e.g., 0.95 means radius where median curve of growth reaches 95% of total flux)
          plot: if True, plot the curve of growth and optimal radius
          
     Returns:
          r_opt: optimal aperture radius in pixels (to use with compute_photometry)"""
     
     # Select only the brightest sources to compute the curve of growth
     if brightest is not None:
          sources = sources[np.argsort(sources['flux'])[-brightest:]]
          print(f"Using only {len(sources)} sources.")

     print("Calculating optimal aperture...")
     positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
     radii = np.arange(1, max_r)

     # Define in and outer annuli for local background estimation
     # TODO: optimize values for the sky annulus 
     ann_in, ann_out = max_r + 2, max_r + 8
     ann = CircularAnnulus(positions, r_in=ann_in, r_out=ann_out)

     # Get local backgrounds
     # TODO: consider impact of extended emission on local background. 
     ann_phot = aperture_photometry(data, ann)
     bkg_mean = np.asarray(ann_phot["aperture_sum"]) / ann.area

     # At each radius, compute photometry
     fluxes = []
     for r in radii:
          ap = CircularAperture(positions, r=r)
          phot = aperture_photometry(data, ap)

          # Subtract local background
          src = np.asarray(phot["aperture_sum"]) - bkg_mean * ap.area
          fluxes.append(src)

     # Normalize fluxes for computing the curve of growth
     fluxes = np.asarray(fluxes).T

     norm = fluxes / fluxes[:, [-1]]
     norm[~np.isfinite(norm)] = np.nan

     # comptue median normalized flux
     median_curve = np.nanmedian(norm, axis=0)  
     # Get the index of the radius where the curve of growth reaches the specified fraction of total flux
     idx = np.where(median_curve >= frac)[0]
     r_opt = radii[idx[0]] if len(idx) else radii[np.nanargmax(median_curve)]
     print(f"Optimal aperture radius: {r_opt}")

     if plot:
          plt.figure()
          plt.plot(radii, median_curve, marker='o')
          plt.axvline(r_opt, color='red')
          plt.xlabel("Aperture radius (pixels)")
          plt.ylabel("Normalized flux")
          plt.title("Curve of growth")
          plt.grid(True)

     return r_opt


# ------- Main photometry function ------------------------------------------------ 
def compute_photometry(data, 
          err,
          header, 
          sources, 
          gal, 
          band,
          radius=10, 
          radius_sky_in=12, 
          radius_sky_out=18, 
          use_brightest=False, 
          sigma_to_clip_bkg=3.0,
          maxiters_for_bkg_clip=5,
          phot_method='exact',
          write=False, 
          overwrite=False,
          apcorr_step=True, 
          local_bkg_subtract=True,
          out_dir='./',
          cat_filetype="fits"):
     """Compute aperture photometry for sources and return catalog with RA, Dec, magnitudes, etc.
     
     Args:
          data: 2D array of image data (background-subtracted)
          err: 2D array of error data (same shape as data)
          header: FITS header of the image
          sources: Table of sources from source finder 
                   (must contain x_centroid, y_centroid, flux, sharpness, roundness, mag, peak)
          aperture_radius: radius of circular aperture to use for photometry (in pixels)
          radius_sky_in: inner radius of the sky annulus (in pixels)
          radius_sky_out: outer radius of the sky annulus (in pixels)
          phot_method: method to use for photometry (e.g., 'exact', 'subpixel', etc.)
          use_brightest: if True, only use the brightest sources for photometry
          write: if True, write catalog to out_dir with name {gal}_jwst_{band}_cat.fits
          out_dir: directory to write catalog if write=True
          cat_filetype: anything that astropy table recognizes - csv, fits

     Returns:
          phot_full: Table with photometry results, including RA, Dec, aperture sum, magnitudes, etc.
     """

     if use_brightest is not False:
          # Aperture photometry of only brightest sources
          sources = sources[np.argsort(sources['flux'])[-use_brightest:]]
          print(f"using only {len(sources)} sources")

     if apcorr_step:
          # Get aperture correction parameters from CRDS file
          radius, radius_sky_in, radius_sky_out, apcorr = get_apcorr_params(crds_dir, band, inst='NIRCam', **conf['parameters']['apcorr'])
          print(f"Using aperture correction factor of {apcorr} for radius {radius} pixels.")

     # Do aperture photometry
     print(f"Doing aperture photometry...")
     positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
     apertures = CircularAperture(positions, r=radius)
     aper_stats = ApertureStats(data, apertures, error=err)
     phot_full = aperture_photometry(data, apertures, error=err, method=phot_method)

     # Annulus
     annuli = CircularAnnulus(positions, r_in=radius_sky_in, r_out=radius_sky_out)
     sigma_clip_bkg = SigmaClip(sigma=sigma_to_clip_bkg, maxiters=maxiters_for_bkg_clip)
     # mask = annuli.to_mask(method='exact')
     # Mask the data to exclude NaNs and infs from the background estimation
     mask = ((np.isinf(data)) | (np.isnan(data)))

     # Background annulus stats
     bkg_stats = ApertureStats(data, annuli, sigma_clip=sigma_clip_bkg, mask=mask, sum_method=phot_method)
     bkg_median = bkg_stats.median
     bkg_median[np.isnan(bkg_median)]=0
     area_aper = aper_stats.sum_aper_area.value
     # area_annulus = bkg_stats.sum_aper_area.value
     total_bkg = bkg_median * area_aper

     # Errors on the background estimates
     bkg_err = bkg_stats.std * aper_stats.sum_aper_area.value
     bkg_err_scalefactor = np.sqrt(0.5*np.pi / bkg_stats.sum_aper_area.value)  # scale factor for background error based on area of annulus and aperture

     # Subtract background from aperture sum
     if local_bkg_subtract:
          phot_full['aperture_sum_mjysr'] = phot_full['aperture_sum'] - total_bkg
          phot_full['aperture_sum_mjy'] = phot_full['aperture_sum_mjysr'] * get_pixarea_in_sr(header)
     else:
          phot_full['aperture_sum_mjysr'] = phot_full['aperture_sum']
          phot_full['aperture_sum_mjy'] = phot_full['aperture_sum'] * get_pixarea_in_sr(header)

     # Copy source-finder morphology columns
     if 'flux' in sources.colnames:  # it won't be there for findpeaks method.  TODO could be added in find step
          phot_full['finder_flux'] = np.asarray(sources['flux'])
     if 'sharpness' in sources.colnames:
          phot_full['sharpness'] = np.asarray(sources['sharpness'])
     if 'roundness' in sources.colnames:
          phot_full['roundness'] = np.asarray(sources['roundness'])          
     if 'mag' in sources.colnames:
          phot_full['finder_mag'] = np.asarray(sources['mag'])
     if 'peak' in sources.colnames:
          phot_full['peak'] = np.asarray(sources['peak'])
     elif 'peak_value' in sources.colnames: 
          phot_full['peak'] = np.asarray(sources['peak_value'])   # TODO change peakfinder output to have peak instead of peak_value

     # Include ra, dec
     wcs = WCS(header)
     ra, dec = wcs.all_pix2world(phot_full["xcenter"], phot_full["ycenter"], 0)
     phot_full["ra"] = ra
     phot_full["dec"] = dec

     # Convert flux from the source finder in table (converted to AB magnitudes)
     if 'finder_flux' in phot_full.colnames:
          phot_full['finder_flux_abmag'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['finder_flux'], header=header)
     # Aperture sum from circular aperture photometry (converted to AB magnitudes)
     phot_full['aperture_sum_abmag'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['aperture_sum'], header=header)

     if apcorr_step:
          if apcorr.unit.is_equivalent(u.dimensionless_unscaled):
               phot_full['aperture_sum_abmag_apcorr'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['aperture_sum'] * apcorr, header=header)
          elif apcorr.unit.is_equivalent(u.mag):
               phot_full['aperture_sum_abmag_apcorr'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['aperture_sum'], header=header) + apcorr.value
     
     # Add the errors
     phot_full['bkg_err'] = np.asarray(bkg_err)

     # TODO: Is there a better way to do this than a list?
     if band.lower()=='f335m' or band.lower()=='f770w' or band.lower()=='f1000w' or band.lower()=='f1130w' or band.lower()=='f2100w':
          phot_full['total_aperture_sum_err'] = np.sqrt(phot_full['aperture_sum_err']**2 + bkg_err**2)
     elif band.lower()=='f300m' or band.lower()=='f360m' or band.lower()=='f444w':
          phot_full['total_aperture_sum_err'] = np.sqrt(phot_full['aperture_sum_err']**2 + bkg_err**2 * bkg_err_scalefactor**2)
     else:
          print(f"Band {band} not recognized for error calculation. Setting total_aperture_sum_err to sqrt(aperture_sum_err**2 + bkg_err**2).")
          phot_full['total_aperture_sum_err'] = np.sqrt(phot_full['aperture_sum_err']**2 + bkg_err**2)

     # Sort by aperture flux
     phot_full.sort("aperture_sum")
     phot_full.reverse()

     # Print the column names of the photometry table
     print(phot_full.colnames)

     # Write the catalog if requested
     if write:
          cat_name = f"{gal}_jwst_{band}_cat_cluster_apcorr." + cat_filetype
          print(f"Writing catalog to {out_dir + cat_name}")
          phot_full.write(out_dir + cat_name, overwrite=overwrite)

     return apertures, phot_full


def bkg_error_quantiles(data, annulus_masks, sigma=3.0):
     """Compute local background statistics in the annulus around each source.
     Args:
          data: 2D array of image data (background-subtracted)
          annulus_masks: list of masks for each annulus (from annuli.to_mask())
          sigma_clip: SigmaClip object for sigma clipping the background pixels (optional)
          mask: boolean array where True indicates pixels to exclude from background estimation (e.g., low coverage or bad data)
     Returns:
     """
     
     bkg_10 = np.zeros(len(annulus_masks))
     bkg_90 = np.zeros(len(annulus_masks))
     bkg_10_clip = np.zeros(len(annulus_masks))
     bkg_90_clip = np.zeros(len(annulus_masks))
     npix_annulus = np.zeros(len(annulus_masks))
     npix_annulus_clipped = np.zeros(len(annulus_masks))
     sigma_clip = SigmaClip(sigma=sigma) if sigma is not None else None

     for i, m in enumerate(annulus_masks):
          annulus_data = m.multiply(data)

          if annulus_data is not None:
               # Flatten and remove zeros, NaNs, and infs
               annulus_data_1d = annulus_data[(annulus_data != 0) & np.isfinite(annulus_data) & ~np.isnan(annulus_data)] 

               if len(annulus_data_1d) > 0:
                    annulus_data_filtered = sigma_clip(annulus_data_1d) if sigma_clip is not None else annulus_data_1d
                    bkg_low, bkg_hi = np.quantile(annulus_data_1d, [0.1,0.9])
                    bkg_low_clip, bkg_hi_clip = np.quantile(annulus_data_filtered, [0.1,0.9])
                    # Update results
                    bkg_10[i] = bkg_low
                    bkg_90[i] = bkg_hi
                    bkg_10_clip[i] = bkg_low_clip
                    bkg_90_clip[i] = bkg_hi_clip
                    npix_annulus[i] = len(annulus_data_1d)
                    npix_annulus_clipped[i] = len(annulus_data_filtered)
          else:
               continue
     return bkg_10, bkg_90, bkg_10_clip, bkg_90_clip, npix_annulus, npix_annulus_clipped
                    


def get_apcorr_params(crds_dir, band, inst, eefraction_value=0.8, apcorr_method='crds'):
     """Get the aperture correction parameters from the CRDS apcorr file for a given filter and eefraction.
     Args:
          crds_dir: directory where the CRDS apcorr files are stored
          band: the filter
          eefraction_value: the fraction of total flux enclosed within the aperture.
     Returns:
          radius: the aperture radius in pixels
          sky_in: the inner sky annulus radius in pixels
          sky_out: the outer sky annulus radius in pixels
          apcorr: the aperture correction factor
     """
     # TODO: add option to use multiple methods, each of which ends up with its own column. 

     # Aperture correction for point sources based on the encircled energy fraction
     # from the crds calibration files. Multiply the flux by apcorr.
     if apcorr_method == 'crds':

          print(f"Getting parameters from CRDS for {band} with eefraction {eefraction_value}...")
          apcorr_files = glob.glob(crds_dir + f"*apcorr*")

          # Check that the files exist
          if len(apcorr_files) == 0:
               raise FileNotFoundError(f"No apcorr files found for {band} at {crds_dir}")
          else:
               print(f"Found apcorr files: {apcorr_files} in {crds_dir}")

          # Load the most recent apcorr file (final in the list)
          apcorr_data = fits.getdata(apcorr_files[-1], ext=1)

          # Get data for a specified eefraction and filter
          row = apcorr_data[apcorr_data['eefraction'] == eefraction_value]
          row = row[(row['filter'] == band.upper())]
          # Extract values
          radius  = row['radius'][0]   # in pixels
          sky_in  = row['skyin'][0]    # in pixels
          sky_out = row['skyout'][0]   # in pixels
          apcorr  = row['apcorr'][0] * u.dimensionless_unscaled  # factor to multiply enclosed flux to get total flux

     # Aperture correction using factors derived in Rodriguez et al. 2025, based on Deger et al. (2022).
     elif apcorr_method == 'cluster':

          print(f"Using aperture correction values from Rodriguez et al. 2025")

          # Load parameters from apcorr_rodriguez.ecsv
          if not os.path.exists('apcorr_rodriguez.ecsv'):
               raise FileNotFoundError("apcorr_rodriguez.ecsv not found. Please make sure the file is in the current directory.")
          
          if inst.lower() == 'nircam':
               apcorr_val = Table.read('apcorr_rodriguez.ecsv', format='ascii.ecsv')
               if band.lower() not in apcorr_val['band'].data:
                    raise ValueError(
                         f"Filter {band} not found in Rodriguez et al. 2025 apcorr file."
                         "\n Available filters: {apcorr_val['band'].data}"
                         "\n Please chose a different aperture correction method in the config file [i.e. 'crds']."
               )
               
               # TODO: get the pixel scale properly from header info
               # These correction factors are only valid for a specific radius.
               # pixel_scale = 0.031
               radius = 4 #* pixel_scale
               an_in = 2.
               an_out = 3. 
               sky_in = an_in * radius
               sky_out = an_out * radius
               apcorr_vega_mag = apcorr_val[apcorr_val['band'] == band.lower()]['apcorr']
               
               # Get the zero point from SVO
               filter_info = SvoFps.get_filter_list(facility='JWST', instrument=inst)
               zero_point_vega = filter_info[filter_info['filterID'] == f'JWST/{inst}.{band.upper()}']['ZeroPoint']
               # delta_mag = - 2.5*np.log10(zero_point_vega/3631.0)
               # Because the aperture correction is a flux ratio, mag system doesn't matter
               apcorr_abmag = apcorr_vega_mag
               apcorr = apcorr_abmag * u.mag

          elif inst.lower() == 'miri':
               print("Cluster aperture corrections not computed for MIRI. Use eefraction = 0.5 (50%) with CRDS.")
               apcorr_files = glob.glob(crds_dir + f"*apcorr*")

               # Check that the files exist
               if len(apcorr_files) == 0:
                    raise FileNotFoundError(f"No apcorr files found for {band} at {crds_dir}")
               else:
                    print(f"Found apcorr files: {apcorr_files} in {crds_dir}")

               # Load the most recent apcorr file (final in the list)
               apcorr_data = fits.getdata(apcorr_files[-1], ext=1)

               # Get data for a specified eefraction and filter
               row = apcorr_data[apcorr_data['eefraction'] == 0.5]
               row = row[(row['filter'] == band.upper())]
               # Extract values
               radius  = row['radius'][0]   # in pixels
               sky_in  = row['skyin'][0]    # in pixels
               sky_out = row['skyout'][0]   # in pixels
               apcorr  = row['apcorr'][0] * u.dimensionless_unscaled 


     # TODO: add method for correction based on curve of growth. 

     # If nothing is recognised, use a simplified approximation. 
     else:
          print(f"Using default (basic) aperture correction parameters for {band} with eefraction {eefraction_value}...")
          radius = filter_fwhm.get(band.upper(), 2.0)  # default to 2 pixels if filter not found
          sky_in = radius + 2
          sky_out = radius + 8
          apcorr = (1.0 / eefraction_value) * u.dimensionless_unscaled # simple correction factor based on eefraction

     return radius, sky_in, sky_out, apcorr

# ------------------------------------------------
# Other useful functions
# ------------------------------------------------

# Load in the catalogs that are produced by the image3pipeline
def get_image3_catalog(filedir, filter, galaxy, level='lv3'):
    cat_dir = filedir
    # cat_dir = dir + f"{galaxy}/{filter.upper()}/{level}"
    cat_filename = f"{galaxy}_nircam_{level}_{filter.lower()}_cat_align.ecsv"
    cat_name = cat_dir + "/" + cat_filename
    return cat_name


# Cross match the catalog that we have made with the outputs of the image3pipeline
def cross_match_catalogs(dir, filter, galaxy, phot_full, cat_image3):
    cat_name = get_image3_catalog(dir, filter, galaxy=galaxy)
    calib_cat = Table.read(cat_name, format='ascii.ecsv')

    # Use proximity based approach to cross match the catalogs
    calib_coords = SkyCoord(ra=calib_cat['ra'] * u.deg, dec=calib_cat['dec'] * u.deg)
    # My photometry into Sky Coords
    phot_coords = SkyCoord(ra=phot_full['ra'] * u.deg, dec=phot_full['dec'] * u.deg)
    # Match coordinates
    ind_2d_cat, dist_2d, _ = match_coordinates_sky(phot_coords, calib_coords)
    return ind_2d_cat, dist_2d, phot_full



# Empirical filter FWHM
# NIRCAM from https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions#gsc.tab=0
# MIRI from https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-performance/miri-point-spread-functions#gsc.tab=0
filter_fwhm = {
    'F150W': 1.613,
    'F164N': 1.806,
    'F187N': 2.065,
    'F200W': 2.129,
    'F212N': 2.323,
    'F277W': 1.460,
    'F300M': 1.587,
    'F335M': 1.762,
    'F360M': 1.905,
    'F405N': 2.159,
    'F444W': 2.302,
    'F770W': 2.445,
    'F1000W': 2.982,
    'F1130W': 3.409,
    'F2100W': 6.127,
}

# Directories
jwst_dir = local['jwst_dir']
out_dir = local['out_dir']
crds_dir = local['crds_dir']
cat_path = local['out_dir']  
# TODO: add cat_path to local.toml if we want to load in an external catalog for photometry instead of running a source finder.

print("-----------------------------------------")
print(f"JWST data directory: {jwst_dir}")
print(f"Output directory: {out_dir}")
print("-----------------------------------------")

# Check that input data directory exists
if not os.path.exists(jwst_dir):
     raise FileNotFoundError(f"JWST data directory {jwst_dir} does not exist. Please check the path in the config file.")
     exit()

# Check that out_dir exists
if not os.path.exists(out_dir):
     raise FileNotFoundError(f"Output directory {out_dir} does not exist.")
     exit()


# This is only still here temporarily
use_filter_fwhm = True 

def do_photometry(
          steps, 
          targets,
          use_filter_fwhm,
          conf,
     ):
     """Main function to run the photometry steps for each galaxy and filter.
     Args:
          steps: list of steps to run (e.g., ['bkg_subtract', 'subtract_bkg', 'source_find', 'r_opt', 'aperture_photometry'])
          targets: list of galaxy names to process
          use_filter_fwhm: this will eventually go into the config
          conf: dictionary of parameters from the config file."""

     print(" ")
     catalogs = {}

     for gal in targets:
          # Get the path to the data
          path = get_path_to_file(
               wdir=jwst_dir, 
               version=version, 
               project=projects[0], 
               galaxy=targets[0],
               ptype=ptype[0],
               filter=conf['bands'][0])

          # Now loop through the filters for this galaxy
          for band in bands:
               print(f"Processing {gal} at {band}...")
               # Initialise catalogs dict to store the photometry results for each galaxy and filter
               if gal not in catalogs:
                    catalogs[gal] = {}
               if band not in catalogs[gal]:
                    catalogs[gal][band] = {}

               # Open the JWST data file 
               img, err, snr_map, coverage_mask, header = open_jwst(
                    mosaic_ext = conf['product'],
                    path = path, 
                    gal = gal, 
                    dir = jwst_dir, 
                    band = band
               )

               # TODO get distance from the galaxy sample table intead of the config file
               # Subtract background 
               if 'subtract_bkg' in steps:
                    if 'box_size_pix' not in conf['parameters']['bkg_subtract']:
                         # Convert box size from pc to pixels using the pixel scale from the header
                         pix_scale = get_pixarea_in_sr(header) ** 0.5 * (180/np.pi) * 3600  # arcsec/pixel
                         box_size_pc = conf['parameters']['bkg_subtract']['box_size_pc']
                         box_size_pix = int(box_size_pc * 206265 / (pix_scale * conf['parameters']['bkg_subtract']['dist_Mpc'] * 1e6 ))
                         conf['parameters']['bkg_subtract']['box_size_pix'] = box_size_pix

                    if 'filter_size_pix' not in conf['parameters']['bkg_subtract']:
                         # Convert filter size from pc to pixels using the pixel scale from the header
                         pix_scale = get_pixarea_in_sr(header) ** 0.5 * (180/np.pi) * 3600  # arcsec/pixel
                         filter_size_pc = conf['parameters']['bkg_subtract']['filter_size_pc']
                         filter_size_pix = int(filter_size_pc * 206265/ (pix_scale * conf['parameters']['bkg_subtract']['dist_Mpc'] * 1e6 ))
                         conf['parameters']['bkg_subtract']['filter_size_pix'] = filter_size_pix

                    img_sub, bkg_mean, bkg_rms, bkg_background = subtract_bkg(
                         img=img, 
                         **conf['parameters']['bkg_subtract'],
                    )

               if 'source_find' in steps:
                    # Get sources using the source finder
                    if 'subtract_bkg' in steps:
                         use_image = img_sub
                    else:
                         use_image = img

                    sources = run_source_finder(
                         img=use_image, 
                         header=header, 
                         bkg=bkg_background, 
                         cat_path=cat_path,
                         **conf['parameters']['source_find'],
                    )
               
               else:
                    print("Importing sources from external catalog...")
                    # Load the external source catalog
                    sources = Table.read(cat_path + cat_filename)

                    # Checks that the colnames include x_centroid, y_centroid, flux, sharpness, roundness, mag, peak, etc. 
                    # and print a warning if any are missing
                    required_cols = ['xcentroid', 'ycentroid', 'flux']
                    x_to_search_for = ['xcentroid', 'x_center', 'x_centroid', 'xcenter']
                    y_to_search_for = ['ycentroid', 'y_center', 'y_centroid', 'ycenter']
                    # Cycle through
                    for col in required_cols:
                         if col not in sources.colnames:
                              # Check whether there is a xcenter and ycenter column instead of x_centroid and y_centroid, and if so, rename them
                              if col == 'xcentroid' and 'x_center' in sources.colnames:
                                   sources['xcentroid'] = sources['x_center']
                              elif col == 'ycentroid' and 'y_center' in sources.colnames:
                                   sources['ycentroid'] = sources['y_center']
                              else:
                                   print(f"Warning: Column '{col}' is missing from the external catalog."
                                        f"\n Please make sure the catalog has the required columns: {required_cols}.")

               # **** Alternatively, load in a catalog computed by another method here ****
               # TODO: if loading in another catalog, need a path to it in local.toml. 

               # Either get the optimum radius based on curve of growth...
               if 'r_opt' in steps:
                    print(f"Computing optimal aperture for photometry...")
                    r_opt = get_optimal_aperture(
                         data = img_sub, 
                         sources = sources,
                         **conf['parameters']['r_opt']
                    )
               else:
                    r_opt = filter_fwhm[band.upper()]*2.5 if use_filter_fwhm else conf['parameters']['photometry']['aperture_radius']
                    print(f"Using fixed aperture radius of {r_opt} pixels for photometry.")

               # Update the fwhm according to the filter if use_filter_fwhm is True.
               # If use_filter_fwhm is False, stay at specified value.
               if use_filter_fwhm:
                    try:
                         fwhm = filter_fwhm[band.upper()]
                         print(f"Using FWHM of {fwhm} pixels for source detection based on JWST PSF for {band.upper()}.")
                    except KeyError:
                         print(f"Warning: FWHM for {band.upper()} not found in filter_fwhm dictionary. Using default FWHM of {fwhm} pixels for source detection.")

               # TODO: is there a better way of doing this?
               if "apcorr_step" in steps:
                    apcorr_step = True
               else:
                    apcorr_step = False

               # # ...or just set it to a fixed value (e.g., based on the PSF FWHM)
               # print(f"Setting aperture radius to {r_opt} pixels.")
               # # Check r_opt relative to the FWHM of the filter:
               # if r_opt > 3 * fwhm:
               #      print("Large r_opt. Using PSF FWHM rather than curve of growth for photometry.")
               #      r_opt = 2.5 * fwhm

               # Perform photometry with circular apertures
               if 'aperture_photometry' in steps:
                    print(f"Performing photometry on {len(sources)} sources with aperture radius of {r_opt} pixels.")
                    apertures, catalog = compute_photometry(
                         data = img_sub, 
                         err = err,
                         header = header, 
                         gal = gal, 
                         band = band,
                         radius = r_opt,
                         sources = sources,
                         apcorr_step = apcorr_step,
                         out_dir = local['out_dir'],
                         **conf['parameters']['photometry']
                    )

                    print(f"Photometry complete. Catalog has {len(catalog)} sources.")

                    # Store the catalog in the catalogs dict
                    print(catalog)
                    print(catalog.colnames)
                    catalogs[gal][band] = catalog

     return catalogs


catalogs = do_photometry(
               steps=steps, 
               targets=targets, 
               use_filter_fwhm=use_filter_fwhm,
               conf=conf
          )


exit()


# ---------------------------------------------------------------------------------------------------------

# Alternative approach using standardised aperture corrections from the JWST CRDS.
path_to_crds = "/nexus/posix0/MIA-astro-env/eschinner/jgonzalez/jwst_pipeline/crds_cache/jwst_ops/references/jwst/" + 'nircam' + "/"

# Get the apcorr file using glob
apcorr_files = glob.glob(path_to_crds + f"*apcorr*")
if len(apcorr_files) == 0:
    raise FileNotFoundError(f"No apcorr files found for {band} in {inst} at {path_to_crds}")
else:
    print(f"Found apcorr files: {apcorr_files}")

# Load the file
apcorr_data = fits.getdata(apcorr_files[0], ext=1)
print(f"APCORR data columns: {apcorr_data.columns.names}")

# Print all the unique filters in the apcorr file
print("Unique eefraction values:", np.unique(apcorr_data['eefraction']))

# Get data for a specific eefraction
# The eefraction is the fraction of the total flux that is enclosed within the aperture radius.
eefraction_value = 0.70
row = apcorr_data[apcorr_data['eefraction'] == eefraction_value]

# Limit to a specific filter 
row = row[(row['filter'] == band.upper())]

# Extract values
wcs_apcorr = WCS(header)
radius = row['radius'][0]   # in pixels
sky_in = row['skyin'][0]    # in pixels
sky_out = row['skyout'][0]  # in pixels
apcorr = row['apcorr'][0]   # factor to multiply enclosed flux to get total flux
print(f"Using aperture correction factor of {apcorr} for radius {radius} pixels and eefraction {eefraction_value}")

# Create apertures for aperture correction, not using the curve of growth
positions = np.transpose((sources['x_centroid'], sources['y_centroid']))

# Redo the aperture photometry using the radius, sky_in, and sky_out from the apcorr file
aperture = CircularAperture(positions, r=radius)
sky_annulus = CircularAnnulus(positions, r_in=sky_in, r_out=sky_out)
phot_table_apcorr = aperture_photometry(img_sub, aperture, wcs=wcs_apcorr, method=phot_method)
sky_table_apcorr = aperture_photometry(img_sub, sky_annulus, wcs=wcs_apcorr, method=phot_method)

# Extract flux and sky
fluxes = phot_table_apcorr['aperture_sum'].value  # MJy/sr
sky_mean = sky_table_apcorr['aperture_sum'].value / sky_annulus.area  
sky_total = sky_mean * aperture.area  

# Correct for sky
net_fluxes = fluxes - sky_total  

# Apply aperture correction 
total_fluxes = net_fluxes * apcorr  

# Convert to AB magnitudes using the pixel area in steradians from the header
pixarea_sr = header['PIXAR_SR']  # in steradians
# total_flux_jy = total_fluxes * 1e6  # MJy → Jy
abmag_apcorr = convert_aperture_sum_Jy_per_sr_to_abmag(total_fluxes, header=header)

# Add column to the phot_table_apcorr with the aperture-corrected AB magnitudes
phot_table_apcorr['ABmag_apcorr'] = abmag_apcorr

# Create merged table on the x and y coordinates of the sources to compare 
# the aperture-corrected magnitudes with the original photometry catalog
merged_table = join(phot_table_apcorr, catalog, keys=['xcenter', 'ycenter'])


# Make a histogram of the aperture sums in the catalog
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(catalog['aperture_sum_abmag'], bins=30, alpha=0.5, label='Original photometry')
ax.hist(phot_table_apcorr['ABmag_apcorr'], bins=30, alpha=0.5, label='Aperture-corrected')
ax.set_xlabel('Aperture Sum')
ax.set_ylabel('Frequency')
ax.legend()
plt.show()


