# Created by rebeccahoughton on 01.06.2026
import tomllib
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, LogStretch
from astropy.io import fits
import glob
from sys import exit
import os

# Load the config file
config_file = 'config/config.toml'     # Photometry parameters
local_file = 'config/local.toml'       # Paths to directories

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
band = bands[0]  # Assuming we are working with the first band for now


def get_path_to_file(wdir, version, project, galaxy, ptype):
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
          print(f"Found file for {galaxy} {filter} in {path}")
     else:
          raise FileNotFoundError(f"No file found for {galaxy} {filter} in {path}. Please check the path and file naming conventions.")
     return path


def load_catalog(path, filename, **kwargs):
    """Load the catalog as an astropy Table."""
    return Table.read(f"{path}/{filename}")


def plot_background():
    return


def plot_cmd(catalog1, catalog2):
    """Plot a colour-magnitude diagram comparing two catalogs."""
    # Implementation for plotting CMD
    fig, ax = plt.subplots()
    x = catalog1['aperture_sum_abmag'] - catalog2['aperture_sum_abmag']
    y = catalog1['aperture_sum_abmag']
    ax.scatter(x, y)
    return


def plot_sharp_round(catalog1):
    """Plot sharpness vs roundness for a catalog."""
    fig, ax = plt.subplots()
    ax.scatter(catalog1['sharpness'], catalog1['roundness'])
    ax.set_xlabel('Sharpness')
    ax.set_ylabel('Roundness')
    return

# ------------------------------------------------------------------------------
# Making source cutouts
#-------------------------------------------------------------------------------

def make_cutout(row, radius, img_sub):
    """Make a cutout of the image centered on the given coordinates."""
    x, y = row['x_center'], row['y_center']
    cutout_size = radius * 5
    x_min, x_max = int(x - cutout_size), int(x + cutout_size)
    y_min, y_max = int(y - cutout_size), int(y + cutout_size)
    img_cutout = img_sub[y_min:y_max, x_min:x_max]
    return img_cutout, cutout_size


def make_source_profile(cutout):
    """Make a line plot of the source profile in the cutout."""
    mid_x = cutout.shape[1] // 2
    mid_y = cutout.shape[0] // 2
    x_profile = cutout[mid_y, :]
    y_profile = cutout[:, mid_x]
    return x_profile, y_profile


def plot_annuli(ax, x, y, radius, sky_in, sky_out):
    """Plot the optimal aperture radius and sky annulus on a cutout."""
    # Draw a circle with radius equal to the optimal aperture radius
    circle = plt.Circle((x, y), radius, edgecolor='red', facecolor='none', lw=1.5, alpha=1.0)
    sky_in_circle = plt.Circle((x, y), sky_in, edgecolor='magenta', facecolor='none', lw=1.5, alpha=0.5)
    sky_out_circle = plt.Circle((x, y), sky_out, edgecolor='orange', facecolor='none', lw=1.5, alpha=0.5)  
    ax.add_patch(circle)
    ax.add_patch(sky_in_circle)
    ax.add_patch(sky_out_circle)
    return
     

# Load the catalog
merged_table = load_catalog(local['out_dir'], f"{targets[0]}_jwst_{bands[0]}_cat.fits")
# radius = conf['parameters']['photometry']['aperture_radius']
# sky_in = conf['parameters']['photometry']['radius_sky_in']
# sky_out = conf['parameters']['photometry']['radius_sky_out']

filename = glob.glob(f"{local['jwst_dir']}{version}/{projects[0]}/release/{targets[0]}/images/*{bands[0].lower()}*")[0]
with fits.open(filename) as hdul:
    img_sub = hdul['SCI'].data
    header = hdul['SCI'].header


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
print(f"Optimal aperture radius: {radius} pixels")
print(f"Sky annulus inner radius: {sky_in} pixels")
print(f"Sky annulus outer radius: {sky_out} pixels")


print(merged_table.colnames)
# exit()

n_cutouts = 2
# Make the 6x6 plot of source cutouts using sources from the aperture corrected catalog
brightest_sources_apcorr = merged_table[np.argsort(merged_table['aperture_sum_abmag'])][:n_cutouts]

for i in range(n_cutouts):
    row = brightest_sources_apcorr[i]
    cutout, cutout_size = make_cutout(row, radius, img_sub)
    fig, ax = plt.subplots(2, 1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
    # Top plot: x profile
    x_profile, y_profile = make_source_profile(cutout)
    x_to_plot = np.linspace(0, cutout_size, len(x_profile))
    ax[0].plot(x_to_plot, x_profile)
    ax[0].axvline(radius + cutout.shape[1]//2, c='red', ls='--')
    ax[0].axvline(sky_in + cutout.shape[1]//2, c='orange', ls='--')
    ax[0].axvline(sky_out + cutout.shape[1]//2, c='magenta', ls='--')
    # Bottom plot: cutout with radii
    x = cutout.shape[1]//2
    y = cutout.shape[0]//2
    norm_cutout = ImageNormalize(vmin=np.nanpercentile(cutout, 2), 
                                vmax=np.nanpercentile(cutout, 93))
    ax[1].imshow(cutout, origin='lower', cmap='inferno', norm=norm_cutout)
    plot_annuli(ax[1], x, y, radius=radius, sky_in=sky_in, sky_out=sky_out)
plt.show()


    



# fig, axes = plt.subplots(4, 4, figsize=(15, 15), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
# for i, (ax, row) in enumerate(zip(axes.flatten(), brightest_sources_apcorr)):
#     x, y = row['x_center'], row['y_center']
#     cutout_size = radius*5
#     x_min, x_max = int(x - cutout_size), int(x + cutout_size)
#     y_min, y_max = int(y - cutout_size), int(y + cutout_size)
#     cutout = img_sub[y_min:y_max, x_min:x_max]
#     norm_cutout = ImageNormalize(vmin=np.nanpercentile(cutout, 2), 
#                                 vmax=np.nanpercentile(cutout, 93))
#     ax.imshow(img_sub, origin='lower', cmap='inferno', norm=norm_cutout)
#     # Plot horizontal and vertical lines through the center of the cutout
#     ax.axhline(y, color='cyan', ls='--', lw=1.0)
#     ax.axvline(x, color='cyan', ls='--', lw=1.0)
#     # Draw a circle with radius equal to the optimal aperture radius
#     circle = plt.Circle((x, y), radius, edgecolor='red', facecolor='none', lw=1.5, alpha=1.0)
#     sky_in_circle = plt.Circle((x, y), sky_in, edgecolor='magenta', facecolor='none', lw=1.5, alpha=0.5)
#     sky_out_circle = plt.Circle((x, y), sky_out, edgecolor='orange', facecolor='none', lw=1.5, alpha=0.5)  
#     ax.add_patch(circle)
#     ax.add_patch(sky_in_circle)
#     ax.add_patch(sky_out_circle)
#     ax.axis('off')
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.text(0.5, 0.9, f"ABmag={row['aperture_sum_abmag']:.2f}", color='white', fontsize=8, ha='center', va='center', transform=ax.transAxes)
#     ax.text(0.08, 0.93, f"{i+1}", color='cyan', fontsize=8, ha='center', va='center', transform=ax.transAxes)
#     # Select all sources in the catalog that are within the cutout region
#     sources_in_cutout = merged_table[(merged_table['x_center'] > x_min) & (merged_table['x_center'] < x_max) & (merged_table['y_center'] > y_min) & (merged_table['y_center'] < y_max)]
#     ax.scatter(sources_in_cutout['x_center'], sources_in_cutout['y_center'], s=50, edgecolor='cyan', facecolor='none', lw=1.0)
# # plt.show()

# fig, axes = plt.subplots(4, 4, figsize=(15, 15), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
# # Plot the PSFs for each od the cutout sources
# for i, (ax, row) in enumerate(zip(axes.flatten(), brightest_sources_apcorr)):
#     # Start at x_center and y_center and 
#     # do a cutout in x that goes through central pixel
#         x, y = row['x_center'], row['y_center']
#         cutout_size = radius*5
#         mid_x = int(cutout_size // 2)
#         mid_y = int(cutout_size // 2)
#         x_min, x_max = int(x - cutout_size), int(x + cutout_size)
#         y_min, y_max = int(y - cutout_size), int(y + cutout_size)
#         cutout = img_sub[y_min:y_max, x_min:x_max]
#         # Take the cutout and get the x profile
#         x_profile = cutout[mid_x, :]
#         y_profile = cutout[:, mid_y]
#         # Line plot
#         x_to_plot = np.linspace(0, cutout_size, len(x_profile))
#         y_to_plot = np.linspace(0, cutout_size,len(y_profile))
#         ax.plot(x_to_plot, x_profile)
#         ax.axvline(radius, color='red', ls='--', lw=1.5, alpha=1.0, label='Optimal aperture radius')
#         ax.axvline(sky_in, color='magenta', ls='--', lw=1.5, alpha=0.5, label='Sky annulus inner radius')
#         ax.axvline(sky_out, color='orange', ls='--', lw=1.5, alpha=0.5, label='Sky annulus outer radius')
#         ax.set_xlabel('Pixel')
#         ax.set_ylabel('Flux')
#         ax.set_title(f"Source {i+1} (ABmag={row['aperture_sum_abmag']:.2f})")
#     # ax.legend()
# plt.show()

# # And the mosaic with these sources highlighted
# fig, ax = plt.subplots(figsize=(10, 10))
# norm_mosaic = ImageNormalize(vmin=np.nanpercentile(img_sub, 25.00), 
#                             vmax=np.nanpercentile(img_sub, 99.9),
#                             stretch=LogStretch())
# ax.imshow(img_sub, origin='lower', cmap='inferno', norm=norm_mosaic)
# # ax.scatter(merged_table['x_center'], merged_table['y_center'], s=20, edgecolor='cyan', facecolor='none', lw=1.5, label='Sources in aperture-corrected catalog')
# for i in range(len(brightest_sources_apcorr)):
#     ax.scatter(brightest_sources_apcorr['x_center'][i], 
#                brightest_sources_apcorr['y_center'][i], 
#                s=100, edgecolor='cyan', facecolor='none', lw=2, 
#                label='Brightest sources in aperture-corrected catalog' if i==0 else "")
#     ax.text(brightest_sources_apcorr['x_center'][i], 
#             brightest_sources_apcorr['y_center'][i]+105, 
#             f"{i+1}", color='cyan', fontsize=10, ha='center', va='center')
# plt.show()