# Created by rebeccahoughton on 01.06.2026
import tomllib
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

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