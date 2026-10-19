import sys
from importlib.metadata import version, PackageNotFoundError

# Check package version
try:
    __version__ = version("jwst_aperture_photometry")
except PackageNotFoundError:
    __version__ = "unknown"

if sys.version_info < (3, 11):
    raise ImportError("jwst_aperture_photometry required Python 3.11 and above.")

# All function imports
from .io import get_file

__all__ = [
    "get_file",
]
