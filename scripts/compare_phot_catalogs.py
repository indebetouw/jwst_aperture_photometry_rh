#!/usr/bin/env python3
"""Compare two photometry catalogs by sky position and flux."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


phot_dir = Path("/Users/ri3e/cv/galaxies/phot")
cat1_path = phot_dir / "ngc5068_jwst_convolved_pah_f360m_phot_cat_r1.91.csv"
cat2_path = phot_dir / "ngc5068_jwst_convolved_pah_f360m_phot_cat_r3.81.csv"
out_plot = phot_dir / "ngc5068_fwh2rad1_vs_2_flux_compare.png"
flux_col = "aperture_flux_mJy"
flux_col2 = "aperture_flux_mJy_apcorr"

cat1 = Table.read(cat1_path)
cat2 = Table.read(cat2_path)

if len(cat1) != len(cat2):
    raise ValueError(f"Catalog lengths differ: {len(cat1)} vs {len(cat2)}")

u=np.argsort(cat1['id'])
cat1 = cat1[u]
u=np.argsort(cat2['id'])
cat2 = cat2[u]

ra_diff_arcsec = np.abs(np.asarray(cat1["ra"], dtype=float) - np.asarray(cat2["ra"], dtype=float)) * 3600.0
dec_diff_arcsec = np.abs(np.asarray(cat1["dec"], dtype=float) - np.asarray(cat2["dec"], dtype=float)) * 3600.0

print(f"Max |delta RA|  (arcsec): {np.nanmax(ra_diff_arcsec):.6e}")
print(f"Max |delta Dec| (arcsec): {np.nanmax(dec_diff_arcsec):.6e}")
assert np.nanmax(ra_diff_arcsec) < 1.0, "RA difference too large"
assert np.nanmax(dec_diff_arcsec) < 1.0, "Dec difference too large"

f1 = np.asarray(cat1[flux_col], dtype=float)
f2 = np.asarray(cat2[flux_col], dtype=float)

ok = np.isfinite(f1) & np.isfinite(f2)
f1 = f1[ok]
f2 = f2[ok]

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(f1, f2, "k.", ms=2, alpha=0.3)
f1 = np.asarray(cat1[flux_col2], dtype=float)
f2 = np.asarray(cat2[flux_col2], dtype=float)
ok = np.isfinite(f1) & np.isfinite(f2)
f1 = f1[ok]
f2 = f2[ok] 
ax.plot(f1, f2, "g.", ms=2, alpha=0.3)

lo = min(np.nanmin(f1), np.nanmin(f2))
hi = max(np.nanmax(f1), np.nanmax(f2))
ax.plot([lo, hi], [lo, hi], "r--", lw=1)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Flux from fwh2rad1.0 catalog (mJy)")
ax.set_ylabel("Flux from fwh2rad2.0 catalog (mJy)")
ax.set_title("NGC5068 convolved_pah_f360m flux comparison")

plt.tight_layout()
plt.savefig(out_plot, dpi=200)
plt.close(fig)

print(f"Saved plot: {out_plot}")
