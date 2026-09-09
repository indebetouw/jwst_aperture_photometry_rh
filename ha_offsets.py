from astropy.table import Table
import numpy as np
import matplotlib.pyplot as pl
pl.ion()
pl.clf()

pdir="/Users/ri3e/cv/galaxies/phot/"
# t0=Table.read(pdir+"ngc3351_F657_phot_cat_r3.00.ecsv")
t1=Table.read(pdir+"ngc3351_F657_amppos.ecsv")

z=np.where(t1['nfitted_amppos']>0)[0]

pl.clf()
pl.plot((t1['ra']-t1['ra_orig'])[z]*3600, (t1['dec']-t1['dec_orig'])[z]*3600, '.')
pl.xlabel('RA offset (arcsec)')
pl.ylabel('Dec offset (arcsec)')
pl.title('Ha peak - 33Pah peak')
pl.grid(True)
