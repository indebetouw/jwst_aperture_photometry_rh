from astropy.table import Table
import matplotlib.pyplot as pl
pl.ion()
pl.clf()
import numpy as np

ft=Table.read("/Users/ri3e/cv/galaxies/phot/ngc3351_jwst_convolved_pah_f360m_find_cat.csv")

ft=ft[np.argsort(ft['peak_value'])[::-1]]

nmax=2000
pl.plot(ft['x_centroid'], ft['peak_value'], '.')
z=np.where(np.isnan(ft['x_centroid']))[0]
pl.plot(ft['x_peak'][z], ft['peak_value'][z], 'x')

with open("test_centroiding.reg", "w") as f:
    z=np.where((np.isfinite(ft['x_centroid']))*(ft['peak_value']>.3))[0]
    for i in range(np.min([nmax,len(z)])):
        f.write("circle({},{},{}) # color=blue\n".format(ft['x_centroid'][z[i]], ft['y_centroid'][z[i]], 5))
    z=np.where((np.isnan(ft['x_centroid']))*(ft['peak_value']>5))[0]
    for i in range(len(z)):
        f.write("circle({},{},{}) # color=red\n".format(ft['x_peak'][z[i]], ft['y_peak'][z[i]], 5))
    z=np.where((np.isnan(ft['x_centroid']))*(ft['peak_value']>2)*(ft['peak_value']<5))[0]
    #for i in range(len(z)):
        #f.write("circle({},{},{}) # color=green\n".format(ft['x_peak'][z[i]], ft['y_peak'][z[i]], 5))



pl.show()


