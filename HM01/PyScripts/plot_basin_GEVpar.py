#!/usr/bin/env python

from pylab import rcParams
from pandas import Series
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gp
import fiona
from itertools import chain
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

### Plot settings
font = {'family' : 'CMU Sans Serif'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 25,
          'grid.linewidth': 0.2,
          'font.size': 25,
          'legend.fontsize': 18,
          'legend.frameon': False,
          'xtick.labelsize': 20,
          'xtick.direction': 'out',
          'ytick.labelsize': 20,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'text.usetex': False}
rcParams.update(params)

### Get the station ID
bas_dir = '../USGS_GageII/boundaries-shapefiles-by-aggeco'                            # Basin shapefile

### Get basins from GAGEII
shp = fiona.open('%s/bas_nonref_NorthEast_wgs84.shp' % (bas_dir))
bds = shp.bounds
shp.close()
extra = 0.01
ll = (bds[0], bds[1])
ur = (bds[2], bds[3])
coords = list(chain(ll, ur))
w, h = coords[2] - coords[0], coords[3] - coords[1]

### Plot
m = Basemap(
    projection='stere',
    lon_0=-80,
    lat_0=50,
    llcrnrlon=coords[0] - extra * w,
    llcrnrlat=coords[1] - extra + 0.01 * h,
    urcrnrlon=coords[2] + extra * w,
    urcrnrlat=coords[3] + extra + 0.01 * h,
    resolution='i',
    suppress_ticks=True)

m.readshapefile(
    '%s/bas_nonref_NorthEast_wgs84' % (bas_dir),
    'bas_nonref_NorthEast_wgs84',
    color='none',
    zorder=2)

# set up a map dataframe
df_map = pd.DataFrame({
    'poly': [Polygon(xy) for xy in m.bas_nonref_NorthEast_wgs84],
    'gage_ID': [basin['GAGE_ID'] for basin in m.bas_nonref_NorthEast_wgs84_info]})

### Plot the GEV parameters
df_map['rand']=np.random.rand(1055)

fig = plt.figure()
ax = fig.add_subplot(111, axisbg='w', frame_on=False)

# use a blue colour ramp - we'll be converting it to a map using cmap()
cmap = plt.get_cmap('Blues')
# draw wards with grey outlines
df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#555555', lw=.2, alpha=1., zorder=4))
pc = PatchCollection(df_map['patches'], match_original=True)
# impose our colour map onto the patch collection
norm = Normalize()
pc.set_facecolor(cmap(norm(df_map['rand'].values)))
ax.add_collection(pc)

# this will set the image width to 722px at 100dpi
plt.tight_layout()
fig.set_size_inches(7.22, 5.25)
plt.savefig('../Figures/test.png', dpi=100, alpha=True)
plt.show()

