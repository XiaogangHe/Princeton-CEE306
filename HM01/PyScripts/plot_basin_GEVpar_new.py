#!/usr/bin/env python

from pylab import rcParams
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import fiona
from itertools import chain
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import matplotlib as mpl
import matplotlib.cm as cm

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

def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    This is a convenience function to stop you making off-by-one errors
    Takes a standard colour ramp, and discretizes it,
    then draws a colour bar with correctly aligned labels
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    #mappable.set_clim(-0.5, ncolors+0.5)
    mappable.set_clim(0, ncolors)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors+1))
    colorbar.set_ticklabels(range(ncolors))
    if labels:
        colorbar.set_ticklabels(labels)
    return colorbar

def cmap_discretize(cmap, N):
    """
    Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)

    """
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

### Get the shape file
### Get the station ID
bas_dir = '../USGS_GageII/boundaries-shapefiles-by-aggeco/Boundary_New'                            # Basin shapefile

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
fig = plt.figure()
ax = fig.add_subplot(111, axisbg='w', frame_on=False)

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
    'GAGE_ID': [basin['GAGE_ID'] for basin in m.bas_nonref_NorthEast_wgs84_info]})

df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(
    x,
    ec='#555555',
    lw=.25, alpha=.5,
    zorder=4))

# Add value associated with each polygon
pop = np.random.rand(682)*100
vmin = pop.min()
vmax = pop.max()
df_map['pop'] = np.nan
df_map.set_index('GAGE_ID', inplace=True)
gageID=list(set(df_map.index))

for i, igageID in enumerate(gageID):
    df_map['pop'].ix[igageID]=pop[i]

# For discrete colorbar, bin the values for each polygon
ncolors = 10
bins = np.linspace(vmin, vmax+0.1, ncolors+1)
df_map['bins'] = np.digitize(df_map['pop'], bins) - 1

# Use a blue colour ramp - we'll be converting it to a map using cmap()
cmap = plt.get_cmap('Blues')
pc = PatchCollection(df_map['patches'], match_original=True)

# Impose our colour map onto the patch collection
norm = Normalize()
pc.set_facecolor(cmap(norm(df_map['bins'].values)))
ax.add_collection(pc)

# Create the label for each bin
bin_lab = np.linspace(vmin, vmax, ncolors+1)
bin_lab = ['%.2f' % (i) for i in bin_lab]
cb = colorbar_index(ncolors=ncolors, cmap=cmap, shrink=0.5, labels=bin_lab, format='%.2f')
cb.ax.tick_params(labelsize=15)

plt.tight_layout()
#fig.set_size_inches(7.22, 5.25)
#plt.savefig('./temp.png', dpi=100)

plt.show()
