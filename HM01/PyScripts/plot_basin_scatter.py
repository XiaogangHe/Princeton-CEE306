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

# Convenience functions for working with colour ramps and bars
def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    This is a convenience function to stop you making off-by-one errors
    Takes a standard colour ramp, and discretizes it,
    then draws a colour bar with correctly aligned labels
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
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

# draw basin patches from polygons
df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(
    x,
    fc='#555555',
    ec='#787878', lw=.25, alpha=.9,
    zorder=4))

# Create Point objects in map coordinates from dataframe lon and lat values
map_points_st = pd.Series(
    [Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(lon[ind_st], lat[ind_st])])    # Stationary, lon & lat should be input
map_points_up = pd.Series(
    [Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(lon[ind_up], lat[ind_up])])    # Increasing trend
map_points_dw = pd.Series(
    [Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(lon[ind_dw], lat[ind_dw])])    # Decreasing trend
gage_points_st = MultiPoint(list(map_points_st.values))
gage_points_up = MultiPoint(list(map_points_up.values))
gage_points_dw = MultiPoint(list(map_points_dw.values))

fig = plt.figure()
ax = fig.add_subplot(111, axisbg='w', frame_on=False)

# we don't need to pass points to m() because we calculated using map_points and shapefile polygons
dev = m.scatter(
    [geom.x for geom in gage_points_st],
    [geom.y for geom in gage_points_st],
    45, marker='o', lw=.25,
    facecolor='#33ccff', edgecolor='w',
    alpha=0.9, antialiased=True,
    label='Stationary', zorder=3)

dev_up = m.scatter(
    [geom.x for geom in gage_points_up],
    [geom.y for geom in gage_points_up],
    45, trend[ind_up], cmap='YlOrRd', marker='^', lw=.25,
    edgecolor='w',
    alpha=0.9, antialiased=True,
    label='Increasing trend', zorder=3)

dev_dw = m.scatter(
    [geom.x for geom in gage_points_dw],
    [geom.y for geom in gage_points_dw],
    45, trend[ind_dw], cmap='spectral', marker='v', lw=.25,
    edgecolor='w',
    alpha=0.9, antialiased=True,
    label='Decreasing trend', zorder=3)

# plot boroughs by adding the PatchCollection to the axes instance
ax.add_collection(PatchCollection(df_map['patches'].values, match_original=True))
plt.legend(loc=2)
plt.title("USGS GagesII basins")
plt.colorbar(dev_up, orientation='horizontal')
plt.tight_layout()
# this will set the image width to 722px at 100dpi
#fig.set_size_inches(7.22, 5.25)
plt.show()
#plt.savefig('./basin_test.png', dpi=400, alpha=True)
