import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import colors

HCDN_locs = np.loadtxt('./basin_centroid_HCDN.txt')
basin_GEV_loc_KED = np.loadtxt('./basin_GEV_loc_KED_stationary.txt')
basin_FID_2d = np.loadtxt('../USGS_GageII/GageII_NE_basin_raster_250m_wgs84.txt', skiprows=6)
basin_FID_1d = np.array(map(np.unique, basin_FID_2d.reshape(1,-1))).squeeze()[1:]
basin_GEV_2d = basin_FID_2d.copy()
for i, iBasin_FID in enumerate(basin_FID_1d):
    basin_GEV_2d[basin_GEV_2d==iBasin_FID] = basin_GEV_loc_KED[i]

cv_error = np.loadtxt('./cv_residual_OK_location.txt')
cv_error_abs_max = 0.25    # location
#cv_error_abs_max = 0.1    # scale
#cv_error_abs_max = 0.35    # shape
size_max = 300
size_pts = size_max*np.absolute(cv_error)/cv_error_abs_max

plt.figure()
mcolors = plt.cm.Spectral(np.linspace(0.1, 0.5, 100))
cmap_stdev = colors.ListedColormap(mcolors)
m =Basemap(llcrnrlon=-82.953,llcrnrlat=37.853,urcrnrlon=-66.395,urcrnrlat=49.166, projection='mill')
x,y= m(HCDN_locs[:,0],HCDN_locs[:,1])
m.imshow(np.ma.masked_equal(basin_GEV_2d[::-1],-9999.), cmap=cmap_stdev, alpha=0.8)
m.scatter(x[cv_error>0], y[cv_error>0], size_pts[cv_error>0], color='#10a674', alpha=0.8, marker='o', edgecolor='w', lw=.25)
m.scatter(x[cv_error<0], y[cv_error<0], size_pts[cv_error<0], color='#a50055', alpha=0.8, marker='o', edgecolor='w', lw=.25)

l1 = plt.scatter([],[], s=100, edgecolors='none', c='#10a674')
l2 = plt.scatter([],[], s=200, edgecolors='none', c='#10a674')
l3 = plt.scatter([],[], s=300, edgecolors='none', c='#10a674')
l4 = plt.scatter([],[], s=100, edgecolors='none', c='#a50055')
l5 = plt.scatter([],[], s=200, edgecolors='none', c='#a50055')
l6 = plt.scatter([],[], s=300, edgecolors='none', c='#a50055')

labels_size = [cv_error_abs_max/i for i in range(1,4)]
labels_pos = ["{:4.2f}".format(size) for size in labels_size][::-1]
labels_neg = ["{:4.2f}".format(-size) for size in labels_size]

leg1 = plt.legend([l4, l5, l6], labels_pos, handlelength=2, loc = 2, bbox_to_anchor=(0,0.8), borderpad = 1, handletextpad=0.5, scatterpoints = 1, frameon=False)
leg2 = plt.legend([l3, l2, l1], labels_neg, handlelength=2, loc = 2, bbox_to_anchor=(0,1), borderpad = 1, handletextpad=0.5, scatterpoints = 1, frameon=False)
plt.gca().add_artist(leg1)

plt.show()

