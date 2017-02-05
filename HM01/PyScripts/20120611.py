#! /usr/local/bin/python
import os,sys
from cf.io import gtLoader
from cf.util.LOGGER import *
from pylab import *
from mpl_toolkits.basemap import Basemap
from cf.util import roller

@ETA
def drawMap(aSrc):
    aSrc    = roller(aSrc)*86400 # change to mm/d
    aSrc    = ma.masked_equal(aSrc,0)
    Fig     = figure()
#Ax      = Fig.add_subplot(111)
    M       =   Basemap(resolution='c')
    
    M.imshow(aSrc,cmap=cm.Spectral_r)
    M.drawcoastlines()
    M.drawrivers()
    colorbar(orientation='h')
    title('LHFLUX')

    Fig.savefig('./LHFLUX.png')
    print aSrc
    show()
    return
@ETA
def main(*args):
    print args 
    varName =args[1]
    Year   =int(args[2])
    baseDir = '/data/hjkim/ELSE/JL1/out/JL1.Prcp_GPCC/JL1.%i/%s'
    srcPath = baseDir%(Year,varName)
  
    gt      =gtLoader(srcPath)
    print gt._header[0]
    print gt._data.shape

    aAve  = gt._data.mean(0)[0]
    drawMap(aAve)
    return

if __name__=='__main__':
    LOG = LOGGER()
    main(*sys.argv)
    
