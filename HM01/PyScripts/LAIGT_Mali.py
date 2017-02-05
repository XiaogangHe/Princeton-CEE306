#! /usr/local/bin/python

# This is a program to convert LAI data from netCDF to Gtool
# Coded by Xiaogang HE on 2012-07-14
# Modified by Xiaogang HE on 2014-04-16


import os,sys
from pylab          import *
from cf.io.gtIO     import *
from cf.io          import gtLoader
from cf.util.LOGGER import *
from netCDF4        import Dataset

@ETA
def main(*args):
    '''
    ./LAIGT.py srcPath
    '''

    srcPath = args[1]

    nlat    = 14                                                                        # need to modify
    nlon    = 14                                                                        # need to modify

    ncData  = Dataset(srcPath,'r')
    lai     = ncData.variables['lai'][:]
    monlai  = [lai[3*i:(3*i+3),].mean(0) for i in range(12)]
    
    print shape(monlai)
    
    gt      = gtLoader('/data2/hexg/MATSIRO_cutJPN/data_med/jpn/grlai_tmp.gt')
    gtHdr   = gt._header
#   gtData  = array(monlai)[:,::-1].reshape(12,1,nlat,nlon) 
    gtData  = array(monlai).reshape(12,1,nlat,nlon) 
    
    print gtData.shape
    
    for Hdr in gtHdr:
        for i,h in enumerate(Hdr):
            print '%02d: %16s'%(i,h)

    print gtData.shape

    year    = 2008                                                                      # need to modify
    DTIME   = [datetime(year,mon,15) for mon in range(1,13)]

    UTIM    = 'MONTH'
    varName = 'LAI'

    save2gt(gtData,DTIME,varName,gtHdr[0][2],20,'./')
    

    return

    
def save2gt(VAR,DT,varName,gt_item,gt_unit,outDir,ATTR=None):
    os.environ['F_UFMTENDIAN']='big'

    ###
    # handle output file (GT)
    if os.path.isdir(outDir) != True:
        raise IOError

    outPath = os.path.join(outDir,'%s_%s.gt'%(varName,str(DT[0])[:4]))
    gt = GT(outPath,'w')

    gt.UTIM = 'MONTH'

    if len(DT)>1:
        gt.TDUR = (DT[1]-DT[0]).days*24+(DT[1]-DT[0]).seconds/3600 
    else:
        gt.TDUR = 0


    if varName in ['Prcpf','Rainf','Snowf','LWdown','SWdown','CCOV']:
        gt.DATE = '%i%02d%02d %02d0000'%(DT[0]-(DT[1]-DT[0])/2).timetuple()[:4]
    else:
        gt.DATE = '%04d%02d%02d %02d0000'%DT[0].timetuple()[:4]


    if ATTR != None:
        for attr in ATTR:
            attr    = attr.split(':')

            gt.__dict__[attr[0]]= attr[1]

    gt.unit = gt_unit
    gt.TITL1= varName
    gt.ITEM = gt_item
    
#    VAR = concatenate(VAR,0)
    
    gt.AITM1= 'GLON14M'                                     # need to modify
    gt.AITM2= 'GLAT14IM'                                    # need to modify
    gt.AITM3= 'SFC'

    gt.UNIT = 'm**2/m**2'
    
    gt.data = VAR

    gt.save()

    ###


if __name__=='__main__':
    LOG = LOGGER()
    main(*sys.argv)
