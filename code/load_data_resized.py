from load_data import NTLSoftLoader 
import numpy as np
import rasterio
import os
from utils import get_max_resized


"""    def getShape(self):
        path = os.path.join("../data",self.data,self.data+"_2000.tif")
        with rasterio.open(path) as img:
            return img.shape"""


class NTLSoftLoaderResized(NTLSoftLoader):
    def __init__(self, data, ntl_type="DMSP"):
        super().__init__(data, ntl_type)
        self.path_to_ntls = '../dataResized'

    def load_ntls(self):    
        ntls = [self.load_one_ntl(2000,self.ntl_type)]
        shape = (len(ntls[0]),len(ntls[0][0]))
        for year in range(2001, 2021):
            ntl = self.load_one_ntl(year,self.ntl_type)
            assert shape == (len(ntl),len(ntl[0])),f"Current image size is : {(len(ntl),len(ntl[0]))}, image size is {shape}"
            ntls.append(ntl)         
        self.shape = shape
        self.ntls = np.array(ntls)
    
    def load_one_ntl(self, year,ntl_type): 
        fname = os.path.join(self.path_to_ntls,
                        self.data,
                        ntl_type,
                        f'ntl_{year}.npy')
        return np.load(fname)
    
    def setMax(self):
        self.max = get_max_resized(self.data)
    
