from load_data import NTLSoftLoader 
import numpy as np
import os

class NTLSoftLoaderResized(NTLSoftLoader):

    def __init__(self, data, ntl_type):
        super().__init__(data, ntl_type)
        self.path_to_ntls = '../dataResized'
    
    def load_ntls(self):
        ntls = [self.load_one_ntl(2000)]
        shape = (len(ntls[0]),len(ntls[0][0]))
        for year in range(2001, 2021):
            ntl = self.load_one_ntl(year)
            assert shape == (len(ntl),len(ntl[0])),f"Current image size is : {(len(ntl),len(ntl[0]))}, image size is {shape}"
            ntls.append(ntl)         
        self.shape = shape
        self.ntls = np.array(ntls)
    
    def load_one_ntl(self, year): 
        fname = os.path.join(self.path_to_ntls,
                        self.data,
                        self.ntl_type,
                        f'ntl_{year}.npy')
        return np.load(fname)