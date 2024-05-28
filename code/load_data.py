import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2  
import json

class NTLSoftLoader():

    def __init__(self, data, ntl_type):
        self.path_to_ntls = '../data'
        self.data = data  # dodoma, Madagascar, south_mada, syria, dar_es_salam
        self.path_to_vis = os.path.join(self.path_to_ntls, self.data, "visu")
        os.makedirs(self.path_to_vis, exist_ok=True)
        self.ntl_type = ntl_type

    def resize_image(self, img):
        height, width = img.shape[:2]
        new_height, new_width = height // 32, width // 32
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        return img_resized
    
    """
    def getShape(self,img):
        return img.shape

        """
    

    

    
    def load_ntls(self):
        ntls = []

        for year in range(2000, 2021):
            img = self.load_one_ntl(year, ntl_type=self.ntl_type)

            if year == 2000:
                img_size = img.shape
                print("Image size is : ", img_size)

            assert img_size == img.shape, f"Current image size is : {img.shape}, image size is {img_size}"
            ntls.append(img)

        self.ntls = np.array(ntls)
    
    def load_sits(self):
        sits = []

        for year in range(2000, 2021):
            img = self.load_image(year)

            if year == 2000:
                img_size = img.shape
                print("Image size is : ", img_size)

            assert img_size == img.shape, f"Current image size is : {img.shape}, image size is {img_size}"
            sits.append(img)

        self.sits = np.array(sits)

    def load_image(self, year):
        fname = os.path.join(self.path_to_ntls,
                             self.data,
                             f'{self.data}_{year}.tif')
        with rasterio.open(fname) as tif:
            img = tif.read()
            img = np.moveaxis(img, 0, -1)
            img = self.resize_image(img)  
        return img

    def load_one_ntl(self, year, ntl_type):
        fname = os.path.join(self.path_to_ntls,
                             self.data,
                             ntl_type,
                             f'ntl_{year}.tif')

        with rasterio.open(fname) as tif:
            img = np.squeeze(tif.read())
            img = self.resize_image(img)  

        return img

    def normalize(self, img, rgb=True, contrast=1., brightness=0.):
        # to RGB
        if rgb:
            img = img[..., :3][..., ::-1]

        # Adjust range to min and max in [0, 1]
        img_min, img_max = img.min(), img.max()
        img = img.astype(np.float32)
        img = (img - img_min) / (img_max - img_min)

        # scale contrast
        img *= contrast

        # scale brightness
        img += brightness

        # Adjust out of range pixels
        img[img > 1] = 1
        img[img < 0] = 0

        return img  

    def plot_optical_image(self, contrast, brightness, year):
        img = self.load_image(year)

        img = self.normalize(img, contrast=contrast, brightness=brightness)
        ntl_dmsp = self.load_one_ntl(year, ntl_type="DMSP")
        ntl_viirs = self.load_one_ntl(year, ntl_type="VIIRS")

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

        ax0.imshow(img)

        im1 = ax1.imshow(ntl_dmsp)
        # ax1.set_title("DMSP")
        im2 = ax2.imshow(ntl_viirs)
        # ax2.set_title("VIIRS")
        
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)  # adjust size and pad as needed
        cbar1 = plt.colorbar(im1, cax=cax1)

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)  # adjust size and pad as needed
        cbar2 = plt.colorbar(im2, cax=cax2)

        fig.suptitle(f"{self.data}, {year}")
        file_name = self.path_to_vis + f"/{self.data}_{year}.png"
        fig.tight_layout()
        print("Data saved at : ", file_name)
        fig.savefig(self.path_to_vis + f"/{self.data}_{year}.png")

    def evaluate(self, d_mat, preds, rand):
        # Debug prints
        print(f"Size of preds: {preds.size}")
        print(f"Max index in rand: {max(rand)}")

        if max(rand) >= preds.size:
            raise IndexError("Index in rand exceeds the size of preds")

        rand_preds = preds[rand]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast",
                        default=1.,
                        type=float, 
                        help="Contrast factor to apply, must be gt 0.")
    parser.add_argument("--brightness", 
                        default=0., 
                        type=float, 
                        help="Brightness factor to apply, bust be in [0, 1]")
    parser.add_argument("--year",
                        default=2010,
                        type=int,
                        help="The year to plot the data")
    parser.add_argument("--data",
                        help="Data to use check in Optical_images")  

    parser.add_argument("--ntl_type",
                        help="ntl type. DMSP ou VIIRS")
    

    args = parser.parse_args()
    dataset = NTLSoftLoader(data=args.data,
                            ntl_type=args.ntl_type)

    dataset.plot_optical_image(args.contrast, args.brightness, args.year)
