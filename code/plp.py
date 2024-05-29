import matplotlib.pyplot  as plt
import rasterio
import os
import argparse
import numpy as np
import time

def isOn(pix):
    palier = 1.5
    return pix >= palier

class lit_pixel(): 
    def __init__(self,country,sat,pth="../data",out="lit_pixel"):
        self.sat = sat
        self.pathToData = pth
        self.country = country
        self.out = out

    def fetchImg(self,year):
        imgPath = os.path.join(self.pathToData,
                                 self.country,
                                 self.sat,
                                 f'ntl_{year}.tif')
        with rasterio.open(imgPath) as img:
            return np.squeeze(img.read())

    def sumAndNbPixelOn(self,year):
        sum,nb = 0,0
        img = self.fetchImg(year)
        for pixels in img:
            for pixel in pixels:
                if(isOn(pixel)):
                    assert pixel >=0 , "Devrait être positif"
                    sum+=pixel
                    nb += 1

        return (sum,nb)

    def makeGraph(self,show):
        yearList = [x for x in range(2000,2021)]
        nbs = []
        sums = []
        for year in yearList:
            ti = time.time()
            (sum,nb) = self.sumAndNbPixelOn(year)
            nbs.append(nb)
            sums.append(sum)
            print(f"Calcul terminé pour l'année {year} en {time.time() - ti} s")

        fig,ax1 = plt.subplots()
        ax1.plot(yearList, sums, 'b-')
        ax1.set_xlabel('Années')
        ax1.set_ylabel('Somme des pixels allumées', color='b')

        tickYears = yearList[::5]
        ax1.set_xticks(tickYears)
        ax1.set_xticklabels(tickYears)
        
        ax2 = ax1.twinx()
        ax2.plot(yearList, nbs, 'r-')
        ax2.set_ylabel('Nombres de pixel allumées', color='r')

        self.fig = fig
        if show : plt.show()
    
    def saveFig(self,):
        print("Lancement de la sauvegarde du graph ...")
        expe_path = "./lit_pixel_analysis/" + self.country + "/" + self.sat
        os.makedirs(expe_path, exist_ok=True)
        self.fig.savefig(expe_path + f'/{self.out}.png', bbox_inches='tight')
        self.fig.savefig(expe_path + f'/{self.out}.svg')
        print("Sauvegarde réussit !")

    def __call__(self,show=True):
        self.makeGraph(show)
        self.saveFig()

class lit_pixel_resized(lit_pixel):
    def __init__(self,country,sat,pth='../dataResized',out="lit_pixel_resized"):
        
        super().__init__(country,sat,pth=pth,out=out)

    def fetchImg(self,year):
        imgPath = os.path.join(self.pathToData,
                                    self.country,
                                    self.sat,
                                    f'ntl_{year}.npy')
        return np.load(imgPath)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name","-n", help="dataset name to use")
    parser.add_argument("--ntl_type","-t", help="dataset name to use")
    parser.add_argument("--noResize","-nr",action='store_true',help ="tell the prog to not use resized data")
    parser.add_argument("--noShow","-ns",action='store_false',help ="tell the end graph to not be shown")
    args = parser.parse_args()

    if args.noResize :
        print("noResize : Génération d'un plot avec des données normales")
        plot =  lit_pixel(args.name,args.ntl_type)
    else :
        print("Génération d'un plot avec les données redimensionnées")
        plot =  lit_pixel_resized(args.name,args.ntl_type)

    print("Création du graph ...")
    plot(args.noShow)