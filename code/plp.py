import matplotlib.pyplot  as plt
import rasterio
import os
import argparse
import numpy as np
import time


class lit_pixel(): 
    def __init__(self,country,sat,floor,pth="../data",out="lit_pixel"):
        self.sat = sat
        self.pathToData = pth
        self.country = country
        self.out = out
        self.floor = floor

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
                if(pixel>self.floor):
                    assert pixel >=0 , "Devrait être positif"
                    sum+=pixel
                    nb += 1

        return (sum,nb)
    
    def AllsumAndNbPixelOn(self,yearList):
        nbs = []
        sums = []
        for year in yearList:
            ti = time.time()
            (sum,nb) = self.sumAndNbPixelOn(year)
            nbs.append(nb)
            sums.append(sum)
            print(f"Calcul terminé pour l'année {year} en {time.time() - ti} s")
        return (nbs,sums)

    def getMasks(self,yearList):
        print(self.sat,"DMSP"==self.sat)
        mask = [np.ma.masked_where((np.array(yearList) > 2013), yearList)
                    ,np.ma.masked_where((np.array(yearList) < 2013), yearList)]
        if(self.sat == "DMSP"):
            return (mask[0],mask[1])
        else :
            return  (mask[1],mask[0])
        
        

    def makeGraph(self,show):
        yearList = [x for x in range(2000,2021)]
        (nbs,sums) = self.AllsumAndNbPixelOn(yearList)

        fig,ax1 = plt.subplots()
        (mask1,mask2) = self.getMasks(yearList)

        ax1.plot(yearList, np.ma.masked_array(sums, mask1.mask), 'b-o')
        ax1.plot(mask2, np.ma.masked_array(sums, mask2.mask), 'b--o')  
        
        ax1.set_xlabel('Années')
        ax1.set_ylabel('Somme des pixels allumées', color='b')
        ax1.set_ylim(bottom=0)

        tickYears = yearList[::5] + [2013]
        tickYears.sort()
        ax1.set_xticks(tickYears)
        ax1.set_xticklabels(tickYears)
        
        ax2 = ax1.twinx()
        ax2.plot(yearList, nbs, 'r-')
        ax2.set_ylabel('Nombres de pixel allumées', color='r')
        ax2.set_ylim(bottom=0)
        plt.title(f"{self.sat} NTL - {self.country} - palier :{self.floor}")

        self.fig = fig
        if show : plt.show()
    
    def saveFig(self):
        print("Lancement de la sauvegarde du graph ...")
        expe_path = "./lit_pixel_analysis/" + self.country + "/" + self.sat + "/" + str(self.floor)
        os.makedirs(expe_path, exist_ok=True)
        self.fig.savefig(expe_path + f'/{self.out}.png', bbox_inches='tight')
        self.fig.savefig(expe_path + f'/{self.out}.svg')
        print("Sauvegarde réussit !")

    def __call__(self,show=True):
        self.makeGraph(show)
        self.saveFig()

class lit_pixel_resized(lit_pixel):
    def __init__(self,country,sat,floor,pth='../dataResized',out="lit_pixel_resized"):
        
        super().__init__(country,sat,floor,pth=pth,out=out)

    def fetchImg(self,year):
        imgPath = os.path.join(self.pathToData,
                                    self.country,
                                    self.sat,
                                    f'ntl_{year}.npy')
        return np.load(imgPath)
    
    
class combined(lit_pixel_resized):
    def __init__(self, country, sat, floor, pth='../dataResized', out="lit_pixel_combined"):
        super().__init__(country, sat, floor, pth=pth, out=out)

    def makeGraph(self,show):
        yearList = [x for x in range(2000,2021)]
        self.sat = "DMSP"
        print("--- LANCEMENT CALCUL DMSP ---")
        (nbs,sums) = self.AllsumAndNbPixelOn(yearList)

        fig,ax1 = plt.subplots()
        plt.title(f"DMSP et VIIRS - {self.country} - palier :{self.floor}")
        maskAfter = np.ma.masked_where((np.array(yearList) > 2013), yearList)
        maskBefore = np.ma.masked_where((np.array(yearList) < 2013), yearList)
        mask = self.getMask(yearList)
        ax1.plot(yearList, np.ma.masked_array(sums, maskAfter.mask), 'b-o',label="DMSP : Somme allumés")
        ax1.plot(maskBefore, np.ma.masked_array(sums, maskBefore.mask), 'b--o')  

        ax1.set_xlabel('Années')
        ax1.set_ylabel('SOMME', color='b')

        tickYears = yearList[::5] + [2013]
        tickYears.sort
        ax1.set_xticks(tickYears)
        ax1.set_xticklabels(tickYears)
        
        ax2 = ax1.twinx()
        ax2.plot(yearList, nbs, 'r-',label="DMSP : Nombre allumés")
        ax2.set_ylabel('NOMBRE', color='r')


        self.sat = "VIIRS"
        print("--- LANCEMENT CALCUL VIIRS ---")
        (nbs,sums) = self.AllsumAndNbPixelOn(yearList)

        ax1.plot(yearList, np.ma.masked_array(sums, maskAfter.mask), 'g--o')
        ax1.plot(maskBefore, np.ma.masked_array(sums, maskBefore.mask), 'g-o',label="VIIRS : Somme allumés")  
            
        ax2.plot(yearList, nbs, 'purple',label="VIIRS : Nombre allumé ")

        fig.legend(loc='lower right',fontsize=10, bbox_transform=fig.transFigure)
        self.fig = fig
        
        if show : plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name","-n", help="dataset name to use")
    parser.add_argument("--ntl_type","-t", help="dataset type to use")
    parser.add_argument("--noResize","-nr",action='store_true',help ="tell the prog to not use resized data")
    parser.add_argument("--noShow","-ns",action='store_false',help ="tell the end graph to not be shown")
    parser.add_argument("--combined","-c",action='store_true',help ="Combined VIIRS and DMSP on same graph")
    parser.add_argument("--floor","-f",default=0,type=int,help="ignore all pixel under this floor")
    args = parser.parse_args()
    n = args.name.lower()
    t =  args.ntl_type.upper()
    if args.combined :
        print("Combined : Génération d'un plot resized avec VIIRS et DMSP")
        plot = combined(n,t,args.floor)
    elif args.noResize :
        print("noResize : Génération d'un plot avec des données normales")
        plot =  lit_pixel(n,t,args.floor)
    else :
        print("Génération d'un plot avec les données redimensionnées")
        plot =  lit_pixel_resized(n,t,args.floor)

    print("Création du graph ...")
    plot(args.noShow)
