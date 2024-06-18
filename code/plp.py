import matplotlib.pyplot  as plt
import rasterio
import os
import argparse
import numpy as np
import time
import math

def normalize(li):
    try:
        m = max(li)
        for i in range(len(li)):
            li[i] = li[i]/m
        return li
    except ZeroDivisionError:  # dans le cas d'une liste composée uniquement de zéros
        return li

class lit_pixel(): 
    def __init__(self,country,sat,floor=0,pth="../data",out="lit_pixel", first_year=2000, last_year=2020):
        self.sat = sat
        self.pathToData = pth
        self.country = country
        self.out = out
        self.floor = floor
        self.first_year=first_year
        self.last_year=last_year

        # Définir la taille de police par défaut pour tous les éléments
        plt.rc('font', size=14)           # taille de la police par défaut pour le texte
        plt.rc('axes', labelsize=15)      # taille de la police pour les labels des axes
        plt.rc('xtick', labelsize=12)     # taille de la police pour les ticks de l'axe X
        plt.rc('ytick', labelsize=12)     # taille de la police pour les ticks de l'axe Y
        plt.rc('legend', fontsize=10)     # taille de la police pour la légende
        plt.rc('figure', titlesize=18)    # taille de la police pour les titres des figures

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
        sums = normalize(sums)
        nbs = normalize(nbs)
        return (nbs,sums)

    def makeGraph(self,show):
        yearList = [x for x in range(self.first_year,self.last_year+1)]
        (nbs,sums) = self.AllsumAndNbPixelOn(yearList)

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

        plt.title(f"{self.sat} NTL - {self.country}")
        self.fig = fig
        if show : plt.show()
        

    def getMax(self):
        print(os.path.join(self.pathToData,
                                 self.country,
                                 self.sat,
                                 f'ntl_année.tif'))
        return  max(math.floor(pixel) for year in range(self.first_year, self.last_year+1) for pixels in self.fetchImg(year) for pixel in pixels) + 1

    def  makeYearlyHistogram(self, show):
        repartitions = []
        max_pixel_count = 0 #np.max([np.max(rep) for rep in repartitions])
        max_val = self.getMax()

        for year in range(self.first_year, self.last_year + 1):
            img = self.fetchImg(year)
            repartition = [0] * max_val

            for i in range(len(img)):
                for j in range(len(img[i])):
                    repartition[math.floor(img[i][j])] += 1
                temp = max(repartition)
                if(temp>max_pixel_count):
                    max_pixel_count = temp
            repartitions.append(repartition)

        max_pixel_count += max_pixel_count * 1/100
        for year, repartition in zip(range(self.first_year, self.last_year + 1), repartitions):
            self.fig = plt.figure(figsize=(10, 6))
            plt.bar(range(len(repartition)), repartition)
            
            plt.xlabel('Intensité des pixels')
            plt.ylabel('Nombre de pixels')
            plt.title(f'Intensité des pixels {self.sat} - {self.country} - {year}')
            
            plt.ylim(0, max_pixel_count)
            if show:
                plt.show()
            self.saveFig("histogramme", out=f"histogramme_{self.country}_{self.sat}_{year}")
            print(f"{self.country} : histogramme {self.sat} {year - 1999}/{self.last_year - 1999} complété !")
            plt.close()

    def makeHistogram(self,show):
        maximum = self.getMax()
        print("Intensité max :",maximum)
        repartition = [0]*maximum
        
        for year in range(self.first_year, self.last_year+1):
            img = self.fetchImg(year)
            for i in range(len(img)):
                for j in range(len(img[i])):
                    repartition[math.floor(img[i][j])] += 1
                    if img[i][j] > 150:
                        pass
                        #print(f"Pixel de luminosité {img[i][j]} en {i*32},{j*32} pour l'année {year}")
        self.fig = plt.figure(figsize=(10, 6))
        plt.bar(range(len(repartition)), repartition)

        plt.xlabel('Intensité des pixels')
        plt.ylabel('Nombre de pixels')
        plt.title(f'Intensité des pixels {self.sat} - {self.country}')
        
        if show :plt.show()
        self.saveFig("histogramme",out=f"histogramme_{self.country}_{self.sat}")
        plt.close()
    
    def get_expe_path(self,end_dir):
        return  os.path.join("../analysis",self.country,"lit_pixel_analysis",self.sat,end_dir)
    
    def saveFig(self,end_dir,out = ""):
        if out == "": 
            out = self.out
        expe_path = self.get_expe_path(end_dir)
        os.makedirs(expe_path, exist_ok=True)
        self.fig.savefig(expe_path + f'/{out}.png', bbox_inches='tight')
        self.fig.savefig(expe_path + f'/{out}.svg')

    def __call__(self,graphs=["g"],show=True):
        for g in graphs:
            if g in ["g", "graph", "graphique"]:
                self.makeGraph(show)
                self.saveFig(str(self.floor))
                plt.close()
            elif g in ["h", "hist", "histogramme"]:
                self.makeHistogram(show)
            elif g in ["ha", "multi", "multiHist","yearly","histogramme_annuel","annuel"]:
                self. makeYearlyHistogram(False)
                    

class lit_pixel_resized(lit_pixel):
    def __init__(self,country,sat,floor=0,pth='../dataResized',out="lit_pixel_resized"):
        super().__init__(country,sat,floor,pth=pth,out=out)

    def fetchImg(self,year):
        imgPath = os.path.join(self.pathToData,
                                    self.country,
                                    self.sat,
                                    f'ntl_{year}.npy')
        return np.load(imgPath)
    
    
class combined(lit_pixel_resized):
    def __init__(self, country,sat="DMSP",floor=0 , pth='../dataResized', out="lit_pixel_combined", first_year=2000, last_year=2020):
        super().__init__(country, sat, floor, pth, out)
        self.first_year=first_year
        self.last_year=last_year

    def makeHistogram(self, show):
        li_sat = ["DMSP","VIIRS"]
        for sat in li_sat:
            self.sat = sat
            super().makeHistogram(show)

    def  makeYearlyHistogram(self, show):
        li_sat = ["DMSP","VIIRS"]
        for sat in li_sat:
            self.sat = sat
            super(). makeYearlyHistogram(show)

    def makeGraph(self,show):
        yearList = [x for x in range(self.first_year,self.last_year+1)]
        self.sat = "DMSP"
        print("--- LANCEMENT CALCUL DMSP ---")
        (nbs,sums) = self.AllsumAndNbPixelOn(yearList)
        fig,ax1 = plt.subplots()
        plt.title(f"DMSP et VIIRS  - normalisé - {self.country} - palier :{self.floor}")
        maskAfter = np.ma.masked_where((np.array(yearList) > 2013), yearList)
        maskBefore = np.ma.masked_where((np.array(yearList) < 2013), yearList)
        ax1.plot(yearList, np.ma.masked_array(sums, maskAfter.mask), 'b-o',label="DMSP : Somme allumés")
        ax1.plot(maskBefore, np.ma.masked_array(sums, maskBefore.mask), 'b--o')  

        ax1.set_xlabel('Années')
        
        tickYears = yearList[::5]
        ax1.set_xticks(tickYears)
        ax1.set_xticklabels(tickYears)
        
        ax2 = ax1.twinx()
        ax2.plot(yearList, nbs, 'r-',label="Nombres de pixel allumées DMSP")
        ax2.set_ylabel('', color='r')

        self.sat = "VIIRS"
        print("--- LANCEMENT CALCUL VIIRS ---")
        (nbs,sums) = self.AllsumAndNbPixelOn(yearList)
        ax1.plot(yearList, np.ma.masked_array(sums, maskAfter.mask), 'g--o')
        ax1.plot(maskBefore, np.ma.masked_array(sums, maskBefore.mask), 'g-o',label="VIIRS : Somme allumés")  
            
        ax2.plot(yearList, nbs, 'purple',label="Nombres de pixel allumées VIIR")

        fig.legend(loc="lower right")
        self.fig = fig        
        if show : plt.show()
    
    def get_expe_path(self,end_dir):
        return  os.path.join("../analysis",self.country,"lit_pixel_analysis","DMSP_et_VIIRS",end_dir)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name","-n", help="dataset name to use")
    parser.add_argument("--ntl_type","-t", help="dataset type to use")
    parser.add_argument("--noResize","-nr",action='store_true',help ="tell the prog to not use resized data")
    parser.add_argument("--noShow","-ns",action='store_false',help ="tell the end graph to not be shown")
    parser.add_argument("--combined","-c",action='store_true',help ="Combined VIIRS and DMSP on same graph")
    parser.add_argument("--graph","-g",nargs='+',default=["g"],help="graphique à créer , ex: graph hist")
    parser.add_argument("--floor","-f",type=int,default=0,help="palier à partir du quel ignorer les pixels")
    args = parser.parse_args()
    n = args.name.lower() 
    t =  args.ntl_type.upper() 
    f = int(args.floor)
    if args.combined :
        print("Combined : Génération d'un plot resized avec VIIRS et DMSP")
        plot = combined(n,t,f)
    elif args.noResize :
        print("noResize : Génération d'un plot avec des données normales")
        plot =  lit_pixel(n,t,f)
    else :
        print("Génération d'un plot avec les données redimensionnées")
        plot =  lit_pixel_resized(n,t,f)

    print("Création du graph ...")
    plot(show=args.noShow,graphs=args.graph)