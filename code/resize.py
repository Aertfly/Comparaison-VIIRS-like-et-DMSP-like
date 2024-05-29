import os
import cv2
import numpy as np
import rasterio
import argparse

# def resizeImagesSavedRec(dir,dirName,dirOut,ignore=[], fact=32):
#     dirOut += "/"+ dirName
#     with os.scandir(dir) as files:
#         for file in files:
#             if file.is_dir():
#                 print(f"On s'attaque  au dir {file.name}")
#                 createDir(file,dirOut)
#                 resizeImagesSaved(file.path,file.name,dirOut,ignore,fact)
#             elif file.is_file() and isImage(file.path) and dir not in ignore:
#                 resizeSaved(file,dirOut,fact)
#     print(f"Fin du dir : {dirName}")

def isImage(pth):
    char = pth[-1]
    temp = ""
    i = 1
    while(char != "." and i<5):
        temp += char
        i += 1
        char = pth[-i]
    temp = temp[::-1]
    return temp == "tif"

def createDir(fname,dirOut):
    dir = dirOut+"/"+ fname
    if not(os.path.exists(dir) and os.path.isdir(dir)):
        os.mkdir(dir)

def isAccepted(d,r,i):
    if (len(i)>0):
        return d.name not in i
    if(len(r)>0):
        return d.name in r
    return True

def startResizeSaved(initDir,dirOut,fact=32,requested=[],ignore=[]):
    with os.scandir(initDir) as dirs:
        for dir in dirs:
            if(dir.is_dir()) and isAccepted(dir,requested,ignore):
                print(f"On retaille les images de {dir.name}")
                createDir(dir.name,dirOut)
                resizeImagesSaved(dir,dirOut+"/"+dir.name,fact)
                print(f"Fin retaillage {dir.name}")
            else :
                print(f"Est ignoré {dir.name}")

def resizeImagesSaved(dir,dirOut,fact):
    satDirs = ["DMSP","VIIRS"]
    for satDir in satDirs:
        print(f"    On passe à {satDir}")
        createDir(satDir,dirOut)
        with os.scandir(dir.path+"/"+satDir) as files :
            for file in files:
                if(file.is_file() and isImage(file.path)):
                    resizeSaved(file,dirOut+"/"+satDir,fact)
                    

def resizeSaved(file,dirOut,fact):
    print(f"        Je vais resize {file.name} avec un facteur de {fact}")
    with rasterio.open(file) as tif:    
        img = np.squeeze(tif.read())
        img = cv2.resize(img, (img.shape[1]//fact, img.shape[0]//fact), interpolation=cv2.INTER_AREA)
        np.save(dirOut+"/"+"/"+file.name[0:-4],img)


if __name__ == "__main__":
    directory_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    initDir = directory_path + "/data"
    dirOut = directory_path + "/dataResized"
    print(f"Chemin du répertoire contenant les images à resize : {initDir}")

    saved = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--fact","-f", help="facteur de division de l'image originale")
    parser.add_argument("--req","-r",nargs='+',default=[],help="Villes à retailler de la forme Ville1-ville2-ville3")
    parser.add_argument("--ignore","-i",nargs='+',default=[],help="Villes à ignorer de la forme Ville1-ville2-ville3")
    args = parser.parse_args()

    requested = args.req
    print("Villes demandées : ",requested)
    ignore = args.ignore
    print("Villes à ignorer : ",ignore)

    try :
        if saved :
            startResizeSaved(initDir,dirOut,fact=int(args.fact),requested=requested,ignore=ignore)  
    except FileNotFoundError:
        print(f"Le dossier {initDir} n'existe pas.")
    except PermissionError:
        print(f"Permission refusée pour accéder à {initDir}.")


