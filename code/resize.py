import os
import cv2
import numpy as np
import rasterio
import argparse

# def resizeImages(dir):
#     dirOut += dir.name
#     with os.scandir(dir) as files:
#         for file in files:
#             if file.is_dir():
#                 print(f"On s'attaque  au dir {file.name}")
#                 resizeImages(file.path)
#             elif file.is_file() and isImage(file.path):
#                 resize(file)
                    
# def resize(file):
#     print(f"      Je vais resize {file.name}")
#     with rasterio.open(file) as tif:    
#         img = tif.read()

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

def createDir(file,dirOut):
    dir = dirOut+"/"+file.name 
    if not(os.path.exists(dir) and os.path.isdir(dir)):
        os.mkdir(dir)

def resizeImagesSaved(dir,dirName,dirOut,ignored=[], fact=32):
    dirOut += "/"+ dirName
    with os.scandir(dir) as files:
        for file in files:
            if file.is_dir():
                print(f"On s'attaque  au dir {file.name}")
                createDir(file,dirOut)
                resizeImagesSaved(file.path,file.name,dirOut,ignored,fact)
            elif file.is_file() and isImage(file.path) and dir not in ignored:
                resizeSaved(file,dirOut,fact)
    print(f"Fin du dir : {dirName}")

       
def resizeSaved(file,dirOut,fact):
    print(f"      Je vais resize {file.name} avec un facteur de {fact}")
    with rasterio.open(file) as tif:    
        img = np.squeeze(tif.read())
        img = cv2.resize(img, (img.shape[1]//fact, img.shape[0]//fact), interpolation=cv2.INTER_AREA)
        np.save(dirOut+"/"+"/"+file.name[0:-4],img)

def ignoreDir(initDir,strongIgnore):
    ignore = []
    with os.scandir(initDir) as dirs:
        for dir in dirs:
            if(dir.is_dir()):
                ignore.append(dir.path)
                if dir.name in strongIgnore :
                    ignore += ignoreDir(dir,strongIgnore)
    return ignore

if __name__ == "__main__":
    directory_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    initDir = directory_path + "/data"
    dirOut = directory_path
    saved = True
    print(f"Chemin du répertoire contenant les images à resize : {initDir}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fact", help="facteur de division de l'image originale")
    parser.add_argument("--ignored",help="Choisir les dossiers à ignorer de la forme dir1_dir2_dir3")
    args = parser.parse_args()
    ignored = ignoreDir(initDir,args.ignored.split("_"))

    try :
        if saved :
            resizeImagesSaved(initDir,"dataResized",dirOut,ignored=ignored,fact=int(args.fact))  
        # else :
        #     resizeImages(initDir)         
    except FileNotFoundError:
        print(f"Le dossier {initDir} n'existe pas.")
    except PermissionError:
        print(f"Permission refusée pour accéder à {initDir}.")


