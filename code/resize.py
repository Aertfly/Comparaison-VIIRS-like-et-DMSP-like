import os
import cv2
import numpy as np
import rasterio


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


def resizeImagesSaved(dir,dirName,dirOut,ignored=["syria"]):
    dirOut += "/"+ dirName
    with os.scandir(dir) as files:
        for file in files:
            if file.is_dir():
                print(f"On s'attaque  au dir {file.name}")
                createDir(file,dirOut)
                resizeImagesSaved(file.path,file.name,dirOut,ignored)
            elif file.is_file() and isImage(file.path) and dirName not in ignored:
                resizeSaved(file,dirOut)
    print(f"Fin du dir : {dirName}")
def createDir(file,dirOut):
    dir = dirOut+"/"+file.name 
    if not(os.path.exists(dir) and os.path.isdir(dir)):
        os.mkdir(dir)

                   
def resizeSaved(file,dirOut):
    print(f"      Je vais resize {file.name}")
    with rasterio.open(file) as tif:    
        img = np.squeeze(tif.read())
        img = cv2.resize(img, (img.shape[1]//32, img.shape[0]//32), interpolation=cv2.INTER_AREA)
        np.save(dirOut+"/"+"/"+file.name[0:-4],img)


def isImage(pth):
    char = pth[-1]
    temp = ""
    i = 1
    while(char != "."):
        temp += char
        i += 1
        char = pth[-i]
    temp = temp[::-1]
    return temp == "tif"

if __name__ == "__main__":
    directory_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    initDir = directory_path + "/data"
    dirOut = directory_path
    saved = True
    print(f"Chemin du répertoire contenant les images à resize : {initDir}")


    try :
        if saved :
            resizeImagesSaved(initDir,"dataResized",dirOut)  
        # else :
        #     resizeImages(initDir)         
    except FileNotFoundError:
        print(f"Le dossier {initDir} n'existe pas.")
    except PermissionError:
        print(f"Permission refusée pour accéder à {initDir}.")
