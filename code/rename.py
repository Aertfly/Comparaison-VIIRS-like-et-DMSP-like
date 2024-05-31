import os
import argparse
import shutil

def getYear(s):
    return s.split("/")[-1].split(".")[0][-4:]

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

def rename(directory_path, old_name, new_name):
    old_dir = os.path.join(directory_path,'data',old_name)
    new_dir = os.path.join(directory_path,'data' ,new_name)

    with os.scandir(old_dir) as files:
        for file in files:
            if file.is_file() and isImage(file.path):
                new_file_name = f"{new_name}_{getYear(file.name)}.tif"
                new_file_path = os.path.join(old_dir,new_file_name)
                os.rename(file.path, new_file_path)
                print(f"Renamed file '{file.name}' to '{new_file_name}'")   
    try:
        shutil.move(str(old_dir), str(new_dir))
        print(f"Renamed directory '{old_name}' to '{new_name}'")
    except OSError as e:
        print(f"Error renaming directory: {e}")


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument("--new_name","-n",
                        help="new name")

    parser.add_argument("--country",
                        "-c",
                        help="country to rename from")


    args = parser.parse_args()

    directory_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    new_name = args.new_name
    rename(directory_path,args.country,new_name)