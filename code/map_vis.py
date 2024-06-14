import numpy as np
import folium
import argparse
import rasterio
import os

from matplotlib.colors import Normalize
from matplotlib.cm import viridis
from utils import get_max_resized
from kmeans import main_kmeans
from plp import combined
from PIL import Image

class map_vis():
    def __init__(self,countries):
        self.countries = []
        for c in countries:
            self.countries.append(country(c))

    def __call__(self):
        m = folium.Map(location=[0, 0], 
                    zoom_start=5,
                    tiles='https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png',
                    attr='OpenStreetMap France',
                    world_copy_jump=True
                )
        initJs= """const handler  = new overlayHandler("""+m.get_name()+""")"""
        for country in self.countries :
            country(m)
            initJs += country.get_init_js()
        self.add_custom_script(m,initJs)
        folium.LayerControl().add_to(m)
        m.save(os.path.join("index.html"))
        print("Map sauvegardée !")

    def add_custom_script(self,m,jsCountriesInit):
        custom_js= """
        <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; z-index: 9999;">
            <label for="opacityRange">Opacité des calques
            <input type="range" id="opacityRange" min="0" max="100" value="60" step="1" 
                oninput="updateOpacity(this.value)"></label>
            <label for="year">Année : <span id="yearValue" name="yearValue">2010</span>
            <br/>
            <input type="range" id="year" name="year" default="2000" min="2000" max="2020" />
            </label>
            <br/>
            <label for="DMSP">DMSP
            <input type="radio" id="DMSP" name="sat" value="DMSP" >
            </label>
            <label for="VIIRS">VIIRS
            <input type="radio" id="VIIRS" name="sat" value="VIIRS" >
            </label>
            <label for="noneSat">Aucun
            <input type="radio" id="noneSat" name="sat" value="null">
            </label>
            <br/>
            <label for="lightmap">Intensité
            <input type="radio" id="lightmap" name="typeVis" value="lightmap">
            </label>
            <label for="kmeans">Kmeans
            <input type="radio" id="kmeans" name="typeVis" value="kmeans">
            </label>
            <label for="noneType">Masquer
            <input type="radio" id="noneType" name="typeVis" value="null">
            </label>
        </div>
        <script src="./mapVis.js"></script>
        <script>
            document.addEventListener("DOMContentLoaded", function() {
                """+jsCountriesInit+"""
            });
        </script>"""
        m.get_root().html.add_child(folium.Element(custom_js))

class country():
    def __init__(self,name,cluster=5,floor=30):
        raster_path = os.path.join("../data", name, "DMSP", "ntl_2000.tif")
        with rasterio.open(raster_path) as dataset:
            width = dataset.width
            height = dataset.height
            transform = dataset.transform
        #transformation de mercator 
        top_left_coords = rasterio.transform.xy(transform, 0, 0)
        bottom_right_coords = rasterio.transform.xy(transform, height - 1, width - 1)
        self.first_year = 2000
        self.last_year = 2021
        self.name = name
        self.floor = floor
        self.cluster = cluster
        self.bounds = [[top_left_coords[1], top_left_coords[0]], [bottom_right_coords[1], bottom_right_coords[0]]]

    def __call__(self,m):
        print("Ajout à la carte de",self.name)
        li_sat = ["DMSP","VIIRS"]
        self.pre_generate_overlays(li_sat)
        self.generate_grp(li_sat).add_to(m)

    def pre_generate_overlays(self,li_sat):
        for sat in li_sat:
            self.pre_generate_kmeans(sat)
            self.pre_generate_lightmaps(sat)
        self.pre_generate_plp()

    def pre_generate_kmeans(self,sat):
        path = os.path.join("../analysis",self.name,"kmeans_analysis",sat,
                            str(self.cluster),f"cluster_img_{self.cluster}.svg")
        if not os.path.exists(path):
            print("Kmeans non trouvé, création de l'image...",path)
            params = {
                'ntl_type': sat,
                'resize': True,
                'show': False,
                'clusters': [self.cluster]
            }
            main_kmeans(self.name, **params)
            print("Image créée, reprise de la suite")

    def pre_generate_plp(self):
        path = os.path.join("../analysis",self.name,"lit_pixel_analysis","DMSP_et_VIIRS",
                            str(self.floor),"lit_pixel_combined.png")
                            
        if not os.path.exists(path) :
            print("Graph non trouvé, lancement de sa création")
            plp = combined(self.name,floor= self.floor)
            plp(show=False,graphs=['g','h'])

    def pre_generate_lightmaps(self,sat):
        vmax = -1
        path = f'../analysis/{self.name}/lightmap/{sat}'
        os.makedirs(path,exist_ok=True)
        for year in range(self.first_year,self.last_year):
            temp_path = path + f"/lightmap_{year}_{sat}.png"
            if not os.path.exists(temp_path):
                print(f"Création de la lightmap pour {sat} de {year}")
                if vmax == -1 :
                    if sat =="DMSP":
                        vmax= 64
                    else:
                        vmax= get_max_resized(self.name)
                self.generate_lightmap(sat,year,temp_path,vmax)

    def generate_lightmap(self,sat,year,path,vmax):
        img_path = os.path.join("../data",self.name,sat,f"ntl_{year}.tif")
        with rasterio.open(img_path) as img:
            raster_data = img.read(1)
        #raster_data = np.load(img_path)
        norm = Normalize(vmin=0, vmax=vmax)
        lightmap_data = viridis(norm(raster_data))

        lightmap_image = (lightmap_data[:, :, :3] * 255).astype(np.uint8)
        lightmap_pil = Image.fromarray(lightmap_image)
        lightmap_pil.save(path)
    
    def generate_grp(self,li_sat):
        grp_country = folium.FeatureGroup(name=f"{self.name}",show=False)
        colors = ['red','green']
        i=0
        for sat in li_sat:
            html_content = """
            <!DOCTYPE html>
            <html>
                <body>
                    <p>"""+sat+"""</p>
                    <img src="../analysis/"""+self.name+"""/kmeans_analysis/"""+sat+"""/"""+str(self.cluster)+"""/clusters_"""+str(self.cluster)+""".png"
                    alt="Image" width="700" height="150">
                </body>
            </html>
            """
            folium.Marker(
                [self.bounds[0][0], self.bounds[i][1]], 
                popup=folium.Popup(html_content, max_width=1400, lazy=True),
                icon=folium.Icon(color=colors[i], icon='info-sign')
            ).add_to(grp_country)
            i+=1
        folium.Marker(
            [self.bounds[1][0], self.bounds[1][1]], 
            icon=folium.Icon(color='blue', icon='info-sign'),
            popup=folium.Popup(
            """<!DOCTYPE html>
            <html>
                <body>
                    <p>Graphique</p>
                    <img src="../analysis/"""+self.name+"""/lit_pixel_analysis/DMSP_et_VIIRS/"""+str(self.floor)+"""/lit_pixel_combined.png"
                    alt="Image" width="598" height="457">
                </body>
            </html>
            """,max_width=1400,lazy=True)
        ).add_to(grp_country)
        folium.Marker(
            [self.bounds[1][0], self.bounds[0][1]], 
            icon=folium.Icon(color='purple', icon='info-sign'),
            popup=folium.Popup(
            """<!DOCTYPE html>
            <html>
                <body>
                    <p>Histogramme</p>
                    <img src="../analysis/"""+self.name+"""/lit_pixel_analysis/DMSP_et_VIIRS/histogramme/histogramme_"""
                    +self.name+"""_DMSP.png" alt="Image" width="852" height="535">
                    <img src="../analysis/"""+self.name+"""/lit_pixel_analysis/DMSP_et_VIIRS/histogramme/histogramme_"""
                    +self.name+"""_VIIRS.png" alt="Image" width="852" height="535">
                </body>
            </html>
            """,lazy=True,max_width=2000)
        ).add_to(grp_country)
        return grp_country

    def get_init_js(self):
        return """
        handler.addOverlay("""+str(self.first_year)+""","""+str(self.last_year)+""",'"""+self.name+"""',"""+str(self.cluster)+""","""+str(self.bounds)+""")"""

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Nom de la zone d'étude")
    parser.add_argument("-a", "--all",action='store_true', help="Prend toute les zones d'étude")
    args = parser.parse_args() 

    
    if args.all:
        temp = []
        with os.scandir("../data") as files:
            for file in files:
                if(file.is_dir()):
                    temp.append(file.name)
    else:
        temp = [args.name]
    main = map_vis(temp)
    main()