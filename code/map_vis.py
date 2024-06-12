import numpy as np
import folium
import argparse
import rasterio
import os

from matplotlib.colors import Normalize
from matplotlib.cm import viridis
from utils import get_max_resized
from kmeans import main_kmeans
from PIL import Image



class map_vis():
    def __init__(self,first_year = 2000, last_year=2021):
        self.first_year = first_year
        self.last_year = last_year

    def __call__(self,countries):
        m = folium.Map(location=[0, 0], 
                    zoom_start=5,
                    #tiles='https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png',
                    attr='OpenStreetMap France',
                    world_copy_jump=True
                )
        initJs= """const handler  = new overlayHandler("""+m.get_name()+""")"""
        for country in countries :
            initJs += self.add_country(m,country)
        self.add_custom_script(m,initJs)
        folium.LayerControl().add_to(m)
        m.save(os.path.join("index.html"))
        print("Map sauvegardée !")

    def add_country(self,m,country):
        print("Lancement de la création de la carte pour",country)
        raster_path = os.path.join("../data", country, "DMSP", "ntl_2000.tif")
        with rasterio.open(raster_path) as dataset:
            width = dataset.width
            height = dataset.height
            transform = dataset.transform

        #transformation de mercator 
        top_left_coords = rasterio.transform.xy(transform, 0, 0)
        bottom_right_coords = rasterio.transform.xy(transform, height - 1, width - 1)
        bounds=[[top_left_coords[1], top_left_coords[0]], [bottom_right_coords[1], bottom_right_coords[0]]]




        clusters = 5
        self.pre_generate_overlays(country,clusters)

        grp_country = folium.FeatureGroup(name=f"{country}",show=False)
        li_sat = ["DMSP","VIIRS"]
        colors = ['red','green']
        i=0
        for sat in li_sat:
            html_content = """
            <!DOCTYPE html>
            <html>
                <body>
                    <p>"""+sat+"""</p>
                    <img src="../analysis/"""+country+"""/"""+sat+"""/kmeans_analysis/"""+str(clusters)+"""/clusters_"""+str(clusters)+""".png" 
                    alt="Image" width="700" height="150">
                </body>
            </html>
            """
            folium.Marker(
                [bounds[0][0], bounds[0][1]+0.1*i], 
                popup=folium.Popup(html_content, max_width=1400),
                icon=folium.Icon(color=colors[i], icon='info-sign')
            ).add_to(grp_country)
            i+=1
        folium.Marker(
            [bounds[1][0], bounds[1][1]], 
            popup='Bottom Right'
        )

        grp_country.add_to(m)
        return """
        handler.addOverlay("""+str(self.first_year)+""","""+str(self.last_year)+""",'"""+country+"""',"""+str(5)+""","""+str(bounds)+""")"""

    def create_lightmaps(self,country,sat):
        vmax = -1
        for year in range(self.first_year,self.last_year):
            path = f'../analysis/{country}/{sat}/lightmap'
            os.makedirs(path,exist_ok=True)
            path +=f"/lightmap_{year}_{sat}.png"
            if not os.path.exists(path):
                print(f"Création de la heat map pour {sat} de {year}")
                if vmax == -1 :
                    if sat =="DMSP":
                        vmax= 64
                    else:
                        vmax= get_max_resized(country)
                self.create_lightmap(sat,country,year,path,vmax)

    def create_lightmap(self,sat,country,year,path,vmax):
        img_path = os.path.join("../data",country,sat,f"ntl_{year}.tif")
        with rasterio.open(img_path) as img:
            raster_data = img.read(1)

        norm = Normalize(vmin=0, vmax=vmax)
        lightmap_data = viridis(norm(raster_data))

        lightmap_image = (lightmap_data[:, :, :3] * 255).astype(np.uint8)
        lightmap_pil = Image.fromarray(lightmap_image)
        lightmap_pil.save(path)

    def create_kmeans(self,sat,country,cluster):
        image= os.path.join("../analysis",country,sat,"kmeans_analysis",str(cluster),f"cluster_img_{cluster}.png")
        if not os.path.exists(image):
            print("Kmeans non trouvé, création de l'image...",image)
            params = {
                'ntl_type': sat,
                'resize': True,
                'show': False,
                'clusters': [cluster]
            }
            main_kmeans(country, **params)
            print("Image créée, reprise de la suite")

    def pre_generate_overlays(self,country,clusters):
        li_sat = ["DMSP","VIIRS"]
        for sat in li_sat:
            self.create_lightmaps(country,sat)
            self.create_kmeans(sat,country,clusters)
        
        #TO DO ajouter legend heat map
        #legend = folium.map.CustomPane('image_legends', z_index=650)
        #legend.add_to(m)
        


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
            <label for="noneType">Cacher
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

    
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Nom de la zone d'étude")
    parser.add_argument("-a", "--all",action='store_true', help="Prend toute les zones d'étude")
    args = parser.parse_args() 

    main = map_vis()
    if args.all:
        temp = []
        with os.scandir("../data") as files:
            for file in files:
                if(file.is_dir()):
                    temp.append(file.name)
        main(temp)
    else:
        main([args.name])