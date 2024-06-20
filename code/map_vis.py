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
import math



class NoSuchFileForThoseYears(Exception):
    def __init__(self, first_year, last_year):
        super().__init__(f"ERREUR aucun fichier pour ses années premiére : {first_year} et derniére : {last_year}")

class map_vis():
    def __init__(self,countries,force=[],first_year=2000, last_year=2020):
        self.force = force
        self.countries = countries

        min_accepted = 2000
        max_accepted = 2020
        if(min_accepted <= first_year <= last_year <= max_accepted):
            self.first_year= first_year
            self.last_year = last_year
        elif min_accepted <= last_year <= first_year <= max_accepted:
            print("Année passée à l'envers, on les inverse")
            self.last_year = first_year
            self.first_year = last_year
        else :
            raise(NoSuchFileForThoseYears(first_year,last_year))
        
    def __call__(self):

        m = folium.Map(location=[0, 0], 
                    tiles= folium.TileLayer(attr="Openstreetmap",show=False,name="Openstreetmap"),
                    zoom_start=3,
                    attr='OpenStreetMap',
                    world_copy_jump=True    
        )
        folium.TileLayer(tiles='https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png',
                        attr="OpenStreetMap France",
                        name='OpenStreetMap France'
        ).add_to(m)

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            name='ESRI World Imagery',
            attr='Tiles © Esri',
            show=False
        ).add_to(m)

        folium.TileLayer(
            tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
            name='OpenTopoMap',
            attr='Map data © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)',
            show=False
        ).add_to(m)






        initJs= """const handler  = new overlayHandler("""+m.get_name()+""")"""
        for name in self.countries:
            country_obj = country(name,
                force=self.force,
                first_year=self.first_year,
                last_year=self.last_year)
            country_obj(m)
            initJs += country_obj.get_init_js()
        self.add_custom_script(m,initJs)
        folium.LayerControl().add_to(m)
                # Liste de différentes icônes et couleurs
        markers = [
            {'location': [45.5236, -122.6750], 'icon': 'info-sign', 'color': 'red', 'popup': 'Red Info Sign'},
            {'location': [45.5244, -122.6699], 'icon': 'cloud', 'color': 'blue', 'popup': 'Blue Cloud'},
            {'location': [45.5215, -122.6764], 'icon': 'heart', 'color': 'green', 'icon_color': 'white', 'popup': 'Green Heart'},
            {'location': [45.5222, -122.6655], 'icon': 'home', 'color': 'black', 'angle': 45, 'popup': 'Black Home'},
            {'location': [45.5250, -122.6780], 'icon': 'star', 'color': 'pink', 'popup': 'Pink Star'}
        ]

        # Ajouter les marqueurs à la carte
        for marker in markers:
            folium.Marker(
                location=marker['location'],
                popup=marker['popup'],
                icon=folium.Icon(
                    icon=marker['icon'],
                    color=marker.get('color', 'blue'),
                    icon_color=marker.get('icon_color', None),
                    angle=marker.get('angle', 0)
                )
            ).add_to(m)
        m.save(os.path.join("index.html"))
        print("Map sauvegardée !")

    def add_custom_script(self,m,jsCountriesInit):
        custom_js= """
        <style>
        .leaflet-image-layer {
            /* old android/safari*/
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges; /* safari */
            image-rendering: pixelated; /* chrome */
            image-rendering: -moz-crisp-edges; /* firefox */
            image-rendering: -o-crisp-edges; /* opera */
            -ms-interpolation-mode: nearest-neighbor; /* ie */
        }
        </style>
        <div style="position: fixed; bottom: 50px; left: 50px; width: 230px; z-index: 9999;">
            <label for="opacityRange">Opacité des calques
            <input type="range" id="opacityRange" min="0" max="100" value="60" step="1" 
                oninput="updateOpacity(this.value)"></label>
            <label for="year">Année : <span id="yearValue" name="yearValue">""" + str(math.ceil((self.first_year+self.last_year)/2)) + """</span>
            <br/>
            <input type="range" id="year" name="year" oninput="updateHist(this.value)" default=\""""+str(self.first_year)+"""" min=\""""+str(self.first_year)+"""" max=\""""+str(self.last_year)+"""" />
            </label>
            <br/>
            <label for="DMSP">DMSP
            <input type="radio" id="DMSP" name="sat" value="DMSP" onClick="updateImage(this.value)">
            </label>
            <label for="VIIRS">VIIRS
            <input type="radio" id="VIIRS" name="sat" value="VIIRS" onClick="updateImage(this.value)" >
            </label>
            <label for="noneSat">Aucun
            <input type="radio" id="noneSat" name="sat" value="null" onClick="updateImage(this.value)">
            </label>
            <br/>
            <label for="ntl_intensity">Intensité
            <input type="radio" id="ntl_intensity" name="typeVis" value="ntl_intensity">
            </label>
            <label for="kmeans">Kmeans
            <input type="radio" id="kmeans" name="typeVis" value="kmeans">
            </label>
            <label for="noneType">Masquer
            <input type="radio"  id="noneType" name="typeVis" value="null">
            </label>
        </div>
        <script src="./mapVis.js"></script>
        <script>
            function updateHist(year){
                var img = document.getElementById("hist_img");
                if (img){
                    let path = img.src.split("/")
                    let file = path[path.length-1].split("_")
                    file[file.length-1] = year + ".png" 
                    path[path.length-1] = file.join("_")
                    img.src = path.join("/");
                }
            }
            function updateImage(sat) {
            console.log(sat)
                if(sat == "null"){
                    var closeButton = document.getElementsByClassName("leaflet-popup-close-button")[0];
                    console.log(closeButton)
                    if(closeButton){
                    console.log("bruh")
                        closeButton.click() = false
                    }
                }
                var img = document.getElementById("hist_img");
                if (img){
                    let path = img.src.split("/")
                    let file = path[path.length-1].split("_")
                    file[2] = sat
                    path[path.length-1] = file.join("_")
                    img.src = path.join("/");
                }
                img = document.getElementById("kmeans_img");
                if (img){
                    let path = img.src.split("/")
                    path[path.length -3] = sat 
                    img.src = path.join("/");
                    document.getElementById("kmeans_sat").innerText = sat == "DMSP" ? "DMSP" : sat == "VIIRS" ? "VIIRS" : "none" ;
                    console.log(document.getElementById("kmeans_sat") , sat == "DMSP" ? "DMSP" : sat == "VIIRS" ? "VIIRS" : "none" )
                }
            }
            document.addEventListener("DOMContentLoaded", function() {
                """+jsCountriesInit+"""
            });
        </script>"""
        m.get_root().html.add_child(folium.Element(custom_js))

class country():
    def __init__(self,name,cluster=5,floor=0,force=[],first_year=2000, last_year=2020):
        raster_path = os.path.join("../data", name, "DMSP", "ntl_2000.tif")
        with rasterio.open(raster_path) as dataset:
            width = dataset.width
            height = dataset.height
            transform = dataset.transform
        #transformation de mercator 
        top_left_coords = rasterio.transform.xy(transform, 0, 0)
        bottom_right_coords = rasterio.transform.xy(transform, height - 1, width - 1)
        self.bounds = [[top_left_coords[1], top_left_coords[0]], [bottom_right_coords[1], bottom_right_coords[0]]]

        self.name = name    
        self.floor = floor
        self.force = force
        self.cluster = cluster
        self.last_year = last_year
        self.first_year  = first_year
        self.cluster_dir_name = f"{cluster}_{first_year}-{last_year}"
        
    def __call__(self,m):
        print("Ajout à la carte de",self.name)
        li_sat = ["DMSP","VIIRS"]
        self.pre_generate_overlays(li_sat)
        self.generate_grp().add_to(m)

    def pre_generate_overlays(self,li_sat):
        for sat in li_sat:
            self.pre_generate_kmeans(sat )
            self.pre_generate_ntl_intensities(sat )
        self.pre_generate_plp()

    def pre_generate_kmeans(self,sat ):
        base_path = os.path.join("../analysis",self.name,"kmeans_analysis",sat,
                            self.cluster_dir_name)
        if (not os.path.exists(os.path.join(base_path,f"cluster_img_{self.cluster}.png"))
            or not os.path.exists(os.path.join(base_path,f"clusters_{self.cluster}.png") 
            or ("kmeans" in self.force))):
            print("Kmeans non trouvé, création de l'image...")
            params = {
                'ntl_type': sat,
                'resize': True,
                'show': False,
                'clusters': [self.cluster]
            }
            main_kmeans(self.name, **params, first_year=self.first_year, last_year=self.last_year)
            print("Image créée, reprise de la suite")

    def pre_generate_plp(self):
        graphs = self.generate_graphs()
        if len(graphs) > 0:
            plp = combined(self.name,floor= self.floor , first_year=self.first_year, last_year=self.last_year)
            plp(show=False,graphs=graphs)

    def generate_graphs(self):  
        if ("plp" in self.force):
            return ["graph","histogramme_annuel"]  
        
        graphs = []
        base_path = os.path.join("../analysis",self.name,"lit_pixel_analysis","DMSP_et_VIIRS")   
        if (not os.path.exists(os.path.join(base_path,str(self.floor),"lit_pixel_combined.png"))
            or ("graph" in self.force)) :
            print("Graph non trouvé, lancement de sa création")
            graphs.append('graph')
        if  ("histogramme_annuel" in self.force) :
            graphs.append('histogramme_annuel')
        else : 
            if(self.is_missing_hist()):
              graphs.append('histogramme_annuel')  
        return graphs
    
    def is_missing_hist(self):
        for sat in ["DMSP","VIIRS"]:
            for year in (self.first_year,self.last_year):
                if not os.path.exists(os.path.join("../analysis",self.name,"lit_pixel_analysis",
                                                        "DMSP_et_VIIRS","histogramme",
                                                        f"histogramme_{self.name}_{sat}_{year}.png")):
                    print(f"histogrammes annuels non trouvés pour {self.name} avec {sat} en {year}, lancement de leur création")
                    return True
        
    def pre_generate_ntl_intensities(self,sat ):
        vmax = -1
        path = f'../analysis/{self.name}/ntl_intensity/{sat}'
        os.makedirs(path,exist_ok=True)
        for year in range(self.first_year,self.last_year+1):
            temp_path = path + f"/ntl_intensity_{year}_{sat}.png"
            if not os.path.exists(temp_path) or ("ntl_intensity" in self.force):
                print(f"{self.name} : Création de l'image des intensité nocturnes pour {sat} de {year}")
                if vmax == -1 :
                    if sat =="DMSP":
                        vmax= 64
                    else:
                        vmax= get_max_resized(self.name)
                self.generate_ntl_intensity(sat,year,temp_path,vmax)

    def generate_ntl_intensity(self,sat,year,path,vmax):
        img_path = os.path.join("../data",self.name,sat,f"ntl_{year}.tif")
        with rasterio.open(img_path) as img:
            raster_data = img.read(1)
        #raster_data = np.load(img_path)
        norm = Normalize(vmin=0, vmax=vmax)
        ntl_intensity_data = viridis(norm(raster_data))
        ntl_intensity_image = (ntl_intensity_data[:, :, :3] * 255).astype(np.uint8)
        ntl_intensity_pil = Image.fromarray(ntl_intensity_image)
        ntl_intensity_pil.save(path)
    
    def generate_grp(self):
        grp_country = folium.FeatureGroup(name=f"{self.name}",show=False)

        folium.Marker(
            [self.bounds[0][0], self.bounds[0][1]], 
            popup=folium.Popup( """
        <!DOCTYPE html>
        <html>
            <body>
                <p>Carte des clusters kmeans <span id="kmeans_sat">DMSP</span></p>
                <img id="kmeans_img" src="../analysis/"""+self.name+"""/kmeans_analysis/DMSP/"""+self.cluster_dir_name+"""/clusters_"""+str(self.cluster)+""".png"
                alt="Image" width="700" height="150">
            </body>
        </html>
        """, max_width=1400, lazy=True),
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(grp_country)
        
        folium.Marker(
            [self.bounds[0][0], self.bounds[0][1] + ((self.bounds[1][1] - self.bounds[0][1]) / 2)], 
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
            [self.bounds[0][0], self.bounds[1][1]], 
            icon=folium.Icon(color='purple', icon='info-sign'),
            popup=folium.Popup(
            """
                <div>
                    <p>Histogramme</p>
                    <img id="hist_img" src="../analysis/"""+self.name+"""/lit_pixel_analysis/DMSP_et_VIIRS/histogramme/histogramme_"""
                    +self.name+"""_DMSP_"""+str(math.ceil((self.first_year+self.last_year)/2))+""".png" alt="Image" width="852" height="535">
                </div>
            """,lazy=True,max_width=2000)
        ).add_to(grp_country)

        folium.Rectangle(
        bounds=[self.bounds[0], self.bounds[1]],
            color='black',
            weight = 2.5
        ).add_to(grp_country)
        return grp_country

    def get_init_js(self ):
        return """
        handler.addOverlay("""+str(self.first_year)+""","""+str(self.last_year)+""",'"""+self.name+"""',"""+str(self.cluster)+""","""+str(self.bounds)+""")"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name",nargs='+', help="Nom de la ou des zones d'études")
    parser.add_argument("-a", "--all",action='store_true', help="Prend toute les zones d'étude")
    parser.add_argument("-f","--force",nargs="+",default=[],help="Force la génération des objets générables(ntl_intensity,graph,kmeans)")
    parser.add_argument("-y","--year",nargs="+",default=[2000,2020],type=int,help="Années de début et de fin")
    args = parser.parse_args() 
    f = []
    for i in args.force:
        if i in ["kmeans", "k"]:
            f.append("kmeans")
        if i in ["plp", "p", "allumées"]:
            f.append("plp")
        if i in ["ntl_intensity", "ntl_i", "ntl", "n", "intensity", "i"]:
            f.append("ntl_intensity")
        if i in ["all", "tous", "a"]:
            f.append("kmeans")
            f.append("plp")
            f.append("ntl_intensity")
    
    if args.all:
        temp = []
        with os.scandir("../data") as files:
            for file in files:
                if file.is_dir():
                    temp.append(file.name)
    else:
        temp = args.name
    main = map_vis(temp,f, first_year=args.year[0], last_year=args.year[1])
    main()