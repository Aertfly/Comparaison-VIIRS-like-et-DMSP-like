import folium
import argparse
import rasterio
import os
from kmeans import main_kmeans
from PIL import Image
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
from utils import get_max_resized

first_year = 2000
last_year = 2021

def create_heat_map(sat,country,year,path,vmax):
    img_path = os.path.join("../data",country,sat,f"ntl_{year}.tif")
    with rasterio.open(img_path) as img:
        raster_data = img.read(1)

    norm = Normalize(vmin=0, vmax=vmax)
    heatmap_data = viridis(norm(raster_data))

    heatmap_image = (heatmap_data[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_image)
    heatmap_pil.save(path)

def create_kmeans_overlay(sat,country,cluster,bounds):
    name=f'Kmeans {sat} {cluster}',
    image= os.path.join("../analysis",country,sat,"kmeans_analysis",str(cluster),f"cluster_img_{cluster}.png")
    try :
        image_overlay = img_overlay(name,image,bounds)
    except(FileNotFoundError):
        print("Image non trouvé, création de l'image...",image)
        params = {
            'ntl_type': sat,
            'resize': True,
            'show': False,
            'clusters': [cluster]
        }
        main_kmeans(country, **params)
        image_overlay = img_overlay(name, image, bounds)
        print("Image créée, reprise de la suite")
    return image_overlay

def img_overlay(name, path, bounds):
    top, bot = bounds
    image_overlay = folium.raster_layers.ImageOverlay(
        name=name,
        image=path,
        bounds=[[bot[0], top[1]], [top[0], bot[1]]],
        opacity=0.6,
        show=False
    )
    return image_overlay

def vis(country):
    global first_year 
    global last_year 
    print("Lancement de la création de la carte")
    raster_path = os.path.join("../data", country, "DMSP", "ntl_2000.tif")
    with rasterio.open(raster_path) as dataset:
        width = dataset.width
        height = dataset.height
        transform = dataset.transform

    center_pixel = (width // 2, height // 2)
    #transformation de mercato
    center_coords = rasterio.transform.xy(transform, center_pixel[1], center_pixel[0])
    top_left_coords = rasterio.transform.xy(transform, 0, 0)
    bottom_right_coords = rasterio.transform.xy(transform, height - 1, width - 1)

    top_left_lat, top_left_lon = top_left_coords[1], top_left_coords[0]
    bottom_right_lat, bottom_right_lon = bottom_right_coords[1], bottom_right_coords[0]
    bounds=[[top_left_lat, top_left_lon], [bottom_right_lat, bottom_right_lon]]

    m = folium.Map(location=[center_coords[1], center_coords[0]], 
                    zoom_start=10,
                    #tiles='https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png',
                    attr='OpenStreetMap France',
                    world_copy_jump=True
                )

    folium.Marker([top_left_lat, top_left_lon], popup='Top Left').add_to(m)
    folium.Marker([bottom_right_lat, bottom_right_lon], popup='Bottom Right').add_to(m)

    overlay_map(m,country,bounds)
    addJs(m,country,[center_coords[1], center_coords[0]],bounds)
    m.save(os.path.join("index.html"))
    print("Map sauvegardée !")


def overlay_map(m,country,bounds):
    li_sat = ["DMSP","VIIRS"]
    global first_year
    global last_year
    print(first_year,last_year)
    for sat in li_sat:
        vmax = -1
        for year in range(first_year,last_year):
            print("YEAR",year)
            path = f'../analysis/{country}/heatmap/{sat}'
            os.makedirs(path,exist_ok=True)
            path +=f"/heatmap_{year}_{sat}.png"
            if not os.path.exists(path):
                print(f"Création de la heat map pour {sat} de {year}")
                if vmax == -1 :
                    if sat =="DMSP":
                        vmax= 64
                    else:
                        vmax= get_max_resized(country)
                create_heat_map(sat,country,year,path,vmax)
        print(f"Création de l'overlay de l'algo kmeans pour {sat}")
        create_kmeans_overlay(sat,country,7,bounds).add_to(m)
        folium.FeatureGroup(name=f"{sat}",show=False).add_to(m)
    
    #TO DO ajouter legend heat map
    #legend = folium.map.CustomPane('image_legends', z_index=650)
    #legend.add_to(m)
    folium.LayerControl().add_to(m)


def addJs(m,country,center,bounds):
    global first_year 
    global last_year 
    path = f'../analysis/{country}/heatmap/'
    custom_js= """

    <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; z-index: 9999;">
        <label for="opacityRange">Opacité des calques</label>
        <input type="range" id="opacityRange" min="0" max="100" value="60" step="1" 
            oninput="updateOpacity(this.value)">
        <label for="year">Année : <span id="yearValue" name="yearValue">2010</span></label>
        <input type="range" id="year" name="year" default="2000" min="2000" max="2020" />
        
    </div>
    <script>
    function updateOpacity(value) {
        var opacity = value / 100;
        var overlays = document.getElementsByClassName('leaflet-image-layer');
        for (var i = 0; i < overlays.length; i++) {
            overlays[i].style.opacity = opacity;
        }
    }
    document.addEventListener("DOMContentLoaded", function() {
        const heatmap = {
            first_year: """ + str(first_year) + """,
            last_year: """ + str(last_year) + """,
            path: '""" + path + """',
            top_map: """ + str(bounds[0]) + """,
            bot_map: """ + str(bounds[1]) + """,
            map: """ + m.get_name() + """,
            imageOverlays: {},
            currentOverlay: null,
            currentSat: null,

            init() {
                const sats = ["DMSP", "VIIRS"];
                for (let sat of sats) {
                    let temp = {}
                    for (let i = this.first_year; i < this.last_year; i++) {
                        let n = "/heatmap_" + i + "_" + sat + ".png";
                        temp[i] = L.imageOverlay(this.path + '/' + sat + n,
                            [[this.bot_map[0], this.top_map[1]], [this.top_map[0], this.bot_map[1]]],
                            {opacity: 0.6}
                        );
                    }
                    this.imageOverlays[sat] = temp;
                }
                this.setListener()
            },

            updateMap(year) {
                if(this.currentSat){
                    this.removeCurrentOverlay();
                    let nextOverlay = this.imageOverlays[this.currentSat][year];
                    this.currentOverlay = nextOverlay ? nextOverlay.addTo(this.map) : null;
                    updateOpacity(document.getElementById('opacityRange').value);
                }
            },

            removeCurrentOverlay() {
                if (this.currentOverlay) this.map.removeLayer(this.currentOverlay);
            },

            setListener() {
                $('#year').on('input', function() {
                    heatmap.updateMap($(this).val());
                    span = document.getElementById("yearValue");
                    span.innerHTML = "";
                    span.appendChild(document.createTextNode($(this).val()));
                });

                var checkboxes = document.getElementsByClassName('leaflet-control-layers-selector');
                for (var i = 0; i < checkboxes.length; i++) {
                    checkboxes[i].addEventListener('change', function() {
                        innerText = this.nextSibling.textContent.trim();
                        if (this.checked) {
                            heatmap.currentSat = innerText;
                            heatmap.updateMap($('#year').val());
                        } else {
                            if(heatmap.currentSat == innerText){
                                heatmap.removeCurrentOverlay();
                                heatmap.currentSat = null
                            }
                        }
                    });
                }
            }
        };
        heatmap.init();
        console.log(heatmap)
    });
    </script>
    """
    m.get_root().html.add_child(folium.Element(custom_js))

    
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Nom de la zone d'étude")
    args = parser.parse_args()
    vis(args.name)
