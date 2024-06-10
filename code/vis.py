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

def create_heat_map_overlay(sat, country, year, bounds, vmax):
    path = f'../analysis/{country}/visu'
    img_path = os.path.join("../data", country, sat, f"ntl_{year}.tif")
    os.makedirs(path, exist_ok=True)
    path += f"/heatmap_{year}.png"
    with rasterio.open(img_path) as img:
        raster_data = img.read(1)

    norm = Normalize(vmin=0, vmax=vmax)
    heatmap_data = viridis(norm(raster_data))

    heatmap_image = (heatmap_data[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_image)
    heatmap_pil.save(path)

    return img_overlay(f"heatmap {sat} {year}", path, bounds)

def create_kmeans_overlay(sat, country, cluster, bounds):
    name = f'Kmeans {sat} {cluster}'
    image = os.path.join("../analysis", country, sat, "kmeans_analysis", str(cluster), f"cluster_img_{cluster}.png")
    try:
        image_overlay = img_overlay(name, image, bounds)
    except FileNotFoundError:
        print("Image non trouvée, création de l'image...", image)
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
    print("Lancement de la création de la carte")
    raster_path = os.path.join("../data", country, "DMSP", "ntl_2000.tif")
    with rasterio.open(raster_path) as dataset:
        width = dataset.width
        height = dataset.height
        transform = dataset.transform

        center_pixel = (width // 2, height // 2)
        center_coords = rasterio.transform.xy(transform, center_pixel[1], center_pixel[0])
        top_left_coords = rasterio.transform.xy(transform, 0, 0)
        bottom_right_coords = rasterio.transform.xy(transform, height - 1, width - 1)
        top_left_lat, top_left_lon = top_left_coords[1], top_left_coords[0]
        bottom_right_lat, bottom_right_lon = bottom_right_coords[1], bottom_right_coords[0] 

    bounds = [[top_left_lat, top_left_lon], [bottom_right_lat, bottom_right_lon]]
    m = folium.Map(location=[center_coords[1], center_coords[0]], zoom_start=10, tiles='https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png', attr='OpenStreetMap France')

    folium.Marker([top_left_lat, top_left_lon], popup='Top Left').add_to(m)
    folium.Marker([bottom_right_lat, bottom_right_lon], popup='Bottom Right').add_to(m)

    overlay_map(m, country, bounds)

    # Ajouter un contrôle de slider pour ajuster l'opacité
    slider_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; z-index: 9999;">
        <label for="opacityRange">Opacité des calques:</label>
        <input type="range" id="opacityRange" min="0" max="100" value="60" step="1" 
            oninput="updateOpacity(this.value)">
    </div>
    <script>
        function updateOpacity(value) {
            var opacity = value / 100;
            var overlays = document.getElementsByClassName('leaflet-image-layer');
            for (var i = 0; i < overlays.length; i++) {
                overlays[i].style.opacity = opacity;
            }
        }
    </script>
    """
    m.get_root().html.add_child(folium.Element(slider_html))

    m.save(os.path.join("index.html"))
    print("Map sauvegardée !")

def overlay_map(m, country, bounds):
    li_sat = ["DMSP", "VIIRS"]
    fg = folium.FeatureGroup(name='test Layer')
    
    for sat in li_sat:
        if (sat == "DMSP"):
            vmax = 64
        else:
            vmax = get_max_resized(country)
        for year in range(2000, 2021):
            print(f"Création de l'overlay de la heat map pour {sat} de {year}")
            create_heat_map_overlay(sat, country, year, bounds, vmax).add_to(m)
        print(f"Création de l'overlay de l'algo kmeans pour {sat}")
        create_kmeans_overlay(sat, country, 7, bounds).add_to(m)
    folium.LayerControl().add_to(m)
    
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Nom de la zone d'étude")
    args = parser.parse_args()
    vis(args.name)
