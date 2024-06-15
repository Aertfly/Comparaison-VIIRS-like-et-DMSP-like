import argparse
import numpy as np
from kmeans import Kmeans  
from load_data_resized import NTLSoftLoaderResized  
def merge_clusters(preds, groups_to_merge):

    new_preds = preds.copy()
    for new_label, group in enumerate(groups_to_merge):
        print(f"Merging clusters: {group} into new label: {new_label}")
        for cluster in group:
            new_preds[new_preds == cluster] = new_label
    return new_preds

def main_kmeans(name, ntl_type="DMSP", clusters=[-1], show=False, groups_to_merge=None):
    print("Paramétre passé à kmeans",
          "name", name,
          "ntl", ntl_type,
          "clusters", clusters,
          "show", show,
          "groups_to_merge", groups_to_merge
    )
    n = name.lower()
    t = ntl_type.upper()

    print("On utilise les données redimensionnées")
    data = NTLSoftLoaderResized(n, ntl_type=t)

    data.load_ntls()
    ntls = data.ntls
    ntls = np.moveaxis(ntls, 0, -1)
    ntls = np.reshape(ntls, (-1, 21))

   
    if groups_to_merge:
        preds = np.arange(ntls.shape[0]) 
        preds = merge_clusters(preds, groups_to_merge)  

        unique_clusters = np.unique(preds)
        new_cluster_count = len(unique_clusters)
        print(f"Clusters après fusion : {new_cluster_count}")

        
        if clusters == [-1]:
            clusters = [new_cluster_count]  # Utiliser le nouveau nombre de clusters après fusion si K=-1

    # Si aucun cluster n'est spécifié, utilisez les clusters définis dans l'argument
    if clusters == [-1]:
        clusters = [10]  

   
    km = Kmeans()
    shape = ntls.shape  
    km(ntls, samples_for_distance_matrix=1000, shape=shape, show=show)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de fusion et d'exécution de K-means")
    parser.add_argument("--name", type=str, help="Nom du fichier")
    parser.add_argument("--ntl_type", type=str, help="Type de données")
    parser.add_argument("-c", "--clusters", nargs="+", type=int, default=[-1], help="Nombre de clusters")
    parser.add_argument("--merge", nargs="+", type=str, default=[], help="Liste de groupes de clusters à fusionner")
    parser.add_argument("--show", action="store_true", help="Afficher les résultats")

    args = parser.parse_args()

    groups_to_merge = []
    for group in args.merge:
        groups_to_merge.append([int(c) for c in group.split(",")])

    main_kmeans(args.name, ntl_type=args.ntl_type, clusters=args.clusters, show=args.show, groups_to_merge=groups_to_merge)
