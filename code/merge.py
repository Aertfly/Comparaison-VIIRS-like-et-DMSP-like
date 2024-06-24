import numpy as np
import os
import argparse
from kmeans import Kmeans,main_kmeans

#numpy.where machin =0 nous donne les indices des valeurs de machin qui sont égales à 0



def load_cluster_data(name,base_path=".",ntl_type="DMSP",nb_clusters=5):
    file_path = os.path.join(base_path, f'../analysis/{name}/kmeans_analysis/{ntl_type}/{nb_clusters}_{name}.npy')
    if not os.path.exists(file_path):
        print(f'Le fichier {file_path} n\'existe pas. Appel à kmeans')
        main_kmeans(name=name
                    ,ntl_type=ntl_type,
                    clusters=[nb_clusters],
                    show=False,
                    resize=True,
                    first_year=2000,
                    last_year=2020)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Même aprés l'appel à Kmeans le fichier {file_path} n'existe pas, vérifiez kmeans")
    return np.load(file_path)


def merge(name, merge_lists, base_path=".", ntl_type="DMSP", nb_clusters=5):
    # Créer une nouvelle matrice pour les clusters fusionnés
    cluster_data = load_cluster_data(name,base_path=base_path,ntl_type=ntl_type,nb_clusters=nb_clusters)

    new_cluster_data = np.copy(cluster_data)

    # Parcourir la liste de listes pour fusionner les classes
    for new_class, clusters_to_merge in enumerate(merge_lists):
            for cluster in clusters_to_merge:
                new_cluster_data[cluster_data == cluster] = new_class

    # Sauvegarder les nouvelles données dans un fichier .npy
    new_file_path = os.path.join(base_path, f'../analysis/{name}/kmeans_analysis/{ntl_type}/{nb_clusters    }_{name}_merged.npy')
    np.save(new_file_path, new_cluster_data)
    print(f'Les données fusionnées ont été sauvegardées dans {new_file_path}.')
    X = np.load(os.path.join(".",f'../analysis/{name}/kmeans_analysis/{ntl_type}/X_{nb_clusters}_{name}.npy'))
    return new_cluster_data,X

# Exemple d'utilisation avec arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fusionner les clusters d\'une ville.')
    parser.add_argument('--name',"-n", type=str, help='Le nom de la ville.')
    parser.add_argument('--NBclusters','-c', type=int,default=5, help='nb de clusters.')
    parser.add_argument('--merge_lists','-l', type=str, help='La liste des clusters à fusionner. Exemple: "[[0,1,3],[2,4]]".')
    parser.add_argument('--base_path', type=str,default= ".", help='Le chemin de base où les fichiers .npy sont stockés.')
    parser.add_argument('--ntl_type','-t', type=str, help='Le type de données NTL (viirs ou dmsp).')

    args = parser.parse_args()

    # Conversion de merge_lists de str à list of lists
    merge_lists = eval(args.merge_lists)

    # Appel de la fonction merge avec les arguments fournis
    new_preds,X = merge(args.name, merge_lists, args.base_path, args.ntl_type,args.NBclusters)
    
    sum=0
    j=len(merge_lists)
    for i in range(j):
        sum+=len(merge_lists[i])

    params = {'n_clusters': args.NBclusters-sum+j,
                'ntimes': 1,
                    'dist': 'euc',
                    'norm': "local",
                    'ntl' : args.ntl_type,
                    'dataset_name': args.name,
                    'x_range': range(2000, 2020+1)}
    
    km = Kmeans(**params, first_year=2000, last_year=2020)
    km.set_expe_path(km.expe_path[:-1] + "_merged/")
    km.vis(X,new_preds,(40,73),"SERT A RIEN",refc=None,show=True)

    # km(new_X, samples_for_distance_matrix=1000, shape= (72,101),show=True,raw=False)

    
