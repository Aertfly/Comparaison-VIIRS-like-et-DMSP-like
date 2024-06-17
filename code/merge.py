import datetime
import os
import time

import numpy as np
import pandas as pd
import scipy.spatial
from tqdm import tqdm
import tslearn.clustering
import tslearn.metrics
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn
import sklearn.metrics
import matplotlib.gridspec as gridspec
import argparse
from metrics import set_size, tex_witdh_in_pt
from matplotlib import patches

from load_data_resized import NTLSoftLoaderResized
from load_data import NTLSoftLoader
from kneed import KneeLocator

def norm_global(X):
    mean = X.mean(axis=0)
    return X - mean, mean

def norm_local(X):
    mean = X.mean(axis=1).reshape(-1, 1)
    return X - mean, mean

def compute_dist_dtw(X, centroids):
    distance = tslearn.metrics.cdist_dtw(X, centroids, n_jobs=6)
    return distance

def compute_dist_euc(X, centroids):
    distance = scipy.spatial.distance.cdist(X, centroids, 'euclidean')
    return distance

def compute_dist_cos(X, centroids):
    distance = scipy.spatial.distance.cdist(X, centroids, 'cosine')
    return distance

def compute_dist_jen(X, centroids):
    distance = scipy.spatial.distance.cdist(X, centroids, 'jensenshannon')
    return distance

all_dist = {'cos': compute_dist_cos,
            'euc': compute_dist_euc,
            'jen': compute_dist_jen,
            'dtw': compute_dist_dtw}

all_norm = {'global': norm_global,
            'local': norm_local,
            'no_norm': lambda  X: (X, 0),
            'plus_one': lambda X: (X + 1, 0)}

def merge_clusters(preds, groups_to_merge):
    new_preds = preds.copy()
    for new_label, group in enumerate(groups_to_merge):
        print(f"Merging clusters: {group} into new label: {new_label}")
        for cluster in group:
            new_preds[new_preds == cluster] = new_label
    return new_preds

class Kmeans():

    def __init__(self, n_clusters=-1, ntimes=1, dist='euc', norm='local',ntl="DMSP", dataset_name='test', path='', x_range=None, aois=None, mask=None):
        self.n_clusters = n_clusters
        self.ntimes = ntimes

        # distance
        self.dist = all_dist[dist]
        self.dist_name = dist

        # norm
        self.norm = all_norm[norm]
        self.norm_name = norm

        # saving path
        self.dataset_name = dataset_name
        self.src_path = path + f'../analysis/merged/{dataset_name}/kmeans_analysis/{ntl}/'
        self.distance_matrix_path = self.src_path #+ f'{self.dist_name}/' + f'{self.norm_name}/'
        self.expe_path = self.distance_matrix_path + f'{self.n_clusters}/'
        os.makedirs(self.expe_path, exist_ok=True)

        # visualization
        self.x_range = x_range if x_range is not None else np.arange(X.shape[1])
        self.aois = aois
        self.mask = mask

    def optimal_k(self, X,max_clusters=15):  
        print("Nombre de cluster optimal en cours de recherche...")
        inertias = []
        K = range(1, max_clusters + 1)
        
        for k in K:
            km = Kmeans(n_clusters=k, ntimes=10, dist=self.dist_name, norm=self.norm_name,
                        dataset_name=self.dataset_name, x_range=self.x_range,
                        aois=self.aois, mask=self.mask)
            centroids, _, _ = km.kmeans_fit(X)
            inertias.append(km.inertia(X, centroids, self.dist))
        
        kn = KneeLocator(K, inertias, curve='convex', direction='decreasing')
        optimal_k = kn.knee
        
        return optimal_k

    def kmeans_plusplus(self, X, n_clusters, dist):
        index = np.random.randint(0, X.shape[0])
        center0 = X[index]
        X_copy = np.copy(X)
        X_copy = np.delete(X_copy, index, 0)
        centroids = np.array([center0])

        for _ in range(n_clusters - 1):
            distances = dist(X_copy, centroids)
            distances.sort(axis=1)
            index = np.argmax(distances[:, 0])

            center1 = X_copy[index]
            centroids = np.append(centroids, [center1], axis=0)
            X_copy = np.delete(X_copy, index, 0)

        return centroids

    def kmeans_fit(self, X):
        cond = None
        centroid_list = []
        inertia_list = []
        is_cond_reached = []
        Ps = []

        for _ in tqdm(range(self.ntimes)):
            centroids = self.kmeans_plusplus(X, self.n_clusters, self.dist)
            P = np.argmin(self.dist(X, centroids), axis=1)
            for _ in range(100):
                centroids = np.vstack([X[P == i, :].mean(axis=0) for i in range(self.n_clusters)])
                tmp = np.argmin(self.dist(X, centroids), axis=1)
                cond = False
                if np.array_equal(P, tmp):
                    cond = True
                    break
                P = tmp
            centroid_list.append(centroids)
            inertia_list.append(self.inertia(X, centroids, self.dist))
            is_cond_reached.append(cond)
            Ps.append(P)

        good_kmeans_index = np.argmin(inertia_list)

        return centroid_list[good_kmeans_index], Ps[good_kmeans_index], is_cond_reached[good_kmeans_index]

    def kmeans_fit_dtw(self, X, n_clusters, ntimes=1):
        km = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', n_init=ntimes)
        km.fit_transform(X)
        return km.cluster_centers_.reshape(n_clusters, X.shape[1]), km.labels_

    def compute_and_save_dist_mat(self, X, dist, n_samples):
        rand = np.random.randint(0, X.shape[0], size=n_samples)
        np.save(self.distance_matrix_path + 'random_indexes', rand)
        X = X[rand]
        d_mat = dist(X, X)
        np.save(self.distance_matrix_path + 'distance_matrix', d_mat)
        return d_mat, rand

    def load_distance_matrix(self):
        rand = np.load(self.distance_matrix_path + 'random_indexes.npy')
        d_mat = np.load(self.distance_matrix_path + 'distance_matrix.npy')
        return d_mat, rand

    def inertia(self, X, centroids, dist):
        return np.sum(np.square(np.min(dist(X, centroids), axis=1)))

    def predict(self, X, centroids, dist):
        return np.argmin(dist(X, centroids), axis=1)

    def evaluate(self, distance_matrix, preds, rand):
        rand_preds=preds[rand]
        sil_scores = sklearn.metrics.silhouette_samples(distance_matrix, rand_preds, metric='precomputed')
        np.save(self.expe_path + 'silhouette_scores', sil_scores)
        MEAN = []
        MAX = []
        MIN = []
        STD = []
        CLUSTERS = []
        CARD = []
        FIRST_DEC = []
        LAST_DEC = []

        for k in range(self.n_clusters):
            cluster = sil_scores[rand_preds == k]
            max_ = np.max(cluster) if cluster.shape[0] != 0 else np.nan
            min_ = np.min(cluster) if cluster.shape[0] != 0 else np.nan
            mean = np.mean(cluster) if cluster.shape[0] != 0 else np.nan
            std = np.std(cluster) if cluster.shape[0] != 0 else np.nan
            first_dec = np.quantile(cluster, 0.1) if cluster.shape[0] != 0 else np.nan
            last_dec = np.quantile(cluster, 0.9) if cluster.shape[0] != 0 else np.nan

            MAX.append(max_)
            MIN.append(min_)
            MEAN.append(mean)
            STD.append(std)
            CLUSTERS.append(k)
            CARD.append(preds[preds == k].shape[0])
            FIRST_DEC.append(first_dec)
            LAST_DEC.append(last_dec)

        df_scores = pd.DataFrame({'clusters': CLUSTERS,
                                  'min': MIN,
                                  'max': MAX,
                                  'mean': MEAN,
                                  'std': STD,
                                  'card': CARD,
                                  'first_dec': FIRST_DEC,
                                  'last_dec': LAST_DEC,
                                  'centroids': [i for i in range(self.n_clusters)]})

        df_scores.to_csv(self.expe_path + 'averaged_silhouette_scores')
        self.df_scores = df_scores
        return df_scores, sil_scores
    def plot_a_class(self, X_vis, preds, ax, i, color):
        example = X_vis[preds == i].mean(axis=0)
        ex_vis_params = dict(linewidth=2, label=f'Cluster {i}', color=color)
        ax.plot(self.x_range, example, **ex_vis_params)

    def plot_classes(self, X_vis, preds):
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.Spectral(np.linspace(0, 1, self.n_clusters))
        for i, color in enumerate(colors):
            self.plot_a_class(X_vis, preds, ax, i, color)
        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity')
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(self.expe_path + 'plot_classes.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='merge.py')
    parser.add_argument('--n_clusters', type=int, default=2, help='Number of clusters')
    parser.add_argument('--dist', type=str, default='euc', help='Distance used for clustering (euc, cos, jen, dtw)')
    parser.add_argument('--norm', type=str, default='local', help='Normalization method (global, local, no_norm, plus_one)')
    parser.add_argument('--dataset_name', type=str, default='test_dataset', help='Name of the dataset')
    parser.add_argument('--ntl', type=str, default='DMSP', help='Name of the data')
    parser.add_argument('--path', type=str, default='./results/', help='Path to save')
    parser.add_argument('--merge_clusters', type=str, help='Name of the data')
    args = parser.parse_args()

    loader = NTLSoftLoader()

    data = loader.load(args.dataset_name, args.ntl)
    X = data[0]
    km = Kmeans(n_clusters=args.n_clusters, dist=args.dist, norm=args.norm, dataset_name=args.dataset_name, path=args.path, x_range=np.arange(X.shape[1]))
    centroids, preds, _ = km.kmeans_fit(X)
    km.compute_and_save_dist_mat(X, km.dist, 10000)
    km.evaluate(km.load_distance_matrix()[0], preds, km.load_distance_matrix()[1])
    km.plot_classes(X, preds)

    groups_to_merge = []
    if args.merge_clusters:
        merge_clusters = args.merge_clusters.split(',')
        for merge in merge_clusters:
            groups_to_merge.append(list(map(int, merge.split('-'))))
        new_preds = merge_clusters(preds, groups_to_merge)
        km.evaluate(km.load_distance_matrix()[0], new_preds, km.load_distance_matrix()[1])
        km.plot_classes(X, new_preds)

if __name__ == '__main__':
    main()
