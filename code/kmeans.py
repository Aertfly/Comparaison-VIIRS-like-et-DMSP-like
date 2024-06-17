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


class Kmeans():

    def __init__(self, n_clusters=-1, ntimes=1, dist='euc', norm='local',ntl="DMSP", dataset_name='test', path='', x_range=None, aois=None, mask=None, first_year=2020, last_year=2020):
        self.n_clusters = n_clusters
        self.ntimes = ntimes

        # distance
        self.dist = all_dist[dist]
        self.dist_name = dist

        # norm
        self.norm = all_norm[norm]
        self.norm_name = norm

        # Années concernées
        self.first_year = first_year
        self.last_year = last_year

        # saving path
        self.dataset_name = dataset_name
        self.src_path = path + f'../analysis/{dataset_name}/kmeans_analysis/{ntl}/'
        self.distance_matrix_path = self.src_path #+ f'{self.dist_name}/' + f'{self.norm_name}/'
        self.expe_path = self.distance_matrix_path + f'{self.n_clusters}_{self.first_year}-{self.last_year}/'
        os.makedirs(self.expe_path, exist_ok=True)

        # visualization
        self.x_range = x_range
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

        #plt.figure(figsize=(8, 6))
        #plt.plot(K, inertias, 'bx-')
        #plt.xlabel('k')
        #plt.ylabel('Inertia')
        #plt.title('Elbow Method showing the optimal k')
        #plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
        #plt.show()
        
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
        # source : https://gdcoder.com/implementation-of-k-means-from-scratch-in-python-9-lines/

        '''
        X: multidimensional data
        max_iterations: number of repetitions before clusters are established

        Steps:
        1. Pick indices of k random point without replacement
        2. Find class (P) of each data point using distance
        3. Stop when max_iteration are reached or P matrix doesn't change

        Return:
        np.array: containg class of each data point
        '''

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

    def plot_a_class(self, X_vis, preds, ax, cls, color=(0, 0, 0)):

        centroid_vis_params = {'c': 'orange',
                               'label': 'Centroid',
                               'alpha': 1.}
        ex_vis_params = {'c': color,
                         'label': 'Examples in Cluster',
                         'alpha': 0.1}
        quantiles_vis_params = {'c': 'red',
                                'label': '1st and 9th decile',
                                'alpha': 1.
                                }

        c_cls = X_vis[preds == int(cls)]
        if c_cls.shape[0] > 0:
            indexes = np.random.randint(0, c_cls.shape[0], size=500)
            for ex in c_cls[indexes]:
                ax.plot(self.x_range, ex, **ex_vis_params)

            ax.plot(self.x_range, c_cls.mean(axis=0), **centroid_vis_params)
            #ax.plot(self.x_range, np.quantile(c_cls, 0.1, axis=0), **quantiles_vis_params)
            #ax.plot(self.x_range, np.quantile(c_cls, 0.9, axis=0), **quantiles_vis_params)
        else:
            ax.plot(self.x_range, [0]*len(self.x_range))
        
        ax.set_ylim(-30, 30)
        ax.set_xticks(self.x_range[::5])

        return ax

    def plot_kmeans_label(self, X_vis, preds, refc=None):

        # Sort the centroid to have the same color than a reference
        if refc is not None:
            idx = []
            for c in refc:
                c = c - c.mean()
                cntr = self.centroid - self.centroid.mean().reshape(-1, 1)
                distance = self.dist(np.expand_dims(c, axis=0), centroids=cntr)
                arg_dist = np.squeeze(np.argsort(distance))
                for amin in arg_dist:
                    if amin not in idx:
                        idx.append(amin)
                        break
        else:
            idx = range(self.n_clusters)
        self.idx = idx 

        print(self.idx)
        self.fig.legend(
            [mpatches.Patch(color=self.colors[i]) for i in range(self.n_clusters)],
            # [f"#={df_scores['card'][i]} - s={np.round(df_scores['mean'][i], 3)}" for i in self.idx]
            [f"{i} : #={self.df_scores['card'][self.idx[i]]}" for i in  range(self.n_clusters)],
            loc='lower center',
            ncol=self.n_clusters,
            bbox_to_anchor=(0.5, -0.05),
        )
        
        square = int(np.sqrt(self.n_clusters))

        nrow, ncol = square, square

        if square ** 2 < self.n_clusters:
            nrow += 1
            if square ** 2 < self.n_clusters - square:
                ncol += 1
        
        gs00 = gridspec.GridSpecFromSubplotSpec(nrow, ncol, subplot_spec=self.main_grid[0], hspace=0.5, wspace=0.3)


        for i, cls in enumerate(self.idx):
            ax = self.fig.add_subplot(gs00[i])
            ax = self.plot_a_class(X_vis, preds, ax, cls, self.colors[i])
            ax.set_title(f"{i}")

        return self.fig
    
    def plot_kmeans_label_only(self, X_vis, preds):

        fig, axes = plt.subplots(1, self.n_clusters,
                                figsize=set_size(width=tex_witdh_in_pt, 
                                                    subplots=(1, self.n_clusters), 
                                                    fact=2),
                                dpi=150
                                )

        for cls, ax in zip(range(self.n_clusters), axes):
            ax = self.plot_a_class(X_vis, preds, ax, cls, self.colors[cls])
            ax.set_title(f"{cls}")
        fig.tight_layout()
        return fig

    def make_cluster_map(self, preds, shape, df_scores):
        print("SHAPE",shape)   
        gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self.main_grid[1])

        ax = self.fig.add_subplot(gs01[0])

        if shape[0] is not None:
            img = np.zeros((shape[0], shape[1], 4))
            img[:, :, 0] = preds.reshape(shape)
            for i, cls in enumerate(self.idx):
                img[img[:, :, 0] == cls] = self.colors[i]
        else :
            img = np.zeros((200, 150, 4))

        # ax.legend()
        if self.mask is not None:
            print('masking...')
            print(self.mask.min(), self.mask.max())
            img[self.mask==255] = [1.]*4
            print("Done.")
        ax.imshow(img)
        # Create a Rectangle patch

        if self.aois is not None:
            
            for aoi_name, aoi in self.aois.items():
                if aoi_name[-1] == '*':
                    aoi_name = aoi_name[:-1]
                    c = 'g'
                else:
                    c = 'r'
                rect = patches.Rectangle(
                    (aoi[2], aoi[0]),
                    aoi[3]-aoi[2], 
                    aoi[1]-aoi[0], 
                    linewidth=1, 
                    edgecolor=c, 
                    facecolor='none'
                )

                # Add the patch to the Axes
                ax.add_patch(rect)
          
        plt.imsave(self.expe_path + f'cluster_img_{self.n_clusters}.png', img)
        plt.imsave(self.expe_path + f'cluster_img_{self.n_clusters}.svg', img)
        return self.fig

    def vis(self, X_vis, preds, shape, df_scores, refc=None, show = True):

        # gridspec inside gridspec
        self.fig = plt.figure(
            figsize=set_size(width=tex_witdh_in_pt, 
                             subplots=(3, self.n_clusters), 
                             fact=1.5),
            dpi=150
            )


        self.main_grid = gridspec.GridSpec(1, 2, figure=self.fig)
 
        subfig = self.plot_kmeans_label(X_vis, preds, refc)

        subfig = self.make_cluster_map(preds, shape, df_scores=df_scores)

        # plt.suptitle(f'Norm : {self.norm_name}, dist : {self.dist_name}, K = {self.n_clusters}')

        self.main_grid.tight_layout(figure=self.fig)
        
        self.fig.savefig(self.expe_path + f'plot_kmeans_{self.n_clusters}.png', bbox_inches='tight')
        self.fig.savefig(self.expe_path + f'plot_kmeans_{self.n_clusters}.svg')

        if show : plt.show()

        print("saving graph...")
        fig = self.plot_kmeans_label_only(X_vis, preds)
        fig.savefig(self.expe_path + f'clusters_{self.n_clusters}.png')
        fig.savefig(self.expe_path + f'clusters_{self.n_clusters}.svg')

    def __call__(self, X_raw, samples_for_distance_matrix, shape, refc=None , show = True):

        print('--------------' + f'K = {self.n_clusters}' + '---------------')
        print('Starting at ', str(datetime.datetime.now()))
        tic = time.time()

        X, mean = self.norm(X_raw)
        if self.n_clusters == -1 :
            self.n_clusters = self.optimal_k(X)
        colors = np.uint8((np.arange(0, self.n_clusters)/self.n_clusters)*255)
        self.colors = plt.get_cmap('terrain')(colors)
 
        if self.dist_name == 'dtw':
            self.centroid, preds = self.kmeans_fit_dtw(X, self.n_clusters, ntimes=self.ntimes)
        else:
            self.centroid, preds, _ = self.kmeans_fit(X)

        np.save(self.expe_path + 'centroids', self.centroid)

        if 'distance_matrix.npy' in os.listdir(self.distance_matrix_path):
            d_mat, rand = self.load_distance_matrix()

        else:
            print('Distance matrix not computed.')
            print('Computing distance matrix ...')
            d_mat, rand = self.compute_and_save_dist_mat(X, self.dist, samples_for_distance_matrix)
            print('Done')

        df_scores, global_sil_scores = self.evaluate(d_mat, preds, rand)

        print('Inertia : ', self.inertia(X, self.centroid, self.dist))
        print('Average Silhouette score : ', global_sil_scores.mean())
        print('Averaged per Class Silhouette Score : ', df_scores['mean'].mean())
 
        self.vis(X, preds, shape, df_scores=df_scores, refc=refc,show=show)

        print('Ending at ', str(datetime.datetime.now()))
        toc = time.time()
        print('Completion time : ', time.strftime('%H:%M:%S', time.gmtime(toc - tic)))

        return self.centroid, df_scores
    
def main_kmeans(name,ntl_type="DMSP",clusters=[-1],show=False,resize=False, first_year=2000, last_year=2020):
    print("Paramètres passés à kmeans :",
          "|name :",name,
          "|ntl :",ntl_type,
          "|clusters :",clusters,
          "|show :",show,
          "|resize :",resize,
          "|first_year :",first_year,
          "|last_year :",last_year
    )
    n = name.lower()
    t =  ntl_type.upper()

    if resize :
        print("On utilise les données redimensionnées")
        data = NTLSoftLoaderResized(n, ntl_type=t, first_year=first_year, last_year=last_year)
    else :
        print("On utilise les données normales")
        data = NTLSoftLoader(n, ntl_type=t, first_year=first_year, last_year=last_year)
        

    data.load_ntls()
    ntls = data.ntls
    ntls = np.moveaxis(ntls, 0, -1)
    ntls = np.reshape(ntls, (-1, last_year-first_year+1))

    for norm in ['local']:
        for i in clusters:

            params = {'n_clusters': i,
                      'ntimes': 10,
                      'dist': 'euc',
                      'norm': norm,
                      'ntl' : ntl_type,
                      'dataset_name': name,
                      'x_range': range(first_year, last_year+1)}

            km = Kmeans(**params, first_year=first_year, last_year=last_year)

            km(ntls, samples_for_distance_matrix=1000, shape=data.getShape(),show=show)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name","-n", help="dataset name to use")
    parser.add_argument("--ntl_type","-t", help="dataset type to use")
    parser.add_argument("-c","--clusters",nargs='+',default=["-1"],help="List of cluster ")
    parser.add_argument("--noResize","-nr",action='store_true',help ="tell the prog to not use resized data")
    parser.add_argument("--noShow","-ns",action='store_true',help ="tell the end graph to not be shown")
    args = parser.parse_args()
    clusters = [int(a) for a in args.clusters]
    params = {
        'ntl_type':args.ntl_type,
        'resize': not args.noResize,
        'show': not args.noShow,
        'clusters' : clusters
    }
    main_kmeans(args.name,**params)
