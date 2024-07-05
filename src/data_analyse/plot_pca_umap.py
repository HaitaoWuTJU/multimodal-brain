import pandas as pd
from sklearn.decomposition import PCA
import umap
from os.path import join as pjoin
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import torch,os
import open_clip
from torch.utils.data import DataLoader
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


loaded_data = torch.load('../results/image_features_intersection.pt')
all_features = loaded_data['all_features']
all_features_norm = loaded_data['all_features_norm']
all_labels = loaded_data['all_labels']
all_img_names = pd.read_csv('../results/train_intersection.csv')['Img'].values

k = 4
n_components_list=[10,50,100,200]
n_neighbors_list=[10,50,100,200]

# n_components_list=[10,50]
# n_neighbors_list=[10]

cartesian_product = list(itertools.product(n_components_list, n_neighbors_list))

plt.figure(figsize=(16, 16),dpi=900) 

for i,(n_components, n_neighbors) in enumerate(cartesian_product):
    print(i)
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(all_features)
    reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=0)
    embedding_2d = reducer.fit_transform(data_pca)

    
    kmeans = KMeans(n_clusters=k)
    knn_prediction = kmeans.fit_predict(data_pca)

    cluster_centers = kmeans.cluster_centers_

    distances = distance.cdist(cluster_centers, data_pca,'euclidean')
    nearses_k_index = np.argsort(distances, axis=1)[:,:50]


    plt.subplot(len(n_components_list), len(n_neighbors_list), i+1)
    for label in set(knn_prediction):
        indices = knn_prediction == label
        plt.scatter(embedding_2d[indices, 0], embedding_2d[indices, 1],label=label,s=1)  
        
        img_names = all_img_names[nearses_k_index[label]]
        img_names = [ele.rsplit('_',1)[0] for ele in img_names]
        for (point,img_name) in zip (embedding_2d[nearses_k_index[label]],img_names):
            plt.scatter(point[0], point[1],c='black',s=1)
    plt.title(f'{n_components}, {n_neighbors}')
    
plt.savefig('../results/UMAP_Embedding_pca(kmeans)_intersection.pdf', format='pdf', bbox_inches='tight')
plt.savefig('../results/UMAP_Embedding_pca(kmeans)intersection.png', bbox_inches='tight')