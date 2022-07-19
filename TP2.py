#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:37:43 2019

@author: af.rosa
"""

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, FeatureAgglomeration
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier

from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import numpy as np
import pandas 
import math

from bissecting_kmeans import bissecting_kmeans
import tp2_aux
import metrics
import utils

np.seterr('raise')
    
# Load images
images = tp2_aux.images_as_matrix()

# Load labels
txt = np.loadtxt("labels.txt", delimiter=',')
labels = txt[:,1]
img_indexes = txt[:,0]

# Extract Features
try:
    features = np.loadtxt("features.txt", delimiter='\t')
    print("Features loaded sucessfully!")
except IOError:
    print("Extracting features...")
    features = utils.extract_features(images, 6)
    print("Saving features...")
    np.savetxt("features.txt", features, delimiter='\t')
    print("Done!")

# Standardizar 
features,_,_ = utils.standardize(features)

cluster_features = True
if(cluster_features):
    # Plot the graph to help choosing the threshold
    utils.sorted_kdist_graph(features.transpose(), k=3)

    f_neighs = kneighbors_graph(features.transpose(), 3, mode='connectivity')
    fa = FeatureAgglomeration(linkage="ward", distance_threshold=31.5, n_clusters=None, connectivity=f_neighs)
    f = fa.fit_transform(features)
    print("Reduced from", features.shape[1], "features to ", f.shape[1])
    features = f

analize_probs = True
if(analize_probs):
    labeled_samples = features[labels != 0]
    true_labels = labels[labels != 0]
    f,prob = f_classif(labeled_samples, true_labels)
    print("Probabilities of f-test")
    print(prob)
    
plot_features = True
if(plot_features):
    data = pandas.DataFrame(features)
    plt.figure(figsize=(20,20))
    data.hist(color='k', alpha=0.5, bins=20) 
    plt.tight_layout()
    #plt.savefig('features.png', dpi=300)
    plt.show()
    plt.close()


def worst_score(direction):
    return -1*math.inf if direction == 'max' else math.inf

def optimize(alg_name, getAlg, getScore, direction, getInterval):

    best_features = None
    best_score = worst_score(direction)
    best_k = -1
    best_clusters = None
    best_scores = None
    best_interval = None
    for f in range(1, features.shape[1]+1):
        print("features:", f)
    
        # Select the best features
        selected_features = utils.select_best_features(features, labels, f)
        
        interval = getInterval(selected_features)
    
        scores = []
        save_scores = False
    
        for k in range(0,len(interval)):
            alg = getAlg(interval[k], selected_features)
            clusters = alg.fit_predict(selected_features)
            score = getScore(selected_features, clusters, labels, direction)
            
            scores.append(score)
        
            
            if( (best_k == -1) or (direction == 'max' and score > best_score) or (direction == 'min' and score < best_score) ):
                best_score = score
                best_k = k
                best_features = selected_features
                best_clusters = clusters
                save_scores = True
                
        if(save_scores):
            best_scores = scores
            best_interval = interval
        
    print("\n")
    print(alg_name)
    print("best features:", best_features.shape[1])
    print("best param:", interval[best_k])
    print("n clusters:", len(set(best_clusters)))
    print("best score:", best_score)

    metrics.print_metrics(best_features, best_clusters, labels)
    
    tp2_aux.report_clusters(img_indexes, best_clusters, "%s.html" % alg_name)
    
    # Print Plot 
    utils.plot(alg_name, best_features.shape[1], best_interval, best_scores, best_score, best_interval[best_k])
        
def kmeans(value, selected_features):
    return KMeans(n_clusters=value)

def kmeans_interval(selected_features):
    return np.linspace(round(features.shape[0]/70), round(features.shape[0]/20), num=15, dtype=int)    

def dbscan(value, selected_features):
    return DBSCAN(eps=value)

def dbscan_interval(features):
    k=5
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(features, np.zeros(features.shape[0]))
    sorted_kdist = np.sort(neigh.kneighbors()[0][:,k-1])[::-1]
    
    first = math.floor(0.1*features.shape[0])
    last = math.floor(0.4*features.shape[0])
    indexes = np.linspace(first, last, num=30, dtype=int)
    
    # ignoring the plot because the method in the article is bad in this case
    #sorted_kdist_graph(features, k=5)
    
    return sorted_kdist[indexes]

def get_n_clusters(clusters):
    aux = set(clusters)
    aux.discard(-1)
    n_clusters = len(aux)
    return n_clusters

def opt_score_kmeans(features, clusters, labels, direction):
    n_clusters = get_n_clusters(clusters)
    
    m = metrics.compute_matrix(clusters[labels != 0], labels[labels != 0])
    precision = metrics.get_precision(m)
    cohesion = metrics.normalizedAvgCohesion(features, clusters)
    
    if(precision >= 0.75 and cohesion <= 0.35 and n_clusters >= 2):
        return metrics.get_f1(m)
    else:
        return worst_score(direction)

def opt_score_dbscan(features, clusters, labels, direction):
    n_clusters = get_n_clusters(clusters)
    
    m = metrics.compute_matrix(clusters[labels != 0], labels[labels != 0])
    precision = metrics.get_precision(m)
    
    if(n_clusters >= 2):
        return precision
    else:
        return worst_score(direction)

def bkmeans(value, selected_features):
    return bissecting_kmeans(n_clusters=value, score='size')

def agg(value, selected_features):
    agg_neighs = kneighbors_graph(selected_features, 5, mode='connectivity')
    return AgglomerativeClustering(linkage='ward', n_clusters=value, connectivity=agg_neighs)

optimize("KMEANS", kmeans, opt_score_kmeans, 'max', kmeans_interval) 
optimize("DBSCAN", dbscan, opt_score_dbscan, 'max', dbscan_interval) 
optimize("B-KMEANS", bkmeans, opt_score_kmeans, 'max', kmeans_interval) 
optimize("AGG", agg, opt_score_kmeans, 'max', kmeans_interval) 



# not used
#tp2_aux.report_clusters_hierarchical(label_indexes,bkmeans_clusters,"B-KMEANS")