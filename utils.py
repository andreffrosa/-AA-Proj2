#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:27:45 2019

@author: af.rosa
"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap 
from sklearn.feature_selection import f_classif,SelectKBest
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

def standardize(data, means=None, stdevs=None):
    """
    Standardizes the data. Outputs standardized data, means and std deviation computed.
    """
    if means is None:
        means = np.mean(data[:,:], axis=0)
    
    if stdevs is None:
        stdevs = np.std(data[:,:], axis=0)
        
    data[:,:] = (data[:,:] - means)/stdevs
    return data,means,stdevs

def select_best_features(features, labels, n):
    labeled_samples = features[labels != 0]
    true_labels = labels[labels != 0]

    best_features_indexes = SelectKBest(f_classif, k=n).fit(labeled_samples, true_labels).get_support(indices=True)

    best_features = np.zeros((features.shape[0], best_features_indexes.shape[0]))
    for i in range(best_features_indexes.shape[0]):
        best_features[:,i] = features[:,best_features_indexes[i]]
    
    return best_features

def sorted_kdist_graph(X, k=5):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, np.zeros(X.shape[0]))
    sorted_kdist = np.sort(neigh.kneighbors()[0][:,k-1])[::-1]
    
    order = np.linspace(1, X.shape[0], X.shape[0], dtype=int)
    
    plt.figure()
    plt.title("Sorted distances to %dth neighbor (descending)" % k)
    plt.plot(order, sorted_kdist, 'b-')
    plt.xlabel("order")
    #plt.xticks(order, order)
    plt.ylabel("Distance to %dth neighbor" % k)
    plt.show()
    plt.close()

def plot_clusters(X,y_pred,title=''):
    """Plotting function; y_pred is an integer array with labels"""    
    plt.figure()
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.axis('equal')
    plt.show()
    

def extract_features(images, n_components):
    
    extractors = [PCA(n_components=n_components), 
                  TSNE(n_components=n_components, method='exact'),
                  Isomap(n_components=n_components)]
    
    total_features = np.zeros((images.shape[0], n_components*len(extractors)))
    
    for ix in range(len(extractors)):
        start = ix*n_components
        end = (ix+1)*n_components
        total_features[:,start:end] = extractors[ix].fit_transform(images)
        
    return total_features

def plot(alg_name, n_features, interval, scores, best_score, best_value):
    plt.figure()
    plt.plot(interval, scores, 'b-')
    
    title = '%s-%dF' % (alg_name, n_features)
    plt.title(title)
    plt.xlabel("range")
    plt.ylabel("score")
    
    plt.axhline(y=best_score)
    plt.axvline(x=best_value)
    
    plt.plot(interval, scores, linestyle='', color='red', markersize=8, marker='*')
    
    plt.savefig('%s.png' % title, dpi=300)
    plt.show()




