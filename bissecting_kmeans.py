#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:12:46 2019

@author: andreffrosa
"""

from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import heapq 

class bissecting_kmeans:
        
    def _getSize(self, X):
        return -1*len(X)
    
    def _getCohesion(self, X):
        cohesion = 0
        dist = euclidean_distances(X, X)
        for i in range(1, X.shape[0]):
            for j in range(0, i):
                cohesion += dist[i,j]
        
        n = X.shape[0]
        n_dists = (n*(n-1))/2 # the number of points in the clusters must matter
        
        return cohesion / n_dists # avg dist between points of same cluster
    
    def _getSizeNCohesion(self, X):
        size = self._getSize(X)
        cohesion = self._getCohesion(X)
        
        return 0.5*size + 0.5*cohesion
    
    def __init__(self, n_clusters=8, score='size', min_size=10):
        self.n_clusters = n_clusters
        self.min_size = min_size
        
        if score == 'size':
            self.getScore = self._getSize
        elif score == 'cohesion':
            self.getScore = self._getCohesion
        elif score == 'size_and_cohesion':
            self.getScore = self._getSizeNCohesion
        else:
            raise ValueError("\'%s\' is not a valid score function" % (score))
    
    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.

        y : Ignored
            not used, present here for API consistency by convention.

        sample_weight : Ignored

        """
        labels = []
        for i in range(X.shape[0]):
                labels.append([])
        
        n_clusters = 1
        queue = []
        heapq.heapify(queue) 
        
        mask = np.ones(X.shape[0], dtype=int)
        indexes = np.where(mask)[0]

        counter = 0

        heapq.heappush(queue, (self.getScore(X[indexes]), counter, indexes) ) 
        
        counter += 1
        
        while len(queue) > 0 and n_clusters < self.n_clusters:
            ix = heapq.heappop(queue)[2]
            
            kmeans = KMeans(n_clusters=2) 
            bin_clusters = kmeans.fit_predict(X[ix])
        
            for i in range(len(bin_clusters)):
                labels[ix[i]].append(1 if bin_clusters[i] > 0 else 0)
            
            c1 = ix[bin_clusters > 0]
            c2 = ix[bin_clusters == 0]
            
            if len(c1) > self.min_size:
                to_push1 = (self.getScore(X[c1]), counter, c1)
                heapq.heappush(queue, to_push1 )
                counter += 1
            
            if len(c2) > self.min_size:
                to_push2 = (self.getScore(X[c2]), counter, c2)
                heapq.heappush(queue, to_push2 ) 
                counter += 1
            
            n_clusters += 1  
        
        self.tree_labels_ = labels
        self.labels_ = self._getLabels()
        return self
    
    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        y : Ignored
            not used, present here for API consistency by convention.

        sample_weight : Ignored

        Returns
        -------
        labels : list of lists, shape [[],]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_
    
    def _getLabels(self):
        
        def multiline_lambda(l):
            result = 0
            for i in np.linspace(len(l)-1, 0, num=len(l), dtype=int):
                result += l[i]*(2**i)
            return result
        
        aux_tags = list(map(lambda l: multiline_lambda(l), self.tree_labels_))
        
        cluster_tags = np.array(list(set(aux_tags)))

        final_tags = np.array(list(map(lambda i: np.where(cluster_tags==i)[0][0], aux_tags)), dtype=int)
        return final_tags
    
    def getTreeLabels(self):
        return self.tree_labels_
