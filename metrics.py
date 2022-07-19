#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:10:35 2019

@author: af.rosa
"""

from sklearn.metrics import silhouette_score,adjusted_rand_score
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import math

def compute_matrix(clusters, labels):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for i in range(1, len(clusters)):
        for j in range(0, i):
            if labels[i] == labels[j]:
                if clusters[i] == clusters[j]:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if clusters[i] == clusters[j]:
                    false_positives += 1
                else:
                    true_negatives += 1
    return np.array([[true_positives, true_negatives],[false_positives, false_negatives]])        

def get_rand_index(m):
    denominator = m[0,0] + m[0,1] + m[1,0] + m[1,1]
    return float(m[0,0] + m[0,1]) / (denominator)
    
def get_precision(m):
    return float(m[0,0]) / (m[1,0] + m[0,0])

def get_recall(m):
    return float(m[0,0]) / (m[1,1] + m[0,0])

def get_f1(m):
    precision = get_precision(m)
    recall = get_recall(m)
    return 2.0*(float(precision*recall)/(precision+recall))

def getCohesion(X):
    cohesion = 0
    dist = euclidean_distances(X, X)
    for i in range(1, X.shape[0]):
        for j in range(0, i):
            cohesion += dist[i,j]
        
    n = X.shape[0]
    n_dists = (n*(n-1))/2 # the number of points in the clusters must matter
        
    return cohesion / n_dists if n_dists > 0 else 0 # avg dist between points of same cluster

def avgCohesion(X, clusters):
    labels = set(clusters)
    labels.discard(-1)
    avg_cohesion = 0.0
    for l in labels:
        avg_cohesion += getCohesion(X[clusters == l])
    return avg_cohesion/len(labels)

def normalizedAvgCohesion(X, clusters):
    labels = set(clusters)
    labels.discard(-1)
    sum_cohesion = 0.0
    max_cohesion = -1*math.inf
    min_cohesion = math.inf
    for l in labels:
        coh = getCohesion(X[clusters == l])
        sum_cohesion += coh
        if(coh > max_cohesion):
            max_cohesion = coh
        if(coh < min_cohesion):
            min_cohesion = coh
    
    avg = sum_cohesion/len(labels)
    
    return (avg - min_cohesion)/(max_cohesion - min_cohesion)  

def print_metrics(features, clusters, labels):
    print("n clusters: \t", len(set(clusters)))
    print("clusters without labels: ", len(set(clusters[labels == 0]).difference(set(clusters[labels != 0]))))
    
    m = compute_matrix(clusters[labels != 0], labels[labels != 0])

    print("silhouette: \t", silhouette_score(features, clusters))
    print("rand index: \t", get_rand_index(m))
    print("precision: \t", get_precision(m))
    print("recall: \t", get_recall(m))
    print("f1: \t\t", get_f1(m))
    print("adjusted rand: \t", adjusted_rand_score(labels[labels != 0], clusters[labels != 0])) 
    print("avg cohesion: \t", normalizedAvgCohesion(features, clusters))
    print("\n")

