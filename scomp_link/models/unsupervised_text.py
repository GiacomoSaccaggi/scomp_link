# -*- coding: utf-8 -*-
"""

██╗   ██╗███╗  ██╗ ██████╗██╗   ██╗██████╗ ███████╗██████╗ ██╗   ██╗██╗ ██████╗███████╗██████╗ 
██║   ██║████╗ ██║██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗██║   ██║██║██╔════╝██╔════╝██╔══██╗
██║   ██║██╔██╗██║╚█████╗ ██║   ██║██████╔╝█████╗  ██████╔╝╚██╗ ██╔╝██║╚█████╗ █████╗  ██║  ██║
██║   ██║██║╚████║ ╚═══██╗██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗ ╚████╔╝ ██║ ╚═══██╗██╔══╝  ██║  ██║
╚██████╔╝██║ ╚███║██████╔╝╚██████╔╝██║     ███████╗██║  ██║  ╚██╔╝  ██║██████╔╝███████╗██████╔╝
 ╚═════╝ ╚═╝  ╚══╝╚═════╝  ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝╚═════╝ ╚══════╝╚═════╝ 

████████╗███████╗██╗  ██╗████████╗
╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝
   ██║   █████╗   ╚███╔╝    ██║   
   ██║   ██╔══╝   ██╔██╗    ██║   
   ██║   ███████╗██╔╝╚██╗   ██║   
   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class TextEmbeddingClustering(BaseEstimator, ClusterMixin):
    """
    Text clustering using embeddings
    
    Parameters:
    -----------
    model_name : str
        Sentence transformer model name
    n_clusters : int
        Number of clusters
    method : str
        Clustering method ('kmeans' or 'hierarchical')
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', n_clusters=5, method='kmeans'):
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.method = method
        self.model = None
        self.clusterer = None
        
    def _initialize_model(self):
        """Initialize sentence transformer"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
    
    def fit(self, X, y=None):
        """
        Fit clustering model
        
        Parameters:
        -----------
        X : array-like of strings
            Text data
        """
        self._initialize_model()
        
        # Get embeddings
        embeddings = self.model.encode(X)
        
        # Cluster
        if self.method == 'kmeans':
            from sklearn.cluster import KMeans
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
        else:
            from sklearn.cluster import AgglomerativeClustering
            self.clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
        
        self.clusterer.fit(embeddings)
        return self
    
    def predict(self, X):
        """
        Predict cluster labels
        
        Parameters:
        -----------
        X : array-like of strings
            Text data
            
        Returns:
        --------
        labels : array
            Cluster labels
        """
        self._initialize_model()
        embeddings = self.model.encode(X)
        return self.clusterer.predict(embeddings)
    
    def fit_predict(self, X, y=None):
        """
        Fit and predict in one step
        
        Parameters:
        -----------
        X : array-like of strings
            Text data
            
        Returns:
        --------
        labels : array
            Cluster labels
        """
        self.fit(X, y)
        self._initialize_model()
        embeddings = self.model.encode(X)
        return self.clusterer.fit_predict(embeddings)
