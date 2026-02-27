# -*- coding: utf-8 -*-
"""

██╗   ██╗███╗  ██╗ ██████╗██╗   ██╗██████╗ ███████╗██████╗ ██╗   ██╗██╗ ██████╗███████╗██████╗ 
██║   ██║████╗ ██║██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗██║   ██║██║██╔════╝██╔════╝██╔══██╗
██║   ██║██╔██╗██║╚█████╗ ██║   ██║██████╔╝█████╗  ██████╔╝╚██╗ ██╔╝██║╚█████╗ █████╗  ██║  ██║
██║   ██║██║╚████║ ╚═══██╗██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗ ╚████╔╝ ██║ ╚═══██╗██╔══╝  ██║  ██║
╚██████╔╝██║ ╚███║██████╔╝╚██████╔╝██║     ███████╗██║  ██║  ╚██╔╝  ██║██████╔╝███████╗██████╔╝
 ╚═════╝ ╚═╝  ╚══╝╚═════╝  ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝╚═════╝ ╚══════╝╚═════╝ 

██╗███╗   ███╗ ██████╗ 
██║████╗ ████║██╔════╝ 
██║██╔████╔██║██║  ██╗ 
██║██║╚██╔╝██║██║  ╚██╗
██║██║ ╚═╝ ██║╚██████╔╝
╚═╝╚═╝     ╚═╝ ╚═════╝ 
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans


class ClusterImg(BaseEstimator, ClusterMixin):
    """
    Simplified Image Clustering wrapper.
    
    Uses KMeans on flattened image features.
    For production, use CNN feature extraction + clustering.
    
    Example:
        model = ClusterImg(n_clusters=5)
        model.fit(X)  # X: image arrays
        labels = model.predict(X)
    """
    
    def __init__(self, n_clusters=8, **kwargs):
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.kmeans_ = None
        self.labels_ = None
    
    def fit(self, X, y=None):
        """
        Fit clustering model on image data.
        
        Args:
            X: Image data (n_samples, height, width, channels) or flattened features
            y: Ignored (unsupervised)
        """
        print(f"ClusterImg: Clustering {len(X)} images into {self.n_clusters} clusters")
        print("⚠️ Note: This is a placeholder. For production, use CNN feature extraction.")
        
        # Flatten images if needed
        if len(X.shape) > 2:
            X_flat = X.reshape(len(X), -1)
        else:
            X_flat = X
        
        # Apply KMeans
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels_ = self.kmeans_.fit_predict(X_flat)
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for images.
        
        Args:
            X: Image data
            
        Returns:
            Cluster labels
        """
        if self.kmeans_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Flatten images if needed
        if len(X.shape) > 2:
            X_flat = X.reshape(len(X), -1)
        else:
            X_flat = X
        
        return self.kmeans_.predict(X_flat)
    
    def fit_predict(self, X, y=None):
        """
        Fit and predict in one step.
        
        Args:
            X: Image data
            y: Ignored
            
        Returns:
            Cluster labels
        """
        self.fit(X, y)
        return self.labels_
