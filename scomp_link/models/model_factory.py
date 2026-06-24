# -*- coding: utf-8 -*-
"""

███╗   ███╗ █████╗ ██████╗ ███████╗██╗             
████╗ ████║██╔══██╗██╔══██╗██╔════╝██║             
██╔████╔██║██║  ██║██║  ██║█████╗  ██║             
██║╚██╔╝██║██║  ██║██║  ██║██╔══╝  ██║             
██║ ╚═╝ ██║╚█████╔╝██████╔╝███████╗███████╗        
╚═╝     ╚═╝ ╚════╝ ╚═════╝ ╚══════╝╚══════╝        

███████╗ █████╗  █████╗ ████████╗ █████╗ ██████╗ ██╗   ██╗
██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗╚██╗ ██╔╝
█████╗  ███████║██║  ╚═╝   ██║   ██║  ██║██████╔╝ ╚████╔╝ 
██╔══╝  ██╔══██║██║  ██╗   ██║   ██║  ██║██╔══██╗  ╚██╔╝  
██║     ██║  ██║╚█████╔╝   ██║   ╚█████╔╝██║  ██║   ██║   
╚═╝     ╚═╝  ╚═╝ ╚════╝    ╚═╝    ╚════╝ ╚═╝  ╚═╝   ╚═╝   
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, SGDClassifier, SGDRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, MeanShift
import numpy as np

from .classifier_optimizer import ClassifierOptimizer
from .regressor_optimizer import RegressorOptimizer, Boruta
from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)

class ModelFactory:
    """
    Factory to create models based on the scomp-link decision tree.
    """
    @staticmethod
    def get_model(model_type: str, **kwargs):
        logger.info(f"Creating model of type: {model_type}")
        
        # New automated optimizers
        if model_type == "ClassifierOptimizer":
            return ClassifierOptimizer(**kwargs)
        elif model_type == "RegressorOptimizer":
            return RegressorOptimizer(**kwargs)
        
        # Text and Img models (lazy imports — torch/transformers loaded only here)
        if "Contrastive Text" in model_type or "Contrastive Learning" in model_type:
            try:
                from .contrastive_text import ContrastiveTextClassifier
                return ContrastiveTextClassifier(**kwargs)
            except ImportError:
                logger.info("⚠️ ContrastiveTextClassifier requires torch and transformers")
                return SpacyEmbeddingModel(**kwargs)  # Fallback
        elif "Spacy" in model_type or "Supervised Text" in model_type:
            from .supervised_text import SpacyEmbeddingModel
            return SpacyEmbeddingModel(**kwargs)
        elif "LDA" in model_type or "Unsupervised Text" in model_type:
            try:
                from .unsupervised_text import LDAText
                return LDAText(**kwargs)
            except ImportError:
                logger.info("⚠️ LDAText not available")
                return None
        
        # Image models
        elif "CNN" in model_type or "Supervised Img" in model_type:
            from .supervised_img import CNNImg
            return CNNImg(**kwargs)
        elif "Cluster Img" in model_type or "Unsupervised Img" in model_type:
            from .unsupervised_img import ClusterImg
            return ClusterImg(**kwargs)
        
        # Classification
        elif "Naive Bayes" in model_type:
            return GaussianNB()
        elif "Classification Tree" in model_type:
            return DecisionTreeClassifier()
        elif "SVC" in model_type:
            return SVC(probability=True)
        elif "K-Neighbors" in model_type:
            return KNeighborsClassifier()
        elif "SGD" in model_type and "Classifier" in model_type:
            return SGDClassifier()
        elif "Gradient Boosting" in model_type and "Classifier" in model_type:
            return GradientBoostingClassifier()
        elif "Random Forest" in model_type and "Classifier" in model_type:
            return RandomForestClassifier()
            
        # Regression
        elif "Ridge" in model_type:
            return Ridge()
        elif "Lasso" in model_type or "Elastic Net" in model_type:
            # Defaulting to ElasticNet as it covers both
            return ElasticNet()
        elif "SVR" in model_type:
            return SVR()
        elif "SGD Regressor" in model_type:
            return SGDRegressor()
        elif "Gradient Boosting" in model_type:
            return GradientBoostingRegressor()
        elif "Random Forest" in model_type:
            return RandomForestRegressor()
            
        # Clustering
        elif "KMeans" in model_type:
            return KMeans(n_clusters=kwargs.get("n_clusters", 8))
        elif "Mean-Shift" in model_type:
            return MeanShift()
            
        elif "Econometric Model" in model_type or "Theorical Psychometric Model" in model_type:
            # For econometric/psychometric models, we can use a basic LinearRegression as a starting point
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
            
        logger.info(f"Warning: Model type '{model_type}' not explicitly matched. Returning None.")
        return None
