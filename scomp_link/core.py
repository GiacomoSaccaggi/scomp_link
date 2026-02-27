# -*- coding: utf-8 -*-
"""

 ██████╗ ██████╗ ██████╗ ███████╗    ██████╗ ██╗██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝    ██╔══██╗██║██╔══██╗██╔════╝██║     ██║████╗  ██║██╔════╝
██║     ██║   ██║██████╔╝█████╗      ██████╔╝██║██████╔╝█████╗  ██║     ██║██╔██╗ ██║█████╗  
██║     ██║   ██║██╔══██╗██╔══╝      ██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝  
╚██████╗╚██████╔╝██║  ██║███████╗    ██║     ██║██║     ███████╗███████╗██║██║ ╚████║███████╗
 ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝

"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union
from .preprocessing.data_processor import Preprocessor
from .models.model_factory import ModelFactory
from .validation.model_validator import Validator

class ScompLinkPipeline:
    """
    ScompLinkPipeline: The "Astromech arm" for Python data projects.
    Automates the workflow from problem formulation to model evaluation.
    """

    def __init__(self, problem_description: str):
        self.problem_description = problem_description
        self.objectives = []
        self.df = None
        self.target_col = None
        self.feature_cols = []
        self.model_type = None
        self.model = None
        self.results = {}
        self.preprocessor = None

    def set_objectives(self, objectives: List[str]):
        """Formulation of objectives (Obiettivi)."""
        self.objectives = objectives
        print(f"Objectives formulated: {self.objectives}")

    def import_and_clean_data(self, df: pd.DataFrame):
        """Importation and cleaning of data (P3-P4)."""
        print("Importing and cleaning data...")
        self.preprocessor = Preprocessor(df)
        self.df = self.preprocessor.clean_data()
        print(f"Data imported. Rows after cleaning: {len(self.df)}")

    def select_variables(self, target_col: str, feature_cols: Optional[List[str]] = None):
        """Selection of target and objective variables (Sel)."""
        self.target_col = target_col
        if feature_cols:
            self.feature_cols = feature_cols
        else:
            self.feature_cols = [c for c in self.df.columns if c != target_col]
        print(f"Target selected: {self.target_col}")

    def choose_model(self, objective_type: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Logic based on the Mermaid graph decision tree.
        """
        metadata = metadata or {}
        print(f"Choosing model for objective: {objective_type}")
        
        if objective_type == "categorical_known":
            if metadata.get("data_type") == "images":
                if metadata.get("count_per_category", 0) < 500:
                    self.model_type = "Pre-trained model"
                else:
                    self.model_type = "CNN (ResNet/Inception)"
            elif metadata.get("exogenous_type") == "categorical":
                if metadata.get("num_features", 0) < 5:
                    self.model_type = "Theorical Psychometric Model"
                else:
                    self.model_type = "Naive Bayes / Classification Tree"
            else: # Mixed Categorical and Numerical
                if metadata.get("records_per_category", 0) < 300:
                    self.model_type = "SVC / K-Neighbors / Naive Bayes"
                else:
                    self.model_type = "SGD / Gradient Boosting / Random Forest"
        
        elif objective_type == "categorical_unknown":
            if metadata.get("categories_known"):
                self.model_type = "KMeans / Hierarchical Clustering"
            else:
                self.model_type = "Mean-Shift Clustering"
        
        elif objective_type == "numerical_study":
            if metadata.get("geospatial"):
                self.model_type = "Geostatistical Model / Kriging"
            elif metadata.get("time_series"):
                self.model_type = "UCM State Space"
            else:
                self.model_type = "Randomized PCA / Statistical Tests"
                
        elif objective_type == "numerical_prediction":
            num_records = len(self.df)
            if num_records < 1000:
                self.model_type = "Econometric Model"
            elif num_records < 100000:
                if metadata.get("only_numerical_exogenous"):
                    if metadata.get("all_variables_important"):
                        self.model_type = "Ridge / SVR"
                    else:
                        self.model_type = "Lasso / Elastic Net"
                else:
                    self.model_type = "Gradient Boosting / Random Forest"
            else: # > 100k
                if metadata.get("only_numerical_exogenous"):
                    self.model_type = "SGD Regressor"
                else:
                    self.model_type = "Gradient Boosting / Random Forest"
        
        elif objective_type == "multi_numerical_prediction":
            if metadata.get("time_series"):
                self.model_type = "VAR / VARMA"
            else:
                self.model_type = "MLP"
        
        print(f"Selected model type: {self.model_type}")
        self.model = ModelFactory.get_model(self.model_type, **metadata)

    def run_pipeline(self, test_size=0.2, task_type="regression", models_to_test=None, 
                     text_col=None, image_col=None, n_clusters=None, epochs=3, batch_size=32,
                     text_model='bert-base-uncased', text_language='en', use_contrastive=True,
                     use_ensemble=False, ensemble_strategy='voting', 
                     advanced_cv=False, cv_methods=['bootstrap'], bootstrap_iterations=1000):
        """
        Split, Modeling, and Evaluation.
        Supports: regression, classification, clustering, text, images
        
        New Parameters:
        - use_ensemble: Enable ensemble learning (voting/stacking)
        - ensemble_strategy: 'voting' or 'stacking'
        - advanced_cv: Enable advanced cross-validation (LOOCV, Bootstrap)
        - cv_methods: List of methods ['loocv', 'bootstrap']
        - bootstrap_iterations: Number of bootstrap iterations
        """
        if self.df is None or self.target_col is None:
            raise ValueError("Data and target must be defined before running the pipeline.")

        # CLUSTERING PATH
        if task_type == "clustering":
            print("MODELLAZIONE: Running Clustering...")
            from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
            from sklearn.metrics import silhouette_score
            
            X = self.df[self.feature_cols].values
            
            if "KMeans" in self.model_type or n_clusters:
                n_clusters = n_clusters or 3
                model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(X)
                metric_name = "Inertia"
                metric_value = model.inertia_
            else:  # Mean-Shift
                bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=min(500, len(X)))
                model = MeanShift(bandwidth=bandwidth)
                clusters = model.fit_predict(X)
                metric_name = "Clusters Found"
                metric_value = len(np.unique(clusters))
            
            silhouette = silhouette_score(X, clusters)
            
            self.results = {
                "status": "success",
                "model_type": self.model_type,
                "clusters": clusters.tolist(),
                "n_clusters": len(np.unique(clusters)),
                "metrics": {
                    metric_name: metric_value,
                    "silhouette_score": silhouette
                }
            }
            return self.results
        
        # TEXT CLASSIFICATION PATH
        if task_type == "text" and text_col:
            print("MODELLAZIONE: Running Text Classification...")
            try:
                if use_contrastive:
                    from .models.contrastive_text import ContrastiveTextClassifier
                    
                    classifier = ContrastiveTextClassifier(
                        model_name=text_model,
                        embedding_dim=128,
                        use_faiss=True
                    )
                    
                    print(f"Training contrastive text classifier for {epochs} epochs...")
                    classifier.train_contrastive(
                        self.df,
                        text_col=text_col,
                        label_col=self.target_col,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=test_size
                    )
                    
                    self.model = classifier
                    self.results = {
                        "status": "success",
                        "model_type": "Contrastive Text Classifier",
                        "metrics": {"trained": True, "epochs": epochs}
                    }
                else:
                    from .models.supervised_text import SpacyEmbeddingModel
                    from sklearn.model_selection import train_test_split
                    
                    # Prepare data for Spacy model
                    to_tag = {}
                    tagged = {}
                    for idx, row in self.df.iterrows():
                        to_tag[idx] = row[text_col]
                        tagged[idx] = row[self.target_col]
                    
                    categories = list(self.df[self.target_col].unique())
                    
                    classifier = SpacyEmbeddingModel(lan=text_language, model_name=text_model)
                    
                    print(f"Training Spacy text classifier with CNN for {epochs} epochs...")
                    nlp_model, scores = classifier.training(to_tag, tagged, categories)
                    
                    self.model = classifier
                    self.results = {
                        "status": "success",
                        "model_type": "Spacy Text Classifier (CNN)",
                        "metrics": scores
                    }
                
                return self.results
            except ImportError:
                print("⚠️  NLP dependencies not installed. Install with: pip install .[nlp]")
                raise
        
        # IMAGE CLASSIFICATION PATH
        if task_type == "image" and image_col:
            print("MODELLAZIONE: Running Image Classification...")
            try:
                from .models.supervised_img import CNNImg
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score, classification_report
                
                X = np.array(self.df[image_col].tolist())
                y = self.df[self.target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                n_classes = len(np.unique(y))
                img_shape = X[0].shape if len(X[0].shape) > 1 else (28, 28, 1)
                
                cnn = CNNImg(input_shape=img_shape, n_classes=n_classes)
                cnn.fit(X_train, y_train)
                y_pred = cnn.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                
                self.model = cnn
                self.results = {
                    "status": "success",
                    "model_type": "CNN Image Classifier",
                    "metrics": {"accuracy": accuracy}
                }
                return self.results
            except ImportError:
                print("⚠️  Image/CV dependencies not installed. Install with: pip install .[img]")
                raise
        
        # IMAGE CLUSTERING PATH
        if task_type == "image_clustering" and image_col:
            print("MODELLAZIONE: Running Image Clustering...")
            try:
                from .models.unsupervised_img import ClusterImg
                from sklearn.metrics import silhouette_score
                
                X = np.array(self.df[image_col].tolist())
                n_clusters = n_clusters or 4
                
                clusterer = ClusterImg(n_clusters=n_clusters, method='kmeans')
                clusters = clusterer.fit_predict(X)
                
                silhouette = silhouette_score(X, clusters)
                
                self.model = clusterer
                self.results = {
                    "status": "success",
                    "model_type": "Image Clustering",
                    "clusters": clusters.tolist(),
                    "n_clusters": len(np.unique(clusters)),
                    "metrics": {"silhouette_score": silhouette}
                }
                return self.results
            except ImportError:
                print("⚠️  Image/CV dependencies not installed. Install with: pip install .[img]")
                raise

        # STANDARD REGRESSION/CLASSIFICATION PATH
        print("P12: Preparing datasets...")
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_datasets(self.target_col, test_size=test_size)

        if task_type == "regression" and models_to_test:
            print("MODELLAZIONE: Running RegressorOptimizer...")
            from .models.regressor_optimizer import RegressorOptimizer
            optimizer = RegressorOptimizer(self.df, self.target_col, self.feature_cols, 
                                          x_complexity_col=self.feature_cols[0],
                                          models_to_test=models_to_test)
            optimizer.test_models_regression()
            
            # Ensemble Learning
            if use_ensemble and len(optimizer.model_results) > 1:
                print(f"\n🎯 Creating {ensemble_strategy} ensemble from {len(optimizer.model_results)} models...")
                from .models.ensemble_optimizer import EnsembleOptimizer
                
                base_models = [(name, result['Model']) for name, result in optimizer.model_results.items()]
                ensemble = EnsembleOptimizer(base_models, task_type='regression', strategy=ensemble_strategy)
                ensemble.fit(X_train, y_train)
                
                ensemble_scores = ensemble.evaluate_ensemble(X_train, y_train, cv=5)
                print(f"✅ Ensemble CV Score: {ensemble_scores['mean_score']:.4f} (±{ensemble_scores['std_score']:.4f})")
                
                self.model = ensemble.ensemble_model
                self.results = {
                    "optimizer_results": optimizer.model_results,
                    "ensemble_scores": ensemble_scores
                }
            else:
                self.model = list(optimizer.model_results.values())[0]['Model']
                self.results = {"optimizer_results": optimizer.model_results}
            return self.results

        if task_type == "classification" and models_to_test:
            print("MODELLAZIONE: Running ClassifierOptimizer...")
            from .models.classifier_optimizer import ClassifierOptimizer
            optimizer = ClassifierOptimizer(self.df, self.target_col, self.feature_cols, 
                                           models_to_test=models_to_test)
            optimizer.test_models_classification()
            
            # Ensemble Learning
            if use_ensemble and len(optimizer.model_results) > 1:
                print(f"\n🎯 Creating {ensemble_strategy} ensemble from {len(optimizer.model_results)} models...")
                from .models.ensemble_optimizer import EnsembleOptimizer
                
                base_models = [(name, result['Model']) for name, result in optimizer.model_results.items()]
                ensemble = EnsembleOptimizer(base_models, task_type='classification', strategy=ensemble_strategy)
                ensemble.fit(X_train, y_train)
                
                ensemble_scores = ensemble.evaluate_ensemble(X_train, y_train, cv=5)
                print(f"✅ Ensemble CV Score: {ensemble_scores['mean_score']:.4f} (±{ensemble_scores['std_score']:.4f})")
                
                self.model = ensemble.ensemble_model
                self.results = {
                    "optimizer_results": optimizer.model_results,
                    "ensemble_scores": ensemble_scores
                }
            else:
                self.model = list(optimizer.model_results.values())[0]['Model']
                self.results = {"optimizer_results": optimizer.model_results}
            return self.results

        if self.model is None:
            raise ValueError("Model must be chosen before running the pipeline.")

        print(f"MODELLAZIONE: Training {self.model_type}...")
        self.model.fit(X_train, y_train)
        
        print("VALUTAZIONE: Evaluating results...")
        y_pred = self.model.predict(X_test)
        y_proba = None
        if task_type == "classification" and hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_test)
        
        validator = Validator(self.model)
        metrics = validator.evaluate(y_test, y_pred, task_type=task_type, y_proba=y_proba)
        
        # Advanced Cross-Validation
        advanced_cv_results = None
        if advanced_cv:
            print("\n🔬 Running Advanced Cross-Validation...")
            from .validation.advanced_cv import AdvancedCV
            
            X = self.df[self.feature_cols]
            y = self.df[self.target_col]
            
            include_loocv = 'loocv' in cv_methods
            include_bootstrap = 'bootstrap' in cv_methods
            
            advanced_cv_results = AdvancedCV.evaluate_all(
                self.model, X, y,
                include_loocv=include_loocv,
                include_bootstrap=include_bootstrap,
                bootstrap_iterations=bootstrap_iterations
            )
            
            print("\n✅ Advanced CV Results:")
            for method, result in advanced_cv_results.items():
                print(f"  {result['method']}: {result['mean_score']:.4f} (±{result['std_score']:.4f})")
        
        validator.generate_validation_report(y_test, y_pred, task_type=task_type, 
                                             y_proba=y_proba, report_name="ScompLink_Validation_Report.html")
        
        self.results = {
            "status": "success",
            "model_type": self.model_type,
            "metrics": metrics,
            "advanced_cv": advanced_cv_results,
            "report_path": "ScompLink_Validation_Report.html"
        }
        
        return self.results

    def save_model(self, path='./staging'):
        """
        Save trained model to disk
        
        Parameters:
        -----------
        path : str
            Directory path to save model
        """
        import os
        import pickle
        
        os.makedirs(path, exist_ok=True)
        
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_path = os.path.join(path, 'model.pkl')
        
        # Handle different model types
        if hasattr(self.model, 'save'):
            # Contrastive text classifier
            self.model.save(path)
            print(f"✅ Model saved to {path}")
        else:
            # Standard sklearn models
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✅ Model saved to {model_path}")
        
        return path
    
    def load_model(self, path='./staging'):
        """
        Load trained model from disk
        
        Parameters:
        -----------
        path : str
            Directory path or file path to load model
        """
        import os
        import pickle
        
        # Check if it's a contrastive text model
        if os.path.isdir(path) and os.path.exists(os.path.join(path, 'model_config.json')):
            from .models.contrastive_text import ContrastiveTextClassifier
            classifier = ContrastiveTextClassifier()
            classifier.load(path)
            self.model = classifier
            print(f"✅ Contrastive model loaded from {path}")
        else:
            # Standard sklearn model
            model_path = os.path.join(path, 'model.pkl') if os.path.isdir(path) else path
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Model loaded from {model_path}")
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions using loaded model
        
        Parameters:
        -----------
        X : array-like
            Input data for prediction
            
        Returns:
        --------
        predictions : array
            Model predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first with load_model()")
        
        return self.model.predict(X)
