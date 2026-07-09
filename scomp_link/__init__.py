# -*- coding: utf-8 -*-
"""
scomp-link: The Astromech Arm for Your Python Projects.

All public classes are lazily imported via PEP 562 (__getattr__).
Only the logger is loaded eagerly. Each class loads its dependencies
on first access, keeping `import scomp_link` near-instant.
"""

from .utils.logger import set_verbosity

__version__ = "1.2.15"

__all__ = [
    "set_verbosity",
    "__version__",
    "ScompLinkPipeline",
    "Preprocessor",
    "FeatureEngineer",
    "DataQualityReport",
    "ModelFactory",
    "RegressorOptimizer",
    "ClassifierOptimizer",
    "AnomalyDetector",
    "TimeSeriesAnomalyDetector",
    "TimeSeriesForecaster",
    "OptunaOptimizer",
    "HalvingSearchOptimizer",
    "EarlyStoppingCV",
    "Validator",
    "FairnessMetrics",
    "ShapExplainer",
    "LimeExplainer",
    "DriftDetector",
    "ScompArtifact",
]

_LAZY_IMPORTS = {
    # models
    "RegressorOptimizer": (".models.regressor_optimizer", "RegressorOptimizer"),
    "ClassifierOptimizer": (".models.classifier_optimizer", "ClassifierOptimizer"),
    "ModelFactory": (".models.model_factory", "ModelFactory"),
    "AnomalyDetector": (".models.anomaly_detector", "AnomalyDetector"),
    "TimeSeriesAnomalyDetector": (".models.ts_anomaly_detector", "TimeSeriesAnomalyDetector"),
    "TimeSeriesForecaster": (".models.forecaster", "TimeSeriesForecaster"),
    # tuning
    "OptunaOptimizer": (".models.advanced_tuning", "OptunaOptimizer"),
    "HalvingSearchOptimizer": (".models.advanced_tuning", "HalvingSearchOptimizer"),
    "EarlyStoppingCV": (".models.advanced_tuning", "EarlyStoppingCV"),
    # core
    "ScompLinkPipeline": (".core", "ScompLinkPipeline"),
    # preprocessing
    "Preprocessor": (".preprocessing.data_processor", "Preprocessor"),
    "FeatureEngineer": (".preprocessing.feature_engineer", "FeatureEngineer"),
    "DataQualityReport": (".preprocessing.data_quality", "DataQualityReport"),
    # validation
    "Validator": (".validation.model_validator", "Validator"),
    "FairnessMetrics": (".validation.fairness", "FairnessMetrics"),
    # explainability
    "ShapExplainer": (".explainability", "ShapExplainer"),
    "LimeExplainer": (".explainability", "LimeExplainer"),
    # monitoring
    "DriftDetector": (".monitoring", "DriftDetector"),
    # persistence
    "ScompArtifact": (".persistence", "ScompArtifact"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path, __package__)
        obj = getattr(module, attr_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'scomp_link' has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys()) + ["set_verbosity", "__version__"]
