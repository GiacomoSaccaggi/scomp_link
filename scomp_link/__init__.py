from .models.regressor_optimizer import RegressorOptimizer
try:
    from .models.url_to_app_model import URLToAppNameExtractor
except ImportError:
    pass
from .models.classifier_optimizer import ClassifierOptimizer
from .models.anomaly_detector import AnomalyDetector
from .models.ts_anomaly_detector import TimeSeriesAnomalyDetector
from .core import ScompLinkPipeline
from .preprocessing.data_processor import Preprocessor
from .models.model_factory import ModelFactory
from .validation.model_validator import Validator

# New modules
try:
    from .explainability import ShapExplainer, LimeExplainer
except ImportError:
    pass
try:
    from .models.advanced_tuning import OptunaOptimizer, HalvingSearchOptimizer, EarlyStoppingCV
except ImportError:
    pass
from .monitoring import DriftDetector
from .persistence import ScompArtifact
from .preprocessing.feature_engineer import FeatureEngineer
from .preprocessing.data_quality import DataQualityReport
from .models.forecaster import TimeSeriesForecaster
from .validation.fairness import FairnessMetrics

from .utils.logger import set_verbosity

__version__ = "1.1.5"
