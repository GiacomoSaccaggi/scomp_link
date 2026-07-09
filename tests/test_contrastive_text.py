# -*- coding: utf-8 -*-
"""
Tests for ContrastiveTextClassifier, EmbeddingSelector, InfoNCELoss, ContrastiveDataset.
Uses mocks to avoid downloading BERT — tests logic, not model weights.
"""
import json
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

torch_available = True
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(not torch_available, reason="torch/transformers not installed")


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_classifier():
    """Create a ContrastiveTextClassifier with mocked BERT weights."""
    from scomp_link.models.contrastive_text import ContrastiveTextClassifier

    with patch.object(AutoTokenizer, 'from_pretrained') as mock_tok, \
         patch.object(AutoModel, 'from_pretrained') as mock_model:

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.side_effect = lambda *args, **kwargs: {
            'input_ids': torch.randint(0, 1000, (len(args[0]) if isinstance(args[0], list) else 1, 128)),
            'attention_mask': torch.ones(len(args[0]) if isinstance(args[0], list) else 1, 128, dtype=torch.long),
        }
        mock_tok.return_value = tokenizer

        # Mock BERT model
        bert = MagicMock()
        bert.config = MagicMock()
        bert.config.hidden_size = 768
        bert.return_value = MagicMock(last_hidden_state=torch.randn(1, 128, 768))
        bert.train = MagicMock(return_value=bert)
        bert.eval = MagicMock(return_value=bert)
        bert.parameters = MagicMock(return_value=iter([torch.randn(10)]))
        bert.named_parameters = MagicMock(return_value=iter([("weight", torch.randn(10))]))
        mock_model.return_value = bert

        clf = ContrastiveTextClassifier(model_name='bert-base-uncased', use_faiss=False, embedding_dim=64)
    return clf


@pytest.fixture
def sample_df():
    """Sample text classification DataFrame."""
    return pd.DataFrame({
        'text': [
            "Machine learning is great", "Deep neural networks are powerful",
            "Football match today", "Basketball game results",
            "Stock market crash news", "Economic indicators rising",
            "Python programming tips", "JavaScript frameworks compared",
            "Tennis tournament score", "Hockey final results",
            "AI research breakthroughs", "Data science pipeline",
            "Soccer world cup update", "Rugby championship",
            "Crypto market analysis", "Banking regulations",
        ],
        'label': [
            "tech", "tech", "sports", "sports",
            "finance", "finance", "tech", "tech",
            "sports", "sports", "tech", "tech",
            "sports", "sports", "finance", "finance",
        ]
    })


# ═══════════════════════════════════════════════════════════════════
# INFO NCE LOSS
# ═══════════════════════════════════════════════════════════════════

class TestInfoNCELoss:
    def test_forward_no_labels(self):
        """InfoNCE loss without labels uses diagonal as positives."""
        from scomp_link.models.contrastive_net import InfoNCELoss
        loss_fn = InfoNCELoss(temperature=0.07)
        
        emb1 = torch.randn(8, 64)
        emb2 = torch.randn(8, 64)
        emb1 = torch.nn.functional.normalize(emb1, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, dim=1)
        
        loss = loss_fn(emb1, emb2)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_forward_with_labels(self):
        """InfoNCE loss with labels uses same-label pairs as positives."""
        from scomp_link.models.contrastive_net import InfoNCELoss
        loss_fn = InfoNCELoss(temperature=0.1)
        
        emb1 = torch.randn(8, 64)
        emb2 = torch.randn(8, 64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
        
        loss = loss_fn(emb1, emb2, labels)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_perfect_alignment_low_loss(self):
        """When pairs are perfectly aligned, loss should be lower."""
        from scomp_link.models.contrastive_net import InfoNCELoss
        loss_fn = InfoNCELoss(temperature=0.07)
        
        # Create embeddings where emb1[i] == emb2[i] (perfect pairs)
        emb = torch.randn(8, 64)
        emb = torch.nn.functional.normalize(emb, dim=1)
        
        loss_aligned = loss_fn(emb, emb)
        loss_random = loss_fn(emb, torch.nn.functional.normalize(torch.randn(8, 64), dim=1))
        
        assert loss_aligned < loss_random


# ═══════════════════════════════════════════════════════════════════
# CONTRASTIVE DATASET
# ═══════════════════════════════════════════════════════════════════

class TestContrastiveDataset:
    def test_init_custom_columns(self, sample_df):
        """ContrastiveDataset accepts custom column names."""
        from scomp_link.models.contrastive_net import ContrastiveDataset
        
        tokenizer = MagicMock()
        tokenizer.side_effect = lambda *args, **kwargs: {
            'input_ids': torch.randint(0, 1000, (1, 128)),
            'attention_mask': torch.ones(1, 128, dtype=torch.long),
        }
        
        ds = ContrastiveDataset(sample_df, tokenizer, text_col='text', label_col='label')
        assert len(ds) == len(sample_df)
        assert len(ds.unique_labels) == 3

    def test_getitem_returns_correct_shapes(self, sample_df):
        """Each item returns (text_ids, text_mask, pair_ids, pair_mask, target)."""
        from scomp_link.models.contrastive_net import ContrastiveDataset
        
        tokenizer = MagicMock()
        tokenizer.side_effect = lambda *args, **kwargs: {
            'input_ids': torch.randint(0, 1000, (1, 128)),
            'attention_mask': torch.ones(1, 128, dtype=torch.long),
        }
        
        ds = ContrastiveDataset(sample_df, tokenizer, text_col='text', label_col='label')
        item = ds[0]
        assert len(item) == 5
        assert item[0].shape == (128,)  # text_ids
        assert item[4].item() in (0.0, 1.0)  # target

    def test_sample_weights(self, sample_df):
        """Sample weights are inversely proportional to class frequency."""
        from scomp_link.models.contrastive_net import ContrastiveDataset
        weights = ContrastiveDataset.get_sample_weights(sample_df, label_col='label')
        assert len(weights) == len(sample_df)
        assert weights.sum() == pytest.approx(len(sample_df), rel=0.01)


# ═══════════════════════════════════════════════════════════════════
# EMBEDDING SELECTOR
# ═══════════════════════════════════════════════════════════════════

class TestEmbeddingSelector:
    def test_offline_mode(self, sample_df):
        """EmbeddingSelector works with precomputed embeddings (no model download)."""
        from scomp_link.models.contrastive_text import EmbeddingSelector
        
        # Simulate precomputed embeddings for two "models"
        n = len(sample_df)
        precomputed = {
            'model_a': np.random.randn(n, 128).astype('float32'),
            'model_b': np.random.randn(n, 64).astype('float32'),
        }
        
        selector = EmbeddingSelector(candidates=['model_a', 'model_b'])
        results = selector.find_best_backbone(
            sample_df, text_col='text', label_col='label',
            precomputed_embeddings=precomputed
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2
        assert 'model' in results.columns
        assert 'loss' in results.columns
        assert results.iloc[0]['loss'] <= results.iloc[1]['loss']  # sorted

    def test_sample_loss_computation(self):
        """_compute_sample_loss returns a positive float."""
        from scomp_link.models.contrastive_text import EmbeddingSelector
        
        selector = EmbeddingSelector()
        
        # Create embeddings with clear cluster separation
        emb = np.vstack([
            np.random.randn(10, 32) + np.array([5, 0] + [0]*30),  # cluster A
            np.random.randn(10, 32) + np.array([-5, 0] + [0]*30), # cluster B
        ]).astype('float32')
        labels = ['A'] * 10 + ['B'] * 10
        
        loss = selector._compute_sample_loss(emb, labels, n_pairs=200)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_well_separated_has_lower_loss(self):
        """Well-separated clusters should produce lower loss than random."""
        from scomp_link.models.contrastive_text import EmbeddingSelector
        
        selector = EmbeddingSelector()
        
        # Well separated
        emb_good = np.vstack([
            np.random.randn(50, 32) * 0.1 + 5,
            np.random.randn(50, 32) * 0.1 - 5,
        ]).astype('float32')
        labels = ['A'] * 50 + ['B'] * 50
        
        # Random (no separation)
        emb_bad = np.random.randn(100, 32).astype('float32')
        
        loss_good = selector._compute_sample_loss(emb_good, labels)
        loss_bad = selector._compute_sample_loss(emb_bad, labels)
        
        assert loss_good < loss_bad


# ═══════════════════════════════════════════════════════════════════
# EMBED METHOD
# ═══════════════════════════════════════════════════════════════════

class TestEmbed:
    def test_embed_returns_correct_shape(self, mock_classifier):
        """embed() returns (n_texts, embedding_dim) array."""
        # Mock forward_one to return correct dim
        mock_classifier.siamese_model.eval = MagicMock()
        mock_classifier.siamese_model.forward_one = MagicMock(
            return_value=torch.randn(3, 64)
        )
        
        result = mock_classifier.embed(["hello", "world", "test"])
        assert result.shape == (3, 64)
        assert result.dtype == np.float32

    def test_embed_single_text(self, mock_classifier):
        """embed() works with a single text."""
        mock_classifier.siamese_model.eval = MagicMock()
        mock_classifier.siamese_model.forward_one = MagicMock(
            return_value=torch.randn(1, 64)
        )
        
        result = mock_classifier.embed(["single text"])
        assert result.shape == (1, 64)


# ═══════════════════════════════════════════════════════════════════
# FIT HEAD
# ═══════════════════════════════════════════════════════════════════

class TestFitHead:
    def test_fit_head_logreg(self, mock_classifier, sample_df):
        """fit_head with LogisticRegression produces a working head."""
        # Mock embed to return random but consistent embeddings
        np.random.seed(42)
        mock_classifier.embed = MagicMock(
            return_value=np.random.randn(len(sample_df), 64).astype('float32')
        )
        
        result = mock_classifier.fit_head(sample_df, text_col='text', label_col='label', head='logreg')
        
        assert result['head_type'] == 'LogisticRegression'
        assert 'cv_accuracy' in result
        assert result['n_classes'] == 3
        assert mock_classifier._head_model is not None

    def test_fit_head_auto(self, mock_classifier, sample_df):
        """fit_head with auto tries multiple and picks best."""
        np.random.seed(42)
        # Create embeddings with some class structure
        n = len(sample_df)
        labels = sample_df['label'].tolist()
        emb = np.zeros((n, 64), dtype='float32')
        for i, label in enumerate(labels):
            if label == 'tech':
                emb[i] = np.random.randn(64) + 2
            elif label == 'sports':
                emb[i] = np.random.randn(64) - 2
            else:
                emb[i] = np.random.randn(64)
        
        mock_classifier.embed = MagicMock(return_value=emb)
        
        result = mock_classifier.fit_head(sample_df, text_col='text', label_col='label', head='auto')
        
        assert result['head_type'] in ('LogisticRegression', 'LinearSVC', 'RandomForest', 'LightGBM', 'XGBoost')
        assert result['cv_accuracy'] > 0

    def test_predict_uses_head(self, mock_classifier):
        """predict() uses head classifier when fitted."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        
        # Manually set up a head
        le = LabelEncoder()
        le.fit(['tech', 'sports', 'finance'])
        mock_classifier._head_label_encoder = le
        mock_classifier._head_model = LogisticRegression(random_state=42)
        mock_classifier._head_model.fit(np.random.randn(30, 64), np.array([0]*10 + [1]*10 + [2]*10))
        mock_classifier._head_type = 'LogisticRegression'
        
        # Mock the embedding
        mock_classifier.siamese_model.eval = MagicMock()
        mock_classifier.siamese_model.forward_one = MagicMock(
            return_value=torch.randn(1, 64)
        )
        
        result = mock_classifier.predict("test text")
        assert result in ['tech', 'sports', 'finance']


# ═══════════════════════════════════════════════════════════════════
# PREDICT (NN FALLBACK)
# ═══════════════════════════════════════════════════════════════════

class TestPredictNN:
    def test_predict_without_head(self, mock_classifier):
        """predict() falls back to NN when no head is fitted."""
        mock_classifier.labels = ["tech", "sports", "finance"]
        mock_classifier.label_embeddings = np.random.randn(3, 64).astype('float32')
        mock_classifier._head_model = None
        
        mock_classifier.siamese_model.eval = MagicMock()
        mock_classifier.siamese_model.forward_one = MagicMock(
            return_value=torch.randn(1, 64)
        )

        result = mock_classifier.predict("test text", top_k=1)
        assert result in ["tech", "sports", "finance"]

    def test_predict_top_k(self, mock_classifier):
        """predict with top_k > 1 returns list."""
        mock_classifier.labels = ["tech", "sports", "finance"]
        mock_classifier.label_embeddings = np.random.randn(3, 64).astype('float32')
        mock_classifier._head_model = None
        
        mock_classifier.siamese_model.eval = MagicMock()
        mock_classifier.siamese_model.forward_one = MagicMock(
            return_value=torch.randn(1, 64)
        )

        result = mock_classifier.predict("test text", top_k=3)
        assert len(result) == 3

    def test_predict_with_confidence(self, mock_classifier):
        """predict with return_confidence returns dict."""
        mock_classifier.labels = ["tech", "sports", "finance"]
        mock_classifier.label_embeddings = np.random.randn(3, 64).astype('float32')
        mock_classifier._head_model = None
        
        mock_classifier.siamese_model.eval = MagicMock()
        mock_classifier.siamese_model.forward_one = MagicMock(
            return_value=torch.randn(1, 64)
        )

        result = mock_classifier.predict("test text", top_k=2, return_confidence=True)
        assert "predictions" in result
        assert "confidences" in result
        assert len(result["predictions"]) == 2


# ═══════════════════════════════════════════════════════════════════
# BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════

class TestPredictBatch:
    def test_batch_with_head(self, mock_classifier, sample_df):
        """predict_batch uses head when fitted."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        le.fit(['tech', 'sports', 'finance'])
        mock_classifier._head_label_encoder = le
        mock_classifier._head_model = LogisticRegression(random_state=42)
        mock_classifier._head_model.fit(np.random.randn(30, 64), np.array([0]*10 + [1]*10 + [2]*10))
        mock_classifier._head_type = 'LogisticRegression'
        
        # Mock embed
        mock_classifier.embed = MagicMock(
            return_value=np.random.randn(len(sample_df), 64).astype('float32')
        )
        
        result = mock_classifier.predict_batch(sample_df['text'], top_k=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)
        assert "prediction" in result.columns
        assert "confidence" in result.columns

    def test_batch_nn_fallback(self, mock_classifier):
        """predict_batch uses NN when no head."""
        mock_classifier.labels = ["tech", "sports", "finance"]
        mock_classifier.label_embeddings = np.random.randn(3, 64).astype('float32')
        mock_classifier._head_model = None
        
        mock_classifier.siamese_model.eval = MagicMock()
        mock_classifier.siamese_model.forward_one = MagicMock(
            return_value=torch.randn(3, 64)
        )
        
        texts = pd.Series(["hello", "world", "test"])
        result = mock_classifier.predict_batch(texts, top_k=1)
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════

class TestSaveLoad:
    def test_save_creates_files(self, mock_classifier, tmp_path):
        """save() creates model.pt, metadata.json, embeddings.csv."""
        mock_classifier.labels = ["tech", "sports", "finance"]
        mock_classifier.label_embeddings = np.random.randn(3, 64).astype('float32')
        mock_classifier.label_freq_cache = {"tech": 0.4, "sports": 0.4, "finance": 0.2}

        save_path = str(tmp_path / "model")
        mock_classifier.save(save_path)

        assert os.path.exists(os.path.join(save_path, "model.pt"))
        assert os.path.exists(os.path.join(save_path, "metadata.json"))
        assert os.path.exists(os.path.join(save_path, "embeddings.csv"))

    def test_save_with_head(self, mock_classifier, tmp_path):
        """save() includes head_model.pkl when head is fitted."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        
        mock_classifier.labels = ["tech", "sports"]
        mock_classifier.label_embeddings = np.random.randn(2, 64).astype('float32')
        mock_classifier.label_freq_cache = {"tech": 0.5, "sports": 0.5}
        mock_classifier._head_model = LogisticRegression()
        mock_classifier._head_model.fit(np.random.randn(10, 64), [0]*5 + [1]*5)
        mock_classifier._head_label_encoder = LabelEncoder().fit(['tech', 'sports'])
        mock_classifier._head_type = 'LogisticRegression'

        save_path = str(tmp_path / "model_with_head")
        mock_classifier.save(save_path)

        assert os.path.exists(os.path.join(save_path, "head_model.pkl"))
        assert os.path.exists(os.path.join(save_path, "head_label_encoder.pkl"))
        
        # Verify metadata includes head_type
        with open(os.path.join(save_path, "metadata.json")) as f:
            metadata = json.load(f)
        assert metadata['head_type'] == 'LogisticRegression'

    def test_metadata_embedding_dim(self, mock_classifier, tmp_path):
        """Metadata correctly stores embedding_dim."""
        mock_classifier.labels = ["a", "b"]
        mock_classifier.label_embeddings = np.random.randn(2, 64).astype('float32')
        mock_classifier.label_freq_cache = {}

        save_path = str(tmp_path / "model3")
        mock_classifier.save(save_path)

        with open(os.path.join(save_path, "metadata.json")) as f:
            metadata = json.load(f)
        assert metadata["embedding_dim"] == 64


# ═══════════════════════════════════════════════════════════════════
# GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════

class TestGenerateReport:
    def test_report_creates_html(self, mock_classifier, tmp_path):
        """generate_report produces an HTML file."""
        mock_classifier.labels = ["tech", "sports", "finance"]
        mock_classifier.label_embeddings = np.random.randn(3, 64).astype('float32')
        mock_classifier._head_type = 'LogisticRegression'
        
        y_true = ["tech", "sports", "finance", "tech", "sports"]
        y_pred = ["tech", "sports", "tech", "tech", "sports"]
        
        report_path = str(tmp_path / "report.html")
        result = mock_classifier.generate_report(y_true, y_pred, report_path=report_path)
        
        assert os.path.exists(result)
        with open(result) as f:
            html = f.read()
        assert "Contrastive Text Classifier" in html
        assert "F1" in html

    def test_report_with_embeddings(self, mock_classifier, tmp_path):
        """generate_report includes t-SNE when embeddings are provided."""
        mock_classifier.labels = ["tech", "sports"]
        mock_classifier.label_embeddings = np.random.randn(2, 64).astype('float32')
        mock_classifier._head_type = 'RF'
        
        y_true = ["tech", "sports"] * 10
        y_pred = ["tech", "sports"] * 10
        embeddings = np.random.randn(20, 64).astype('float32')
        
        report_path = str(tmp_path / "report_tsne.html")
        result = mock_classifier.generate_report(y_true, y_pred, report_path=report_path,
                                                  embeddings=embeddings)
        assert os.path.exists(result)

    def test_report_with_selector_results(self, mock_classifier, tmp_path):
        """generate_report includes backbone comparison when selector_results provided."""
        mock_classifier.labels = ["a", "b"]
        mock_classifier.label_embeddings = np.random.randn(2, 64).astype('float32')
        mock_classifier._head_type = None
        
        selector_df = pd.DataFrame({
            'model': ['model_a', 'model_b'],
            'loss': [0.5, 0.8],
            'embedding_dim': [128, 64],
            'encode_time_s': [1.2, 0.5],
        })
        
        report_path = str(tmp_path / "report_selector.html")
        result = mock_classifier.generate_report(
            ["a", "b", "a"], ["a", "b", "b"],
            report_path=report_path, selector_results=selector_df
        )
        assert os.path.exists(result)


# ═══════════════════════════════════════════════════════════════════
# CONTRASTIVE LOSS (ORIGINAL)
# ═══════════════════════════════════════════════════════════════════

class TestContrastiveLoss:
    def test_positive_pairs_low_loss(self):
        """Positive pairs with identical embeddings should have ~0 loss."""
        from scomp_link.models.contrastive_net import ContrastiveLoss
        loss_fn = ContrastiveLoss(margin=1.0)
        
        emb = torch.randn(4, 64)
        labels = torch.ones(4)  # all positive
        
        loss = loss_fn(emb, emb, labels)
        assert loss.item() < 0.01

    def test_negative_pairs_beyond_margin(self):
        """Negative pairs beyond margin should have 0 loss."""
        from scomp_link.models.contrastive_net import ContrastiveLoss
        loss_fn = ContrastiveLoss(margin=1.0)
        
        emb1 = torch.ones(4, 64) * 5
        emb2 = torch.ones(4, 64) * -5  # very far apart
        labels = torch.zeros(4)  # all negative
        
        loss = loss_fn(emb1, emb2, labels)
        assert loss.item() == 0.0


# ═══════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════

class TestEarlyStopping:
    def test_no_stop_on_improvement(self):
        """EarlyStopping doesn't trigger when loss improves."""
        from scomp_link.models.contrastive_net import EarlyStopping
        es = EarlyStopping(patience=3)
        
        assert es(1.0) is False
        assert es(0.9) is False
        assert es(0.8) is False

    def test_stops_after_patience(self):
        """EarlyStopping triggers after patience epochs without improvement."""
        from scomp_link.models.contrastive_net import EarlyStopping
        es = EarlyStopping(patience=3, min_delta=0.01)
        
        es(0.5)  # best
        assert es(0.5) is False  # no improvement, counter=1
        assert es(0.5) is False  # counter=2
        assert es(0.5) is True   # counter=3 = patience → stop
