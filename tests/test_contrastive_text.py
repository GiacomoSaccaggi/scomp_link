# -*- coding: utf-8 -*-
"""
Tests for ContrastiveTextClassifier using mocks.
Avoids downloading BERT model — mocks torch and transformers.
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


@pytest.fixture
def mock_classifier():
    """Create a ContrastiveTextClassifier with mocked BERT weights."""
    from scomp_link.models.contrastive_text import ContrastiveTextClassifier

    with patch.object(AutoTokenizer, 'from_pretrained') as mock_tok, \
         patch.object(AutoModel, 'from_pretrained') as mock_model:

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 128)),
            'attention_mask': torch.ones(1, 128, dtype=torch.long),
        }
        tokenizer.side_effect = lambda *args, **kwargs: {
            'input_ids': torch.randint(0, 1000, (len(args[0]) if isinstance(args[0], list) else 1, 128)),
            'attention_mask': torch.ones(len(args[0]) if isinstance(args[0], list) else 1, 128, dtype=torch.long),
        }
        mock_tok.return_value = tokenizer

        # Mock BERT model (returns a real-shaped tensor)
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
            "Machine learning is great",
            "Deep neural networks",
            "Football match today",
            "Basketball game results",
            "Stock market crash",
            "Economic indicators rising",
            "Python programming tips",
            "JavaScript frameworks",
            "Tennis tournament score",
            "Hockey final results",
        ],
        'label': [
            "tech", "tech", "sports", "sports",
            "finance", "finance", "tech", "tech",
            "sports", "sports",
        ]
    })


class TestContrastiveTextInit:
    def test_init_creates_model(self, mock_classifier):
        assert mock_classifier.tokenizer is not None
        assert mock_classifier.siamese_model is not None
        assert mock_classifier.labels == []
        assert mock_classifier.label_embeddings is None

    def test_init_faiss_disabled(self, mock_classifier):
        assert mock_classifier.use_faiss is False


class TestContrastiveTextPredict:
    def test_predict_after_setup(self, mock_classifier):
        """Test predict with manually set embeddings (skip training)."""
        # Manually set up embeddings as if trained
        mock_classifier.labels = ["tech", "sports", "finance"]
        mock_classifier.label_embeddings = np.random.randn(3, 64).astype('float32')

        # Mock the forward_one method to return a fixed embedding
        mock_classifier.siamese_model.eval = MagicMock()
        mock_classifier.siamese_model.forward_one = MagicMock(
            return_value=torch.randn(1, 64)
        )

        result = mock_classifier.predict("test text", top_k=1)
        assert result in ["tech", "sports", "finance"]

    def test_predict_top_k(self, mock_classifier):
        """Test predict with top_k > 1."""
        mock_classifier.labels = ["tech", "sports", "finance"]
        mock_classifier.label_embeddings = np.random.randn(3, 64).astype('float32')
        mock_classifier.siamese_model.eval = MagicMock()
        mock_classifier.siamese_model.forward_one = MagicMock(
            return_value=torch.randn(1, 64)
        )

        result = mock_classifier.predict("test text", top_k=3)
        assert len(result) == 3
        assert all(r in ["tech", "sports", "finance"] for r in result)

    def test_predict_with_confidence(self, mock_classifier):
        """Test predict with return_confidence=True."""
        mock_classifier.labels = ["tech", "sports", "finance"]
        mock_classifier.label_embeddings = np.random.randn(3, 64).astype('float32')
        mock_classifier.siamese_model.eval = MagicMock()
        mock_classifier.siamese_model.forward_one = MagicMock(
            return_value=torch.randn(1, 64)
        )

        result = mock_classifier.predict("test text", top_k=2, return_confidence=True)
        assert "predictions" in result
        assert "confidences" in result
        assert len(result["predictions"]) == 2


class TestContrastiveTextBatch:
    def test_predict_batch(self, mock_classifier):
        """Test batch prediction."""
        mock_classifier.labels = ["tech", "sports", "finance"]
        mock_classifier.label_embeddings = np.random.randn(3, 64).astype('float32')
        mock_classifier.siamese_model.eval = MagicMock()
        mock_classifier.siamese_model.forward_one = MagicMock(
            return_value=torch.randn(3, 64)
        )

        texts = pd.Series(["hello world", "test text", "another one"])
        result = mock_classifier.predict_batch(texts, top_k=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "prediction" in result.columns
        assert "confidence" in result.columns


class TestContrastiveTextSaveLoad:
    def test_save_and_load(self, mock_classifier, tmp_path):
        """Test save/load cycle."""
        # Set up state
        mock_classifier.labels = ["tech", "sports", "finance"]
        mock_classifier.label_embeddings = np.random.randn(3, 64).astype('float32')
        mock_classifier.label_freq_cache = {"tech": 0.4, "sports": 0.4, "finance": 0.2}

        save_path = str(tmp_path / "model")
        mock_classifier.save(save_path)

        # Verify files exist
        assert os.path.exists(os.path.join(save_path, "model.pt"))
        assert os.path.exists(os.path.join(save_path, "metadata.json"))
        assert os.path.exists(os.path.join(save_path, "embeddings.csv"))

        # Verify metadata content
        with open(os.path.join(save_path, "metadata.json")) as f:
            metadata = json.load(f)
        assert metadata["labels"] == ["tech", "sports", "finance"]
        assert metadata["embedding_dim"] == 64

    def test_embeddings_saved_correctly(self, mock_classifier, tmp_path):
        """Test that embeddings CSV has correct shape."""
        mock_classifier.labels = ["a", "b"]
        mock_classifier.label_embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_classifier.label_freq_cache = {"a": 0.5, "b": 0.5}

        save_path = str(tmp_path / "model2")
        mock_classifier.save(save_path)

        loaded = np.loadtxt(os.path.join(save_path, "embeddings.csv"), delimiter=",")
        assert loaded.shape == (2, 3)
        np.testing.assert_array_almost_equal(loaded, mock_classifier.label_embeddings)


class TestContrastiveTextHelpers:
    def test_build_embeddings(self, mock_classifier, sample_df):
        """Test _calculate_label_embeddings sets up label_embeddings."""
        # Mock the tokenizer and forward_one for embedding calculation
        mock_classifier.siamese_model.eval = MagicMock()
        mock_classifier.siamese_model.forward_one = MagicMock(
            return_value=torch.randn(1, 64)
        )

        mock_classifier._calculate_label_embeddings(
            sample_df, label_col='label'
        )

        assert mock_classifier.label_embeddings is not None
        assert len(mock_classifier.labels) == 3  # tech, sports, finance
        assert mock_classifier.label_embeddings.shape[0] == 3
