# -*- coding: utf-8 -*-
"""
Contrastive Network components for Siamese text classification.

Provides:
- ContrastiveSiameseModel: BERT + projection layer for contrastive embeddings
- ContrastiveLoss: Contrastive loss function with margin
- SiameseDataset: PyTorch Dataset for generating contrastive pairs
- EarlyStopping: Training early stopping callback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset


class ContrastiveSiameseModel(nn.Module):
    """
    Siamese network with BERT backbone and projection layer.
    
    Args:
        bert_model: Pre-trained BERT model
        embedding_dim: Output embedding dimension
    """

    def __init__(self, bert_model, embedding_dim=256):
        super().__init__()
        self.bert = bert_model
        hidden_size = bert_model.config.hidden_size
        self.projection_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, embedding_dim)
        )

    def forward_one(self, input_ids, attention_mask):
        """Encode a single input through BERT + projection."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        projection = self.projection_layer(cls_output)
        return F.normalize(projection, p=2, dim=1)

    def forward(self, text_ids, text_mask, label_ids, label_mask):
        """Forward pass for both text and label inputs."""
        text_emb = self.forward_one(text_ids, text_mask)
        label_emb = self.forward_one(label_ids, label_mask)
        return text_emb, label_emb


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    
    For positive pairs (label=1): minimizes distance.
    For negative pairs (label=0): pushes apart beyond margin.
    
    Args:
        margin: Margin for negative pairs
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, labels):
        """
        Args:
            emb1: First embedding batch
            emb2: Second embedding batch
            labels: 1 for positive pair, 0 for negative pair
        """
        distances = F.pairwise_distance(emb1, emb2)
        labels = labels.float()
        loss = labels * distances.pow(2) + \
               (1 - labels) * F.relu(self.margin - distances).pow(2)
        return loss.mean()


class SiameseDataset(Dataset):
    """
    Dataset for generating contrastive pairs from text data.
    
    Expects DataFrame with 'url' (text) and 'app_name' (label) columns.
    Generates positive pairs (same label) and negative pairs (different label).
    
    Args:
        df: DataFrame with 'url' and 'app_name' columns
        tokenizer: HuggingFace tokenizer
        augment_prob: Probability of text augmentation
    """

    def __init__(self, df, tokenizer, augment_prob=0.5):
        self.df = df
        self.tokenizer = tokenizer
        self.url_tokens = df['url'].tolist()
        self.app_names = df['app_name'].tolist()
        self.hard_negative_ratio = 0.3
        self.augment_prob = augment_prob
        self.app_frequencies = df['app_name'].value_counts().to_dict()
        self.popular_apps = list(df['app_name'].value_counts().head(1000).index)
        
        # Group indices by label for efficient pair generation
        self._label_to_indices = {}
        for idx, label in enumerate(self.app_names):
            self._label_to_indices.setdefault(label, []).append(idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.url_tokens[idx]
        label = self.app_names[idx]

        # 50% positive pairs, 50% negative pairs
        if random.random() > 0.5:
            # Positive pair: same label
            same_indices = self._label_to_indices[label]
            pair_idx = random.choice(same_indices)
            pair_text = self.app_names[pair_idx] if random.random() > 0.5 else label
            target = torch.tensor(1, dtype=torch.float32)
        else:
            # Negative pair: different label
            neg_label = random.choice(self.popular_apps)
            while neg_label == label:
                neg_label = random.choice(self.popular_apps)
            pair_text = neg_label
            target = torch.tensor(0, dtype=torch.float32)

        # Augment text
        if self.augment_prob > 0 and random.random() < self.augment_prob:
            text = self._augment(text)

        # Tokenize
        text_enc = self.tokenizer(str(text), return_tensors='pt', truncation=True,
                                  padding='max_length', max_length=128)
        label_enc = self.tokenizer(str(pair_text), return_tensors='pt', truncation=True,
                                   padding='max_length', max_length=128)

        return (text_enc['input_ids'].squeeze(0),
                text_enc['attention_mask'].squeeze(0),
                label_enc['input_ids'].squeeze(0),
                label_enc['attention_mask'].squeeze(0),
                target)

    def _augment(self, text):
        """Simple text augmentation: random word dropout."""
        words = str(text).split()
        if len(words) <= 2:
            return text
        n_drop = max(1, int(len(words) * 0.1))
        indices = random.sample(range(len(words)), min(n_drop, len(words) - 1))
        return ' '.join(w for i, w in enumerate(words) if i not in indices)

    @staticmethod
    def get_sample_weights(df):
        """Calculate sample weights for balanced sampling."""
        label_counts = df['app_name'].value_counts()
        weights = 1.0 / label_counts[df['app_name']].values
        return weights / weights.sum() * len(weights)


class EarlyStopping:
    """
    Early stopping callback for training.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
    """

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        """
        Returns True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience
