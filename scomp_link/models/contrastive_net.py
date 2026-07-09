# -*- coding: utf-8 -*-
"""
 ██████╗ ██████╗ ███╗   ██╗████████╗██████╗  █████╗ ███████╗████████╗██╗██╗   ██╗███████╗
██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██║██║   ██║██╔════╝
██║     ██║   ██║██╔██╗ ██║   ██║   ██████╔╝███████║███████╗   ██║   ██║╚██╗ ██╔╝█████╗  
██║     ██║   ██║██║╚████║   ██║   ██╔══██╗██╔══██║╚════██║   ██║   ██║ ╚████╔╝ ██╔══╝  
╚██████╗╚██████╔╝██║ ╚███║   ██║   ██║  ██║██║  ██║███████║   ██║   ██║  ╚██╔╝  ███████╗
 ╚═════╝ ╚═════╝ ╚═╝  ╚══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝   ╚═╝   ╚══════╝

███╗   ██╗███████╗████████╗
████╗  ██║██╔════╝╚══██╔══╝
██╔██╗ ██║█████╗     ██║   
██║╚████║██╔══╝     ██║   
██║ ╚███║███████╗   ██║   
╚═╝  ╚══╝╚══════╝   ╚═╝   

Contrastive Network components for text classification.

Provides:
- ContrastiveSiameseModel: BERT + projection layer for contrastive embeddings
- ContrastiveLoss: Contrastive loss function with margin
- InfoNCELoss: Temperature-scaled cross-entropy (NT-Xent) for many-class scenarios
- SiameseDataset: PyTorch Dataset for generating contrastive pairs (legacy, url/app_name columns)
- ContrastiveDataset: Generic contrastive pair dataset (configurable column names)
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
    Contrastive loss function (margin-based).
    
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
            emb1: First embedding batch [B, D]
            emb2: Second embedding batch [B, D]
            labels: 1 for positive pair, 0 for negative pair [B]
        """
        distances = F.pairwise_distance(emb1, emb2)
        labels = labels.float()
        loss = labels * distances.pow(2) + \
               (1 - labels) * F.relu(self.margin - distances).pow(2)
        return loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent loss for contrastive learning.
    
    More stable than margin-based loss for many classes.
    Uses in-batch negatives: each positive pair is contrasted against
    all other samples in the batch as negatives.
    
    L = -log( exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )
    
    Args:
        temperature: Scaling temperature τ (lower = sharper distribution)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb1, emb2, labels=None):
        """
        Compute InfoNCE loss using in-batch negatives.
        
        Args:
            emb1: Anchor embeddings [B, D] (normalized)
            emb2: Positive embeddings [B, D] (normalized)
            labels: Optional — if provided, uses supervised contrastive
                    (same-label pairs are positives). If None, treats
                    (emb1[i], emb2[i]) as the only positive pair.
        
        Returns:
            Scalar loss
        """
        batch_size = emb1.shape[0]
        
        # Cosine similarity matrix [B, B]
        sim_matrix = torch.mm(emb1, emb2.T) / self.temperature
        
        if labels is None:
            # Standard InfoNCE: diagonal = positive pairs
            targets = torch.arange(batch_size, device=emb1.device)
            loss = F.cross_entropy(sim_matrix, targets)
        else:
            # Supervised contrastive: same-label pairs are positives
            labels = labels.view(-1)
            # Mask: 1 where labels match (positive pairs)
            mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            # Remove self-contrast from diagonal
            mask.fill_diagonal_(0)
            
            # Log-softmax for numerical stability
            log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
            
            # Mean log-prob over positive pairs
            n_positives = mask.sum(dim=1).clamp(min=1)
            loss = -(mask * log_prob).sum(dim=1) / n_positives
            loss = loss.mean()
        
        return loss


class SiameseDataset(Dataset):
    """
    Dataset for generating contrastive pairs from text data (legacy).
    
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


class ContrastiveDataset(Dataset):
    """
    Generic contrastive pair dataset with configurable column names.
    
    Generates positive pairs (same label) and negative pairs (different label).
    More flexible than SiameseDataset — accepts any text/label column names.
    
    Args:
        df: DataFrame with text and label columns
        tokenizer: HuggingFace tokenizer
        text_col: Name of the text column
        label_col: Name of the label column
        max_length: Max token length for tokenizer
        augment_prob: Probability of text augmentation
    """

    def __init__(self, df, tokenizer, text_col='text', label_col='label',
                 max_length=128, augment_prob=0.3):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length
        self.augment_prob = augment_prob
        
        self.texts = df[text_col].tolist()
        self.labels = df[label_col].tolist()
        self.unique_labels = list(set(self.labels))
        
        # Group indices by label for efficient pair generation
        self._label_to_indices = {}
        for idx, label in enumerate(self.labels):
            self._label_to_indices.setdefault(label, []).append(idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 50% positive pairs, 50% negative pairs
        if random.random() > 0.5:
            # Positive pair: another text with same label
            same_indices = self._label_to_indices[label]
            pair_idx = random.choice(same_indices)
            pair_text = self.texts[pair_idx]
            target = torch.tensor(1, dtype=torch.float32)
        else:
            # Negative pair: text with different label
            neg_label = random.choice(self.unique_labels)
            while neg_label == label:
                neg_label = random.choice(self.unique_labels)
            neg_indices = self._label_to_indices[neg_label]
            pair_idx = random.choice(neg_indices)
            pair_text = self.texts[pair_idx]
            target = torch.tensor(0, dtype=torch.float32)

        # Optional augmentation
        if self.augment_prob > 0 and random.random() < self.augment_prob:
            text = self._augment(text)

        # Tokenize both texts
        text_enc = self.tokenizer(str(text), return_tensors='pt', truncation=True,
                                  padding='max_length', max_length=self.max_length)
        pair_enc = self.tokenizer(str(pair_text), return_tensors='pt', truncation=True,
                                  padding='max_length', max_length=self.max_length)

        return (text_enc['input_ids'].squeeze(0),
                text_enc['attention_mask'].squeeze(0),
                pair_enc['input_ids'].squeeze(0),
                pair_enc['attention_mask'].squeeze(0),
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
    def get_sample_weights(df, label_col='label'):
        """Calculate sample weights for balanced sampling."""
        label_counts = df[label_col].value_counts()
        weights = 1.0 / label_counts[df[label_col]].values
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
