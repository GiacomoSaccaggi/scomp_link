# -*- coding: utf-8 -*-
"""

 █████╗  █████╗ ███╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗██╗██╗   ██╗███████╗
██╔══██╗██╔══██╗████╗ ██║╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██║██║   ██║██╔════╝
██║  ╚═╝██║  ██║██╔██╗██║   ██║   ██████╔╝███████║╚█████╗    ██║   ██║╚██╗ ██╔╝█████╗  
██║  ██╗██║  ██║██║╚████║   ██║   ██╔══██╗██╔══██║ ╚═══██╗   ██║   ██║ ╚████╔╝ ██╔══╝  
╚█████╔╝╚█████╔╝██║ ╚███║   ██║   ██║  ██║██║  ██║██████╔╝   ██║   ██║  ╚██╔╝  ███████╗
 ╚════╝  ╚════╝ ╚═╝  ╚══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝   ╚═╝   ╚══════╝

███╗  ██╗███████╗████████╗
████╗ ██║██╔════╝╚══██╔══╝
██╔██╗██║█████╗     ██║   
██║╚████║██╔══╝     ██║   
██║ ╚███║███████╗   ██║   
╚═╝  ╚══╝╚══════╝   ╚═╝   
"""

import os
import io
import json
import time
import pickle
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from functools import wraps
from typing import Optional, List, Dict, Union
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel


from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from .contrastive_net import (ContrastiveSiameseModel, ContrastiveLoss, InfoNCELoss,
                                   SiameseDataset, ContrastiveDataset, EarlyStopping)
except (ImportError, ValueError):
    try:
        from scomp_link.models.contrastive_net import (ContrastiveSiameseModel, ContrastiveLoss, InfoNCELoss,
                                                        SiameseDataset, ContrastiveDataset, EarlyStopping)
    except (ImportError, ValueError):
        ContrastiveSiameseModel = None
        ContrastiveLoss = None
        InfoNCELoss = None
        SiameseDataset = None
        ContrastiveDataset = None
        EarlyStopping = None


# ═══════════════════════════════════════════════════════════════════
# DEFAULT BACKBONE CANDIDATES
# ═══════════════════════════════════════════════════════════════════

DEFAULT_BACKBONE_CANDIDATES = [
    "all-MiniLM-L6-v2",           # 22M params, fastest
    "all-mpnet-base-v2",          # 109M params, best quality
    "BAAI/bge-small-en-v1.5",    # 33M params, good balance
    "intfloat/e5-small-v2",      # 33M params, instruction-tuned
    "paraphrase-MiniLM-L3-v2",   # 17M params, ultra-fast
]


def print_throughput(func):
    """Decorator to print iterations per second"""
    @wraps(func)
    def wrapper(self, text_series, *args, **kwargs):
        n_items = len(text_series)
        logger.info(f"\n🚀 Processing {n_items} texts...")
        start = time.time()
        result = func(self, text_series, *args, **kwargs)
        elapsed = time.time() - start
        throughput = n_items / elapsed if elapsed > 0 else 0
        logger.info(f"✅ Completed in {elapsed:.2f}s ({throughput:.1f} texts/s)\n")
        return result
    return wrapper


# ═══════════════════════════════════════════════════════════════════
# EMBEDDING SELECTOR
# ═══════════════════════════════════════════════════════════════════

class EmbeddingSelector:
    """
    Auto-select the best pretrained embedding backbone for a dataset.
    
    Computes contrastive loss on a sample of data for each candidate model
    to find the best starting point before full training.
    
    Dependencies: sentence-transformers, torch
    
    Usage example:
        selector = EmbeddingSelector()
        results = selector.find_best_backbone(df, text_col='text', label_col='label')
        best_model = results.iloc[0]['model']
    """

    def __init__(self, candidates: Optional[List[str]] = None):
        """
        Args:
            candidates: List of sentence-transformer model names to evaluate.
                        Defaults to DEFAULT_BACKBONE_CANDIDATES.
        """
        self.candidates = candidates or DEFAULT_BACKBONE_CANDIDATES

    def find_best_backbone(self, df: pd.DataFrame, text_col: str = 'text',
                           label_col: str = 'label', sample_size: int = 500,
                           precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None
                           ) -> pd.DataFrame:
        """
        Evaluate backbone candidates by computing contrastive loss on a sample.
        
        Args:
            df: DataFrame with text and label columns
            text_col: Name of text column
            label_col: Name of label column
            sample_size: Number of samples to use (capped at len(df))
            precomputed_embeddings: Dict {model_name: embeddings_array} for offline mode.
                                    If provided, skips encoding for those models.
        
        Returns:
            DataFrame with columns: model, loss, embedding_dim, encode_time_s
            Sorted by loss ascending (best first).
        """
        precomputed_embeddings = precomputed_embeddings or {}
        
        # Sample the dataset
        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            sample_df = df.reset_index(drop=True)
        
        texts = sample_df[text_col].tolist()
        labels = sample_df[label_col].tolist()
        
        results = []
        for model_name in self.candidates:
            logger.info(f"  🔍 Evaluating: {model_name}")
            try:
                if model_name in precomputed_embeddings:
                    # Offline mode: use pre-computed embeddings
                    embeddings = precomputed_embeddings[model_name]
                    if len(embeddings) > sample_size:
                        embeddings = embeddings[:sample_size]
                    encode_time = 0.0
                else:
                    # Online mode: encode with sentence-transformers
                    from sentence_transformers import SentenceTransformer
                    st_model = SentenceTransformer(model_name)
                    t0 = time.perf_counter()
                    embeddings = st_model.encode(texts, show_progress_bar=False, 
                                                  normalize_embeddings=True)
                    encode_time = time.perf_counter() - t0
                
                # Compute contrastive loss on pairs from the sample
                loss = self._compute_sample_loss(embeddings, labels)
                
                results.append({
                    'model': model_name,
                    'loss': float(loss),
                    'embedding_dim': embeddings.shape[1],
                    'encode_time_s': round(encode_time, 3),
                })
                logger.info(f"    loss={loss:.4f}, dim={embeddings.shape[1]}, time={encode_time:.2f}s")
                
            except Exception as e:
                logger.info(f"    ⚠️ Failed: {e}")
                results.append({
                    'model': model_name,
                    'loss': float('inf'),
                    'embedding_dim': 0,
                    'encode_time_s': 0.0,
                })
        
        results_df = pd.DataFrame(results).sort_values('loss').reset_index(drop=True)
        logger.info(f"\n🏆 Best backbone: {results_df.iloc[0]['model']} (loss={results_df.iloc[0]['loss']:.4f})")
        return results_df

    def _compute_sample_loss(self, embeddings: np.ndarray, labels: list,
                             n_pairs: int = 2000) -> float:
        """
        Compute average contrastive loss on random pairs.
        
        Generates n_pairs/2 positive pairs + n_pairs/2 negative pairs
        and computes mean contrastive distance.
        """
        label_to_indices = {}
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(label, []).append(idx)
        
        unique_labels = list(label_to_indices.keys())
        if len(unique_labels) < 2:
            return 0.0
        
        pos_distances = []
        neg_distances = []
        
        # Generate positive pairs (same label)
        for _ in range(n_pairs // 2):
            label = unique_labels[np.random.randint(len(unique_labels))]
            indices = label_to_indices[label]
            if len(indices) < 2:
                continue
            i, j = np.random.choice(indices, size=2, replace=False)
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            pos_distances.append(dist)
        
        # Generate negative pairs (different label)
        for _ in range(n_pairs // 2):
            l1, l2 = np.random.choice(len(unique_labels), size=2, replace=False)
            i = np.random.choice(label_to_indices[unique_labels[l1]])
            j = np.random.choice(label_to_indices[unique_labels[l2]])
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            neg_distances.append(dist)
        
        if not pos_distances or not neg_distances:
            return float('inf')
        
        # Loss = mean positive distance - mean negative distance
        # (lower = better separation)
        margin = 1.0
        pos_loss = np.mean(pos_distances) ** 2
        neg_loss = np.mean([max(0, margin - d) ** 2 for d in neg_distances])
        return pos_loss + neg_loss


# ═══════════════════════════════════════════════════════════════════
# CONTRASTIVE TEXT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

class ContrastiveTextClassifier:
    """
    Contrastive Learning Text Classifier with optional weak learner head.
    
    Uses Siamese Networks with BERT for embedding learning, then optionally
    fits a lightweight classifier (head) on top of frozen embeddings.
    
    Flow:
        1. train_contrastive() — learn embeddings via contrastive loss
        2. fit_head() — fit a weak learner on embeddings (optional but recommended)
        3. predict() / predict_batch() — classify using head (or NN fallback)
    
    Dependencies: torch, transformers, faiss-cpu (optional), scikit-learn
    
    Usage example:
        clf = ContrastiveTextClassifier(model_name='bert-base-uncased')
        clf.train_contrastive(df, text_col='text', label_col='category', epochs=5)
        clf.fit_head(df, text_col='text', label_col='category', head='auto')
        predictions = clf.predict_batch(test_df['text'])
    """
    
    def __init__(self, model_name='bert-base-uncased', use_faiss=True, embedding_dim=256):
        """
        Initialize BERT model and tokenizer.
        
        Args:
            model_name: HuggingFace model name
            use_faiss: Use FAISS for fast nearest-neighbor inference
            embedding_dim: Dimension of projection layer output
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.train()
        self.siamese_model = ContrastiveSiameseModel(self.model, embedding_dim=embedding_dim)
        
        # FAISS index for nearest-neighbor fallback
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index = None
        if self.use_faiss:
            logger.info("✅ FAISS enabled for fast inference")
        else:
            logger.info("⚠️ FAISS disabled - using numpy (slower)")
        
        # Storage for embeddings and labels
        self.label_embeddings = None
        self.labels = []
        self.label_freq_cache = {}
        
        # Head classifier (weak learner)
        self._head_model = None
        self._head_label_encoder = None
        self._head_type = None

    # ─── TRAINING ─────────────────────────────────────────────────────

    def train_contrastive(self, df, text_col='text', label_col='label', 
                         epochs=5, batch_size=64, lr=2e-5,
                         loss_fn='contrastive',
                         use_weighted_sampling=True, accumulation_steps=2,
                         validation_split=0.1, early_stopping_patience=5):
        """
        Train embeddings with contrastive learning.
        
        Args:
            df: DataFrame with text and labels
            text_col: Column name for text
            label_col: Column name for labels
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            loss_fn: 'contrastive' (margin) or 'infonce' (NT-Xent)
            use_weighted_sampling: Balance classes via weighted sampler
            accumulation_steps: Gradient accumulation steps
            validation_split: Fraction for validation set
            early_stopping_patience: Patience for early stopping
        """
        logger.info("🚀 Preparing contrastive training...")
        
        # Use ContrastiveDataset (generic column names)
        df_train = df.copy().reset_index(drop=True)
        
        # Split train/validation
        if validation_split > 0:
            val_size = int(len(df_train) * validation_split)
            train_df = df_train.iloc[:-val_size].reset_index(drop=True)
            val_df = df_train.iloc[-val_size:].reset_index(drop=True)
            logger.info(f"📊 Train: {len(train_df)}, Validation: {len(val_df)}")
        else:
            train_df = df_train
            val_df = None
        
        train_dataset = ContrastiveDataset(train_df, self.tokenizer, 
                                            text_col=text_col, label_col=label_col)
        
        # Weighted sampling for class balance
        if use_weighted_sampling:
            sample_weights = ContrastiveDataset.get_sample_weights(train_df, label_col=label_col)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            logger.info("✅ Weighted sampling enabled")
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation loader
        val_loader = None
        if val_df is not None:
            val_dataset = ContrastiveDataset(val_df, self.tokenizer, 
                                             text_col=text_col, label_col=label_col, augment_prob=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function
        if loss_fn == 'infonce':
            criterion = InfoNCELoss(temperature=0.07)
            logger.info("✅ Using InfoNCE loss (temperature=0.07)")
        else:
            criterion = ContrastiveLoss(margin=1.0)
            logger.info("✅ Using contrastive loss (margin=1.0)")
        
        # Optimizer with differentiated learning rates
        optimizer = optim.AdamW([
            {'params': self.siamese_model.bert.parameters(), 'lr': lr},
            {'params': self.siamese_model.projection_layer.parameters(), 'lr': lr * 10}
        ], weight_decay=0.01)
        
        logger.info(f"✅ Differentiated LR: BERT={lr:.2e}, Projection={lr*10:.2e}")
        logger.info(f"✅ Gradient accumulation: {accumulation_steps} steps")
        
        early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.001)
        
        self.siamese_model.train()
        logger.info(f"\n🎯 Starting training for {epochs} epochs...\n")
        
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            
            for i, data in enumerate(train_loader):
                text_ids, text_mask, label_ids, label_mask, targets = data
                
                text_emb, label_emb = self.siamese_model(text_ids, text_mask, label_ids, label_mask)
                loss = criterion(text_emb, label_emb, targets)
                
                loss = loss / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
            
            avg_train_loss = total_loss / max(len(train_loader), 1)
            
            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader, criterion)
                logger.info(f"Epoch {epoch + 1}/{epochs} — Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                if early_stopping(val_loss):
                    logger.info(f"⚠️ Early stopping at epoch {epoch + 1}")
                    break
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} — Train Loss: {avg_train_loss:.4f}")
        
        self.siamese_model.eval()
        logger.info("\n✅ Training completed!\n")
        
        # Calculate label embeddings and build FAISS index
        self._calculate_label_embeddings(df_train, label_col)
        if self.use_faiss:
            self._build_faiss_index()
    
    def _validate(self, val_loader, criterion):
        """Validation loss computation."""
        self.siamese_model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in val_loader:
                text_ids, text_mask, label_ids, label_mask, targets = data
                text_emb, label_emb = self.siamese_model(text_ids, text_mask, label_ids, label_mask)
                loss = criterion(text_emb, label_emb, targets)
                total_loss += loss.item()
        self.siamese_model.train()
        return total_loss / max(len(val_loader), 1)
    
    def _calculate_label_embeddings(self, df, label_col):
        """Calculate embeddings for all unique labels."""
        logger.info("Calculating label embeddings...")
        self.siamese_model.eval()
        
        unique_labels = df[label_col].unique().tolist()
        self.labels = unique_labels
        
        embeddings_list = []
        with torch.no_grad():
            for label in tqdm(unique_labels, desc="Encoding labels"):
                tokenized = self.tokenizer(str(label), return_tensors='pt', 
                                          truncation=True, padding='max_length', max_length=128)
                emb = self.siamese_model.forward_one(tokenized['input_ids'], 
                                                     tokenized['attention_mask']).cpu().numpy()
                embeddings_list.append(emb.flatten())
        
        self.label_embeddings = np.array(embeddings_list)
        
        # Build frequency cache
        label_counts = df[label_col].value_counts()
        total = label_counts.sum()
        for label in unique_labels:
            self.label_freq_cache[label] = label_counts.get(label, 0) / total
        
        logger.info(f"✅ Embeddings calculated for {len(unique_labels)} labels")
    
    def _build_faiss_index(self):
        """Build FAISS index for fast nearest-neighbor search."""
        if not self.use_faiss or self.label_embeddings is None:
            return
        logger.info("🔨 Building FAISS index...")
        embedding_dim = self.label_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.faiss_index.add(self.label_embeddings.astype('float32'))
        logger.info(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")

    # ─── EMBEDDING ────────────────────────────────────────────────────

    def embed(self, texts: Union[List[str], pd.Series], batch_size: int = 512) -> np.ndarray:
        """
        Compute embeddings for a list of texts using the trained model.
        
        Args:
            texts: List or Series of text strings
            batch_size: Encoding batch size
        
        Returns:
            np.ndarray of shape (n_texts, embedding_dim)
        """
        self.siamese_model.eval()
        texts = list(texts)
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                tokenized = self.tokenizer(batch_texts, return_tensors='pt', truncation=True,
                                          padding='max_length', max_length=128)
                batch_emb = self.siamese_model.forward_one(
                    tokenized['input_ids'], tokenized['attention_mask']
                ).cpu().numpy()
                all_embeddings.append(batch_emb)
        
        return np.vstack(all_embeddings).astype('float32')

    # ─── HEAD CLASSIFIER (WEAK LEARNER) ──────────────────────────────

    def fit_head(self, df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label',
                 head: str = 'auto', batch_size: int = 512) -> Dict:
        """
        Fit a weak learner head on top of frozen contrastive embeddings.
        
        This is faster and often more accurate than pure nearest-neighbor,
        especially with many classes and overlapping embedding clusters.
        
        Args:
            df: DataFrame with text and labels
            text_col: Column name for text
            label_col: Column name for labels
            head: Classifier type. Options:
                  'logreg' — LogisticRegression (fast, linear)
                  'svm' — LinearSVC (fast, linear, margin-based)
                  'rf' — RandomForestClassifier (non-linear, robust)
                  'lgbm' — LightGBM (gradient boosted, best accuracy)
                  'xgb' — XGBoost (gradient boosted, alternative)
                  'auto' — tries all available, picks best by CV
            batch_size: Batch size for embedding computation
        
        Returns:
            Dict with head_type, cv_accuracy, n_classes
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score
        
        logger.info(f"🎯 Fitting head classifier (strategy='{head}')...")
        
        # Encode all texts
        texts = df[text_col].tolist()
        labels = df[label_col].tolist()
        
        X_emb = self.embed(texts, batch_size=batch_size)
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels)
        self._head_label_encoder = le
        
        # Get available classifiers
        candidates = self._get_head_candidates(head)
        
        if head == 'auto':
            # Cross-validate all candidates, pick best
            logger.info(f"  🔬 Auto-selecting from {len(candidates)} candidates...")
            best_score = -1
            best_name = None
            best_model = None
            
            for name, model in candidates:
                try:
                    scores = cross_val_score(model, X_emb, y, cv=min(5, len(np.unique(y))),
                                            scoring='accuracy', n_jobs=-1)
                    mean_score = scores.mean()
                    logger.info(f"    {name}: {mean_score:.4f} (±{scores.std():.4f})")
                    if mean_score > best_score:
                        best_score = mean_score
                        best_name = name
                        best_model = model
                except Exception as e:
                    logger.info(f"    {name}: ⚠️ failed ({e})")
            
            if best_model is None:
                raise RuntimeError("All head classifiers failed")
            
            logger.info(f"\n  🏆 Best head: {best_name} (accuracy={best_score:.4f})")
            best_model.fit(X_emb, y)
            self._head_model = best_model
            self._head_type = best_name
        else:
            # Use specified head
            name, model = candidates[0]
            scores = cross_val_score(model, X_emb, y, cv=min(5, len(np.unique(y))),
                                    scoring='accuracy', n_jobs=-1)
            logger.info(f"  {name}: {scores.mean():.4f} (±{scores.std():.4f})")
            model.fit(X_emb, y)
            self._head_model = model
            self._head_type = name
            best_score = scores.mean()
        
        logger.info(f"✅ Head fitted: {self._head_type}")
        return {
            'head_type': self._head_type,
            'cv_accuracy': float(best_score),
            'n_classes': len(le.classes_),
        }

    def _get_head_candidates(self, head: str) -> List[tuple]:
        """Build list of (name, model) tuples for head selection."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier
        
        candidates = []
        
        if head in ('auto', 'logreg'):
            candidates.append(('LogisticRegression', 
                              LogisticRegression(max_iter=1000, C=1.0, random_state=42)))
        if head in ('auto', 'svm'):
            candidates.append(('LinearSVC', 
                              LinearSVC(max_iter=2000, C=1.0, random_state=42)))
        if head in ('auto', 'rf'):
            candidates.append(('RandomForest', 
                              RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)))
        
        # Optional: LightGBM
        if head in ('auto', 'lgbm'):
            try:
                from lightgbm import LGBMClassifier
                candidates.append(('LightGBM', 
                                  LGBMClassifier(n_estimators=200, learning_rate=0.1,
                                                 random_state=42, verbosity=-1, n_jobs=-1)))
            except ImportError:
                if head == 'lgbm':
                    logger.info("⚠️ LightGBM not installed, falling back to LogReg")
                    candidates.append(('LogisticRegression', 
                                      LogisticRegression(max_iter=1000, random_state=42)))
        
        # Optional: XGBoost
        if head in ('auto', 'xgb'):
            try:
                from xgboost import XGBClassifier
                candidates.append(('XGBoost', 
                                  XGBClassifier(n_estimators=200, learning_rate=0.1,
                                                random_state=42, verbosity=0, n_jobs=-1)))
            except ImportError:
                if head == 'xgb':
                    logger.info("⚠️ XGBoost not installed, falling back to LogReg")
                    candidates.append(('LogisticRegression', 
                                      LogisticRegression(max_iter=1000, random_state=42)))
        
        return candidates

    # ─── PREDICTION ───────────────────────────────────────────────────

    def predict(self, text: str, top_k: int = 1, return_confidence: bool = False):
        """
        Predict label for a single text.
        Uses head classifier if fitted, else nearest-neighbor.
        
        Args:
            text: Input text string
            top_k: Number of top predictions to return
            return_confidence: Return confidence scores alongside predictions
        
        Returns:
            str (top_k=1) or list[str] (top_k>1) or dict (return_confidence=True)
        """
        self.siamese_model.eval()
        
        # Encode text
        tokenized = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                   padding='max_length', max_length=128)
        with torch.no_grad():
            text_emb = self.siamese_model.forward_one(
                tokenized['input_ids'], tokenized['attention_mask']
            ).cpu().numpy().reshape(1, -1).astype('float32')
        
        # Use head if available
        if self._head_model is not None and top_k == 1 and not return_confidence:
            pred_idx = self._head_model.predict(text_emb)[0]
            return self._head_label_encoder.inverse_transform([pred_idx])[0]
        
        if self._head_model is not None and hasattr(self._head_model, 'predict_proba'):
            proba = self._head_model.predict_proba(text_emb)[0]
            top_indices = np.argsort(proba)[-top_k:][::-1]
            predictions = self._head_label_encoder.inverse_transform(top_indices).tolist()
            confidences = proba[top_indices].tolist()
            if return_confidence:
                return {'predictions': predictions, 'confidences': confidences}
            return predictions[0] if top_k == 1 else predictions
        
        # Nearest-neighbor fallback
        return self._predict_nn(text_emb, top_k, return_confidence)

    def _predict_nn(self, text_emb, top_k, return_confidence):
        """Nearest-neighbor prediction (fallback when no head is fitted)."""
        if self.use_faiss and self.faiss_index is not None:
            distances, indices = self.faiss_index.search(text_emb, top_k)
            similarities = 1 - (distances[0] ** 2 / 2)
            top_indices = indices[0]
        else:
            dot_product = np.dot(text_emb, self.label_embeddings.T)
            norms = np.linalg.norm(text_emb) * np.linalg.norm(self.label_embeddings, axis=1)
            similarities = (dot_product / norms).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            similarities = similarities[top_indices]
        
        predictions = [self.labels[idx] for idx in top_indices]
        
        if return_confidence:
            return {'predictions': predictions, 'confidences': similarities.tolist()}
        return predictions[0] if top_k == 1 else predictions
    
    @print_throughput
    def predict_batch(self, text_series, batch_size: int = 512, top_k: int = 1) -> pd.DataFrame:
        """
        Batch predict labels for multiple texts.
        Uses head classifier if fitted, else nearest-neighbor.
        
        Args:
            text_series: Series or list of texts
            batch_size: Batch size for encoding
            top_k: Return top K predictions per text
        
        Returns:
            DataFrame with columns: text, prediction, confidence, [top_k_predictions, top_k_confidences]
        """
        self.siamese_model.eval()
        texts = list(text_series)
        
        # Encode all texts
        embeddings = self.embed(texts, batch_size=batch_size)
        
        # Use head if available
        if self._head_model is not None:
            if hasattr(self._head_model, 'predict_proba'):
                proba = self._head_model.predict_proba(embeddings)
                pred_indices = np.argsort(proba, axis=1)[:, -top_k:][:, ::-1]
                confidences = np.take_along_axis(proba, pred_indices, axis=1)
                
                results = []
                for i in range(len(texts)):
                    preds = self._head_label_encoder.inverse_transform(pred_indices[i]).tolist()
                    confs = confidences[i].tolist()
                    results.append({
                        'text': texts[i],
                        'prediction': preds[0],
                        'confidence': confs[0],
                        'top_k_predictions': preds if top_k > 1 else None,
                        'top_k_confidences': confs if top_k > 1 else None,
                    })
                return pd.DataFrame(results)
            else:
                preds = self._head_model.predict(embeddings)
                pred_labels = self._head_label_encoder.inverse_transform(preds)
                return pd.DataFrame({
                    'text': texts,
                    'prediction': pred_labels,
                    'confidence': [1.0] * len(texts),
                    'top_k_predictions': [None] * len(texts),
                    'top_k_confidences': [None] * len(texts),
                })
        
        # Nearest-neighbor fallback
        logger.info(f"Searching top-{top_k} predictions (NN)...")
        if self.use_faiss and self.faiss_index is not None:
            distances, indices = self.faiss_index.search(embeddings, top_k)
            similarities = 1 - (distances ** 2 / 2)
        else:
            dot_product = np.dot(embeddings, self.label_embeddings.T)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) * np.linalg.norm(self.label_embeddings, axis=1)
            similarities = dot_product / norms
            indices = np.argsort(similarities, axis=1)[:, -top_k:][:, ::-1]
            similarities = np.take_along_axis(similarities, indices, axis=1)
        
        results = []
        for i in range(len(texts)):
            pred_labels = [self.labels[idx] for idx in indices[i]]
            pred_confs = similarities[i].tolist()
            results.append({
                'text': texts[i],
                'prediction': pred_labels[0],
                'confidence': pred_confs[0],
                'top_k_predictions': pred_labels if top_k > 1 else None,
                'top_k_confidences': pred_confs if top_k > 1 else None,
            })
        return pd.DataFrame(results)

    # ─── REPORT GENERATION ────────────────────────────────────────────

    def generate_report(self, y_true, y_pred, report_path: str = "contrastive_report.html",
                        embeddings: Optional[np.ndarray] = None,
                        selector_results: Optional[pd.DataFrame] = None) -> str:
        """
        Generate an HTML validation report for the contrastive classifier.
        
        Includes: confusion matrix heatmap, per-class F1 scores, backbone
        selection results (if provided), and embedding visualization.
        
        Args:
            y_true: True labels (array-like)
            y_pred: Predicted labels (array-like)
            report_path: Output HTML file path
            embeddings: Optional embeddings for t-SNE visualization
            selector_results: Optional DataFrame from EmbeddingSelector
        
        Returns:
            Path to the generated HTML report
        """
        from sklearn.metrics import (classification_report, confusion_matrix,
                                      accuracy_score, f1_score)
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        from scomp_link.utils.plotly_utils import barchart
        import plotly.figure_factory as ff
        import plotly.express as px
        import plotly.graph_objects as go
        
        y_true = list(y_true)
        y_pred = list(y_pred)
        
        report = ScompLinkHTMLReport(title='Contrastive Text Classifier — Report')
        
        # Section 1: Overall metrics
        report.open_section("Overall Performance")
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics_df = pd.DataFrame([{
            'Accuracy': f"{acc:.4f}",
            'F1 (weighted)': f"{f1:.4f}",
            'Head': self._head_type or 'Nearest Neighbor',
            'Embedding Dim': self.label_embeddings.shape[1] if self.label_embeddings is not None else '—',
            'N Classes': len(set(y_true)),
        }])
        report.add_dataframe(metrics_df, 'metrics')
        report.close_section()
        
        # Section 2: Per-class F1
        report.open_section("Per-Class Performance")
        cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cls_df = pd.DataFrame(cls_report).T
        cls_df = cls_df[cls_df.index.isin(set(y_true))].sort_values('f1-score', ascending=False)
        
        fig_f1 = go.Figure(go.Bar(
            x=cls_df.index.tolist(),
            y=cls_df['f1-score'].tolist(),
            marker_color='#6E37FA'
        ))
        fig_f1.update_layout(title="F1-Score per Class", xaxis_tickangle=-45,
                            yaxis_title="F1", height=400)
        report.add_graph_to_report(fig_f1, "Per-Class F1-Score")
        report.close_section()
        
        # Section 3: Confusion Matrix
        report.open_section("Confusion Matrix")
        labels_sorted = sorted(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        # Only show if not too many classes
        if len(labels_sorted) <= 20:
            fig_cm = px.imshow(cm, x=labels_sorted, y=labels_sorted,
                              color_continuous_scale='Purples',
                              labels=dict(x="Predicted", y="True", color="Count"))
            fig_cm.update_layout(title="Confusion Matrix", height=500)
            report.add_graph_to_report(fig_cm, "Confusion Matrix")
        else:
            report.add_text(f"Confusion matrix omitted ({len(labels_sorted)} classes — too large for visualization)")
        report.close_section()
        
        # Section 4: Backbone Selection (if provided)
        if selector_results is not None:
            report.open_section("Backbone Selection")
            report.add_dataframe(selector_results, 'backbone_ranking')
            fig_sel = go.Figure(go.Bar(
                x=selector_results['model'].tolist(),
                y=selector_results['loss'].tolist(),
                marker_color='#34d399'
            ))
            fig_sel.update_layout(title="Contrastive Loss by Backbone (lower = better)",
                                xaxis_tickangle=-45, yaxis_title="Loss", height=350)
            report.add_graph_to_report(fig_sel, "Backbone Comparison")
            report.close_section()
        
        # Section 5: Embedding Visualization (t-SNE)
        if embeddings is not None and len(embeddings) <= 5000:
            report.open_section("Embedding Space (t-SNE)")
            try:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
                coords = tsne.fit_transform(embeddings)
                viz_df = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'label': y_true[:len(embeddings)]})
                fig_tsne = px.scatter(viz_df, x='x', y='y', color='label',
                                     title="t-SNE of Contrastive Embeddings", height=500)
                fig_tsne.update_traces(marker=dict(size=4, opacity=0.7))
                report.add_graph_to_report(fig_tsne, "t-SNE Embedding Space")
            except Exception as e:
                report.add_text(f"t-SNE visualization failed: {e}")
            report.close_section()
        
        report.save_html(report_path)
        logger.info(f"✅ Report saved: {report_path}")
        return report_path

    # ─── SAVE / LOAD ──────────────────────────────────────────────────

    def save(self, path: str = './ContrastiveTextModel'):
        """Save model, embeddings, and head classifier."""
        os.makedirs(path, exist_ok=True)
        
        # Save siamese model weights
        torch.save(self.siamese_model.state_dict(), os.path.join(path, 'model.pt'))
        
        # Save metadata
        embedding_dim = self.siamese_model.projection_layer[-1].out_features
        metadata = {
            'labels': self.labels,
            'label_freq_cache': self.label_freq_cache,
            'embedding_dim': embedding_dim,
            'model_name': self.model_name,
            'head_type': self._head_type,
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save embeddings
        if self.label_embeddings is not None:
            np.savetxt(os.path.join(path, 'embeddings.csv'), self.label_embeddings, delimiter=',')
        
        # Save head classifier (if fitted)
        if self._head_model is not None:
            with open(os.path.join(path, 'head_model.pkl'), 'wb') as f:
                pickle.dump(self._head_model, f)
            with open(os.path.join(path, 'head_label_encoder.pkl'), 'wb') as f:
                pickle.dump(self._head_label_encoder, f)
        
        logger.info(f"✅ Model saved to {path}")
    
    def load(self, path: str = './ContrastiveTextModel', model_name: Optional[str] = None):
        """Load model, embeddings, and head classifier."""
        # Load metadata
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        embedding_dim = metadata.get('embedding_dim', 256)
        loaded_model_name = model_name or metadata.get('model_name', 'bert-base-uncased')
        
        # Reinitialize with correct dimensions
        self.__init__(model_name=loaded_model_name, use_faiss=self.use_faiss, embedding_dim=embedding_dim)
        
        # Load weights
        self.siamese_model.load_state_dict(
            torch.load(os.path.join(path, 'model.pt'), map_location=torch.device('cpu'))
        )
        self.siamese_model.eval()
        
        self.labels = metadata['labels']
        self.label_freq_cache = metadata.get('label_freq_cache', {})
        
        # Load embeddings
        emb_path = os.path.join(path, 'embeddings.csv')
        if os.path.exists(emb_path):
            self.label_embeddings = np.loadtxt(emb_path, delimiter=',')
        
        # Build FAISS index
        if self.use_faiss:
            self._build_faiss_index()
        
        # Load head classifier (if exists)
        head_path = os.path.join(path, 'head_model.pkl')
        if os.path.exists(head_path):
            with open(head_path, 'rb') as f:
                self._head_model = pickle.load(f)
            with open(os.path.join(path, 'head_label_encoder.pkl'), 'rb') as f:
                self._head_label_encoder = pickle.load(f)
            self._head_type = metadata.get('head_type')
            logger.info(f"✅ Head loaded: {self._head_type}")
        
        logger.info(f"✅ Model loaded from {path}")
