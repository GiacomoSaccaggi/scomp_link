# -*- coding: utf-8 -*-
"""
Example: Contrastive Embedding Pipeline with Weak Learner Head

Demonstrates the full contrastive text classification workflow:
1. EmbeddingSelector — find the best pretrained backbone (offline mode)
2. ContrastiveTextClassifier — train contrastive embeddings
3. fit_head() — fit a weak learner (auto-selects best)
4. Compare: head accuracy vs nearest-neighbor accuracy
5. Generate an HTML report

Uses synthetic data — no model download required for this demo.
For real usage, replace with your dataset and remove precomputed_embeddings.
"""
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.metrics import accuracy_score

from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)


def create_synthetic_dataset(n_per_class=50, n_classes=5, seed=42):
    """Create a synthetic text classification dataset."""
    np.random.seed(seed)
    categories = [f"category_{i}" for i in range(n_classes)]
    texts = []
    labels = []
    
    word_banks = {
        'category_0': ['machine', 'learning', 'algorithm', 'neural', 'deep', 'model', 'training'],
        'category_1': ['football', 'basketball', 'tennis', 'match', 'score', 'player', 'game'],
        'category_2': ['stock', 'market', 'trading', 'finance', 'bank', 'economy', 'invest'],
        'category_3': ['recipe', 'cooking', 'ingredient', 'kitchen', 'food', 'dish', 'flavor'],
        'category_4': ['movie', 'film', 'actor', 'director', 'cinema', 'scene', 'plot'],
    }
    
    for cat in categories:
        words = word_banks[cat]
        for _ in range(n_per_class):
            n_words = np.random.randint(4, 10)
            text = ' '.join(np.random.choice(words, size=n_words, replace=True))
            texts.append(text)
            labels.append(cat)
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    logger.info("\n" + "=" * 60)
    logger.info("  Contrastive Embedding Pipeline — Example")
    logger.info("=" * 60)
    
    # ─── Step 1: Create dataset ────────────────────────────────
    logger.info("\n📊 Step 1: Creating synthetic dataset...")
    df = create_synthetic_dataset(n_per_class=50, n_classes=5)
    logger.info(f"   Dataset: {len(df)} rows, {df['label'].nunique()} classes")
    logger.info(f"   Classes: {df['label'].unique().tolist()}")
    
    # ─── Step 2: Backbone Selection (offline mode) ─────────────
    logger.info("\n🔍 Step 2: Selecting best backbone (offline mode)...")
    from scomp_link.models.contrastive_text import EmbeddingSelector
    
    # Simulate precomputed embeddings for two "models"
    # In real usage: remove precomputed_embeddings to download actual models
    np.random.seed(42)
    n = len(df)
    labels = df['label'].tolist()
    
    # "model_good": embeddings with class structure
    emb_good = np.zeros((n, 64), dtype='float32')
    for i, label in enumerate(labels):
        class_idx = int(label.split('_')[1])
        emb_good[i] = np.random.randn(64) * 0.5 + class_idx * 2
    
    # "model_bad": random embeddings (no class structure)
    emb_bad = np.random.randn(n, 128).astype('float32')
    
    precomputed = {
        'model_with_structure': emb_good,
        'model_random': emb_bad,
    }
    
    selector = EmbeddingSelector(candidates=['model_with_structure', 'model_random'])
    selector_results = selector.find_best_backbone(
        df, text_col='text', label_col='label',
        precomputed_embeddings=precomputed
    )
    logger.info(f"\n   Results:\n{selector_results.to_string(index=False)}")
    
    # ─── Step 3: Fit head on best embeddings ───────────────────
    logger.info("\n🎯 Step 3: Fitting head classifier on embeddings...")
    
    # Split train/test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Use the "good" embeddings directly to demo fit_head without actual BERT
    train_emb = emb_good[train_df.index]
    test_emb = emb_good[test_df.index]
    
    # Fit multiple weak learners and auto-select
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['label'])
    y_test = le.transform(test_df['label'])
    
    heads = [
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
        ('LinearSVC', LinearSVC(max_iter=2000, random_state=42)),
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
    
    logger.info("\n   Cross-validating heads:")
    best_score = -1
    best_name = None
    best_model = None
    
    for name, model in heads:
        scores = cross_val_score(model, train_emb, y_train, cv=5, scoring='accuracy')
        logger.info(f"     {name}: {scores.mean():.4f} (±{scores.std():.4f})")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name
            best_model = model
    
    logger.info(f"\n   🏆 Best: {best_name} (CV accuracy={best_score:.4f})")
    
    # Train final model
    best_model.fit(train_emb, y_train)
    
    # ─── Step 4: Evaluate ──────────────────────────────────────
    logger.info("\n📈 Step 4: Evaluating on test set...")
    
    # Head prediction
    y_pred_head = best_model.predict(test_emb)
    acc_head = accuracy_score(y_test, y_pred_head)
    
    # Nearest-neighbor prediction (for comparison)
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_emb, y_train)
    y_pred_nn = knn.predict(test_emb)
    acc_nn = accuracy_score(y_test, y_pred_nn)
    
    logger.info(f"   Head ({best_name}): accuracy = {acc_head:.4f}")
    logger.info(f"   Nearest Neighbor (k=5): accuracy = {acc_nn:.4f}")
    logger.info(f"   Improvement: +{(acc_head - acc_nn)*100:.1f}%")
    
    # ─── Step 5: Generate Report ──────────────────────────────
    logger.info("\n📄 Step 5: Generating HTML report...")
    from scomp_link.models.contrastive_text import ContrastiveTextClassifier
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    # Create classifier instance for report generation (mocked)
    with patch.object(AutoTokenizer, 'from_pretrained') as mock_tok, \
         patch.object(AutoModel, 'from_pretrained') as mock_model:
        mock_tok.return_value = MagicMock()
        bert = MagicMock()
        bert.config = MagicMock(hidden_size=768)
        bert.train = MagicMock(return_value=bert)
        bert.parameters = MagicMock(return_value=iter([]))
        mock_model.return_value = bert
        clf = ContrastiveTextClassifier(use_faiss=False, embedding_dim=64)
    
    clf.labels = le.classes_.tolist()
    clf.label_embeddings = np.random.randn(len(le.classes_), 64).astype('float32')
    clf._head_type = best_name
    
    y_true_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred_head)
    
    report_path = clf.generate_report(
        y_true_labels, y_pred_labels,
        report_path="staging/contrastive_pipeline_report.html",
        embeddings=test_emb,
        selector_results=selector_results
    )
    
    logger.info(f"\n   ✅ Report: {report_path}")
    
    # ─── Summary ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  ✅ Pipeline Complete!")
    logger.info(f"     Best backbone: {selector_results.iloc[0]['model']}")
    logger.info(f"     Best head: {best_name}")
    logger.info(f"     Test accuracy: {acc_head:.4f}")
    logger.info(f"     Report: {report_path}")
    logger.info("=" * 60 + "\n")


if __name__ == '__main__':
    import os
    os.makedirs('staging', exist_ok=True)
    main()
