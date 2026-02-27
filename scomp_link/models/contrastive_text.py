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
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from functools import wraps
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from .contrastive_net import ContrastiveSiameseModel, ContrastiveLoss, SiameseDataset, EarlyStopping
except (ImportError, ValueError):
    try:
        from scomp_link.models.contrastive_net import ContrastiveSiameseModel, ContrastiveLoss, SiameseDataset, EarlyStopping
    except (ImportError, ValueError):
        ContrastiveSiameseModel = None
        ContrastiveLoss = None
        SiameseDataset = None
        EarlyStopping = None


def print_throughput(func):
    """Decorator to print iterations per second"""
    @wraps(func)
    def wrapper(self, text_series, *args, **kwargs):
        n_items = len(text_series)
        print(f"\n🚀 Processing {n_items} texts...")
        start = time.time()
        result = func(self, text_series, *args, **kwargs)
        elapsed = time.time() - start
        throughput = n_items / elapsed if elapsed > 0 else 0
        print(f"✅ Completed in {elapsed:.2f}s ({throughput:.1f} texts/s)\n")
        return result
    return wrapper


class ContrastiveTextClassifier:
    """
    Generalized Contrastive Learning Text Classifier.
    
    Uses Siamese Networks with BERT for text classification tasks.
    Suitable for:
    - Text categorization
    - Semantic similarity
    - Multi-class classification with many classes
    - Few-shot learning scenarios
    
    Example:
        # Train
        classifier = ContrastiveTextClassifier(model_name='bert-base-uncased')
        classifier.train_contrastive(df, text_col='text', label_col='category', epochs=5)
        
        # Predict
        predictions = classifier.predict(test_texts)
        
        # Batch predict
        results_df = classifier.predict_batch(test_df['text'], top_k=3)
    """
    
    def __init__(self, model_name='bert-base-uncased', use_faiss=True, embedding_dim=256):
        """
        Initialize BERT model and tokenizer
        
        Args:
            model_name: HuggingFace model name
            use_faiss: Use FAISS for fast inference
            embedding_dim: Dimension of projection layer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.train()
        self.siamese_model = ContrastiveSiameseModel(self.model, embedding_dim=embedding_dim)
        
        # FAISS index for fast inference
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index = None
        if self.use_faiss:
            print("✅ FAISS enabled for fast inference")
        else:
            print("⚠️ FAISS disabled - using numpy (slower)")
        
        # Storage for embeddings and labels
        self.label_embeddings = None
        self.labels = []
        self.label_freq_cache = {}

    def train_contrastive(self, df, text_col='text', label_col='label', 
                         epochs=5, batch_size=64, lr=2e-5,
                         use_weighted_sampling=True, accumulation_steps=2,
                         validation_split=0.1, early_stopping_patience=5):
        """
        Train with contrastive learning
        
        Args:
            df: DataFrame with text and labels
            text_col: Column name for text
            label_col: Column name for labels
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            use_weighted_sampling: Balance classes
            accumulation_steps: Gradient accumulation
            validation_split: Validation set ratio
            early_stopping_patience: Early stopping patience
        """
        print("🚀 Preparing contrastive training...")
        
        # Rename columns for compatibility with SiameseDataset
        df_train = df.copy()
        df_train['url'] = df_train[text_col]  # Reuse 'url' field
        df_train['app_name'] = df_train[label_col]  # Reuse 'app_name' field
        
        # Split train/validation
        if validation_split > 0:
            val_size = int(len(df_train) * validation_split)
            train_df = df_train.iloc[:-val_size].reset_index(drop=True)
            val_df = df_train.iloc[-val_size:].reset_index(drop=True)
            print(f"📊 Train: {len(train_df)}, Validation: {len(val_df)}")
        else:
            train_df = df_train
            val_df = None
        
        # Create dataset (reuse SiameseDataset with identity tokenizer)
        class TextDataset(SiameseDataset):
            def __init__(self, df, tokenizer, augment_prob=0.5):
                self.df = df
                self.tokenizer = tokenizer
                self.url_tokens = df['url'].tolist()  # No tokenization, use raw text
                self.app_names = df['app_name'].tolist()
                self.hard_negative_ratio = 0.3
                self.augment_prob = augment_prob
                self.app_frequencies = df['app_name'].value_counts().to_dict()
                self.popular_apps = list(df['app_name'].value_counts().head(1000).index)
        
        train_dataset = TextDataset(train_df, self.tokenizer)
        
        # Weighted sampling
        if use_weighted_sampling:
            sample_weights = SiameseDataset.get_sample_weights(train_df)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            print("✅ Weighted sampling enabled")
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation loader
        val_loader = None
        if val_df is not None:
            val_dataset = TextDataset(val_df, self.tokenizer, augment_prob=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = ContrastiveLoss(margin=1.0)
        optimizer = optim.AdamW([
            {'params': self.siamese_model.bert.parameters(), 'lr': lr},
            {'params': self.siamese_model.projection_layer.parameters(), 'lr': lr * 10}
        ], weight_decay=0.01)
        
        print(f"✅ Differentiated LR: BERT={lr:.2e}, Projection={lr*10:.2e}")
        print(f"✅ Gradient accumulation: {accumulation_steps} steps")
        
        # Early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.001)
        
        self.siamese_model.train()
        print(f"\n🎯 Starting training for {epochs} epochs...\n")
        
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            
            for i, data in enumerate(train_loader):
                text_ids, text_mask, label_ids, label_mask, labels = data
                
                text_emb, label_emb = self.siamese_model(text_ids, text_mask, label_ids, label_mask)
                loss = criterion(text_emb, label_emb, labels)
                
                loss = loss / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader, criterion)
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if early_stopping(val_loss):
                    print(f"⚠️ Early stopping at epoch {epoch + 1}")
                    break
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        self.siamese_model.eval()
        print("\n✅ Training completed!\n")
        
        # Calculate embeddings and build index
        self._calculate_label_embeddings(df_train, label_col)
        if self.use_faiss:
            self._build_faiss_index()
    
    def _validate(self, val_loader, criterion):
        """Validation step"""
        self.siamese_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data in val_loader:
                text_ids, text_mask, label_ids, label_mask, labels = data
                text_emb, label_emb = self.siamese_model(text_ids, text_mask, label_ids, label_mask)
                loss = criterion(text_emb, label_emb, labels)
                total_loss += loss.item()
        
        self.siamese_model.train()
        return total_loss / len(val_loader)
    
    def _calculate_label_embeddings(self, df, label_col):
        """Calculate embeddings for all unique labels"""
        print("Calculating label embeddings...")
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
        
        print(f"✅ Embeddings calculated for {len(unique_labels)} labels")
    
    def _build_faiss_index(self):
        """Build FAISS index for fast search"""
        if not self.use_faiss or self.label_embeddings is None:
            return
        
        print("🔨 Building FAISS index...")
        embedding_dim = self.label_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.faiss_index.add(self.label_embeddings.astype('float32'))
        print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def predict(self, text, top_k=1, return_confidence=False):
        """
        Predict label for single text
        
        Args:
            text: Input text
            top_k: Return top K predictions
            return_confidence: Return confidence scores
            
        Returns:
            str or dict: Predicted label(s)
        """
        self.siamese_model.eval()
        
        # Encode text
        tokenized = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                   padding='max_length', max_length=128)
        with torch.no_grad():
            text_emb = self.siamese_model.forward_one(
                tokenized['input_ids'], tokenized['attention_mask']
            ).cpu().numpy().reshape(1, -1).astype('float32')
        
        # Search
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
    def predict_batch(self, text_series, batch_size=512, top_k=1):
        """
        Batch predict labels for multiple texts
        
        Args:
            text_series: Series or list of texts
            batch_size: Batch size for encoding
            top_k: Return top K predictions per text
            
        Returns:
            DataFrame with predictions and confidences
        """
        self.siamese_model.eval()
        
        texts = list(text_series)
        all_embeddings = []
        
        print(f"Encoding {len(texts)} texts...")
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
                batch_texts = texts[i:i + batch_size]
                tokenized = self.tokenizer(batch_texts, return_tensors='pt', truncation=True,
                                          padding='max_length', max_length=128)
                batch_emb = self.siamese_model.forward_one(
                    tokenized['input_ids'], tokenized['attention_mask']
                ).cpu().numpy()
                all_embeddings.append(batch_emb)
        
        embeddings = np.vstack(all_embeddings).astype('float32')
        del all_embeddings
        
        print(f"Searching top-{top_k} predictions...")
        if self.use_faiss and self.faiss_index is not None:
            distances, indices = self.faiss_index.search(embeddings, top_k)
            similarities = 1 - (distances ** 2 / 2)
        else:
            dot_product = np.dot(embeddings, self.label_embeddings.T)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) * np.linalg.norm(self.label_embeddings, axis=1)
            similarities = dot_product / norms
            indices = np.argsort(similarities, axis=1)[:, -top_k:][:, ::-1]
            similarities = np.take_along_axis(similarities, indices, axis=1)
        
        # Build results
        results = []
        for i in range(len(texts)):
            pred_labels = [self.labels[idx] for idx in indices[i]]
            pred_confidences = similarities[i].tolist()
            results.append({
                'text': texts[i],
                'prediction': pred_labels[0],
                'confidence': pred_confidences[0],
                'top_k_predictions': pred_labels if top_k > 1 else None,
                'top_k_confidences': pred_confidences if top_k > 1 else None
            })
        
        return pd.DataFrame(results)
    
    def save(self, path='./ContrastiveTextModel'):
        """Save model and embeddings"""
        os.makedirs(path, exist_ok=True)
        
        # Save weights
        torch.save(self.siamese_model.state_dict(), os.path.join(path, 'model.pt'))
        
        # Save metadata
        metadata = {
            'labels': self.labels,
            'label_freq_cache': self.label_freq_cache
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save embeddings
        np.savetxt(os.path.join(path, 'embeddings.csv'), self.label_embeddings, delimiter=',')
        
        print(f"✅ Model saved to {path}")
    
    def load(self, path='./ContrastiveTextModel', model_name='bert-base-uncased'):
        """Load model and embeddings"""
        # Reinitialize
        self.__init__(model_name=model_name, use_faiss=self.use_faiss)
        
        # Load weights
        self.siamese_model.load_state_dict(
            torch.load(os.path.join(path, 'model.pt'), map_location=torch.device('cpu'))
        )
        self.siamese_model.eval()
        
        # Load metadata
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        self.labels = metadata['labels']
        self.label_freq_cache = metadata['label_freq_cache']
        
        # Load embeddings
        self.label_embeddings = np.loadtxt(os.path.join(path, 'embeddings.csv'), delimiter=',')
        
        # Build FAISS index
        if self.use_faiss:
            self._build_faiss_index()
        
        print(f"✅ Model loaded from {path}")
