#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 13: Text Unsupervised - Text Embeddings and Clustering
Use Case: Group similar texts without labels
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline

print("=" * 70)
print("EXAMPLE 13: Text Unsupervised - Text Embeddings and Clustering")
print("=" * 70)

try:
    from scomp_link.models.unsupervised_text import TextEmbeddingClustering
    
    texts = [
        "Machine learning transforms data analysis",
        "Deep learning revolutionizes AI",
        "Stock market shows strong growth",
        "Economic indicators point upward",
        "Football team wins championship",
        "Basketball playoffs begin",
        "New movie breaks records",
        "Actor wins prestigious award",
        "Scientists discover new planet",
        "Research breakthrough in medicine"
    ] * 20
    
    df = pd.DataFrame({'text': texts})
    
    print(f"Dataset size: {len(df)} records")
    
    pipe = ScompLinkPipeline("Text Clustering without Labels")
    pipe.set_objectives(["Discover Text Groups", "Semantic Clustering"])
    pipe.import_and_clean_data(df)
    df['cluster'] = 0
    pipe.select_variables(target_col='cluster')
    pipe.choose_model("categorical_unknown", metadata={"categories_known": True, "n_clusters": 5})
    
    # Use text embeddings for clustering
    clusterer = TextEmbeddingClustering(model_name='bert-base-uncased', n_clusters=5)
    clusters = clusterer.fit_predict(df['text'].values)
    
    df['predicted_cluster'] = clusters
    
    from sklearn.metrics import silhouette_score
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text'].values)
    silhouette = silhouette_score(embeddings, clusters)
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print(f"Clusters found: {len(np.unique(clusters))}")
    print(f"Cluster distribution: {pd.Series(clusters).value_counts().to_dict()}")
    print(f"Silhouette Score: {silhouette:.3f}")
    print("=" * 70)
    
    # Show sample texts per cluster
    print("\nSample texts per cluster:")
    for cluster_id in range(5):
        cluster_texts = df[df['predicted_cluster'] == cluster_id]['text'].head(2).tolist()
        print(f"\nCluster {cluster_id}:")
        for text in cluster_texts:
            print(f"  - {text}")
    
    print("\n✅ Text clustering completed!")
    
    # Save clusterer
    print("\n" + "=" * 70)
    print("SAVING CLUSTERING MODEL...")
    import pickle
    import os
    os.makedirs('./staging/example_13', exist_ok=True)
    with open('./staging/example_13/clusterer.pkl', 'wb') as f:
        pickle.dump(clusterer, f)
    print("✅ Clusterer saved to ./staging/example_13/clusterer.pkl")
    print("=" * 70)
    
    # Load and predict on new texts
    print("\n" + "=" * 70)
    print("LOADING CLUSTERER AND TESTING PREDICTION...")
    with open('./staging/example_13/clusterer.pkl', 'rb') as f:
        loaded_clusterer = pickle.load(f)
    
    test_texts = ["AI transforms industry", "Team wins trophy", "New discovery announced"]
    test_clusters = loaded_clusterer.predict(test_texts)
    print(f"Test cluster assignments: {test_clusters}")
    print("✅ Clusterer saved, loaded, and tested successfully!")
    print("=" * 70)
    
except ImportError as e:
    print("\n⚠️  NLP dependencies not installed!")
    print("Install with: pip install .[nlp] sentence-transformers")
    print(f"Error: {e}")
