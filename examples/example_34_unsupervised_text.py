# -*- coding: utf-8 -*-
"""
Example 34: Unsupervised Text Clustering (TextEmbeddingClustering)
===================================================================

Demonstrates the TextEmbeddingClustering model:
  1. Initialize with sentence-transformers embeddings
  2. Fit on unlabelled text corpus
  3. Predict cluster assignments
  4. Evaluate with silhouette score
  5. Inspect cluster contents

Note: This example requires:
  - sentence-transformers: pip install sentence-transformers

Requirements:
  pip install scomp-link
"""

import numpy as np
from sklearn.metrics import silhouette_score

from scomp_link.utils.decorators import timer, memory_usage


@memory_usage
def generate_text_corpus():
    """Generate a synthetic text corpus with implicit topic clusters."""
    np.random.seed(42)

    # 4 topic clusters (the model should discover them without labels)
    topics = {
        'sports': [
            "The football match ended in a dramatic penalty shootout",
            "Tennis champion wins third consecutive grand slam title",
            "Basketball team breaks scoring record in overtime victory",
            "Olympic swimmers set new world records at championships",
            "Golf tournament final round produces thrilling finish",
            "Marathon runner completes race in under two hours",
            "Boxing heavyweight title fight goes to decision",
            "Cricket team wins series with dominant bowling performance",
            "Formula one driver secures pole position for race",
            "Hockey playoff series goes to game seven",
            "Rugby world cup final attracts record viewership",
            "Volleyball team wins gold medal at asian games",
        ],
        'technology': [
            "New artificial intelligence model achieves human-level performance",
            "Quantum computer solves problem in record time",
            "Startup launches revolutionary battery technology for electric vehicles",
            "Cloud computing services expand to new global regions",
            "Cybersecurity firm discovers critical vulnerability in popular software",
            "Smartphone manufacturer reveals foldable display technology",
            "Open source community releases major framework update",
            "Machine learning algorithm improves medical diagnosis accuracy",
            "Autonomous driving technology passes safety certification",
            "Blockchain platform processes million transactions per second",
            "Virtual reality headset achieves photorealistic graphics",
            "Robotics company demonstrates general purpose humanoid robot",
        ],
        'cooking': [
            "Chef prepares traditional Italian pasta with fresh ingredients",
            "Baking sourdough bread requires patience and proper fermentation",
            "Grilled vegetables taste better with olive oil and herbs",
            "Japanese sushi technique takes years to master properly",
            "Slow cooking transforms tough cuts into tender meals",
            "Fresh herbs elevate simple dishes to restaurant quality",
            "Chocolate tempering requires precise temperature control",
            "Indian curry spices need toasting before adding liquid",
            "French pastry cream is base for many classic desserts",
            "Smoked meat gets flavor from low temperature and wood chips",
            "Homemade pizza dough needs overnight cold fermentation",
            "Wok cooking requires very high heat and fast movement",
        ],
        'finance': [
            "Stock market reaches all time high amid strong earnings",
            "Central bank raises interest rates to combat inflation",
            "Cryptocurrency market experiences significant volatility",
            "Real estate prices continue to climb in major cities",
            "Venture capital funding reaches new quarterly record",
            "Bond yields rise as investors expect tighter monetary policy",
            "Merger and acquisition activity surges in technology sector",
            "Commodity prices fluctuate due to supply chain disruptions",
            "Pension fund adjusts portfolio allocation toward alternatives",
            "Digital banking startup attracts millions of new customers",
            "Insurance industry adopts artificial intelligence for claims",
            "Foreign exchange markets react to geopolitical tensions",
        ],
    }

    # Flatten into a single corpus (lose the labels — unsupervised!)
    corpus = []
    true_labels = []
    for topic, texts in topics.items():
        corpus.extend(texts)
        true_labels.extend([topic] * len(texts))

    # Shuffle
    indices = np.random.permutation(len(corpus))
    corpus = [corpus[i] for i in indices]
    true_labels = [true_labels[i] for i in indices]

    return corpus, true_labels


@timer
def cluster_texts(corpus, n_clusters: int = 4, method: str = 'kmeans'):
    """Fit and predict text clusters."""
    from scomp_link.models.unsupervised_text import TextEmbeddingClustering

    clusterer = TextEmbeddingClustering(
        model_name='all-MiniLM-L6-v2',
        n_clusters=n_clusters,
        method=method,
    )

    labels = clusterer.fit_predict(corpus)
    return clusterer, labels


# --- Main execution ---

if __name__ == '__main__':
    print("=" * 70)
    print("UNSUPERVISED TEXT CLUSTERING — TextEmbeddingClustering")
    print("=" * 70)

    # === 1. Generate corpus ===
    print("\n--- 1. Generating text corpus ---")
    corpus, true_labels = generate_text_corpus()
    print(f"  Total documents: {len(corpus)}")
    print(f"  True topics (hidden): {sorted(set(true_labels))}")
    print(f"\n  Sample documents:")
    for i in range(4):
        print(f"    \"{corpus[i][:60]}...\"")

    # === 2. KMeans clustering ===
    print("\n--- 2. KMeans Clustering (k=4) ---")
    clusterer_km, labels_km = cluster_texts(corpus, n_clusters=4, method='kmeans')

    sil_km = silhouette_score(
        clusterer_km.model.encode(corpus),
        labels_km,
    )
    print(f"  Silhouette Score: {sil_km:.4f}")
    print(f"  Cluster sizes: {dict(zip(*np.unique(labels_km, return_counts=True)))}")

    # === 3. Hierarchical clustering ===
    print("\n--- 3. Hierarchical Clustering (k=4) ---")
    clusterer_hc, labels_hc = cluster_texts(corpus, n_clusters=4, method='hierarchical')

    sil_hc = silhouette_score(
        clusterer_hc.model.encode(corpus),
        labels_hc,
    )
    print(f"  Silhouette Score: {sil_hc:.4f}")
    print(f"  Cluster sizes: {dict(zip(*np.unique(labels_hc, return_counts=True)))}")

    # === 4. Inspect cluster contents ===
    print("\n--- 4. Cluster Content Inspection (KMeans) ---")
    for cluster_id in sorted(np.unique(labels_km)):
        cluster_docs = [corpus[i] for i in range(len(corpus)) if labels_km[i] == cluster_id]
        cluster_true = [true_labels[i] for i in range(len(corpus)) if labels_km[i] == cluster_id]

        # Dominant topic in this cluster
        from collections import Counter
        topic_counts = Counter(cluster_true)
        dominant = topic_counts.most_common(1)[0]

        print(f"\n  Cluster {cluster_id} ({len(cluster_docs)} docs, dominant topic: {dominant[0]})")
        for doc in cluster_docs[:3]:
            print(f"    • \"{doc[:55]}...\"")

    # === 5. Evaluate cluster-topic alignment ===
    print("\n--- 5. Cluster-Topic Alignment ---")
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Map true labels to integers for comparison
    label_map = {l: i for i, l in enumerate(sorted(set(true_labels)))}
    true_int = [label_map[l] for l in true_labels]

    ari = adjusted_rand_score(true_int, labels_km)
    nmi = normalized_mutual_info_score(true_int, labels_km)
    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"  Normalized Mutual Information: {nmi:.4f}")
    print(f"  (1.0 = perfect alignment, 0.0 = random)")

    # === 6. Predict on new texts ===
    print("\n--- 6. Predict on new unseen texts ---")
    new_texts = [
        "The striker scored a hat trick in the cup final",
        "Neural network architecture improves image recognition",
        "Season vegetables with salt pepper and fresh rosemary",
        "Hedge fund reports strong quarterly returns on investments",
    ]
    new_labels = clusterer_km.predict(new_texts)
    for text, label in zip(new_texts, new_labels):
        print(f"  Cluster {label}: \"{text[:50]}...\"")

    # === Summary ===
    print("\n" + "=" * 70)
    best_method = "KMeans" if sil_km > sil_hc else "Hierarchical"
    print("✅ Unsupervised text clustering complete!")
    print(f"   • Corpus: {len(corpus)} documents, 4 latent topics")
    print(f"   • KMeans silhouette: {sil_km:.4f}")
    print(f"   • Hierarchical silhouette: {sil_hc:.4f}")
    print(f"   • Best method: {best_method}")
    print(f"   • Topic alignment (ARI): {ari:.4f}")
    print("=" * 70)
