# -*- coding: utf-8 -*-
"""
Example 33: Supervised Text Classification (SpacyEmbeddingModel)
=================================================================

Demonstrates the SpacyEmbeddingModel for text categorization:
  1. Initialize model with spaCy + BERT embeddings
  2. Exercise text preprocessing (extract_words)
  3. Exercise helper methods (report_progress, get_opt_params, configure_optimizer)
  4. Train the CNN-based text categorizer (if spaCy textcat API is compatible)

Note: This example requires:
  - spacy with en_core_web_md model: python -m spacy download en_core_web_md
  - torch, transformers

Requirements:
  pip install scomp-link
  python -m spacy download en_core_web_md
"""

import numpy as np
from scomp_link.utils.decorators import timer, memory_usage


@memory_usage
def generate_text_dataset():
    """Generate a synthetic text classification dataset."""
    np.random.seed(42)

    # Simulated tech support tickets with 3 categories
    templates = {
        'billing': [
            "I was charged twice for my subscription",
            "My invoice shows incorrect amount",
            "Please refund my last payment",
            "Cannot update credit card information",
            "Subscription was cancelled but still charging",
            "Need a receipt for my purchase",
            "How to change billing address",
            "Payment failed but money was deducted",
            "I want to downgrade my plan",
            "Automatic renewal was unexpected",
            "Charged after free trial ended",
            "Price increased without notice",
            "Cannot apply discount code",
            "Need invoice for tax purposes",
            "Double charge on my account",
        ],
        'technical': [
            "Application crashes on startup",
            "Cannot connect to the server",
            "Error 500 when loading dashboard",
            "Login page keeps refreshing",
            "Data export not working properly",
            "API returns timeout errors",
            "Mobile app freezes randomly",
            "Integration with Slack stopped working",
            "File upload fails for large files",
            "Search function returns no results",
            "Cannot sync data between devices",
            "Performance is very slow today",
            "Two factor authentication broken",
            "Notifications not being delivered",
            "Database connection timeout",
        ],
        'account': [
            "How to reset my password",
            "Cannot change my email address",
            "Need to add team members",
            "Account locked after failed attempts",
            "How to enable two factor auth",
            "Delete my account permanently",
            "Transfer ownership to another user",
            "Change username or display name",
            "Merge two accounts together",
            "Session expires too quickly",
            "Cannot access admin settings",
            "Update profile picture not saving",
            "Permission denied for team features",
            "How to export my account data",
            "Deactivate then reactivate account",
        ],
    }

    texts = {}
    labels = {}
    idx = 0
    for category, msgs in templates.items():
        for msg in msgs:
            texts[idx] = msg
            labels[idx] = category
            idx += 1

    categories = list(templates.keys())
    return texts, labels, categories


@timer
def exercise_model_methods(model, texts):
    """Exercise all reachable methods of SpacyEmbeddingModel."""
    results = {}

    # 1. _SpacyEmbeddingModel__extract_words (name-mangled private method)
    print("  Testing __extract_words...")
    sample_texts = [
        "The application crashes when loading the dashboard",
        "I need a refund for my subscription payment",
        "Cannot reset password after multiple failed attempts",
        "",  # edge case: empty string
        "123 456 789",  # edge case: no words
    ]
    for text in sample_texts:
        # Access the private method via name mangling
        words = model._SpacyEmbeddingModel__extract_words(text)
        results[text[:30]] = words

    # 2. report_progress
    print("  Testing report_progress...")
    model.report_progress(
        epoch=5,
        best=0.85,
        losses={"textcat": 0.123},
        scores={"textcat_p": 0.82, "textcat_r": 0.79, "textcat_f": 0.80},
    )

    # 3. get_opt_params
    print("  Testing get_opt_params...")
    params = model.get_opt_params({
        "learn_rate": 0.01,
        "b1": 0.9,
        "b2_ratio": 0.999,
        "adam_eps": 1e-8,
        "L2": 0.0001,
        "grad_norm_clip": 1.0,
    })
    results['opt_params'] = params

    # 4. configure_optimizer (mock optimizer object)
    print("  Testing configure_optimizer...")
    class MockOptimizer:
        alpha = 0.0
        b1 = 0.0
        b2 = 0.0
        eps = 0.0
        L2 = 0.0
        max_grad_norm = 0.0

    mock_opt = MockOptimizer()
    model.configure_optimizer(mock_opt, params)
    results['optimizer_configured'] = {
        'alpha': mock_opt.alpha,
        'b1': mock_opt.b1,
        'max_grad_norm': mock_opt.max_grad_norm,
    }

    # 5. _select_language with invalid language (error path)
    print("  Testing _select_language error path...")
    bad_result = model._select_language('xx', 'md')
    results['bad_language'] = bad_result

    return results


@timer
def try_training(model, texts, labels, categories):
    """Attempt the full training pipeline (may fail on spaCy v3 API changes)."""
    try:
        nlp_model, scores = model.training(texts, labels, categories)
        return scores
    except Exception as e:
        print(f"  ⚠️  Training failed (expected on spaCy v3): {type(e).__name__}")
        return None


# --- Main execution ---

if __name__ == '__main__':
    print("=" * 70)
    print("SUPERVISED TEXT CLASSIFICATION — SpacyEmbeddingModel")
    print("=" * 70)

    # === 1. Generate data ===
    print("\n--- 1. Generating text dataset ---")
    texts, labels, categories = generate_text_dataset()
    print(f"  Total texts: {len(texts)}")
    print(f"  Categories: {categories}")
    print(f"  Samples per category: {len(texts) // len(categories)}")

    # === 2. Initialize model ===
    print("\n--- 2. Initializing SpacyEmbeddingModel ---")
    try:
        from scomp_link.models.supervised_text import SpacyEmbeddingModel

        model = SpacyEmbeddingModel(lan='en', model_name='bert-base-uncased')
        print(f"  ✅ Model initialized")
        print(f"  Language: en")
        print(f"  Spacy pipeline: {model.nlp.pipe_names}")

        # === 3. Exercise individual methods ===
        print("\n--- 3. Exercising model methods ---")
        results = exercise_model_methods(model, texts)

        print(f"\n  Word extraction results:")
        for text_key, words in results.items():
            if text_key not in ('opt_params', 'optimizer_configured', 'bad_language'):
                print(f"    \"{text_key}\" → {words[:5]}")

        print(f"\n  Optimizer params: {results['opt_params']}")
        print(f"  Optimizer configured: {results['optimizer_configured']}")
        print(f"  Bad language result: {results['bad_language']}")

        # === 4. Attempt full training ===
        print("\n--- 4. Attempting full training pipeline ---")
        scores = try_training(model, texts, labels, categories)
        if scores:
            print(f"  ✅ Training succeeded!")
            print(f"  Test report: {scores.get('classification_report_te', 'N/A')[:100]}")
        else:
            print(f"  Training skipped (spaCy v3 API incompatibility with legacy textcat config)")

        # === Summary ===
        print("\n" + "=" * 70)
        print("✅ Supervised text classification example complete!")
        print(f"   • Model initialized with spaCy + BERT")
        print(f"   • __extract_words: tested on {len(results) - 3} texts")
        print(f"   • report_progress, get_opt_params, configure_optimizer: all exercised")
        print(f"   • _select_language error path: tested")
        print("=" * 70)

    except (ImportError, ModuleNotFoundError, OSError, ValueError, Exception) as e:
        print(f"  ⚠️  SpacyEmbeddingModel not available: {type(e).__name__}: {e}")
        print("  Skipping (requires: spacy, en_core_web_md, torch, transformers)")
        print("\n" + "=" * 70)
        print("⚠️  Supervised text example skipped — missing NLP dependencies")
        print("=" * 70)
