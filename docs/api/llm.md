# LLM Toolkit

End-to-end large language model workflows: fine-tuning, building from scratch, evaluation, RAG, model merging, quantization, and serving.

## Installation

```bash
pip install scomp-link[llm]
```

This pulls in `torch`, `transformers`, `peft`, `bitsandbytes`, `chromadb`, `sentence-transformers`, `llama-cpp-python`, and related dependencies.

---

## Quick Start

```python
from scomp_link.llm import FineTuner, ModelConverter, TextQualityMetrics

# Fine-tune with LoRA
ft = FineTuner("meta-llama/Llama-3-8B", method="lora")
result = ft.train("alpaca.json", epochs=3)
ft.merge_and_save("./merged")

# Convert to GGUF
mc = ModelConverter("./merged")
mc.to_gguf(quantization="Q4_K_M")

# Evaluate
metrics = TextQualityMetrics.evaluate(generated, references=ground_truth)
```

---

## Package Structure

```
scomp_link/llm/
├── core/               # FineTuner, TransformerBuilder, TransformerConfig, FineTuneConfig
├── training/           # LoRA/QLoRA/full training logic, GPT from scratch
├── data/               # DatasetFormatter, TextDeduplicator, TextFilter
├── evaluation/         # BLEUScore, ROUGEScore, NGramAnalyzer, TextQualityMetrics
├── rag/                # RAGPipeline, code-aware chunking, ChromaDB, guardrails
├── serving/            # ModelConverter (GGUF), InferenceServer, ModelMerger, LocalEmbedder
└── dsl.py              # Pipeline DSL steps (LLMFineTuneStep, LLMConvertStep, etc.)
```

---

## Training

### FineTuner

Fine-tune any HuggingFace model with LoRA, QLoRA, or full fine-tuning.

```python
from scomp_link.llm import FineTuner

ft = FineTuner(
    model_name="meta-llama/Llama-3-8B",
    method="lora",          # "lora", "qlora", "full"
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
)

result = ft.train(
    data_path="alpaca.json",
    epochs=3,
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_seq_length=2048,
    fp16=True,
)

# Merge LoRA weights back into base model
ft.merge_and_save("./merged")
```

**FineTuneConfig** fields: `model_name`, `method`, `lora_r`, `lora_alpha`, `lora_dropout`, `epochs`, `batch_size`, `learning_rate`, `max_seq_length`, `gradient_accumulation_steps`, `fp16`, `bf16`, `output_dir`.

### TransformerBuilder

Build and train a GPT-style transformer from scratch.

```python
from scomp_link.llm import TransformerBuilder, TransformerConfig

config = TransformerConfig(
    vocab_size=32000,
    n_layers=12,
    n_heads=12,
    d_model=768,
    d_ff=3072,
    max_seq_length=1024,
    dropout=0.1,
    rope=True,
    kv_cache=True,
)

builder = TransformerBuilder(config)
builder.train("corpus.txt", epochs=10, batch_size=32, learning_rate=3e-4)
builder.save("my_gpt.scomp")
```

---

## Model Conversion & Serving

### ModelConverter

Convert HuggingFace models to GGUF format for local inference.

```python
from scomp_link.llm import ModelConverter

mc = ModelConverter("./merged")
mc.to_gguf(quantization="Q4_K_M")  # 15 levels: IQ2_XXS through f16
mc.estimate_resources("Q4_K_M")     # VRAM + disk estimates
```

**Quantization levels:** `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ2_M`, `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_0`, `Q4_K_S`, `Q4_K_M`, `Q5_K_S`, `Q5_K_M`, `Q8_0`, `f16`.

### InferenceServer

Serve a model as a Flask REST API.

```python
from scomp_link.llm import InferenceServer

server = InferenceServer("./merged", load_in_4bit=True)
server.run(host="0.0.0.0", port=8080)
```

Endpoints: `POST /generate`, `GET /health`.

### ModelMerger

Combine multiple models using different strategies.

```python
from scomp_link.llm import ModelMerger

merger = ModelMerger("base_model")
merged = merger.merge(
    models=["model_a", "model_b"],
    method="ties",       # "linear", "slerp", "ties", "dare"
    weights=[0.6, 0.4],
)
merged.save("./merged_model")
```

- **Linear**: weighted average of parameters
- **SLERP**: spherical interpolation on the unit hypersphere
- **TIES** (Yadav et al. 2023): trim, elect sign, merge — resolves parameter conflicts
- **DARE** (Yu et al. 2023): drop and rescale — sparsifies task vectors before merging

---

## RAG Pipeline

### RAGPipeline

Build retrieval-augmented generation pipelines with code-aware chunking.

```python
from scomp_link.llm import RAGPipeline

rag = RAGPipeline(
    embedding_model="nomic-embed-text-v1.5",
    chunk_strategy="code_aware",   # or "fixed", "semantic"
    vector_store="chromadb",
)

# Index documents
rag.add_documents(["docs/api.md", "src/main.py", "config.yaml"])

# Query
results = rag.query("How do I configure the database?", top_k=5)

# With guardrails (prompt injection detection)
results = rag.query(user_input, guardrails=True)
```

**Code-aware chunking** understands Python AST, YAML structure, Markdown headings, and SQL statements — keeps logical units together instead of splitting mid-function.

### LocalEmbedder

12 built-in model profiles with Matryoshka dimension support.

```python
from scomp_link.llm import LocalEmbedder

embedder = LocalEmbedder("nomic-embed-text-v1.5", dimensions=256)
vectors = embedder.embed(["Hello world", "Ciao mondo"])
```

Profiles: Qwen3-Embedding, E5-large-v2, BGE-base-en, Nomic-embed-text, Jina-embeddings-v5, and more.

---

## Evaluation

### TextQualityMetrics

All-in-one evaluation for generated text.

```python
from scomp_link.llm import TextQualityMetrics

report = TextQualityMetrics.evaluate(
    generated=["The cat sat on the mat."],
    references=["A cat was sitting on the mat."],
)
# Returns: bleu, rouge_1, rouge_2, rouge_l, self_bleu, distinct_1, distinct_2, zipf_coefficient
```

### Individual Metrics

```python
from scomp_link.llm.evaluation import BLEUScore, ROUGEScore, NGramAnalyzer

bleu = BLEUScore.compute(hypothesis="the cat sat", reference="a cat sat")
rouge = ROUGEScore.compute(hypothesis="the cat sat", reference="a cat sat")
diversity = NGramAnalyzer.distinct_n(texts, n=2)
self_bleu = NGramAnalyzer.self_bleu(texts)
```

---

## Data Processing

### DatasetFormatter

Convert between common LLM dataset formats.

```python
from scomp_link.llm import DatasetFormatter

formatter = DatasetFormatter()
formatter.convert("alpaca.json", "train.jsonl", source="alpaca", target="chatml")
```

Supported formats: Alpaca, ShareGPT, OpenAI, ChatML, Llama, Plain.

### TextDeduplicator

Deduplicate text corpora using exact matching (SHA-256) or fuzzy matching (MinHash LSH).

```python
from scomp_link.llm import TextDeduplicator

dedup = TextDeduplicator(method="minhash", threshold=0.8)
clean = dedup.deduplicate(texts)
```

### TextFilter

Filter low-quality text based on length, language, perplexity, and other heuristics.

```python
from scomp_link.llm import TextFilter

filtered = TextFilter.filter(texts, min_length=50, max_perplexity=500)
```

---

## DSL Pipeline Steps

```python
from scomp_link.llm.dsl import (
    LLMFineTuneStep, LLMConvertStep, LLMSaveStep,
    LLMEvalStep, LLMFormatStep, LLMDedupStep,
    LLMMergeStep, LLMRAGBuildStep,
)

# Fine-tune → convert → save
chain = (
    LLMFineTuneStep("meta-llama/Llama-3-8B", method="lora", dataset="data.json")
    >> LLMConvertStep(quantization="Q4_K_M")
    >> LLMSaveStep("model.scomp")
)
chain.run()
```

LLM steps cannot be mixed with ML or Report steps — `TypeError` at chain construction.

---

## CLI Commands

All commands live under `scomp-link llm`:

```bash
scomp-link llm finetune --model <hf_id> --method lora --data train.json --epochs 3
scomp-link llm convert --model ./merged --quantization Q4_K_M
scomp-link llm scratch --config model.yaml --data corpus.txt --epochs 10
scomp-link llm estimate --model ./merged --quantization Q4_K_M
scomp-link llm evaluate --generated output.txt --references ref.txt
scomp-link llm dedup --data corpus.txt --method ngram --threshold 0.8
scomp-link llm merge --base base_model --models m1,m2 --method ties
scomp-link llm serve --model ./merged --port 8080
scomp-link llm format --input data.json --output out.jsonl --source alpaca --target chatml
```

---

## MCP Tools

9 tools exposed via the MCP server:

| Tool | What it does |
|------|-------------|
| `llm_finetune` | Fine-tune a model (LoRA/QLoRA/full) |
| `llm_convert` | Convert to GGUF with quantization |
| `llm_scratch` | Build transformer from scratch |
| `llm_evaluate` | Evaluate text quality metrics |
| `llm_dedup` | Deduplicate text corpus |
| `llm_merge` | Merge multiple models |
| `llm_serve` | Start inference server |
| `llm_format` | Convert dataset formats |
| `llm_rag_build` | Build RAG pipeline from documents |

---

## Artifact Persistence

LLM artifacts use the same `.scomp` format with SHA-256 weight integrity verification:

```python
from scomp_link.llm import FineTuner

ft = FineTuner("meta-llama/Llama-3-8B", method="lora")
ft.train("data.json", epochs=3)
ft.save_artifact("model.scomp")  # includes weights hash

# Load and verify
loaded = FineTuner.load_artifact("model.scomp")  # checks SHA-256
```
