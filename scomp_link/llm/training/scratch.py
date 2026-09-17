# -*- coding: utf-8 -*-
"""
███████╗ ██████╗██████╗  █████╗ ████████╗ ██████╗██╗  ██╗
██╔════╝██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██║  ██║
███████╗██║     ██████╔╝███████║   ██║   ██║     ███████║
╚════██║██║     ██╔══██╗██╔══██║   ██║   ██║     ██╔══██║
███████║╚██████╗██║  ██║██║  ██║   ██║   ╚██████╗██║  ██║
╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝

Build GPT-style transformers from scratch — model architecture + training loop.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

from scomp_link.exceptions import DataValidationError, ModelTrainingError
from scomp_link.llm.core.configs import TrainResult, TransformerConfig

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from transformers import PreTrainedTokenizerFast


def _require_torch():
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError("TransformerBuilder requires torch. " "Install with: pip install scomp-link[llm]")


def _require_transformers():
    try:
        import transformers

        return transformers
    except ImportError:
        raise ImportError("TransformerBuilder requires transformers. " "Install with: pip install scomp-link[llm]")


# ---------------------------------------------------------------------------
# Model architecture — GPT-style decoder-only transformer
# ---------------------------------------------------------------------------


def _build_rope_cache(seq_len: int, head_dim: int, device: "torch.device"):
    torch = _require_torch()
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    dim_pairs = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (dim_pairs / head_dim))
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (seq, head_dim/2)
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor"):
    # x: (B, n_heads, T, head_dim)
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    torch = _require_torch()
    return torch.stack((out1, out2), dim=-1).flatten(-2)


def _make_gpt_block(config: TransformerConfig):
    torch = _require_torch()
    nn = torch.nn

    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, cfg: TransformerConfig):
            super().__init__()
            self.n_heads = cfg.n_heads
            self.head_dim = cfg.d_model // cfg.n_heads
            self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
            self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.attn_drop = nn.Dropout(cfg.dropout)
            self.resid_drop = nn.Dropout(cfg.dropout)
            self.rope = cfg.rope

        def forward(self, x, rope_cos=None, rope_sin=None, kv_cache=None):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, T, D)

            if self.rope and rope_cos is not None and rope_sin is not None:
                q = _apply_rope(q, rope_cos, rope_sin)
                k = _apply_rope(k, rope_cos, rope_sin)

            if kv_cache is not None:
                k_prev, v_prev = kv_cache
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)

            new_cache = (k, v)

            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale

            # causal mask
            kv_len = k.size(2)
            q_len = q.size(2)
            causal = torch.triu(
                torch.ones(q_len, kv_len, device=x.device, dtype=torch.bool),
                diagonal=kv_len - q_len + 1,
            )
            attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            out = (attn @ v).transpose(1, 2).reshape(B, q_len, C)
            return self.resid_drop(self.out_proj(out)), new_cache

    class FeedForward(nn.Module):
        def __init__(self, cfg: TransformerConfig):
            super().__init__()
            self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
            self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
            self.act = nn.GELU()
            self.drop = nn.Dropout(cfg.dropout)

        def forward(self, x):
            return self.drop(self.fc2(self.act(self.fc1(x))))

    class GPTBlock(nn.Module):
        def __init__(self, cfg: TransformerConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.attn = MultiHeadSelfAttention(cfg)
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.ffn = FeedForward(cfg)

        def forward(self, x, rope_cos=None, rope_sin=None, kv_cache=None):
            attn_out, new_cache = self.attn(self.ln1(x), rope_cos, rope_sin, kv_cache)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, new_cache

    return GPTBlock


def _make_gpt_model(config: TransformerConfig):
    torch = _require_torch()
    nn = torch.nn
    GPTBlock = _make_gpt_block(config)

    class GPTModel(nn.Module):
        def __init__(self, cfg: TransformerConfig):
            super().__init__()
            self.cfg = cfg
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_emb = None if cfg.rope else nn.Embedding(cfg.max_seq_len, cfg.d_model)
            self.drop = nn.Dropout(cfg.dropout)
            self.blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layers)])
            self.ln_f = nn.LayerNorm(cfg.d_model)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

            # weight tying
            self.lm_head.weight = self.tok_emb.weight

            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, input_ids, kv_caches=None):
            B, T = input_ids.shape
            x = self.tok_emb(input_ids)

            if self.pos_emb is not None:
                if kv_caches is not None and kv_caches[0] is not None:
                    offset = kv_caches[0][0].size(2)
                else:
                    offset = 0
                positions = torch.arange(offset, offset + T, device=input_ids.device)
                x = x + self.pos_emb(positions)

            x = self.drop(x)

            # RoPE cache
            rope_cos, rope_sin = None, None
            if self.cfg.rope:
                head_dim = self.cfg.d_model // self.cfg.n_heads
                total_len = T
                if kv_caches is not None and kv_caches[0] is not None:
                    total_len += kv_caches[0][0].size(2)
                rope_cos, rope_sin = _build_rope_cache(total_len, head_dim, input_ids.device)

            new_caches: list[tuple] = []
            for i, block in enumerate(self.blocks):
                cache = kv_caches[i] if kv_caches is not None else None
                x, new_cache = block(x, rope_cos, rope_sin, kv_cache=cache)
                new_caches.append(new_cache)

            x = self.ln_f(x)
            logits = self.lm_head(x)
            return logits, new_caches

    return GPTModel


# ---------------------------------------------------------------------------
# TransformerBuilder — public API
# ---------------------------------------------------------------------------


class TransformerBuilder:
    def __init__(self, config: TransformerConfig) -> None:
        torch = _require_torch()
        _require_transformers()

        self.config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        GPTModel = _make_gpt_model(config)
        self.model = GPTModel(config).to(self._device)
        self.tokenizer = None
        self._trained = False

    # -- Tokenizer training ------------------------------------------------

    def train_tokenizer(
        self,
        corpus: Union[str, Path, list[str]],
        vocab_size: int = 32_000,
        algorithm: Literal["bpe", "unigram", "wordpiece"] = "bpe",
    ) -> "PreTrainedTokenizerFast":
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        except ImportError:
            raise ImportError(
                "train_tokenizer requires the 'tokenizers' package. " "Install with: pip install scomp-link[llm]"
            )
        from transformers import PreTrainedTokenizerFast

        special_tokens = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]

        from tokenizers import decoders, processors

        if algorithm == "bpe":
            base = Tokenizer(models.BPE(unk_token="<|unk|>"))
            base.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            base.decoder = decoders.ByteLevel()
            base.post_processor = processors.ByteLevel(trim_offsets=False)
            trainer_obj = trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                show_progress=True,
            )
        elif algorithm == "unigram":
            base = Tokenizer(models.Unigram())
            base.pre_tokenizer = pre_tokenizers.Metaspace()
            base.decoder = decoders.Metaspace()
            trainer_obj = trainers.UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                show_progress=True,
                unk_token="<|unk|>",
            )
        elif algorithm == "wordpiece":
            base = Tokenizer(models.WordPiece(unk_token="<|unk|>"))
            base.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
            base.decoder = decoders.WordPiece()
            trainer_obj = trainers.WordPieceTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                show_progress=True,
            )
        else:
            raise DataValidationError(
                f"Unsupported tokenizer algorithm: {algorithm!r}. " f"Supported: bpe, unigram, wordpiece"
            )

        corpus_path = Path(corpus) if isinstance(corpus, str) and "\n" not in corpus and Path(corpus).exists() else None

        if corpus_path is not None:
            base.train([str(corpus_path)], trainer=trainer_obj)
        elif isinstance(corpus, list):
            if not corpus:
                raise DataValidationError("Corpus is empty (zero items)")
            base.train_from_iterator(corpus, trainer=trainer_obj)
        elif isinstance(corpus, (str, Path)):
            p = Path(corpus)
            if p.exists():
                base.train([str(p)], trainer=trainer_obj)
            else:
                # Treat as raw text string
                base.train_from_iterator([corpus], trainer=trainer_obj)
        else:
            raise DataValidationError(
                f"Unsupported corpus type: {type(corpus).__name__}. "
                f"Supported: str (text or file path), Path, list[str]"
            )

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base,
            eos_token="<|eos|>",
            bos_token="<|bos|>",
            pad_token="<|pad|>",
            unk_token="<|unk|>",
        )
        return self.tokenizer

    # -- Training ----------------------------------------------------------

    def train(
        self,
        dataset: Union[str, Path, list[str], Dataset],
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        callbacks: list | None = None,
    ) -> TrainResult:
        torch = _require_torch()

        if self.tokenizer is None:
            raise ModelTrainingError("Tokenizer not initialized. Call train_tokenizer() before train().")

        callbacks = callbacks or []
        texts = self._load_texts(dataset)

        if not texts:
            raise DataValidationError("Training dataset is empty (zero items)")

        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_seq_len,
            padding=False,
            return_tensors=None,
        )

        # Build a simple dataset of chunks
        all_ids: list[list[int]] = list(encodings["input_ids"])  # type: ignore[assignment]
        # Filter empty sequences
        all_ids = [ids for ids in all_ids if len(ids) > 1]
        if not all_ids:
            raise DataValidationError("All sequences are empty after tokenization")

        # Collate into batches
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset as TorchDataset

        class TextDataset(TorchDataset):
            def __init__(self, token_ids, max_len):
                self.samples = []
                for ids in token_ids:
                    ids_t = torch.tensor(ids, dtype=torch.long)
                    if len(ids_t) > max_len:
                        ids_t = ids_t[:max_len]
                    self.samples.append(ids_t)

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        def collate_fn(batch):
            max_len = max(len(s) for s in batch)
            assert self.tokenizer is not None
            _raw = self.tokenizer.pad_token_id
            pad_id: int = _raw if isinstance(_raw, int) else 0
            padded = torch.stack(
                [torch.cat([s, torch.full((max_len - len(s),), pad_id, dtype=torch.long)]) for s in batch]
            )
            return padded

        ds = TextDataset(all_ids, self.config.max_seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Training loop
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        total_batches = len(loader) * epochs
        warmup_steps = max(1, total_batches // 10)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_batches - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        loss_history: list[float] = []
        global_step = 0
        start_time = time.time()

        _raw_pad = self.tokenizer.pad_token_id
        pad_id: int = _raw_pad if isinstance(_raw_pad, int) else 0
        criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        for epoch in range(epochs):
            self.model.train()
            for batch in loader:
                batch = batch.to(self._device)
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]

                logits, _ = self.model(input_ids)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    last_finite = loss_history[-1] if loss_history else float("nan")
                    raise ModelTrainingError(
                        f"Loss became {'NaN' if torch.isnan(loss) else 'Inf'} "
                        f"at step {global_step}. Last finite loss: {last_finite:.6f}"
                    )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                loss_history.append(loss.item())
                global_step += 1

                for cb in callbacks:
                    if hasattr(cb, "on_step"):
                        cb.on_step(global_step, loss_history[-1])

            # End-of-epoch callback
            for cb in callbacks:
                if hasattr(cb, "on_epoch"):
                    cb.on_epoch(epoch, None)

        training_time = time.time() - start_time
        peak_mem = 0.0
        if torch.cuda.is_available():
            try:
                peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            except Exception:
                pass

        self._trained = True

        return TrainResult(
            loss_history=loss_history,
            eval_loss=None,
            eval_metrics={},
            model_path=Path("."),
            adapter_path=None,
            total_steps=global_step,
            training_time_seconds=training_time,
            peak_memory_gb=peak_mem,
            config=self.config,
        )

    # -- Generation --------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> str:
        if not self._trained:
            raise ModelTrainingError("Model has not been trained. Call train() before generate().")

        torch = _require_torch()
        self.model.eval()

        assert self.tokenizer is not None
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(  # type: ignore[attr-defined]
            self._device
        )
        generated = input_ids

        kv_caches: list[tuple | None] | None = None
        eos_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            # Initial forward pass — process the full prompt
            logits, kv_caches = self.model(input_ids, kv_caches=None)

            for _ in range(max_tokens):
                next_logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    threshold = topk_vals[:, -1].unsqueeze(-1)
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if eos_id is not None and next_token.item() == eos_id:
                    break

                generated = torch.cat([generated, next_token], dim=1)

                # KV-cache: only feed the new token
                logits, kv_caches = self.model(next_token, kv_caches=kv_caches)

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    # -- Internal helpers --------------------------------------------------

    @staticmethod
    def _load_texts(dataset) -> list[str]:
        if isinstance(dataset, list):
            if not dataset:
                raise DataValidationError("Training dataset is empty (zero items)")
            if not all(isinstance(s, str) for s in dataset):
                raise DataValidationError("When dataset is a list, all elements must be strings")
            return dataset

        if isinstance(dataset, (str, Path)):
            p = Path(dataset)
            if not p.exists():
                raise DataValidationError(f"Dataset file not found: {p}")
            suffix = p.suffix.lower()
            if suffix == ".txt":
                text = p.read_text(encoding="utf-8")
                lines = [ln for ln in text.splitlines() if ln.strip()]
                if not lines:
                    raise DataValidationError(f"Training dataset is empty: {p}")
                return lines
            elif suffix in (".csv", ".json", ".jsonl", ".parquet"):
                try:
                    import pandas as pd
                except ImportError:
                    raise ImportError(
                        "Reading CSV/JSON/Parquet datasets requires pandas. " "Install with: pip install pandas"
                    )
                if suffix == ".csv":
                    df = pd.read_csv(p)
                elif suffix in (".json", ".jsonl"):
                    df = pd.read_json(p, lines=(suffix == ".jsonl"))
                else:
                    df = pd.read_parquet(p)

                if len(df) == 0:
                    raise DataValidationError(f"Training dataset is empty: {p}")
                # Pick first text-like column
                text_col = None
                for col in df.columns:
                    if df[col].dtype == object:
                        text_col = col
                        break
                if text_col is None:
                    raise DataValidationError(f"No text column found in {p}. " f"Columns: {list(df.columns)}")
                return df[text_col].dropna().tolist()
            else:
                raise DataValidationError(
                    f"Unsupported dataset format: '{suffix}'. "
                    f"Supported: .txt, .csv, .json, .jsonl, .parquet, "
                    f"or a list of strings"
                )

        # HuggingFace Dataset
        try:
            from datasets import Dataset as HFDataset

            if isinstance(dataset, HFDataset):
                if len(dataset) == 0:
                    raise DataValidationError("Training dataset is empty (zero rows)")
                text_col = None
                for col in dataset.column_names:
                    if dataset.features[col].dtype == "string":
                        text_col = col
                        break
                if text_col is None and dataset.column_names:
                    text_col = dataset.column_names[0]
                if text_col is None:
                    raise DataValidationError("No text column found in HuggingFace Dataset")
                return dataset[text_col]
        except ImportError:
            pass

        raise DataValidationError(
            f"Unsupported dataset type: {type(dataset).__name__}. "
            f"Supported: str/Path (file path), list[str], or HuggingFace Dataset"
        )


if __name__ == "__main__":
    # Show what a tiny GPT config looks like
    cfg = TransformerConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=64,
        dropout=0.1,
        rope=True,
    )
    print(f"Tiny GPT: {cfg.vocab_size} vocab, {cfg.d_model}d, {cfg.n_heads}h, {cfg.n_layers}L")
    print(
        f"  params ≈ {cfg.vocab_size * cfg.d_model + cfg.n_layers * (4 * cfg.d_model**2 + 2 * cfg.d_model * cfg.d_ff):,}"
    )
    print(f"  RoPE: {cfg.rope}, flash_attn: {cfg.flash_attention}")
    print("\nUsage:")
    print("  builder = TransformerBuilder(cfg)")
    print("  builder.train_tokenizer('corpus.txt', vocab_size=256)")
    print("  result = builder.train('corpus.txt', epochs=5)")
    print("  text = builder.generate('Once upon a time', max_tokens=50)")
