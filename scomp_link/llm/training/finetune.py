# -*- coding: utf-8 -*-
"""
███████╗██╗███╗   ██╗███████╗████████╗██╗   ██╗███╗   ██╗███████╗
██╔════╝██║████╗  ██║██╔════╝╚══██╔══╝██║   ██║████╗  ██║██╔════╝
█████╗  ██║██╔██╗ ██║█████╗     ██║   ██║   ██║██╔██╗ ██║█████╗
██╔══╝  ██║██║╚██╗██║██╔══╝     ██║   ██║   ██║██║╚██╗██║██╔══╝
██║     ██║██║ ╚████║███████╗   ██║   ╚██████╔╝██║ ╚████║███████╗
╚═╝     ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝

Fine-tune pretrained HuggingFace models with LoRA, QLoRA, or full training.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

if TYPE_CHECKING:
    import pandas as pd
    from datasets import Dataset

from scomp_link.exceptions import DataValidationError, ModelTrainingError
from scomp_link.llm.core.configs import FineTuneConfig, TrainResult

logger = logging.getLogger(__name__)

_INSTALL_MSG = "FineTuner requires torch and transformers. " "Install with: pip install scomp-link[llm]"


def _require_torch():
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError(_INSTALL_MSG)


def _require_transformers():
    try:
        import transformers

        return transformers
    except ImportError:
        raise ImportError(_INSTALL_MSG)


def _require_peft():
    try:
        import peft

        return peft
    except ImportError:
        raise ImportError("FineTuner with LoRA/QLoRA requires peft. " "Install with: pip install scomp-link[llm]")


def _require_bitsandbytes():
    try:
        import bitsandbytes

        return bitsandbytes
    except ImportError:
        raise ImportError("QLoRA requires bitsandbytes. " "Install with: pip install scomp-link[llm]")


def _resolve_dtype(mixed_precision: str):
    torch = _require_torch()
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def _cleanup_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _release_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class FineTuner:
    """Fine-tune a pretrained HuggingFace model using LoRA, QLoRA, or full."""

    def __init__(
        self,
        model_name_or_path: str,
        method: Literal["lora", "qlora", "full"] = "lora",
        config: FineTuneConfig | None = None,
    ) -> None:
        _require_torch()
        _require_transformers()

        if method not in ("lora", "qlora", "full"):
            raise DataValidationError(f"method must be one of 'lora', 'qlora', 'full', got {method!r}")

        self.config = config or FineTuneConfig(method=method)
        if config is not None:
            self.config.method = method

        self._model_name = model_name_or_path
        self._method = method
        self._model = None
        self._tokenizer = None
        self._trained = False

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        torch = _require_torch()
        _require_transformers()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        output_dir = Path(self.config.output_dir)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        except Exception as exc:
            _cleanup_dir(output_dir)
            raise ModelTrainingError(f"Failed to load tokenizer from '{self._model_name}': {exc}") from exc

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        try:
            if self._method == "qlora":
                self._model = self._load_quantized_model()
            else:
                device_map = "auto" if torch.cuda.is_available() else None
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    trust_remote_code=True,
                    device_map=device_map,
                    torch_dtype=_resolve_dtype(self.config.mixed_precision),
                )
        except Exception as exc:
            _cleanup_dir(output_dir)
            _release_gpu_memory()
            if "CUDA" in str(exc) or "cuda" in str(exc):
                raise ModelTrainingError(f"Failed to load model '{self._model_name}': {exc}") from exc
            raise ModelTrainingError(f"Failed to load model '{self._model_name}': {exc}") from exc

        if self._method in ("lora", "qlora"):
            self._apply_peft_adapters()
        elif self._method == "full":
            self._model.train()

        if self.config.gradient_checkpointing:
            self._model.gradient_checkpointing_enable()

    def _load_quantized_model(self):
        torch = _require_torch()
        _require_bitsandbytes()
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise ModelTrainingError("QLoRA requires a CUDA GPU. " f"Detected device: {_detect_device(torch)}")

        capability = torch.cuda.get_device_capability()
        if capability[0] < 7:
            raise ModelTrainingError(
                f"QLoRA requires a CUDA GPU with compute capability >= 7.0. "
                f"Detected: {capability[0]}.{capability[1]}"
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(self.config.bits == 4),
            load_in_8bit=(self.config.bits == 8),
            bnb_4bit_compute_dtype=_resolve_dtype(self.config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
        )

        return AutoModelForCausalLM.from_pretrained(
            self._model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
        )

    def _apply_peft_adapters(self) -> None:
        _require_peft()
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        assert self._model is not None

        if self._method == "qlora":
            self._model = prepare_model_for_kbit_training(self._model)

        target_modules = self.config.lora_target_modules
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self._model = get_peft_model(self._model, lora_config)  # type: ignore[reportCallIssue]

    def train(
        self,
        dataset: str | Path | pd.DataFrame | Dataset,
        eval_dataset: str | Path | pd.DataFrame | Dataset | None = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        callbacks: list[Any] | None = None,
    ) -> TrainResult:
        _require_torch()
        assert self._model is not None
        assert self._tokenizer is not None
        from scomp_link.llm.data.loader import load_dataset

        if epochs < 1:
            raise DataValidationError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise DataValidationError(f"batch_size must be >= 1, got {batch_size}")
        if learning_rate <= 0:
            raise DataValidationError(f"learning_rate must be > 0, got {learning_rate}")

        train_ds = load_dataset(dataset, text_field=self.config.dataset_text_field)
        eval_ds = (
            load_dataset(eval_dataset, text_field=self.config.dataset_text_field) if eval_dataset is not None else None
        )

        callbacks = callbacks or []
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = self._training_loop(train_ds, eval_ds, epochs, batch_size, learning_rate, callbacks)
        except ModelTrainingError:
            raise
        except Exception as exc:
            _release_gpu_memory()
            _cleanup_dir(output_dir)
            raise ModelTrainingError(f"Training failed: {exc}") from exc

        self._trained = True
        return result

    def _training_loop(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        callbacks: list[Any],
    ) -> TrainResult:
        torch = _require_torch()
        assert self._model is not None
        assert self._tokenizer is not None
        from torch.utils.data import DataLoader

        output_dir = Path(self.config.output_dir)
        config = self.config
        model = self._model
        tokenizer = self._tokenizer

        tokenized_train = self._tokenize_dataset(train_dataset)
        tokenized_eval = self._tokenize_dataset(eval_dataset) if eval_dataset else None

        tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        if tokenized_eval:
            tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        train_loader = DataLoader(
            tokenized_train,  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.weight_decay,
        )

        total_opt_steps = math.ceil(len(train_loader) / config.gradient_accumulation_steps) * epochs
        warmup_steps = int(config.warmup_ratio * total_opt_steps)

        scheduler = self._get_cosine_scheduler(optimizer, warmup_steps, total_opt_steps)
        use_amp = config.mixed_precision != "no" and torch.cuda.is_available()
        amp_dtype = _resolve_dtype(config.mixed_precision)
        scaler = torch.amp.GradScaler("cuda", enabled=(config.mixed_precision == "fp16" and torch.cuda.is_available()))  # type: ignore[reportPrivateImportUsage]

        device = next(model.parameters()).device
        loss_history: list[float] = []
        global_step = 0
        last_finite_loss = 0.0
        eval_loss = None
        eval_metrics: dict[str, float] = {}
        start_time = time.time()

        try:
            for epoch in range(epochs):
                model.train()
                accumulated_loss = 0.0

                for step, batch in enumerate(train_loader):
                    batch = {k: v.to(device) for k, v in batch.items()}

                    try:
                        with torch.amp.autocast(  # type: ignore[reportPrivateImportUsage]
                            device_type=device.type,
                            dtype=amp_dtype,
                            enabled=use_amp,
                        ):
                            outputs = model(**batch)
                            loss = outputs.loss / config.gradient_accumulation_steps

                        if torch.isnan(loss) or torch.isinf(loss):
                            raise ModelTrainingError(
                                f"Loss became NaN/Inf at epoch {epoch}, "
                                f"step {global_step}, "
                                f"last finite loss: {last_finite_loss:.6f}"
                            )

                        scaler.scale(loss).backward()

                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            _release_gpu_memory()
                            _cleanup_dir(output_dir)
                            mem_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
                            raise ModelTrainingError(
                                f"CUDA out of memory at step {global_step} "
                                f"(batch_size={batch_size}, "
                                f"GPU memory used: {mem_mb:.0f} MB). "
                                "Suggestions: reduce batch_size, enable "
                                "gradient_checkpointing=True, or use "
                                "method='qlora' with bits=4"
                            ) from exc
                        raise

                    accumulated_loss += loss.item()

                    if (step + 1) % config.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()

                        loss_history.append(accumulated_loss)
                        last_finite_loss = accumulated_loss
                        accumulated_loss = 0.0
                        global_step += 1

                        for cb in callbacks:
                            cb.on_step(global_step, loss_history[-1])

                        if config.save_steps and global_step % config.save_steps == 0:
                            self._save_checkpoint(output_dir, global_step)

                remaining = (step + 1) % config.gradient_accumulation_steps
                if remaining != 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                    loss_history.append(accumulated_loss)
                    last_finite_loss = accumulated_loss
                    accumulated_loss = 0.0
                    global_step += 1

                if tokenized_eval:
                    eval_loss, eval_metrics = self._evaluate(tokenized_eval, batch_size)

                for cb in callbacks:
                    cb.on_epoch(epoch, eval_loss)

        except ModelTrainingError:
            _release_gpu_memory()
            _cleanup_dir(output_dir)
            raise
        except Exception as exc:
            _release_gpu_memory()
            _cleanup_dir(output_dir)
            raise ModelTrainingError(f"Training failed: {exc}") from exc

        training_time = time.time() - start_time
        peak_memory = 0.0
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                peak_memory = _torch.cuda.max_memory_allocated() / (1024**3)
        except Exception:
            pass

        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        adapter_path = None
        if self._method in ("lora", "qlora"):
            adapter_path = output_dir / "adapter"
            model.save_pretrained(str(adapter_path))

        return TrainResult(
            loss_history=loss_history,
            eval_loss=eval_loss,
            eval_metrics=eval_metrics,
            model_path=output_dir,
            adapter_path=adapter_path,
            total_steps=global_step,
            training_time_seconds=training_time,
            peak_memory_gb=peak_memory,
            config=self.config,
        )

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        assert self._tokenizer is not None
        tokenizer = self._tokenizer
        max_len = self.config.max_seq_length

        def tokenize_fn(examples):
            tokenized = tokenizer(
                examples[self.config.dataset_text_field],
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    def _evaluate(self, eval_dataset: Dataset, batch_size: int) -> tuple[float, dict[str, float]]:
        torch = _require_torch()
        assert self._model is not None
        from torch.utils.data import DataLoader

        model = self._model
        device = next(model.parameters()).device
        config = self.config

        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)  # type: ignore[reportArgumentType]
        model.eval()
        total_loss = 0.0
        total_steps = 0

        use_amp = config.mixed_precision != "no" and torch.cuda.is_available()
        amp_dtype = _resolve_dtype(config.mixed_precision)

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast(  # type: ignore[reportPrivateImportUsage]
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    outputs = model(**batch)
                total_loss += outputs.loss.item()
                total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        perplexity = math.exp(min(avg_loss, 100))

        return avg_loss, {"perplexity": perplexity}

    def _get_cosine_scheduler(self, optimizer, warmup_steps: int, total_steps: int):
        _require_torch()
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    def _save_checkpoint(self, output_dir: Path, step: int) -> None:
        assert self._model is not None
        assert self._tokenizer is not None
        ckpt_dir = output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(ckpt_dir))
        self._tokenizer.save_pretrained(str(ckpt_dir))

    def merge_and_save(self, output_dir: str | Path) -> Path:
        _require_torch()
        _require_peft()
        assert self._model is not None
        assert self._tokenizer is not None

        if self._method not in ("lora", "qlora"):
            raise ModelTrainingError(
                "merge_and_save is only supported for LoRA/QLoRA methods. " f"Current method: '{self._method}'"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        merged_model = self._model.merge_and_unload()  # type: ignore[reportCallIssue]
        merged_model.save_pretrained(str(output_dir))  # type: ignore[reportCallIssue]
        self._tokenizer.save_pretrained(str(output_dir))

        return output_dir

    def push_to_hub(self, repo_id: str, private: bool = True) -> str:
        _require_torch()
        _require_transformers()
        assert self._model is not None
        assert self._tokenizer is not None

        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                from huggingface_hub import HfFolder

                token = HfFolder.get_token()
            except Exception:
                token = None

        if not token:
            raise ModelTrainingError(
                "No HuggingFace credentials found. Set HF_TOKEN environment " "variable or run 'huggingface-cli login'."
            )

        self._model.push_to_hub(repo_id, private=private, token=token)  # type: ignore[reportArgumentType]
        self._tokenizer.push_to_hub(repo_id, private=private, token=token)  # type: ignore[reportArgumentType]

        return f"https://huggingface.co/{repo_id}"


def _detect_device(torch) -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    # Show default config and explain the three methods
    cfg = FineTuneConfig()
    print(f"Default config: method={cfg.method}, lora_r={cfg.lora_r}, alpha={cfg.lora_alpha}")
    print(f"  mixed_precision={cfg.mixed_precision}, grad_accum={cfg.gradient_accumulation_steps}")
    print(f"  grad_checkpointing={cfg.gradient_checkpointing}, max_seq={cfg.max_seq_length}")
    print("\nMethods:")
    print(f"  lora  — freeze base, train small A·B adapter matrices (rank {cfg.lora_r})")
    print(f"  qlora — same as lora but base model quantized to {cfg.bits}-bit (saves ~60% VRAM)")
    print("  full  — update all parameters (best quality, needs big GPU)")
