# -*- coding: utf-8 -*-
"""
 ██████╗ █████╗ ██╗     ██╗     ██████╗  █████╗  ██████╗██╗  ██╗███████╗
██╔════╝██╔══██╗██║     ██║     ██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝
██║     ███████║██║     ██║     ██████╔╝███████║██║     █████╔╝ ███████╗
██║     ██╔══██║██║     ██║     ██╔══██╗██╔══██║██║     ██╔═██╗ ╚════██║
╚██████╗██║  ██║███████╗███████╗██████╔╝██║  ██║╚██████╗██║  ██╗███████║
 ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝

Callback protocol and built-in callbacks for LLM training loops.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Callback(Protocol):
    """Protocol for training callbacks."""

    def on_step(self, step: int, loss: float) -> None: ...
    def on_epoch(self, epoch: int, eval_loss: float | None) -> None: ...


class PrintCallback:
    """Logs training progress to stdout."""

    def on_step(self, step: int, loss: float) -> None:
        print(f"[step {step}] loss={loss:.4f}")

    def on_epoch(self, epoch: int, eval_loss: float | None) -> None:
        if eval_loss is not None:
            print(f"[epoch {epoch}] eval_loss={eval_loss:.4f}")
        else:
            print(f"[epoch {epoch}] completed")


class WandbCallback:
    """Logs training metrics to Weights & Biases.

    Requires ``wandb`` to be installed (``pip install wandb``).
    A new run is initialised lazily on the first ``on_step`` call.
    """

    def __init__(self, project: str = "scomp-link-llm", **init_kwargs):
        try:
            import wandb  # noqa: F401
        except ImportError:
            raise ImportError("WandbCallback requires wandb. Install with: pip install wandb")
        self._wandb = wandb
        self._project = project
        self._init_kwargs = init_kwargs
        self._run = None

    def _ensure_run(self):
        if self._run is None:
            self._run = self._wandb.init(project=self._project, **self._init_kwargs)

    def on_step(self, step: int, loss: float) -> None:
        self._ensure_run()
        self._wandb.log({"step": step, "train/loss": loss}, step=step)

    def on_epoch(self, epoch: int, eval_loss: float | None) -> None:
        self._ensure_run()
        metrics: dict = {"epoch": epoch}
        if eval_loss is not None:
            metrics["eval/loss"] = eval_loss
        self._wandb.log(metrics)


if __name__ == "__main__":
    # Simulate a mini training loop with PrintCallback
    cb = PrintCallback()
    for step in range(1, 6):
        loss = 3.2 / step  # fake decreasing loss
        cb.on_step(step, loss)
    cb.on_epoch(0, eval_loss=1.25)
    cb.on_epoch(1, eval_loss=0.98)

    # Check the protocol works
    assert isinstance(cb, Callback), "PrintCallback should satisfy Callback protocol"
    print("Protocol check passed")
