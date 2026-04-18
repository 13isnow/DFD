from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


def _get_console_logger():
    try:
        from loguru import logger as loguru_logger

        return loguru_logger
    except Exception:
        import logging

        logger = logging.getLogger("cifake_reproduce")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


class TBLogger:
    def __init__(self, log_dir: str | Path, enabled: bool = True, console: bool = True):
        self.log_dir = str(Path(log_dir))
        self._console = _get_console_logger() if console else None
        self._step = 0
        self._start_ts = time.time()

        self._writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(self.log_dir)
            except Exception as e:
                self._writer = None
                self._console_log("warning", f"TensorBoard disabled: {e}")

        self._console_log("info", f"TensorBoard log_dir: {self.log_dir}")
        self.add_text("meta/log_dir", self.log_dir, step=0)

    def _console_log(self, level: str, message: str):
        if not self._console:
            return
        try:
            getattr(self._console, level)(message)
        except Exception:
            try:
                self._console.info(message)
            except Exception:
                pass

    def _resolve_step(self, step: Optional[int]) -> int:
        if step is None:
            self._step += 1
            return self._step
        step_i = int(step)
        if step_i > self._step:
            self._step = step_i
        return step_i

    def print(self, *args: Any, sep: str = " ", end: str = "\n", tag: str = "stdout", step: Optional[int] = None):
        text = sep.join("" if a is None else str(a) for a in args) + end
        self.add_text(tag, text, step=step)

    def add_text(self, tag: str, text: str, step: Optional[int] = None):
        step_i = self._resolve_step(step)
        if self._writer:
            self._writer.add_text(tag, text, global_step=step_i)
            self._writer.flush()
        self._console_log("info", f"{tag}@{step_i}: {text.rstrip()}")

    def add_scalar(self, tag: str, value: float, step: Optional[int] = None):
        step_i = self._resolve_step(step)
        if self._writer:
            self._writer.add_scalar(tag, float(value), global_step=step_i)
            self._writer.flush()
        self._console_log("info", f"{tag}@{step_i}: {value}")

    def add_json(self, tag: str, obj: Any, step: Optional[int] = None):
        self.add_text(tag, json.dumps(obj, ensure_ascii=False, indent=2), step=step)

    def close(self):
        elapsed = time.time() - self._start_ts
        self.add_scalar("meta/elapsed_seconds", elapsed, step=self._step + 1)
        if self._writer:
            self._writer.close()
