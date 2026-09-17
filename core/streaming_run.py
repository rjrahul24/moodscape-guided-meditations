"""Run an auto-generation on a background thread, streaming progress.

Lives in core/ rather than app.py so it can be unit-tested: app.py loads
torch and Gradio and registers an atexit hard-exit hook, so importing it from
a test is not viable.

The thread-and-queue pattern mirrors the manual handler in app.py.
"""

import queue
import threading
from dataclasses import dataclass

from core.auto_generate import AutoResult, ScriptGenerationError
from core.auto_generate import run as default_run


@dataclass(frozen=True)
class ProgressUpdate:
    """One progress tick from the pipeline."""

    fraction: float
    message: str


class StreamingRun:
    """Iterate for progress; read .result or .error when iteration ends."""

    def __init__(self, prompt: str, *, config=None, runner=None, **kwargs):
        self._prompt = prompt
        self._config = config
        self._runner = runner if runner is not None else default_run
        self._kwargs = kwargs
        self.result: AutoResult | None = None
        self.error: str | None = None
        self.invalid_input: bool = False

    def __iter__(self):
        if not self._prompt or not self._prompt.strip():
            self.error = "Enter a prompt first."
            self.invalid_input = True
            return

        updates: queue.Queue = queue.Queue()

        def progress_cb(fraction, message):
            updates.put(ProgressUpdate(fraction=fraction, message=message))

        def worker():
            try:
                self.result = self._runner(
                    self._prompt,
                    config=self._config,
                    progress_cb=progress_cb,
                    **self._kwargs,
                )
            except (ScriptGenerationError, RuntimeError) as exc:
                # ScriptGenerationError and RuntimeError are this system's own
                # contract types: their messages are written to be read as-is
                # by a human (e.g. "ollama returned HTTP 404 for model X"), so
                # prefixing the class name only adds noise. ScriptGenerationError
                # subclasses RuntimeError, but both branches produce the same
                # bare message, so the except order here doesn't change behavior.
                # Any other exception type is unexpected, so its class name is
                # kept below because it may be the only clue to what broke.
                self.error = str(exc)
            except Exception as exc:  # adapter or pipeline failure
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                updates.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = updates.get()
            if item is None:
                break
            yield item

        thread.join()
