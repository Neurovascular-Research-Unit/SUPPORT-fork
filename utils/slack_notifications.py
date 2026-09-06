"""Optional, fail-open Slack notifications for SUPPORT training.

The webhook is deliberately kept out of trainer parameters and command-line
arguments so it cannot be written to run manifests, logs, notebook output, or
Slurm accounting.  Prefer a mode-0600 file named by
``SUPPORT_SLACK_WEBHOOK_FILE``; ``SUPPORT_SLACK_WEBHOOK_URL`` is retained as a
convenient environment-only fallback.
"""

from __future__ import annotations

import math
import os
import re
import socket
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlparse


WEBHOOK_FILE_ENV = "SUPPORT_SLACK_WEBHOOK_FILE"
WEBHOOK_URL_ENV = "SUPPORT_SLACK_WEBHOOK_URL"
_WEBHOOK_PATTERN = re.compile(
    r"https://hooks\.slack(?:-gov)?\.com/services/[^\s<>'\"]+",
    flags=re.IGNORECASE,
)
_SPARKS = "▁▂▃▄▅▆▇█"
_SLACK_WEBHOOK_HOSTS = {"hooks.slack.com", "hooks.slack-gov.com"}


class SlackConfigurationError(ValueError):
    """Raised when notifications were requested without a safe webhook source."""


def redact_slack_webhooks(value: Any) -> str:
    """Return text with Slack webhook bearer secrets removed."""
    return _WEBHOOK_PATTERN.sub("[REDACTED_SLACK_WEBHOOK]", str(value))


def _safe_field(value: Any) -> str:
    """Escape one user-derived field for Slack mrkdwn without mentions."""
    text = " ".join(redact_slack_webhooks(value).split())
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sparkline(values: Sequence[Optional[float]], low: float, high: float) -> str:
    if not values:
        return ""
    if high <= low:
        return "▄" * len(values)
    scale = (len(_SPARKS) - 1) / (high - low)
    output = []
    for value in values:
        if value is None:
            output.append("·")
        else:
            index = max(0, min(len(_SPARKS) - 1, round((value - low) * scale)))
            output.append(_SPARKS[index])
    return "".join(output)


def render_loss_sparklines(
    history: Iterable[Mapping[str, Any]], width: int = 40
) -> str:
    """Render comparable training/validation loss curves for a Slack message."""
    records = list(history)
    if width < 1:
        raise ValueError("sparkline width must be at least 1")
    records = records[-width:]
    if not records:
        return "Loss plot: no completed epochs yet."

    training = [_finite_float(row.get("training_loss")) for row in records]
    validation = [_finite_float(row.get("validation_loss")) for row in records]
    finite_values = [
        value for value in (*training, *validation) if value is not None
    ]
    if not finite_values:
        return "Loss plot: no finite loss values yet."

    low = min(finite_values)
    high = max(finite_values)
    start_epoch = int(records[0].get("epoch", 0)) + 1
    end_epoch = int(records[-1].get("epoch", len(records) - 1)) + 1
    return (
        f"Loss plot (epochs {start_epoch}–{end_epoch}, shared scale)\n"
        f"train      `{_sparkline(training, low, high)}`\n"
        f"validation `{_sparkline(validation, low, high)}`"
    )


def save_loss_plot(
    history: Iterable[Mapping[str, Any]], destination: Path
) -> Optional[Path]:
    """Atomically save a full training/validation loss plot, if data exist."""
    records = list(history)
    if not records:
        return None
    epochs = [int(row.get("epoch", index)) + 1 for index, row in enumerate(records)]
    training = [_finite_float(row.get("training_loss")) for row in records]
    validation = [_finite_float(row.get("validation_loss")) for row in records]
    if not any(value is not None for value in (*training, *validation)):
        return None

    # Import lazily so Slack-disabled training does not gain a new import-time
    # dependency or select a matplotlib backend unexpectedly.
    import matplotlib.pyplot as plt

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".png", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    figure = None
    try:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        if any(value is not None for value in training):
            axis.plot(epochs, training, marker="o", label="Training")
        if any(value is not None for value in validation):
            axis.plot(epochs, validation, marker="o", label="Validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title("SUPPORT training and validation loss")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(temporary_path, format="png", dpi=150)
        os.replace(temporary_path, destination)
    finally:
        if figure is not None:
            plt.close(figure)
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


class SlackNotifier:
    """Post concise SUPPORT status messages through an Incoming Webhook.

    Delivery and plotting failures are intentionally nonfatal.  The warning
    path never includes exception text because ``requests`` errors commonly
    contain the complete secret webhook URL.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        *,
        enabled: bool = False,
        timeout: float = 10,
        post: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.timeout = float(timeout)
        self._post = post
        self._webhook_url = None
        self._run_started_at = None
        self._epoch_started_at = {}
        self._current_epoch = None
        self._history = []
        self._failure_attempted = False
        if self.enabled:
            self._webhook_url = self._validate_webhook_url(webhook_url)

    def __repr__(self) -> str:
        state = "enabled" if self.enabled else "disabled"
        return f"SlackNotifier({state}, webhook=[REDACTED])"

    @staticmethod
    def _validate_webhook_url(webhook_url: Optional[str]) -> str:
        value = (webhook_url or "").strip()
        parsed = urlparse(value)
        try:
            port = parsed.port
        except ValueError as exc:
            raise SlackConfigurationError("Slack webhook URL has an invalid port") from exc
        path_parts = [part for part in parsed.path.split("/") if part]
        if (
            parsed.scheme != "https"
            or parsed.hostname not in _SLACK_WEBHOOK_HOSTS
            or port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or len(path_parts) != 4
            or path_parts[0] != "services"
            or any(not part for part in path_parts[1:])
        ):
            raise SlackConfigurationError(
                "Slack webhook must be a plain hooks.slack.com (or Slack Gov) "
                "HTTPS Incoming Webhook URL"
            )
        return value

    @staticmethod
    def _read_webhook_file(path: Path) -> str:
        path = Path(path).expanduser()
        if path.is_symlink():
            raise SlackConfigurationError("Slack webhook file must not be a symlink")
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_file():
                raise SlackConfigurationError("Slack webhook path is not a file")
            mode = stat.S_IMODE(resolved.stat().st_mode)
            if mode & 0o077:
                raise SlackConfigurationError(
                    "Slack webhook file must be private; run chmod 600 on it"
                )
            lines = resolved.read_text(encoding="utf-8").splitlines()
        except SlackConfigurationError:
            raise
        except (OSError, UnicodeError) as exc:
            raise SlackConfigurationError(
                f"Could not read Slack webhook file ({type(exc).__name__})"
            ) from exc
        nonempty = [line.strip() for line in lines if line.strip()]
        if len(nonempty) != 1:
            raise SlackConfigurationError(
                "Slack webhook file must contain exactly one nonempty line"
            )
        return nonempty[0]

    @classmethod
    def from_environment(
        cls,
        *,
        enabled: bool,
        webhook_file: Optional[Path] = None,
        timeout: float = 10,
        post: Optional[Callable[..., Any]] = None,
    ) -> "SlackNotifier":
        if not enabled:
            return cls(enabled=False, timeout=timeout, post=post)
        selected_file = webhook_file or os.environ.get(WEBHOOK_FILE_ENV)
        if selected_file:
            webhook_url = cls._read_webhook_file(Path(selected_file))
        else:
            webhook_url = os.environ.get(WEBHOOK_URL_ENV)
        if not webhook_url:
            raise SlackConfigurationError(
                "Slack notifications require SUPPORT_SLACK_WEBHOOK_FILE "
                "(preferred) or SUPPORT_SLACK_WEBHOOK_URL"
            )
        return cls(
            webhook_url=webhook_url,
            enabled=True,
            timeout=timeout,
            post=post,
        )

    def _warning(self, action: str, exc: BaseException) -> None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        suffix = f", HTTP {status}" if status is not None else ""
        print(
            f"WARNING: Slack {action} failed ({type(exc).__name__}{suffix}); "
            "training will continue.",
            file=sys.stderr,
        )

    @staticmethod
    def _format_seconds(seconds: Optional[float]) -> str:
        if seconds is None:
            return "unknown"
        seconds = max(0, round(float(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    @staticmethod
    def _metric_lines(prefix: str, metrics: Optional[Mapping[str, Any]]) -> list[str]:
        if not metrics:
            return [f"*{prefix}:* not run"]
        values = []
        for label, key in (("total", "loss_total"), ("L1", "loss_l1"), ("L2", "loss_l2")):
            value = _finite_float(metrics.get(key))
            values.append(f"{label}={value:.6g}" if value is not None else f"{label}=n/a")
        return [f"*{prefix}:* " + ", ".join(values)]

    @staticmethod
    def _job_context() -> str:
        job_id = os.environ.get("SLURM_JOB_ID", "not under Slurm")
        job_name = os.environ.get("SLURM_JOB_NAME")
        job = f"{job_name} ({job_id})" if job_name else job_id
        return f"*Job:* {_safe_field(job)} on {_safe_field(socket.gethostname())}"

    def send(self, message: str) -> bool:
        """Send one text payload, returning success without ever raising."""
        if not self.enabled:
            return False
        safe_message = redact_slack_webhooks(message)
        if len(safe_message) > 3900:
            safe_message = safe_message[:3880] + "\n…(truncated)"
        try:
            post = self._post
            if post is None:
                import requests

                post = requests.post
            response = post(
                self._webhook_url,
                json={"text": safe_message},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            self._warning("notification", exc)
            return False

    def notify_run_started(
        self,
        trainer: Any,
        *,
        training_count: Optional[int] = None,
        validation_count: Optional[int] = None,
    ) -> bool:
        self._run_started_at = time.monotonic()
        self._failure_attempted = False
        self._current_epoch = None
        self._epoch_started_at.clear()
        training_text = (
            str(training_count) if training_count is not None else "unknown"
        )
        validation_text = (
            str(validation_count) if validation_count is not None else "unknown"
        )
        return self.send(
            ":arrow_forward: *SUPPORT training started*\n"
            f"*Experiment:* {_safe_field(trainer.exp_name)}\n"
            f"{self._job_context()}\n"
            f"*Mode/device:* {trainer.dimensionality} / {trainer.device}\n"
            f"*Input:* {_safe_field(trainer.in_path)}\n"
            f"*Output:* {_safe_field(trainer.out_dir)}\n"
            f"*Epoch target:* {trainer.n_epochs}; *training/validation TIFFs:* "
            f"{training_text}/{validation_text}\n"
            f"*Batch/patch:* {trainer.batch_size} / {list(trainer.patch_size)}\n"
            f"*Workers/prefetch:* {trainer.n_cpu} / {trainer.prefetch_factor}; "
            f"*resume:* {trainer.resume}"
        )

    def notify_epoch_started(self, trainer: Any, epoch: int) -> bool:
        self._current_epoch = int(epoch)
        self._epoch_started_at[int(epoch)] = time.monotonic()
        total_batches = len(trainer.dataloader)
        checkpoint = _safe_field(trainer.model_path or "none")
        return self.send(
            f":hourglass_flowing_sand: *Epoch {epoch + 1}/{trainer.n_epochs} started*\n"
            f"*Experiment:* {_safe_field(trainer.exp_name)}\n"
            f"{self._job_context()}\n"
            f"*Batches:* {total_batches}; *batch size:* {trainer.batch_size}; "
            f"*learning rate:* {trainer.lr:.6g}\n"
            f"*Starting checkpoint:* {checkpoint}"
        )

    def notify_epoch_completed(
        self,
        trainer: Any,
        epoch: int,
        training_metrics: Mapping[str, Any],
        validation_metrics: Optional[Mapping[str, Any]],
        history: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> bool:
        started_at = self._epoch_started_at.pop(int(epoch), None)
        if self._current_epoch == int(epoch):
            self._current_epoch = None
        duration = time.monotonic() - started_at if started_at is not None else None
        if history is not None:
            self._history = list(history)
        else:
            record = {
                "epoch": int(epoch),
                "training_loss": training_metrics.get("loss_total"),
                "validation_loss": (
                    validation_metrics.get("loss_total")
                    if validation_metrics is not None
                    else None
                ),
            }
            self._history = [
                row for row in self._history if int(row.get("epoch", -1)) != int(epoch)
            ]
            self._history.append(record)
            self._history.sort(key=lambda row: int(row.get("epoch", 0)))

        plot_path = Path(trainer.out_dir) / f"{trainer.exp_name}_loss.png"
        saved_plot = None
        try:
            saved_plot = save_loss_plot(self._history, plot_path)
        except Exception as exc:
            self._warning("loss-plot save", exc)

        lines = [
            f":white_check_mark: *Epoch {epoch + 1}/{trainer.n_epochs} completed*",
            f"*Experiment:* {_safe_field(trainer.exp_name)}",
            self._job_context(),
            f"*Duration:* {self._format_seconds(duration)}",
            *self._metric_lines("Training loss", training_metrics),
            *self._metric_lines("Validation loss", validation_metrics),
        ]
        if validation_metrics is not None:
            lines.append(
                f"*Validation files:* {len(validation_metrics.get('files', []))}; "
                f"*elements:* {validation_metrics.get('element_count', 'n/a')}"
            )
        lines.append(f"*Checkpoint:* {_safe_field(trainer.model_path or 'none')}")
        lines.append(render_loss_sparklines(self._history))
        if saved_plot is not None:
            lines.append(f"*Full loss plot saved:* {_safe_field(saved_plot)}")
        return self.send("\n".join(lines))

    def notify_run_completed(self, trainer: Any) -> bool:
        duration = (
            time.monotonic() - self._run_started_at
            if self._run_started_at is not None
            else None
        )
        return self.send(
            ":tada: *SUPPORT training completed*\n"
            f"*Experiment:* {_safe_field(trainer.exp_name)}\n"
            f"{self._job_context()}\n"
            f"*Completed epoch target:* {trainer.n_epochs}\n"
            f"*Total runtime:* {self._format_seconds(duration)}\n"
            f"*Latest model:* {_safe_field(trainer.model_path or 'none')}"
        )

    def notify_failure(
        self,
        exc: BaseException,
        *,
        trainer: Optional[Any] = None,
        stage: str = "training",
    ) -> bool:
        if self._failure_attempted:
            return False
        self._failure_attempted = True
        experiment = _safe_field(getattr(trainer, "exp_name", "not initialized"))
        epoch = self._current_epoch
        epoch_text = (
            f"{int(epoch) + 1}/{getattr(trainer, 'n_epochs', '?')}"
            if epoch is not None
            else "not started"
        )
        checkpoint = _safe_field(getattr(trainer, "model_path", None) or "none")
        detail = _safe_field(exc).replace("`", "'")
        if len(detail) > 1000:
            detail = detail[:980] + "…(truncated)"
        return self.send(
            ":x: *SUPPORT training failed*\n"
            f"*Experiment:* {experiment}\n"
            f"{self._job_context()}\n"
            f"*Stage:* {_safe_field(stage)}; *epoch:* {epoch_text}\n"
            f"*Last complete checkpoint:* {checkpoint}\n"
            f"*Error:* `{type(exc).__name__}: {detail}`"
        )
