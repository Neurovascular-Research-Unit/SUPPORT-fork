"""Unit tests for optional, fail-open Slack training notifications."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from utils.slack_notifications import (
    SlackConfigurationError,
    SlackNotifier,
    redact_slack_webhooks,
    render_loss_sparklines,
    save_loss_plot,
)


FAKE_WEBHOOK = "https://hooks.slack.com/services/TEST/ONLY/NOT_A_SECRET"
SENSITIVE_WEBHOOK = "https://hooks.slack.com/services/TEST/ONLY/SENSITIVE_TOKEN"


def successful_response() -> mock.Mock:
    response = mock.Mock()
    response.raise_for_status.return_value = None
    return response


class SlackNotifierTests(unittest.TestCase):
    @mock.patch.dict(
        os.environ,
        {"SUPPORT_SLACK_WEBHOOK_URL": FAKE_WEBHOOK},
        clear=True,
    )
    def test_disabled_notifier_never_contacts_slack(self) -> None:
        post = mock.Mock()
        notifier = SlackNotifier.from_environment(enabled=False, post=post)

        self.assertFalse(notifier.send("training started"))
        post.assert_not_called()

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_requested_notifications_without_webhook_are_configuration_error(self) -> None:
        with self.assertRaisesRegex(
            SlackConfigurationError, "SUPPORT_SLACK_WEBHOOK_FILE"
        ):
            SlackNotifier.from_environment(enabled=True, post=mock.Mock())

    @mock.patch.dict(
        os.environ,
        {"SUPPORT_SLACK_WEBHOOK_URL": FAKE_WEBHOOK},
        clear=True,
    )
    def test_send_posts_expected_payload_with_timeout(self) -> None:
        post = mock.Mock(return_value=successful_response())
        notifier = SlackNotifier.from_environment(enabled=True, post=post)

        self.assertTrue(notifier.send("epoch 1 started"))
        post.assert_called_once_with(
            FAKE_WEBHOOK,
            json={"text": "epoch 1 started"},
            timeout=10,
        )

    @mock.patch.dict(
        os.environ,
        {"SUPPORT_SLACK_WEBHOOK_URL": SENSITIVE_WEBHOOK},
        clear=True,
    )
    def test_delivery_failure_is_nonfatal_and_redacts_webhook(self) -> None:
        post = mock.Mock(
            side_effect=TimeoutError(f"timed out while posting to {SENSITIVE_WEBHOOK}")
        )
        notifier = SlackNotifier.from_environment(enabled=True, post=post)
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            result = notifier.send("epoch 1 finished")

        self.assertFalse(result)
        self.assertIn("training will continue", stderr.getvalue())
        self.assertNotIn(SENSITIVE_WEBHOOK, stderr.getvalue())
        self.assertNotIn("SENSITIVE_TOKEN", stderr.getvalue())
        self.assertNotIn("SENSITIVE_TOKEN", repr(notifier))

    def test_webhook_in_message_is_redacted_before_delivery(self) -> None:
        post = mock.Mock(return_value=successful_response())
        notifier = SlackNotifier(
            SENSITIVE_WEBHOOK,
            enabled=True,
            post=post,
        )

        self.assertTrue(notifier.send(f"failure mentioned {SENSITIVE_WEBHOOK}"))

        delivered_message = post.call_args.kwargs["json"]["text"]
        self.assertIn("[REDACTED_SLACK_WEBHOOK]", delivered_message)
        self.assertNotIn("SENSITIVE_TOKEN", delivered_message)

    def test_non_slack_https_destination_is_rejected(self) -> None:
        with self.assertRaisesRegex(SlackConfigurationError, "hooks.slack.com"):
            SlackNotifier(
                "https://example.com/services/TEST/ONLY/NOT_A_SECRET",
                enabled=True,
                post=mock.Mock(),
            )

    def test_run_start_resets_failure_and_stale_epoch_state(self) -> None:
        post = mock.Mock(return_value=successful_response())
        notifier = SlackNotifier(FAKE_WEBHOOK, enabled=True, post=post)
        notifier._failure_attempted = True
        notifier._current_epoch = 8
        notifier._epoch_started_at[8] = 1.0
        trainer = SimpleNamespace(
            exp_name="experiment",
            dimensionality="2+1D",
            device="cuda",
            in_path="/input",
            out_dir="/output",
            n_epochs=10,
            batch_size=4,
            patch_size=[3, 8, 8],
            n_cpu=0,
            prefetch_factor=2,
            resume=True,
        )

        self.assertTrue(notifier.notify_run_started(trainer))

        self.assertFalse(notifier._failure_attempted)
        self.assertIsNone(notifier._current_epoch)
        self.assertEqual(notifier._epoch_started_at, {})

    def test_user_fields_cannot_create_slack_mentions(self) -> None:
        post = mock.Mock(return_value=successful_response())
        notifier = SlackNotifier(FAKE_WEBHOOK, enabled=True, post=post)
        trainer = SimpleNamespace(
            exp_name="<!channel>",
            dimensionality="2+1D",
            device="cuda",
            in_path="/input/<unsafe>",
            out_dir="/output",
            n_epochs=1,
            batch_size=4,
            patch_size=[3, 8, 8],
            n_cpu=0,
            prefetch_factor=2,
            resume=False,
        )

        self.assertTrue(notifier.notify_run_started(trainer))

        message = post.call_args.kwargs["json"]["text"]
        self.assertNotIn("<!channel>", message)
        self.assertIn("&lt;!channel&gt;", message)
        self.assertIn("&lt;unsafe&gt;", message)

    def test_failure_notification_is_sent_at_most_once(self) -> None:
        post = mock.Mock(return_value=successful_response())
        notifier = SlackNotifier(FAKE_WEBHOOK, enabled=True, post=post)
        trainer = SimpleNamespace(
            exp_name="experiment",
            epoch=2,
            n_epochs=10,
            model_path="/output/model_1.pth",
        )
        notifier._current_epoch = 2

        self.assertTrue(
            notifier.notify_failure(
                RuntimeError(f"worker failed; credential={SENSITIVE_WEBHOOK}"),
                trainer=trainer,
                stage="epoch",
            )
        )
        self.assertFalse(
            notifier.notify_failure(
                RuntimeError("duplicate failure"),
                trainer=trainer,
                stage="shutdown",
            )
        )

        post.assert_called_once()
        delivered_message = post.call_args.kwargs["json"]["text"]
        self.assertIn("SUPPORT training failed", delivered_message)
        self.assertIn("epoch", delivered_message)
        self.assertIn("3/10", delivered_message)
        self.assertIn("[REDACTED_SLACK_WEBHOOK]", delivered_message)
        self.assertNotIn("SENSITIVE_TOKEN", delivered_message)


class SlackWebhookFileTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_private_webhook_file_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            webhook_file = Path(temporary_directory) / "slack-webhook"
            webhook_file.write_text(f"{FAKE_WEBHOOK}\n", encoding="utf-8")
            webhook_file.chmod(0o600)
            post = mock.Mock(return_value=successful_response())

            notifier = SlackNotifier.from_environment(
                enabled=True,
                webhook_file=webhook_file,
                post=post,
            )

            self.assertTrue(notifier.send("started"))
            self.assertEqual(post.call_args.args[0], FAKE_WEBHOOK)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_group_or_world_accessible_webhook_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            webhook_file = Path(temporary_directory) / "slack-webhook"
            webhook_file.write_text(f"{FAKE_WEBHOOK}\n", encoding="utf-8")
            webhook_file.chmod(0o644)

            with self.assertRaisesRegex(SlackConfigurationError, "chmod 600"):
                SlackNotifier.from_environment(
                    enabled=True,
                    webhook_file=webhook_file,
                    post=mock.Mock(),
                )


class LossPlotTests(unittest.TestCase):
    history = [
        {"epoch": 0, "training_loss": 4.0, "validation_loss": 5.0},
        {"epoch": 1, "training_loss": 2.0, "validation_loss": 3.0},
        {"epoch": 2, "training_loss": None, "validation_loss": 1.0},
    ]

    def test_render_loss_sparklines_uses_shared_scale_and_marks_missing_data(self) -> None:
        rendered = render_loss_sparklines(self.history)

        self.assertIn("epochs 1–3, shared scale", rendered)
        self.assertIn("train      `▆▃·`", rendered)
        self.assertIn("validation `█▅▁`", rendered)

    def test_render_loss_sparklines_handles_empty_history(self) -> None:
        self.assertEqual(
            render_loss_sparklines([]),
            "Loss plot: no completed epochs yet.",
        )

    def test_save_loss_plot_writes_png_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "loss.png"

            with mock.patch.dict(
                os.environ,
                {
                    "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": temporary_directory,
                    "XDG_CACHE_HOME": temporary_directory,
                },
            ):
                result = save_loss_plot(self.history, destination)

            self.assertEqual(result, destination.resolve())
            self.assertTrue(destination.is_file())
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertFalse(list(destination.parent.glob(".loss.*.png")))

    def test_save_loss_plot_skips_history_without_finite_losses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "loss.png"

            result = save_loss_plot(
                [{"epoch": 0, "training_loss": float("nan"), "validation_loss": None}],
                destination,
            )

            self.assertIsNone(result)
            self.assertFalse(destination.exists())


class RedactionTests(unittest.TestCase):
    def test_redact_slack_webhooks_removes_secret_from_arbitrary_text(self) -> None:
        redacted = redact_slack_webhooks(f"before {SENSITIVE_WEBHOOK} after")

        self.assertEqual(
            redacted,
            "before [REDACTED_SLACK_WEBHOOK] after",
        )


if __name__ == "__main__":
    unittest.main()
