"""Focused tests for lower-power simulation and photon calibration."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from utils.salt import estimate_photons_per_unit, simulate_lower_power


class _PassthroughRng:
    def poisson(self, values):
        return values


class SimulateLowerPowerTests(unittest.TestCase):
    def _simulate_without_noise(self, video, **kwargs):
        with mock.patch(
            "utils.salt.np.random.default_rng",
            return_value=_PassthroughRng(),
        ):
            return simulate_lower_power(video, **kwargs)

    def test_scales_illumination_background_with_two_photon_signal(self) -> None:
        video = np.full((3, 16, 16), 50, dtype=np.float32)
        video[:, :2, :2] = 10
        original = video.copy()

        result = self._simulate_without_noise(
            video,
            power_ratio=0.5,
            gamma=2.0,
            photons_per_unit=1.0,
        )

        expected = 0.25 * video
        np.testing.assert_allclose(result, expected)
        np.testing.assert_array_equal(video, original)
        self.assertEqual(result.shape, video.shape)
        self.assertEqual(result.dtype, np.float32)


class PhotonTransferEstimatorTests(unittest.TestCase):
    @staticmethod
    def _synthetic_stacks(support_gain=2.5, support_offset=37.0):
        rng = np.random.default_rng(3)
        yy, xx = np.mgrid[:64, :64]
        clean_frame = (200.0 + 12.0 * xx + 18.0 * yy).astype(np.float32)
        clean = np.broadcast_to(clean_frame, (60, 64, 64))
        photons_per_raw_unit = 0.02
        raw = rng.poisson(clean * photons_per_raw_unit).astype(np.float32)
        raw /= photons_per_raw_unit
        support = np.broadcast_to(
            support_offset + support_gain * clean_frame,
            raw.shape,
        ).copy()
        return raw, support, photons_per_raw_unit

    @staticmethod
    def _estimate(raw, support, **overrides):
        options = {
            "max_pairs": 30,
            "sample_count": 4_096,
            "n_bins": 12,
            "min_samples_per_bin": 100,
            "scale_min_samples_per_bin": 20,
            "max_fit_samples": 200_000,
            "random_seed": 7,
        }
        options.update(overrides)
        return estimate_photons_per_unit(raw, support, **options)

    def test_recovers_poisson_scale_in_raw_and_support_units(self) -> None:
        raw, support, expected_raw = self._synthetic_stacks()

        estimate = self._estimate(raw, support)

        self.assertAlmostEqual(
            estimate.photons_per_raw_unit / expected_raw,
            1.0,
            delta=0.10,
        )
        self.assertAlmostEqual(estimate.support_gain / 2.5, 1.0, delta=0.05)
        self.assertAlmostEqual(
            estimate.photons_per_unit / (expected_raw / 2.5),
            1.0,
            delta=0.10,
        )
        self.assertGreater(estimate.linear_r_squared, 0.98)
        self.assertEqual(estimate.warnings, ())
        self.assertAlmostEqual(
            estimate.photons_per_raw_unit,
            1.0 / estimate.variance_slope,
        )
        self.assertAlmostEqual(
            estimate.photons_per_unit,
            estimate.photons_per_raw_unit / estimate.support_gain,
        )

    def test_support_affine_rescaling_transforms_only_support_units(self) -> None:
        raw, support, _ = self._synthetic_stacks(
            support_gain=1.0, support_offset=0.0
        )
        first = self._estimate(raw, support)
        transformed = 41.0 + 3.0 * support

        second = self._estimate(raw, transformed)

        self.assertAlmostEqual(
            first.photons_per_raw_unit,
            second.photons_per_raw_unit,
            delta=1e-8,
        )
        self.assertAlmostEqual(second.support_gain, 3.0 * first.support_gain)
        self.assertAlmostEqual(
            second.support_offset,
            41.0 + 3.0 * first.support_offset,
        )
        self.assertAlmostEqual(
            second.photons_per_unit,
            first.photons_per_unit / 3.0,
            delta=1e-8,
        )
        self.assertAlmostEqual(
            second.effective_support_pedestal,
            41.0 + 3.0 * first.effective_support_pedestal,
            delta=1e-3,
        )

    def test_warns_when_temporal_changes_inflate_primary_pair_variance(self) -> None:
        rng = np.random.default_rng(123)
        yy, xx = np.mgrid[:64, :64]
        base = 200.0 + 12.0 * xx + 18.0 * yy
        fluctuations = (
            rng.normal(size=(60, 64, 64)) * 3.0 * np.sqrt(base)[None]
        )
        clean = np.clip(base[None] + fluctuations, 0, None)
        raw = rng.poisson(0.02 * clean).astype(np.float32) / 0.02
        support = (37.0 + 2.5 * clean).astype(np.float32)

        estimate = self._estimate(raw, support)

        self.assertLess(estimate.photons_per_raw_unit, 0.018)
        self.assertAlmostEqual(
            estimate.quiet_photons_per_raw_unit / 0.02,
            1.0,
            delta=0.05,
        )
        self.assertGreater(
            abs(
                estimate.quiet_photons_per_unit
                / estimate.photons_per_unit
                - 1.0
            ),
            0.15,
        )
        self.assertTrue(
            any("motion or biological dynamics" in item for item in estimate.warnings)
        )

    def test_is_reproducible_and_does_not_modify_inputs(self) -> None:
        raw, support, _ = self._synthetic_stacks()
        raw_before = raw.copy()
        support_before = support.copy()

        first = self._estimate(
            raw,
            support,
            sample_count=2_000,
            max_fit_samples=30_000,
        )
        second = self._estimate(
            raw,
            support,
            sample_count=2_000,
            max_fit_samples=30_000,
        )

        self.assertEqual(first, second)
        np.testing.assert_array_equal(raw, raw_before)
        np.testing.assert_array_equal(support, support_before)

    def test_frame_window_excludes_edges_from_pairs_and_scale_fit(self) -> None:
        raw, support, _ = self._synthetic_stacks()
        raw = np.concatenate((raw, raw, raw), axis=0)
        support = np.concatenate((support, support, support), axis=0)
        raw_with_bad_edges = raw.copy()
        support_with_bad_edges = support.copy()
        raw_with_bad_edges[:30] = 50_000
        raw_with_bad_edges[150:] = -50_000
        support_with_bad_edges[:30] = -1_000_000
        support_with_bad_edges[150:] = 1_000_000

        original_linspace = np.linspace
        with mock.patch(
            "utils.salt.np.linspace", wraps=original_linspace
        ) as linspace_spy:
            windowed = self._estimate(
                raw_with_bad_edges,
                support_with_bad_edges,
                frame_start=30,
                frame_stop=150,
                max_pairs=10,
            )
        cropped = self._estimate(
            raw[30:150],
            support[30:150],
            frame_start=0,
            frame_stop=120,
            max_pairs=10,
        )

        self.assertEqual(windowed, cropped)
        self.assertEqual(windowed.pair_count, 10)
        pair_selection_calls = [
            call
            for call in linspace_spy.call_args_list
            if call.kwargs.get("dtype") is np.int64
        ]
        self.assertEqual(len(pair_selection_calls), 1)
        self.assertEqual(pair_selection_calls[0].args, (0, 59, 10))

    def test_accepts_tzyx_and_rejects_invalid_inputs(self) -> None:
        raw, support, _ = self._synthetic_stacks()
        estimate = self._estimate(
            raw.reshape(60, 2, 32, 64),
            support.reshape(60, 2, 32, 64),
        )
        self.assertGreater(estimate.photons_per_unit, 0)

        with self.assertRaisesRegex(ValueError, "same time-first shape"):
            self._estimate(raw, support[:, :, :-1])
        with self.assertRaisesRegex(ValueError, "shape"):
            self._estimate(raw[0], support[0])
        with self.assertRaisesRegex(ValueError, "two complete frame pairs"):
            self._estimate(raw[:3], support[:3])
        with self.assertRaisesRegex(TypeError, "frame_stop"):
            self._estimate(raw, support, frame_stop=True)
        with self.assertRaisesRegex(ValueError, "frame_stop"):
            self._estimate(raw, support, frame_start=10, frame_stop=10)
        with self.assertRaisesRegex(ValueError, "frame_stop"):
            self._estimate(raw, support, frame_stop=raw.shape[0] + 1)
        with self.assertRaisesRegex(ValueError, "fit_quantiles"):
            self._estimate(raw, support, fit_quantiles=(0.9, 0.1))
        with self.assertRaisesRegex(ValueError, "n_bins must be at least six"):
            self._estimate(raw, support, n_bins=5)
        with self.assertRaisesRegex(ValueError, "distinct intensity"):
            self._estimate(raw, np.ones_like(support))


class AdditionalSimulateLowerPowerTests(unittest.TestCase):
    def _simulate_without_noise(self, video, **kwargs):
        with mock.patch(
            "utils.salt.np.random.default_rng",
            return_value=_PassthroughRng(),
        ):
            return simulate_lower_power(video, **kwargs)

    def test_4d_illumination_background_scales_with_signal(self) -> None:
        video = np.empty((3, 2, 16, 16), dtype=np.float32)
        video[:, 0] = 50
        video[:, 0, :2, :2] = 10
        video[:, 1] = 140
        video[:, 1, -2:, -2:] = 100

        result = self._simulate_without_noise(
            video,
            power_ratio=0.5,
            gamma=2.0,
            photons_per_unit=1.0,
        )

        expected = 0.25 * video
        np.testing.assert_allclose(result, expected)
        self.assertEqual(result.shape, video.shape)

    def test_explicit_roi_does_not_hold_optical_background_fixed(self) -> None:
        video = np.empty((2, 2, 8, 8), dtype=np.float32)
        video[:, 0] = 50
        video[:, 0, 2:4, 2:4] = 10
        video[:, 1] = 140
        video[:, 1, 2:4, 2:4] = 100
        roi = (slice(None), slice(2, 4), slice(2, 4))

        result = self._simulate_without_noise(
            video,
            power_ratio=0.5,
            gamma=1.0,
            photons_per_unit=1.0,
            background_slices=roi,
        )

        expected = 0.5 * video
        np.testing.assert_allclose(result, expected)

    def test_keeps_only_configured_detector_pedestal_fixed(self) -> None:
        video = np.array(
            [[[5.0, 10.0], [30.0, 50.0]]], dtype=np.float32
        )

        result = self._simulate_without_noise(
            video,
            power_ratio=0.5,
            gamma=2.0,
            photons_per_unit=1.0,
            zero_photon_level=10.0,
        )

        expected = 10.0 + 0.25 * np.clip(video - 10.0, 0, None)
        np.testing.assert_allclose(result, expected)

    def test_subtracts_negative_pedestal_before_physical_clipping(self) -> None:
        video = np.full((1, 2, 2), -50.0, dtype=np.float32)

        result = self._simulate_without_noise(
            video,
            power_ratio=0.5,
            gamma=2.0,
            photons_per_unit=1.0,
            zero_photon_level=-100.0,
        )

        np.testing.assert_allclose(result, -87.5)

    def test_is_invariant_to_affine_support_units_with_matching_parameters(self) -> None:
        video = np.linspace(10.0, 250.0, 128, dtype=np.float32).reshape(
            2, 8, 8
        )
        pedestal = 5.0
        gain = 3.0
        offset = 41.0

        first = simulate_lower_power(
            video,
            power_ratio=0.6,
            gamma=2.0,
            photons_per_unit=0.2,
            zero_photon_level=pedestal,
            seed=17,
        )
        second = simulate_lower_power(
            offset + gain * video,
            power_ratio=0.6,
            gamma=2.0,
            photons_per_unit=0.2 / gain,
            zero_photon_level=offset + gain * pedestal,
            seed=17,
        )

        np.testing.assert_allclose(second, offset + gain * first, atol=1e-4)

    def test_rejects_invalid_zero_photon_level(self) -> None:
        video = np.ones((2, 8, 8), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "finite"):
            simulate_lower_power(video, zero_photon_level=np.nan)
        with self.assertRaisesRegex(ValueError, "broadcastable"):
            simulate_lower_power(video, zero_photon_level=np.ones((3, 3)))

    def test_rejects_inputs_without_time_and_spatial_axes(self) -> None:
        for shape in ((8, 8), (1, 1, 1, 8, 8)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(AssertionError, "shape"):
                    simulate_lower_power(np.ones(shape, dtype=np.float32))

    def test_clips_negative_support_predictions_before_power_scaling(self) -> None:
        video = np.full((2, 8, 8), 5, dtype=np.float32)
        video[:, :4, :4] = -2
        original = video.copy()

        result = self._simulate_without_noise(
            video,
            power_ratio=0.5,
            gamma=2.0,
            photons_per_unit=1.0,
            background_slices=(slice(None), slice(0, 4), slice(0, 4)),
        )

        self.assertGreaterEqual(float(result.min()), 0.0)
        np.testing.assert_array_equal(result[:, :4, :4], 0.0)
        np.testing.assert_allclose(result[:, 4:, 4:], 1.25)
        np.testing.assert_array_equal(video, original)


if __name__ == "__main__":
    unittest.main()
