"""Focused tests for lower-power simulation background estimation."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from utils.salt import simulate_lower_power


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

    def test_automatically_finds_dark_border_background(self) -> None:
        video = np.full((3, 16, 16), 50, dtype=np.float32)
        video[:, :2, :2] = 10
        original = video.copy()

        result = self._simulate_without_noise(
            video,
            power_ratio=0.5,
            gamma=2.0,
            photons_per_unit=1.0,
        )

        expected = 10 + 0.25 * (video - 10)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_array_equal(video, original)
        self.assertEqual(result.shape, video.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_4d_background_is_found_independently_per_z_slice(self) -> None:
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

        background = np.array([10, 100], dtype=np.float32)[None, :, None, None]
        expected = background + 0.25 * (video - background)
        np.testing.assert_allclose(result, expected)
        self.assertEqual(result.shape, video.shape)

    def test_explicit_roi_is_applied_separately_to_each_z_slice(self) -> None:
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

        background = np.array([10, 100], dtype=np.float32)[None, :, None, None]
        expected = background + 0.5 * (video - background)
        np.testing.assert_allclose(result, expected)

    def test_rejects_inputs_without_time_and_spatial_axes(self) -> None:
        for shape in ((8, 8), (1, 1, 1, 8, 8)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(AssertionError, "shape"):
                    simulate_lower_power(np.ones(shape, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
