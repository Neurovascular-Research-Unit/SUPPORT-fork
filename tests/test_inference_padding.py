"""Padded inference must retain every original frame and spatial coordinate."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.test_collection import TestingClass


class CenterFrame(torch.nn.Module):
    def __init__(self, patch_frames):
        super().__init__()
        self.in_channels = patch_frames

    def forward(self, patch):
        middle = self.in_channels // 2
        return patch[:, middle:middle + 1]


class InferencePaddingTests(unittest.TestCase):
    def test_padded_volume_inference_keeps_all_frames_and_z_planes(self):
        for patch_frames in (1, 3, 5):
            with self.subTest(patch_frames=patch_frames), tempfile.TemporaryDirectory() as temporary:
                image = np.arange(7 * 2 * 3 * 4, dtype=np.float32).reshape(7, 2, 3, 4)
                original = image.copy()
                tester = TestingClass({
                    "dimensionality": "3+1D", "use_cpu": True, "n_cpu": 0,
                    "pad": True, "patch_size": [patch_frames, 3, 4],
                    "patch_interval": [1, 3, 4], "batch_size": 2,
                    "out_dir": temporary, "exp_name": "padded-volume",
                })
                tester.local_model = CenterFrame(patch_frames)
                result = tester._run_inference_on_4d_stack(image)
                self.assertEqual(result.shape, original.shape)
                np.testing.assert_allclose(result, original, atol=2e-5)
                np.testing.assert_array_equal(image, original)
                self.assertFalse((Path(temporary) / ".support_cache").exists())


if __name__ == "__main__":
    unittest.main()
