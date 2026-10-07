"""Regression tests for SUPPORT normalization boundaries."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

try:
    import torch
    from utils.dataset import DatasetSUPPORT
    from utils.train_collection import TrainingClass
except ModuleNotFoundError:
    torch = None
    DatasetSUPPORT = None
    TrainingClass = None


@unittest.skipIf(torch is None, "PyTorch is required for dataset normalization tests")
class SourceTiffNormalizationTests(unittest.TestCase):
    @staticmethod
    def _volume() -> np.ndarray:
        volume = np.empty((2, 2, 2, 2), dtype=np.float32)
        volume[:, 0, :, :] = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        volume[:, 1, :, :] = 100 + np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        return volume

    @staticmethod
    def _planes(volume: np.ndarray):
        return [
            torch.from_numpy(np.ascontiguousarray(volume[:, z_index, :, :]))
            for z_index in range(volume.shape[1])
        ]

    def test_z_planes_from_one_tiff_share_full_volume_statistics(self) -> None:
        volume = self._volume()
        dataset = DatasetSUPPORT(
            noisy_imgs=self._planes(volume),
            patch_size=[2, 2, 2],
            patch_interval=[1, 1, 1],
            load_to_memory=True,
            random_patch=False,
            normalization_group_ids=[0, 0],
        )

        expected = torch.from_numpy(volume)
        expected_mean = torch.mean(expected)
        expected_std = torch.std(expected)
        torch.testing.assert_close(
            dataset.mean_images,
            expected_mean.repeat(2),
        )
        torch.testing.assert_close(
            dataset.std_images,
            expected_std.repeat(2),
            rtol=1e-6,
            atol=1e-6,
        )

        normalized_volume = torch.stack(dataset.noisy_imgs, dim=1)
        torch.testing.assert_close(
            torch.mean(normalized_volume),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.std(normalized_volume),
            torch.tensor(1.0),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertLess(float(torch.mean(normalized_volume[:, 0])), -0.9)
        self.assertGreater(float(torch.mean(normalized_volume[:, 1])), 0.9)

    def test_lazy_loading_uses_shared_stats_without_modifying_source(self) -> None:
        volume = self._volume()
        planes = [
            np.ascontiguousarray(volume[:, z_index, :, :])
            for z_index in range(volume.shape[1])
        ]
        original_planes = [plane.copy() for plane in planes]
        dataset = DatasetSUPPORT(
            noisy_imgs=planes,
            patch_size=[2, 2, 2],
            patch_interval=[1, 1, 1],
            load_to_memory=False,
            random_patch=False,
            normalization_group_ids=[0, 0],
        )

        for actual, original in zip(planes, original_planes):
            np.testing.assert_array_equal(actual, original)

        patch, _, _ = dataset[0]
        expected_patch = (
            torch.from_numpy(original_planes[0]) - dataset.mean_images[0]
        ) / dataset.std_images[0]
        torch.testing.assert_close(patch, expected_patch.to(torch.float32))

    def test_different_source_tiffs_keep_separate_statistics(self) -> None:
        first = self._volume()
        second = self._volume() + 1000
        planes = self._planes(first) + self._planes(second)
        dataset = DatasetSUPPORT(
            noisy_imgs=planes,
            patch_size=[2, 2, 2],
            patch_interval=[1, 1, 1],
            load_to_memory=True,
            random_patch=False,
            normalization_group_ids=[0, 0, 1, 1],
        )

        self.assertEqual(dataset.mean_images[0], dataset.mean_images[1])
        self.assertEqual(dataset.mean_images[2], dataset.mean_images[3])
        self.assertNotEqual(dataset.mean_images[0], dataset.mean_images[2])

    def test_grouped_stats_cannot_override_generalized_normalization(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            DatasetSUPPORT(
                noisy_imgs=self._planes(self._volume()),
                patch_size=[2, 2, 2],
                patch_interval=[1, 1, 1],
                load_to_memory=True,
                normalize_across_images=True,
                normalization_group_ids=[0, 0],
            )

    def test_training_groups_planes_by_their_original_4d_tiff(self) -> None:
        volume = self._volume()
        trainer = TrainingClass.__new__(TrainingClass)
        trainer.in_path = "synthetic.tif"
        trainer.is_dir = False
        trainer.train_gen = False
        trainer.dimensionality = "3+1D"
        trainer.load_to_memory = True
        trainer.patch_size = [2, 2, 2]
        trainer.patch_interval = [1, 1, 1]
        trainer.normalization_scope = "per_stack"
        trainer._read_stack = mock.Mock(return_value=volume)
        trainer._dataloader_kwargs = mock.Mock(
            return_value={"batch_size": 1, "num_workers": 0}
        )

        trainer.generate_dataloader()

        dataset = trainer.dataloader.dataset
        self.assertEqual(dataset.normalization_group_ids, [0, 0])
        self.assertFalse(dataset.normalize_across_images)
        self.assertEqual(trainer.normalization_scope, "per_stack")

    def test_legacy_3d_training_keeps_the_original_ungrouped_path(self) -> None:
        stack = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        expected_stack = torch.from_numpy(stack.copy())
        trainer = TrainingClass.__new__(TrainingClass)
        trainer.in_path = "synthetic.tif"
        trainer.is_dir = False
        trainer.train_gen = False
        trainer.dimensionality = "3+1D"
        trainer.load_to_memory = True
        trainer.patch_size = [2, 2, 2]
        trainer.patch_interval = [1, 1, 1]
        trainer.normalization_scope = "per_stack"
        trainer._read_stack = mock.Mock(return_value=stack)
        trainer._dataloader_kwargs = mock.Mock(
            return_value={"batch_size": 1, "num_workers": 0}
        )

        trainer.generate_dataloader()

        dataset = trainer.dataloader.dataset
        self.assertIsNone(dataset.normalization_group_ids)
        torch.testing.assert_close(
            dataset.mean_images,
            torch.mean(expected_stack).reshape(1),
        )
        torch.testing.assert_close(
            dataset.std_images,
            torch.std(expected_stack).reshape(1),
        )

    def test_generalized_training_keeps_corpus_global_normalization(self) -> None:
        volume = self._volume()
        trainer = TrainingClass.__new__(TrainingClass)
        trainer.in_path = "unused-directory"
        trainer.is_dir = True
        trainer.train_gen = True
        trainer.dimensionality = "3+1D"
        trainer.load_to_memory = True
        trainer.patch_size = [2, 2, 2]
        trainer.patch_interval = [1, 1, 1]
        trainer.normalization_scope = "dataset"
        trainer.normalization_mean = None
        trainer.normalization_std = None
        trainer.normalization_voxel_count = None
        trainer._frozen_input_paths = ("synthetic.tif",)
        trainer._read_stack = mock.Mock(return_value=volume)
        trainer._dataloader_kwargs = mock.Mock(
            return_value={"batch_size": 1, "num_workers": 0}
        )

        trainer.generate_dataloader()

        dataset = trainer.dataloader.dataset
        self.assertIsNone(dataset.normalization_group_ids)
        self.assertTrue(dataset.normalize_across_images)
        self.assertEqual(trainer.normalization_scope, "dataset")


if __name__ == "__main__":
    unittest.main()
