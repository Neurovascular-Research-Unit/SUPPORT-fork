"""Temporal training edges never enter patches, targets, or normalization."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.train_collection import TrainingClass


class ExhaustiveTemporalStarts:
    """Exercise every valid temporal start through the real patch precomputer."""

    @staticmethod
    def integers(low, high, size):
        return np.resize(np.arange(low, high), size)

    @staticmethod
    def shuffle(values):
        pass


class TrainingFrameExclusionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    @staticmethod
    def image(*, volume=False, frames=9, edge_value=1e8):
        shape = (frames, 2, 2, 3) if volume else (frames, 2, 3)
        image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 10
        image[0] = edge_value
        image[-1] = -edge_value
        if volume:
            image[:, 1] += 1000
        return image

    def write(self, name, image):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, image, imagej=True, photometric="minisblack",
                         metadata={"axes": "TZYX" if image.ndim == 4 else "TYX"})
        return path

    def trainer(self, path, *, resident=True, patch=3, exclude=True, **extra):
        params = {"in_path": str(path), "dimensionality": "3+1D",
                  "load_to_memory": resident, "patch_size": [patch, 2, 3],
                  "patch_interval": [1, 1, 1], "batch_size": 1,
                  "n_cpu": 0, "use_cpu": True, **extra}
        if exclude is not None:
            params["exclude_training_edge_frames"] = exclude
        with redirect_stdout(io.StringIO()):
            return TrainingClass(params)

    @staticmethod
    def load(trainer):
        with redirect_stdout(io.StringIO()):
            trainer.generate_dataloader()
        return trainer.dataloader.dataset

    def assert_all_patches_use(self, dataset, interior, patch=3):
        planes = ([interior[:, z] for z in range(interior.shape[1])]
                  if interior.ndim == 4 else [interior])
        dataset.patch_rng = ExhaustiveTemporalStarts()
        dataset.precompute_indices()
        covered = {index: set() for index in range(len(planes))}
        for index in range(len(dataset)):
            actual, coordinates, ds_index = dataset[index]
            ds_index = int(ds_index)
            start, stop = (int(value) for value in coordinates[0])
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(stop, interior.shape[0])
            recovered = actual * dataset.std_images[ds_index] + dataset.mean_images[ds_index]
            expected = torch.from_numpy(np.array(planes[ds_index][start:stop], copy=True))
            torch.testing.assert_close(recovered, expected, rtol=1e-5, atol=1e-4)
            torch.testing.assert_close(recovered[patch // 2], expected[patch // 2],
                                       rtol=1e-5, atol=1e-4)
            covered[ds_index].update(range(start, stop))
        self.assertTrue(all(frames == set(range(interior.shape[0])) for frames in covered.values()))

    def test_all_context_targets_and_statistics_exclude_edges_on_every_z_plane(self):
        for volume in (False, True):
            for resident in (False, True):
                with self.subTest(volume=volume, resident=resident):
                    image = self.image(volume=volume)
                    interior = image[1:-1].copy()
                    path = self.write(f"{volume}-{resident}.tif", image)
                    before = path.read_bytes()
                    trainer = self.trainer(path, resident=resident)
                    dataset = self.load(trainer)
                    expected = torch.from_numpy(interior)
                    count = 2 if volume else 1
                    torch.testing.assert_close(dataset.mean_images,
                                               expected.mean().repeat(count), rtol=1e-6, atol=1e-5)
                    torch.testing.assert_close(dataset.std_images,
                                               expected.std().repeat(count), rtol=1e-6, atol=1e-5)
                    self.assertAlmostEqual(trainer.normalization_mean, float(expected.mean()), places=4)
                    self.assertAlmostEqual(trainer.normalization_std, float(expected.std()), places=4)
                    self.assertEqual([stack.shape[0] for stack in dataset.noisy_imgs], [7] * count)
                    if volume:
                        self.assertEqual(dataset.normalization_group_ids, [0, 0])
                        self.assertEqual(dataset.normalization_group_stats[0]["voxel_count"], interior.size)
                    self.assert_all_patches_use(dataset, interior)
                    self.assertEqual(path.read_bytes(), before)

    def test_changing_only_excluded_pixels_cannot_change_training_statistics(self):
        for resident in (False, True):
            statistics = []
            for index, edge_value in enumerate((1e4, 1e12)):
                image = self.image(volume=True, edge_value=edge_value)
                path = self.write(f"edge-{resident}-{index}.tif", image)
                dataset = self.load(self.trainer(path, resident=resident))
                statistics.append((dataset.mean_images.clone(), dataset.std_images.clone()))
            torch.testing.assert_close(statistics[0][0], statistics[1][0], rtol=0, atol=0)
            torch.testing.assert_close(statistics[0][1], statistics[1][1], rtol=0, atol=0)

    def test_generalized_normalization_uses_only_the_interior_of_each_tiff(self):
        for resident in (False, True):
            directory = self.root / f"global-{resident}"
            arrays = [self.image(volume=True), self.image(volume=True) + 300]
            for index, image in enumerate(arrays):
                self.write(f"{directory.name}/input-{index}.tif", image)
            trainer = self.trainer(directory, resident=resident, train_gen=True)
            dataset = self.load(trainer)
            interior = torch.from_numpy(np.concatenate([image[1:-1].reshape(-1) for image in arrays]))
            self.assertTrue(dataset.normalize_across_images)
            self.assertEqual(dataset.dataset_count, interior.numel())
            torch.testing.assert_close(dataset.dataset_mean.float(), interior.mean())
            torch.testing.assert_close(dataset.dataset_std.float(), interior.std())
            self.assertTrue(all(stack.shape[0] == 7 for stack in dataset.noisy_imgs))

    def test_offload_restores_only_interior_without_retrimming_or_duplicate_diagnostics(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                path = self.write(f"reload-{cached}.tif", self.image(volume=True))
                before = path.read_bytes()
                trainer = self.trainer(path, epoch_end_callback=lambda *_: None,
                                       training_cache_dir=str(self.root / f"cache-{cached}"))
                if not cached:
                    trainer._resolve_training_cache_base = mock.Mock(return_value=None)
                output = io.StringIO()
                with redirect_stdout(output):
                    trainer.generate_dataloader()
                    expected = [stack.clone() for stack in trainer.dataloader.dataset.noisy_imgs]
                    self.assertTrue(trainer._suspend_training_data_for_validation(True))
                    self.assertTrue(trainer._restore_training_data_after_validation())
                actual = trainer.dataloader.dataset.noisy_imgs
                for expected_stack, actual_stack in zip(expected, actual):
                    torch.testing.assert_close(actual_stack, expected_stack, rtol=0, atol=0)
                    self.assertEqual(actual_stack.shape[0], 7)
                self.assertEqual(output.getvalue().count("Training frame exclusion for"), 1)
                self.assertEqual(path.read_bytes(), before)
                trainer._remove_training_cache()

    def test_short_temporal_stacks_fail_before_creating_a_dataset(self):
        for volume in (False, True):
            path = self.write(f"short-{volume}.tif", self.image(volume=volume, frames=4))
            trainer = self.trainer(path)
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "at least 5 temporal frames.*has 4 frames"):
                    trainer.generate_dataloader()
            self.assertFalse(hasattr(trainer, "dataloader"))

    def test_zero_margin_keeps_all_frames_and_default_false_preserves_legacy_training(self):
        image = self.image(frames=3)
        path = self.write("zero-margin.tif", image)
        for patch, exclude in ((1, True), (3, None)):
            with self.subTest(patch=patch, exclude=exclude):
                trainer = self.trainer(path, patch=patch, exclude=exclude)
                dataset = self.load(trainer)
                self.assertEqual(dataset.noisy_imgs[0].shape[0], 3)
                self.assertEqual(trainer.exclude_training_edge_frames, exclude is True)
                torch.testing.assert_close(dataset.mean_images[0], torch.from_numpy(image).mean())
                if exclude is None:
                    self.assertIsNone(trainer.normalization_mean)
                    self.assertIsNone(trainer.normalization_std)

    def test_exclusion_rejects_static_3d_nonboolean_options_and_invalid_temporal_patch(self):
        path = self.write("validation.tif", self.image())
        with self.assertRaisesRegex(ValueError, "requires temporal"):
            self.trainer(path, dimensionality="3D")
        for invalid in (0, -1, 3.0, True):
            with self.subTest(temporal_patch=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.trainer(path, patch=invalid)
        with self.assertRaisesRegex(TypeError, "exclude_training_edge_frames must be True or False"):
            self.trainer(path, exclude=1)

    def test_2d_time_series_also_crops_the_temporal_axis(self):
        image = self.image()
        path = self.write("two-dimensional-time.tif", image)
        dataset = self.load(self.trainer(path, dimensionality="2+1D"))
        self.assert_all_patches_use(dataset, image[1:-1])


if __name__ == "__main__":
    unittest.main()
