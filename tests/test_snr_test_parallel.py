"""CUDA-free tests for the staged multi-GPU SNR sweep scheduler."""

from __future__ import annotations

import builtins
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from unittest import mock

import numpy as np


def load_snr_test_module():
    module_path = Path(__file__).resolve().parents[1] / "notebooks" / "snr_test.py"
    spec = importlib.util.spec_from_file_location(
        "_support_snr_test_parallel_under_test",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SNR = load_snr_test_module()


class DeviceSelectionTests(unittest.TestCase):
    def test_repository_root_is_added_for_direct_script_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory).resolve()
            path_without_repository = [
                entry for entry in sys.path if entry != str(repository)
            ]
            with mock.patch.object(SNR.sys, "path", path_without_repository):
                SNR._ensure_repository_importable(repository)
                self.assertEqual(sys.path[0], str(repository))
                SNR._ensure_repository_importable(repository)
                self.assertEqual(sys.path.count(str(repository)), 1)

    def test_visible_device_tokens_preserve_slurm_tokens_and_whitespace(self) -> None:
        self.assertEqual(
            SNR.visible_device_tokens(" 3, GPU-acde , MIG-abcd/1/0 "),
            ("3", "GPU-acde", "MIG-abcd/1/0"),
        )
        self.assertEqual(SNR.visible_device_tokens("-1"), ())
        self.assertEqual(SNR.visible_device_tokens("  "), ())
        self.assertEqual(
            SNR.visible_device_tokens(None, device_count=3),
            ("0", "1", "2"),
        )

    def test_visible_device_tokens_reject_malformed_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty device token"):
            SNR.visible_device_tokens("0,,2")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SNR.visible_device_tokens("GPU-a,GPU-a")
        with self.assertRaisesRegex(ValueError, "device_count is required"):
            SNR.visible_device_tokens(None)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            SNR.visible_device_tokens(None, device_count=-1)

    def test_select_gpu_slots_maps_logical_ids_to_inherited_tokens(self) -> None:
        tokens = ("GPU-first", "GPU-second", "MIG-third/0/1")
        self.assertEqual(
            SNR.select_gpu_slots(None, tokens),
            (
                (0, "GPU-first"),
                (1, "GPU-second"),
                (2, "MIG-third/0/1"),
            ),
        )
        self.assertEqual(
            SNR.select_gpu_slots([2, 0], tokens),
            ((2, "MIG-third/0/1"), (0, "GPU-first")),
        )

    def test_select_gpu_slots_rejects_invalid_requests(self) -> None:
        invalid_requests = (
            ([], ValueError, "at least one"),
            ([0, 0], ValueError, "unique"),
            ([-1], ValueError, "zero or greater"),
            ([True], TypeError, "integers"),
            (["0"], TypeError, "integers"),
            ([2], ValueError, "unavailable"),
        )
        for requested, exception_type, message in invalid_requests:
            with self.subTest(requested=requested):
                with self.assertRaisesRegex(exception_type, message):
                    SNR.select_gpu_slots(requested, ("GPU-a", "GPU-b"))
        with self.assertRaisesRegex(ValueError, "no CUDA GPUs"):
            SNR.select_gpu_slots(None, ())

    def test_gpu_cli_accepts_multiple_logical_ids(self) -> None:
        args = SNR.build_parser().parse_args(
            [
                "--gpus",
                "2",
                "0",
                "--photons-per-unit",
                "0.0037",
                "--photon-frame-start",
                "60",
                "--photon-frame-end-margin",
                "60",
                "--photon-max-pairs",
                "40",
                "--photon-zero-level",
                "-12.5",
                "--repository",
                "/data/user/SUPPORT",
                "--input-directory",
                "/data/user/snr-input",
                "--output-directory",
                "/data/user/snr-output",
                "--parameters",
                "/data/user/parameters.json",
                "--dimensionality",
                "3+1D",
                "--power-percent",
                "100",
                "50",
                "--n-cpu",
                "6",
                "--no-train-pin-memory",
                "--no-test-pin-memory",
            ]
        )
        self.assertEqual(args.gpus, [2, 0])
        self.assertEqual(args.photons_per_unit, 0.0037)
        self.assertEqual(args.photon_frame_start, 60)
        self.assertEqual(args.photon_frame_end_margin, 60)
        self.assertEqual(args.photon_max_pairs, 40)
        self.assertEqual(args.photon_zero_level, -12.5)
        self.assertEqual(args.repository, Path("/data/user/SUPPORT"))
        self.assertEqual(args.input_directory, Path("/data/user/snr-input"))
        self.assertEqual(args.output_directory, Path("/data/user/snr-output"))
        self.assertEqual(args.parameters, Path("/data/user/parameters.json"))
        self.assertEqual(args.dimensionality, "3+1D")
        self.assertEqual(args.power_percent, [100, 50])
        self.assertEqual(args.n_cpu, 6)
        self.assertIs(args.train_pin_memory, False)
        self.assertIs(args.test_pin_memory, False)
        self.assertFalse(args.use_estimated_photon_pedestal)
        default_args = SNR.build_parser().parse_args([])
        self.assertIsNone(default_args.train_pin_memory)
        self.assertIsNone(default_args.test_pin_memory)
        estimated_args = SNR.build_parser().parse_args(
            ["--use-estimated-photon-pedestal"]
        )
        self.assertTrue(estimated_args.use_estimated_photon_pedestal)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            SNR.build_parser().parse_args(["--gpus", "-1"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            SNR.build_parser().parse_args(["--photons-per-unit", "0"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            SNR.build_parser().parse_args(["--photon-zero-level", "nan"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            SNR.build_parser().parse_args(["--photon-frame-end-margin", "-1"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            SNR.build_parser().parse_args(["--power-percent", "0"])

    def test_main_uses_every_input_tiff_and_sweep_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = root / "SUPPORT"
            input_root = root / "input"
            output_root = root / "output"
            parameters_path = root / "parameters.json"
            (repository / "notebooks").mkdir(parents=True)
            input_root.mkdir()
            for filename in ("movie_3.tif", "movie_1.tif", "movie_2.tiff"):
                (input_root / filename).write_bytes(b"placeholder")
            (input_root / "ignored.txt").write_bytes(b"not a TIFF")
            parameters_path.write_text("{}", encoding="utf-8")

            def run_staged(original_tasks, power_tasks, selected_gpus, configuration):
                self.assertEqual(
                    [task.source.name for task in original_tasks],
                    ["movie_1.tif", "movie_2.tiff", "movie_3.tif"],
                )
                self.assertEqual(
                    [(task.source.name, task.power) for task in power_tasks],
                    [
                        ("movie_1.tif", 100),
                        ("movie_1.tif", 50),
                        ("movie_2.tiff", 100),
                        ("movie_2.tiff", 50),
                        ("movie_3.tif", 100),
                        ("movie_3.tif", 50),
                    ],
                )
                self.assertEqual(selected_gpus, ((0, "GPU-a"),))
                self.assertEqual(configuration.repository, repository.resolve())
                self.assertEqual(configuration.input_root, input_root.resolve())
                self.assertEqual(configuration.output_root, output_root.resolve())
                self.assertEqual(configuration.parameters_path, parameters_path.resolve())
                self.assertEqual(configuration.dimensionality, "2+1D")
                self.assertEqual(configuration.n_cpu, 3)
                self.assertIs(configuration.train_pin_memory, False)
                self.assertIs(configuration.test_pin_memory, False)
                self.assertEqual(configuration.photons_per_unit, 0.01)
                self.assertEqual(configuration.photon_estimator_frame_start, 60)
                self.assertEqual(
                    configuration.photon_estimator_frame_end_margin, 60
                )
                return tuple(
                    SNR.TaskResult(
                        task.index,
                        0,
                        task.source,
                        task.power,
                        True,
                        output_root / f"clean-{task.index}.tif",
                    )
                    for task in power_tasks
                )

            arguments = [
                "--repository",
                str(repository),
                "--input-directory",
                str(input_root),
                "--output-directory",
                str(output_root),
                "--parameters",
                str(parameters_path),
                "--dimensionality",
                "2+1D",
                "--power-percent",
                "100",
                "50",
                "--n-cpu",
                "3",
                "--gpus",
                "0",
                "--photons-per-unit",
                "0.01",
                "--photon-frame-start",
                "60",
                "--photon-frame-end-margin",
                "60",
                "--no-train-pin-memory",
                "--no-test-pin-memory",
            ]
            with (
                mock.patch.object(
                    SNR, "_discover_visible_devices", return_value=("GPU-a",)
                ),
                mock.patch.object(
                    SNR, "run_staged_sweep", side_effect=run_staged
                ) as staged,
                redirect_stdout(io.StringIO()),
            ):
                exit_code = SNR.main(arguments)

            self.assertEqual(exit_code, 0)
            staged.assert_called_once()


class SweepPlanningTests(unittest.TestCase):
    def test_build_original_tasks_creates_one_distinct_reference_per_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = [root / "b.tiff", root / "a.tif"]

            tasks = SNR.build_original_tasks(paths, root / "outputs")

            self.assertEqual([task.index for task in tasks], [0, 1])
            self.assertEqual([task.source.name for task in tasks], ["b.tiff", "a.tif"])
            self.assertEqual(tasks[0].exp_name, "b_original")
            self.assertEqual(
                tasks[0].output_path,
                (
                    root
                    / "outputs"
                    / "b"
                    / "b_original"
                    / "b_original_SPRT.tif"
                ).resolve(),
            )

    def test_build_sweep_tasks_is_deterministic_and_preserves_power_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = [root / "b.tiff", root / "a.tif"]
            tasks = SNR.build_sweep_tasks(paths, [100, 40], root / "outputs")

            self.assertEqual([task.index for task in tasks], [0, 1, 2, 3])
            self.assertEqual(
                [(task.source.name, task.power) for task in tasks],
                [("b.tiff", 100), ("b.tiff", 40), ("a.tif", 100), ("a.tif", 40)],
            )
            self.assertEqual(tasks[0].exp_name, "b_original_100")
            self.assertEqual(
                tasks[0].reference_path,
                (
                    root
                    / "outputs"
                    / "b"
                    / "b_original"
                    / "b_original_SPRT.tif"
                ).resolve(),
            )
            self.assertEqual(
                tasks[0].simulated_path,
                (
                    root
                    / "outputs"
                    / "b"
                    / "b_original_100"
                    / "b_original_100.tif"
                ).resolve(),
            )

    def test_build_sweep_tasks_rejects_duplicate_destinations_and_powers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "original-image output collision"):
                SNR.build_original_tasks(
                    [root / "first" / "same.tif", root / "second" / "same.tiff"],
                    root / "outputs",
                )
            with self.assertRaisesRegex(ValueError, "output collision"):
                SNR.build_sweep_tasks(
                    [root / "first" / "same.tif", root / "second" / "same.tiff"],
                    [50],
                    root / "outputs",
                )
            with self.assertRaisesRegex(ValueError, "must be unique"):
                SNR.build_sweep_tasks([root / "a.tif"], [50, 50], root / "outputs")

    def test_build_sweep_tasks_validates_power_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "a.tif"
            for powers, exception_type, message in (
                ([], ValueError, "at least one"),
                ([0], ValueError, "between 1 and 100"),
                ([101], ValueError, "between 1 and 100"),
                ([True], TypeError, "integers"),
                ([50.0], TypeError, "integers"),
            ):
                with self.subTest(powers=powers):
                    with self.assertRaisesRegex(exception_type, message):
                        SNR.build_sweep_tasks([path], powers, root / "outputs")

    def test_partition_tasks_balances_serial_gpu_lanes_without_loss(self) -> None:
        tasks = tuple(
            SNR.SweepTask(
                index=index,
                source=Path(f"source-{index}.tif"),
                reference_path=Path(f"original-{index}.tif"),
                power=100 - index,
                exp_name=f"job-{index}",
                out_dir=Path("outputs"),
                simulated_path=Path(f"outputs/job-{index}.tif"),
            )
            for index in range(7)
        )

        assignments = SNR.partition_tasks(tasks, [4, 1, 9])

        self.assertEqual(
            {slot: [task.index for task in lane] for slot, lane in assignments.items()},
            {4: [0, 3, 6], 1: [1, 4], 9: [2, 5]},
        )
        flattened = [task.index for lane in assignments.values() for task in lane]
        self.assertEqual(sorted(flattened), list(range(7)))
        with self.assertRaisesRegex(ValueError, "at least one"):
            SNR.partition_tasks(tasks, [])


class RecordingQueue:
    def __init__(self, initial=()) -> None:
        self.items = list(initial)
        self.closed = False
        self.joined = False

    def put(self, value) -> None:
        self.items.append(value)

    def get(self, timeout=None):
        if not self.items:
            raise Empty
        return self.items.pop(0)

    def close(self) -> None:
        self.closed = True

    def join_thread(self) -> None:
        self.joined = True


class ResourceCleanupTests(unittest.TestCase):
    def test_release_failure_still_attempts_log_gc_and_cuda_cleanup(self) -> None:
        events = []

        class Handler:
            def flush(self):
                events.append("flush")

            def close(self):
                events.append("close")

        handler = Handler()

        class Logger:
            handlers = [handler]

            def removeHandler(self, removed):
                self.handlers.remove(removed)
                events.append("remove_handler")

        manager = SimpleNamespace(logger=Logger(), obj=object())

        class Resource:
            log = manager

            @staticmethod
            def release_resources():
                events.append("release")
                raise RuntimeError("release failed")

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(empty_cache=lambda: events.append("empty_cache"))
        )

        with (
            mock.patch.object(SNR.gc, "collect", side_effect=lambda: events.append("gc")),
            self.assertRaisesRegex(
                SNR.ResourceCleanupError,
                "release_resources: RuntimeError: release failed",
            ),
        ):
            SNR._release_job_resource(Resource(), fake_torch)

        self.assertEqual(
            events,
            ["release", "flush", "close", "remove_handler", "gc", "empty_cache"],
        )
        self.assertEqual(manager.logger.handlers, [])
        self.assertIsNone(manager.obj)


class WorkerBehaviorTests(unittest.TestCase):
    @staticmethod
    def _configuration(root: Path, rm_tmp: bool = True):
        return SNR.SweepConfiguration(
            repository=root,
            input_root=root,
            parameters_path=root / "parameters.json",
            output_root=root / "outputs",
            dimensionality="3+1D",
            n_cpu=2,
            rm_tmp=rm_tmp,
            rm_dir=False,
            checkpoints=False,
            resume=True,
            rm_intermediaries=True,
            photons_per_unit=0.01,
        )

    @staticmethod
    def _tasks(root: Path, count: int = 2):
        return tuple(
            SNR.SweepTask(
                index=index,
                source=root / f"source-{index}.tif",
                reference_path=root / f"original-{index}.tif",
                power=100 - index * 10,
                exp_name=f"source-{index}_{100 - index * 10:03d}",
                out_dir=root / "outputs" / f"source-{index}",
                simulated_path=root / "outputs" / f"simulated-{index}.tif",
            )
            for index in range(count)
        )

    def test_gpu_worker_binds_before_torch_import_and_runs_lane_serially(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tasks = self._tasks(root, count=3)
            configuration = self._configuration(root)
            queue = RecordingQueue()
            events = []

            cuda = SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                set_device=lambda device: events.append(("set_device", device)),
                get_device_name=lambda device: "Fake GPU",
            )
            fake_torch = SimpleNamespace(cuda=cuda)
            real_import = builtins.__import__

            def importing(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "torch":
                    events.append(("import_torch", os.environ.get("CUDA_VISIBLE_DEVICES")))
                    return fake_torch
                return real_import(name, globals, locals, fromlist, level)

            def run_task(task, active_configuration, torch_module, gpu_slot):
                self.assertIs(active_configuration, configuration)
                self.assertIs(torch_module, fake_torch)
                self.assertEqual(gpu_slot, 2)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-token")
                events.append(("task", task.index))
                return root / f"clean-{task.index}.tif"

            with (
                mock.patch("builtins.__import__", side_effect=importing),
                mock.patch.object(SNR, "_run_task", side_effect=run_task),
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "old-token"}),
            ):
                SNR._gpu_worker(2, "GPU-token", tasks, configuration, queue)

            self.assertEqual(events[0], ("import_torch", "GPU-token"))
            self.assertEqual(events[1], ("set_device", 0))
            self.assertEqual(events[2:], [("task", 0), ("task", 1), ("task", 2)])
            self.assertEqual([result.task_index for result in queue.items], [0, 1, 2])
            self.assertTrue(all(result.success for result in queue.items))
            self.assertTrue(all(result.gpu_slot == 2 for result in queue.items))

    def test_gpu_worker_reports_every_preflight_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tasks = self._tasks(root)
            queue = RecordingQueue()
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
            )
            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.dict(os.environ, {}, clear=False),
                self.assertRaisesRegex(RuntimeError, "could not isolate"),
            ):
                SNR._gpu_worker(
                    1,
                    "GPU-missing",
                    tasks,
                    self._configuration(root),
                    queue,
                )

            self.assertEqual([result.task_index for result in queue.items], [0, 1])
            self.assertTrue(all(not result.success for result in queue.items))
            self.assertTrue(all("could not isolate" in result.error for result in queue.items))

    def test_gpu_worker_continues_after_a_task_failure_then_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tasks = self._tasks(root)
            queue = RecordingQueue()
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    device_count=lambda: 1,
                    set_device=lambda device: None,
                    get_device_name=lambda device: "Fake GPU",
                )
            )

            def run_task(task, *args):
                if task.index == 0:
                    raise RuntimeError("job failed")
                return root / "clean.tif"

            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(SNR, "_run_task", side_effect=run_task) as runner,
                redirect_stderr(io.StringIO()),
                self.assertRaisesRegex(SystemExit, "1"),
            ):
                SNR._gpu_worker(
                    0,
                    "0",
                    tasks,
                    self._configuration(root),
                    queue,
                )

            self.assertEqual(runner.call_count, 2)
            self.assertEqual([result.success for result in queue.items], [False, True])
            self.assertIn("job failed", queue.items[0].error)

    def test_gpu_worker_stops_lane_after_cleanup_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tasks = self._tasks(root, count=3)
            queue = RecordingQueue()
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    device_count=lambda: 1,
                    set_device=lambda device: None,
                    get_device_name=lambda device: "Fake GPU",
                )
            )

            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(
                    SNR,
                    "_run_task",
                    side_effect=SNR.ResourceCleanupError("cleanup failed"),
                ) as runner,
                redirect_stderr(io.StringIO()),
                self.assertRaisesRegex(SystemExit, "1"),
            ):
                SNR._gpu_worker(
                    5,
                    "GPU-five",
                    tasks,
                    self._configuration(root),
                    queue,
                )

            runner.assert_called_once_with(tasks[0], mock.ANY, fake_torch, 5)
            self.assertEqual([result.task_index for result in queue.items], [0, 1, 2])
            self.assertTrue(all(not result.success for result in queue.items))
            self.assertIn("cleanup failed", queue.items[0].error)
            self.assertTrue(all("was not started" in result.error for result in queue.items[1:]))


class StagedWorkflowTests(unittest.TestCase):
    def test_original_job_runs_support_directly_on_raw_and_keeps_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            configuration = WorkerBehaviorTests._configuration(root)
            task = SNR.OriginalTask(
                index=0,
                source=root / "raw.tif",
                exp_name="raw_original",
                out_dir=root / "outputs" / "raw",
                output_path=(
                    root
                    / "outputs"
                    / "raw"
                    / "raw_original"
                    / "raw_original_SPRT.tif"
                ),
            )
            fake_torch = object()

            with (
                mock.patch.object(
                    SNR,
                    "_run_support",
                    return_value=task.output_path,
                ) as run_support,
                mock.patch.object(SNR, "_write_simulated_tiff") as writer,
            ):
                result = SNR._run_original_task(
                    task,
                    configuration,
                    fake_torch,
                    gpu_slot=3,
                )

            self.assertEqual(result, task.output_path)
            writer.assert_not_called()
            run_support.assert_called_once_with(
                input_path=task.source,
                exp_name=task.exp_name,
                out_dir=task.out_dir,
                output_path=task.output_path,
                configuration=configuration,
                torch=fake_torch,
            )

    def test_power_job_simulates_from_reference_never_raw_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            configuration = WorkerBehaviorTests._configuration(root)
            task = SNR.SweepTask(
                index=0,
                source=root / "raw.tif",
                reference_path=root / "fresh-original.tif",
                power=40,
                exp_name="raw_original_040",
                out_dir=root / "outputs" / "raw",
                simulated_path=(
                    root
                    / "outputs"
                    / "raw"
                    / "raw_original_040"
                    / "raw_original_040.tif"
                ),
                photons_per_unit=0.0037,
                zero_photon_level=-12.5,
            )
            reference = np.full((2, 1, 2, 2), 7, dtype=np.float32)
            simulated = np.full_like(reference, 3)
            salt_inputs = []

            class FakeTiffReader:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return None

                @staticmethod
                def asarray():
                    return reference

            salt_module = types.ModuleType("utils.salt")

            def simulate(stack, ratio, **kwargs):
                salt_inputs.append((np.array(stack, copy=True), ratio, kwargs))
                return simulated

            salt_module.simulate_lower_power = simulate

            def publish(path, stack, axes):
                self.assertEqual(path, task.simulated_path)
                np.testing.assert_array_equal(stack, simulated)
                self.assertEqual(axes, "TZYX")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"simulated")

            final_output = task.out_dir / task.exp_name / f"{task.exp_name}_SPRT.tif"
            with (
                mock.patch.dict(sys.modules, {"utils.salt": salt_module}),
                mock.patch.object(
                    SNR.tifffile,
                    "TiffFile",
                    return_value=FakeTiffReader(),
                ) as reader,
                mock.patch.object(
                    SNR,
                    "_write_simulated_tiff",
                    side_effect=publish,
                ),
                mock.patch.object(
                    SNR,
                    "_run_support",
                    return_value=final_output,
                ) as run_support,
            ):
                result = SNR._run_task(task, configuration, object(), gpu_slot=1)

            self.assertEqual(result, final_output)
            reader.assert_called_once_with(task.reference_path)
            self.assertEqual(len(salt_inputs), 1)
            np.testing.assert_array_equal(salt_inputs[0][0], reference)
            self.assertEqual(salt_inputs[0][1], 0.4)
            self.assertEqual(salt_inputs[0][2]["gamma"], 2.0)
            self.assertEqual(salt_inputs[0][2]["photons_per_unit"], 0.0037)
            self.assertEqual(salt_inputs[0][2]["zero_photon_level"], -12.5)
            self.assertFalse(task.simulated_path.exists())
            self.assertEqual(
                run_support.call_args.kwargs["input_path"],
                task.simulated_path,
            )

    def test_staged_sweep_uses_fresh_outputs_and_skips_failed_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources = (root / "good.tif", root / "bad.tif")
            original_tasks = SNR.build_original_tasks(sources, root / "outputs")
            power_tasks = SNR.build_sweep_tasks(
                sources,
                [100, 50],
                root / "outputs",
            )
            fresh_reference = root / "fresh-good-original.tif"
            original_results = (
                SNR.TaskResult(
                    0,
                    0,
                    sources[0].resolve(),
                    None,
                    True,
                    fresh_reference,
                    stage="original",
                ),
                SNR.TaskResult(
                    1,
                    1,
                    sources[1].resolve(),
                    None,
                    False,
                    error="baseline exploded",
                    stage="original",
                ),
            )
            configuration = SNR.replace(
                WorkerBehaviorTests._configuration(root),
                photon_zero_level=6.25,
            )

            def run_power(runnable, *_args):
                self.assertEqual(len(runnable), 2)
                self.assertTrue(all(task.source == sources[0].resolve() for task in runnable))
                self.assertTrue(all(task.reference_path == fresh_reference for task in runnable))
                self.assertTrue(all(task.photons_per_unit == 0.01 for task in runnable))
                self.assertTrue(all(task.zero_photon_level == 6.25 for task in runnable))
                return tuple(
                    SNR.TaskResult(
                        task.index,
                        0,
                        task.source,
                        task.power,
                        True,
                        root / f"clean-{task.index}.tif",
                    )
                    for task in runnable
                )

            with (
                mock.patch.object(
                    SNR,
                    "run_parallel_originals",
                    return_value=original_results,
                ) as run_originals,
                mock.patch.object(
                    SNR,
                    "run_parallel_sweep",
                    side_effect=run_power,
                ) as run_sweep,
                mock.patch.object(
                    SNR,
                    "_estimate_source_photons_per_unit",
                ) as estimate_photons,
            ):
                results = SNR.run_staged_sweep(
                    original_tasks,
                    power_tasks,
                    ((0, "GPU-a"), (1, "GPU-b")),
                    configuration,
                )

            run_originals.assert_called_once()
            run_sweep.assert_called_once()
            estimate_photons.assert_not_called()
            self.assertEqual([result.task_index for result in results], [0, 1, 2, 3])
            self.assertEqual([result.success for result in results], [True, True, False, False])
            self.assertTrue(all("not started" in result.error for result in results[2:]))
            self.assertTrue(all("baseline exploded" in result.error for result in results[2:]))

    def test_staged_sweep_estimates_once_and_reuses_value_for_all_powers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = (root / "raw.tif").resolve()
            original_tasks = SNR.build_original_tasks((source,), root / "outputs")
            power_tasks = SNR.build_sweep_tasks(
                (source,), (100, 70, 40), root / "outputs"
            )
            fresh_reference = (root / "fresh-original.tif").resolve()
            original_results = (
                SNR.TaskResult(
                    0,
                    0,
                    source,
                    None,
                    True,
                    fresh_reference,
                    stage="original",
                ),
            )
            configuration = SNR.replace(
                WorkerBehaviorTests._configuration(root),
                photons_per_unit=None,
                use_estimated_photon_pedestal=True,
            )

            def run_power(runnable, *_args):
                self.assertEqual(len(runnable), 3)
                self.assertTrue(
                    all(task.reference_path == fresh_reference for task in runnable)
                )
                self.assertTrue(
                    all(task.photons_per_unit == 0.0037 for task in runnable)
                )
                self.assertTrue(
                    all(task.zero_photon_level == 12.5 for task in runnable)
                )
                return tuple(
                    SNR.TaskResult(
                        task.index,
                        0,
                        task.source,
                        task.power,
                        True,
                        root / f"clean-{task.index}.tif",
                    )
                    for task in runnable
                )

            with (
                mock.patch.object(
                    SNR,
                    "run_parallel_originals",
                    return_value=original_results,
                ),
                mock.patch.object(
                    SNR,
                    "_estimate_source_photons_per_unit",
                    return_value=SimpleNamespace(
                        photons_per_unit=0.0037,
                        effective_support_pedestal=12.5,
                    ),
                ) as estimate_photons,
                mock.patch.object(
                    SNR,
                    "run_parallel_sweep",
                    side_effect=run_power,
                ),
            ):
                results = SNR.run_staged_sweep(
                    original_tasks,
                    power_tasks,
                    ((0, "GPU-a"),),
                    configuration,
                )

            estimate_photons.assert_called_once_with(
                source, fresh_reference, configuration
            )
            self.assertTrue(all(result.success for result in results))

    def test_staged_sweep_keeps_automatic_estimates_isolated_per_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources = tuple(
                (root / name).resolve() for name in ("first.tif", "second.tif")
            )
            original_tasks = SNR.build_original_tasks(sources, root / "outputs")
            power_tasks = SNR.build_sweep_tasks(
                sources, (100, 40), root / "outputs"
            )
            references = {
                sources[0]: (root / "first-reference.tif").resolve(),
                sources[1]: (root / "second-reference.tif").resolve(),
            }
            original_results = tuple(
                SNR.TaskResult(
                    task.index,
                    0,
                    task.source,
                    None,
                    True,
                    references[task.source],
                    stage="original",
                )
                for task in original_tasks
            )
            expected_photons = {sources[0]: 0.0011, sources[1]: 0.0022}
            configuration = SNR.replace(
                WorkerBehaviorTests._configuration(root),
                photons_per_unit=None,
            )

            def estimate(source, reference, _configuration):
                self.assertEqual(reference, references[source])
                return SimpleNamespace(
                    photons_per_unit=expected_photons[source],
                    effective_support_pedestal=17.0,
                )

            def run_power(runnable, *_args):
                self.assertEqual(len(runnable), 4)
                for task in runnable:
                    self.assertEqual(
                        task.photons_per_unit, expected_photons[task.source]
                    )
                    self.assertEqual(task.reference_path, references[task.source])
                return tuple(
                    SNR.TaskResult(
                        task.index,
                        0,
                        task.source,
                        task.power,
                        True,
                        root / f"clean-{task.index}.tif",
                    )
                    for task in runnable
                )

            with (
                mock.patch.object(
                    SNR,
                    "run_parallel_originals",
                    return_value=original_results,
                ),
                mock.patch.object(
                    SNR,
                    "_estimate_source_photons_per_unit",
                    side_effect=estimate,
                ) as estimate_photons,
                mock.patch.object(
                    SNR,
                    "run_parallel_sweep",
                    side_effect=run_power,
                ),
                redirect_stdout(io.StringIO()),
            ):
                results = SNR.run_staged_sweep(
                    original_tasks,
                    power_tasks,
                    ((0, "GPU-a"),),
                    configuration,
                )

            self.assertEqual(estimate_photons.call_count, 2)
            self.assertEqual(
                {call.args[0] for call in estimate_photons.call_args_list},
                set(sources),
            )
            self.assertTrue(all(result.success for result in results))

    def test_tiff_calibration_helper_passes_the_default_central_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = np.zeros((200, 4, 4), dtype=np.float32)
            support = np.ones_like(raw)
            captured = {}
            expected = SimpleNamespace(
                photons_per_unit=0.001,
                photons_per_raw_unit=0.002,
                support_gain=2.0,
                effective_support_pedestal=0.0,
                linear_r_squared=1.0,
                delta_aicc=-1.0,
                scale_r_squared=1.0,
                scale_delta_aicc=-1.0,
                quiet_photons_per_unit=None,
                quiet_linear_r_squared=None,
                warnings=(),
            )
            salt_module = types.ModuleType("utils.salt")

            def estimate(raw_stack, support_stack, **kwargs):
                self.assertIs(raw_stack, raw)
                self.assertIs(support_stack, support)
                captured.update(kwargs)
                return expected

            salt_module.estimate_photons_per_unit = estimate
            configuration = SNR.replace(
                WorkerBehaviorTests._configuration(root),
                photons_per_unit=None,
            )

            with (
                mock.patch.dict(sys.modules, {"utils.salt": salt_module}),
                mock.patch.object(
                    SNR,
                    "_read_time_first_tiff",
                    side_effect=(raw, support),
                ),
                redirect_stdout(io.StringIO()),
            ):
                result = SNR._estimate_source_photons_per_unit(
                    root / "raw.tif",
                    root / "support.tif",
                    configuration,
                )

            self.assertIs(result, expected)
            self.assertEqual(captured["frame_start"], 60)
            self.assertEqual(captured["frame_stop"], 140)
            self.assertEqual(captured["max_pairs"], 50)

    def test_tiff_calibration_helper_rejects_noninteger_end_margin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stack = np.zeros((200, 4, 4), dtype=np.float32)
            salt_module = types.ModuleType("utils.salt")
            salt_module.estimate_photons_per_unit = mock.Mock()

            for margin in (True, 60.5, np.float64(60.0)):
                configuration = SNR.replace(
                    WorkerBehaviorTests._configuration(root),
                    photons_per_unit=None,
                    photon_estimator_frame_end_margin=margin,
                )
                with (
                    self.subTest(margin=margin),
                    mock.patch.dict(sys.modules, {"utils.salt": salt_module}),
                    mock.patch.object(
                        SNR,
                        "_read_time_first_tiff",
                        side_effect=(stack, stack),
                    ),
                    redirect_stdout(io.StringIO()),
                    self.assertRaisesRegex(ValueError, "nonnegative integer"),
                ):
                    SNR._estimate_source_photons_per_unit(
                        root / "raw.tif",
                        root / "support.tif",
                        configuration,
                    )

            salt_module.estimate_photons_per_unit.assert_not_called()

    def test_tiff_calibration_helper_returns_support_unit_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rng = np.random.default_rng(3)
            yy, xx = np.mgrid[:64, :64]
            clean_frame = (200.0 + 12.0 * xx + 18.0 * yy).astype(np.float32)
            clean = np.broadcast_to(clean_frame, (60, 64, 64))
            raw = rng.poisson(clean * 0.02).astype(np.float32) / 0.02
            support = np.broadcast_to(37.0 + 2.5 * clean_frame, raw.shape).copy()
            raw_path = root / "raw.tif"
            support_path = root / "support.tif"
            SNR.tifffile.imwrite(
                raw_path, raw, imagej=True, metadata={"axes": "TYX"}
            )
            SNR.tifffile.imwrite(
                support_path,
                support,
                imagej=True,
                metadata={"axes": "TYX"},
            )
            configuration = SNR.replace(
                WorkerBehaviorTests._configuration(root),
                photons_per_unit=None,
                photon_estimator_frame_start=0,
                photon_estimator_frame_end_margin=0,
                photon_estimator_max_pairs=30,
                photon_estimator_sample_count=4_096,
            )

            result = SNR._estimate_source_photons_per_unit(
                raw_path, support_path, configuration
            )

            self.assertAlmostEqual(
                result.photons_per_unit / (0.02 / 2.5), 1.0, delta=0.10
            )
            self.assertTrue(np.isfinite(result.effective_support_pedestal))

    def test_pedestal_configuration_conflicts_fail_before_original_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = WorkerBehaviorTests._configuration(root)
            cases = (
                SNR.replace(base, use_estimated_photon_pedestal=True),
                SNR.replace(
                    base,
                    photons_per_unit=None,
                    photon_zero_level=3.0,
                    use_estimated_photon_pedestal=True,
                ),
            )
            for configuration in cases:
                with (
                    self.subTest(configuration=configuration),
                    mock.patch.object(SNR, "run_parallel_originals") as runner,
                    self.assertRaisesRegex(ValueError, "estimated|either"),
                ):
                    SNR.run_staged_sweep((), (), (), configuration)
                runner.assert_not_called()


class RunTaskCleanupTests(unittest.TestCase):
    @staticmethod
    def _parameters():
        return {
            "3+1D": {
                "train": {
                    "patch_size": [3, 8, 8],
                    "depth": 3,
                    "bc_channels": 4,
                    "obo_channels": [4, 2],
                    "ll_channels": [4, 2],
                    "unet_channels": [4, 8],
                    "bs_size": [3, 3],
                    "bp": False,
                    "rnd_seed": 17,
                    "pin_memory": True,
                },
                "test": {
                    "patch_interval": [1, 4, 4],
                    "batch_size": 2,
                    "prefetch_factor": 1,
                    "pad": True,
                    "pin_memory": True,
                },
            }
        }

    def _run_case(
        self,
        failure_stage=None,
        *,
        train_pin_memory=None,
        test_pin_memory=None,
    ):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        parameters_path = root / "parameters.json"
        parameters_path.write_text(json.dumps(self._parameters()), encoding="utf-8")
        task = SNR.SweepTask(
            index=0,
            source=root / "source.tif",
            reference_path=root / "source_original_SPRT.tif",
            power=50,
            exp_name="source_050",
            out_dir=root / "outputs" / "source",
            simulated_path=root / "outputs" / "source_050" / "source_050.tif",
        )
        configuration = SNR.SweepConfiguration(
            repository=root,
            input_root=root,
            parameters_path=parameters_path,
            output_root=root / "outputs",
            dimensionality="3+1D",
            n_cpu=2,
            rm_tmp=True,
            rm_dir=False,
            checkpoints=False,
            resume=True,
            rm_intermediaries=True,
            photons_per_unit=0.01,
            train_pin_memory=train_pin_memory,
            test_pin_memory=test_pin_memory,
        )
        events = []

        class FakeTrainer:
            def __init__(self, parameters):
                events.append(("trainer_init", parameters))
                self.model_path = "/vf/users/model.pth"
                self.out_dir = "/vf/users/results"
                self.exp_name = task.exp_name
                self.log = SimpleNamespace(log_params=lambda: events.append("log_params"))

            def vars_dict(self):
                events.append("vars_dict")

            def run(self):
                events.append("trainer_run")
                if failure_stage == "training":
                    raise RuntimeError("training failed")

            def release_resources(self):
                events.append("trainer_release")

        class FakeTester:
            def __init__(self, parameters):
                events.append(("tester_init", parameters))
                self.out_path = root / "clean.tif"

            def run(self):
                events.append("tester_run")
                if failure_stage == "testing":
                    raise RuntimeError("testing failed")

            def release_resources(self):
                events.append("tester_release")

        salt_module = types.ModuleType("utils.salt")
        salt_module.simulate_lower_power = lambda stack, *args, **kwargs: np.asarray(stack)
        testing_module = types.ModuleType("utils.test_collection")
        testing_module.TestingClass = FakeTester
        training_module = types.ModuleType("utils.train_collection")
        training_module.TrainingClass = FakeTrainer
        fake_torch = SimpleNamespace(
            manual_seed=lambda seed: events.append(("seed", seed)),
            cuda=SimpleNamespace(empty_cache=lambda: events.append("empty_cache")),
        )
        class FakeTiffReader:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            @staticmethod
            def asarray():
                return np.ones((2, 1, 2, 2), dtype=np.float32)

        tiff_reader = FakeTiffReader()

        def publish(path, stack, axes):
            events.append(("publish", axes))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"simulated")

        stubs = {
            "utils.salt": salt_module,
            "utils.test_collection": testing_module,
            "utils.train_collection": training_module,
        }
        with (
            mock.patch.dict(sys.modules, stubs),
            mock.patch.object(SNR.tifffile, "TiffFile", return_value=tiff_reader),
            mock.patch.object(SNR, "_write_simulated_tiff", side_effect=publish),
        ):
            if failure_stage is None:
                result = SNR._run_task(task, configuration, fake_torch, gpu_slot=1)
                error = None
            else:
                result = None
                try:
                    SNR._run_task(task, configuration, fake_torch, gpu_slot=1)
                except RuntimeError as exc:
                    error = exc
                else:
                    self.fail(f"{failure_stage} failure was not propagated")
        return task, events, result, error

    def test_run_task_releases_training_before_testing_and_deletes_on_success(self) -> None:
        task, events, result, error = self._run_case()

        self.assertIsNone(error)
        self.assertEqual(result.name, "clean.tif")
        self.assertFalse(task.simulated_path.exists())
        self.assertLess(events.index("trainer_release"), events.index("tester_run"))
        self.assertIn("tester_release", events)
        self.assertEqual(events.count("empty_cache"), 2)
        self.assertIn(("seed", 17), events)
        trainer_parameters = next(
            event[1] for event in events if isinstance(event, tuple) and event[0] == "trainer_init"
        )
        tester_parameters = next(
            event[1] for event in events if isinstance(event, tuple) and event[0] == "tester_init"
        )
        self.assertEqual(trainer_parameters["cuda_device"], "0")
        self.assertEqual(tester_parameters["cuda_device"], "0")
        self.assertIs(trainer_parameters["pin_memory"], True)
        self.assertIs(tester_parameters["pin_memory"], True)

    def test_run_task_applies_pin_memory_overrides_independently(self) -> None:
        cases = (
            (False, None, False, True),
            (None, False, True, False),
            (False, False, False, False),
        )
        for train_override, test_override, expected_train, expected_test in cases:
            with self.subTest(
                train_pin_memory=train_override,
                test_pin_memory=test_override,
            ):
                _, events, _, error = self._run_case(
                    train_pin_memory=train_override,
                    test_pin_memory=test_override,
                )
                self.assertIsNone(error)
                trainer_parameters = next(
                    event[1]
                    for event in events
                    if isinstance(event, tuple) and event[0] == "trainer_init"
                )
                tester_parameters = next(
                    event[1]
                    for event in events
                    if isinstance(event, tuple) and event[0] == "tester_init"
                )
                self.assertIs(trainer_parameters["pin_memory"], expected_train)
                self.assertIs(tester_parameters["pin_memory"], expected_test)

    def test_run_task_keeps_simulated_input_and_releases_trainer_on_train_failure(self) -> None:
        task, events, _, error = self._run_case(failure_stage="training")

        self.assertIn("training failed", str(error))
        self.assertTrue(task.simulated_path.exists())
        self.assertIn("trainer_release", events)
        self.assertNotIn("tester_run", events)
        self.assertEqual(events.count("empty_cache"), 1)

    def test_run_task_keeps_simulated_input_and_releases_tester_on_test_failure(self) -> None:
        task, events, _, error = self._run_case(failure_stage="testing")

        self.assertIn("testing failed", str(error))
        self.assertTrue(task.simulated_path.exists())
        self.assertIn("trainer_release", events)
        self.assertIn("tester_release", events)
        self.assertEqual(events.count("empty_cache"), 2)


class FakeProcess:
    def __init__(self, *, target, args, name, exitcode=0) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.exitcode = exitcode
        self.started = False
        self.joined = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return False

    def join(self) -> None:
        self.joined = True

    def terminate(self) -> None:
        self.exitcode = -15


class FakeContext:
    def __init__(self, results, exitcodes=()) -> None:
        self.queue = RecordingQueue(results)
        self.exitcodes = list(exitcodes)
        self.processes = []

    def Queue(self):
        return self.queue

    def Process(self, *, target, args, name):
        exitcode = self.exitcodes[len(self.processes)] if self.exitcodes else 0
        process = FakeProcess(target=target, args=args, name=name, exitcode=exitcode)
        self.processes.append(process)
        return process


class ParallelSweepAggregationTests(unittest.TestCase):
    def test_results_are_sorted_and_missing_worker_result_becomes_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tasks = WorkerBehaviorTests._tasks(root, count=3)
            configuration = WorkerBehaviorTests._configuration(root)
            initial_results = [
                SNR.TaskResult(2, 3, tasks[2].source, tasks[2].power, True, root / "2.tif"),
                SNR.TaskResult(0, 3, tasks[0].source, tasks[0].power, True, root / "0.tif"),
            ]
            context = FakeContext(initial_results)

            with mock.patch.object(
                SNR.multiprocessing,
                "get_context",
                return_value=context,
            ) as get_context:
                results = SNR.run_parallel_sweep(
                    tasks,
                    ((3, "GPU-a"), (7, "GPU-b")),
                    configuration,
                )

            get_context.assert_called_once_with("spawn")
            self.assertEqual([result.task_index for result in results], [0, 1, 2])
            self.assertTrue(results[0].success)
            self.assertFalse(results[1].success)
            self.assertEqual(results[1].gpu_slot, 7)
            self.assertIn("without reporting", results[1].error)
            self.assertTrue(results[2].success)
            self.assertEqual(
                [process.name for process in context.processes],
                ["snr-gpu-3", "snr-gpu-7"],
            )
            self.assertTrue(
                all(process.started and process.joined for process in context.processes)
            )
            self.assertTrue(context.queue.closed)
            self.assertTrue(context.queue.joined)

    def test_unexplained_abnormal_worker_exit_invalidates_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tasks = WorkerBehaviorTests._tasks(root, count=1)
            configuration = WorkerBehaviorTests._configuration(root)
            reported = SNR.TaskResult(
                0,
                4,
                tasks[0].source,
                tasks[0].power,
                True,
                root / "clean.tif",
            )
            context = FakeContext([reported], exitcodes=[9])

            with mock.patch.object(SNR.multiprocessing, "get_context", return_value=context):
                results = SNR.run_parallel_sweep(
                    tasks,
                    ((4, "GPU-z"),),
                    configuration,
                )

            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].success)
            self.assertEqual(results[0].output_path, root / "clean.tif")
            self.assertIn("exited with status 9", results[0].error)

    def test_result_delayed_until_post_join_drain_is_not_marked_missing(self) -> None:
        class DelayedQueue(RecordingQueue):
            def __init__(self, initial=()) -> None:
                super().__init__(initial)
                self.get_calls = 0

            def get(self, timeout=None):
                self.get_calls += 1
                if self.get_calls == 1:
                    raise Empty
                return super().get(timeout=timeout)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tasks = WorkerBehaviorTests._tasks(root, count=1)
            configuration = WorkerBehaviorTests._configuration(root)
            reported = SNR.TaskResult(
                0,
                6,
                tasks[0].source,
                tasks[0].power,
                True,
                root / "clean.tif",
            )
            context = FakeContext([])
            context.queue = DelayedQueue([reported])

            with mock.patch.object(
                SNR.multiprocessing,
                "get_context",
                return_value=context,
            ):
                results = SNR.run_parallel_sweep(
                    tasks,
                    ((6, "GPU-six"),),
                    configuration,
                )

            self.assertEqual(results, (reported,))
            self.assertEqual(context.queue.get_calls, 2)
            self.assertTrue(context.processes[0].joined)


if __name__ == "__main__":
    unittest.main()
