"""CUDA-free tests for the independent multi-GPU SNR sweep scheduler."""

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
from contextlib import redirect_stderr
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
        args = SNR.build_parser().parse_args(["--gpus", "2", "0"])
        self.assertEqual(args.gpus, [2, 0])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            SNR.build_parser().parse_args(["--gpus", "-1"])


class SweepPlanningTests(unittest.TestCase):
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
            self.assertEqual(tasks[0].exp_name, "b_100")
            self.assertEqual(
                tasks[0].simulated_path,
                (root / "outputs" / "b" / "b_100" / "b_100.tif").resolve(),
            )

    def test_build_sweep_tasks_rejects_duplicate_destinations_and_powers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
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
        )

    @staticmethod
    def _tasks(root: Path, count: int = 2):
        return tuple(
            SNR.SweepTask(
                index=index,
                source=root / f"source-{index}.tif",
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
                },
                "test": {
                    "patch_interval": [1, 4, 4],
                    "batch_size": 2,
                    "prefetch_factor": 1,
                    "pad": True,
                },
            }
        }

    def _run_case(self, failure_stage=None):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        parameters_path = root / "parameters.json"
        parameters_path.write_text(json.dumps(self._parameters()), encoding="utf-8")
        task = SNR.SweepTask(
            index=0,
            source=root / "source.tif",
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
