from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
import unittest

from pose_app.benchmark_timing import (
    SUMMARY_FIELDS,
    TimingCollector,
    percentile_ms,
)


REALTIME_ROOT = Path(__file__).resolve().parents[1]


class TimingCollectorTests(unittest.TestCase):
    def test_measure_records_a_positive_duration(self):
        collector = TimingCollector(gpu_synchronize=False)
        with collector.measure("module_a"):
            time.sleep(0.01)
        summary = collector.summary()
        self.assertIn("module_a", summary)
        self.assertEqual(summary["module_a"]["count"], 1)
        self.assertGreater(summary["module_a"]["mean_ms"], 0.0)
        self.assertGreater(collector.samples_ms("module_a")[0], 0.0)

    def test_measure_records_every_sample_of_a_module(self):
        collector = TimingCollector(gpu_synchronize=False)
        for _ in range(3):
            with collector.measure("module_a"):
                pass
        self.assertEqual(collector.summary()["module_a"]["count"], 3)
        self.assertEqual(len(collector.samples_ms("module_a")), 3)

    def test_measure_still_records_when_the_block_raises(self):
        collector = TimingCollector(gpu_synchronize=False)
        with self.assertRaises(RuntimeError):
            with collector.measure("module_a"):
                raise RuntimeError("boom")
        self.assertEqual(collector.summary()["module_a"]["count"], 1)

    def test_manual_record_ms_statistics_are_correct(self):
        collector = TimingCollector(gpu_synchronize=False)
        for value in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0):
            collector.record_ms("manual", value)
        summary = collector.summary()["manual"]
        self.assertEqual(summary["count"], 10)
        self.assertAlmostEqual(summary["mean_ms"], 55.0, places=9)
        self.assertAlmostEqual(summary["median_ms"], 55.0, places=9)
        self.assertAlmostEqual(summary["p90_ms"], 91.0, places=9)
        self.assertAlmostEqual(summary["p95_ms"], 95.5, places=9)
        self.assertAlmostEqual(summary["min_ms"], 10.0, places=9)
        self.assertAlmostEqual(summary["max_ms"], 100.0, places=9)
        self.assertEqual(set(summary), set(SUMMARY_FIELDS))

    def test_manual_record_ms_median_of_even_sample_is_interpolated(self):
        collector = TimingCollector(gpu_synchronize=False)
        for value in (1.0, 2.0, 3.0, 4.0):
            collector.record_ms("manual", value)
        self.assertAlmostEqual(collector.summary()["manual"]["median_ms"], 2.5, places=9)
        self.assertAlmostEqual(collector.summary()["manual"]["p90_ms"], 3.7, places=9)

    def test_empty_collector_summarises_to_an_empty_mapping(self):
        collector = TimingCollector(gpu_synchronize=False)
        self.assertEqual(collector.summary(), {})
        self.assertEqual(collector.module_names(), [])
        self.assertEqual(collector.total_samples(), 0)
        self.assertEqual(collector.samples_ms("missing"), [])

    def test_percentile_rejects_an_empty_sample_set(self):
        with self.assertRaises(ValueError):
            percentile_ms([], 50.0)

    def test_record_ms_rejects_non_finite_and_negative_values(self):
        collector = TimingCollector(gpu_synchronize=False)
        with self.assertRaises(ValueError):
            collector.record_ms("module_a", float("nan"))
        with self.assertRaises(ValueError):
            collector.record_ms("module_a", -1.0)
        self.assertEqual(collector.summary(), {})

    def test_separate_modules_are_summarised_independently(self):
        collector = TimingCollector(gpu_synchronize=False)
        collector.record_ms("module_a", 1.0)
        collector.record_ms("module_b", 2.0)
        collector.record_ms("module_b", 4.0)
        summary = collector.summary()
        self.assertEqual(summary["module_a"]["count"], 1)
        self.assertEqual(summary["module_b"]["count"], 2)
        self.assertAlmostEqual(summary["module_b"]["mean_ms"], 3.0, places=9)

    def test_opt_out_policy_never_reports_a_synchronisation(self):
        collector = TimingCollector(gpu_synchronize=False)
        self.assertFalse(collector.gpu_synchronize_policy)
        with collector.measure("module_a"):
            pass
        self.assertFalse(collector.gpu_synchronized)
        self.assertEqual(collector.summary()["module_a"]["count"], 1)

    def test_cuda_probe_reports_a_boolean_without_importing_torch_in_process(self):
        # The probe is exercised in a separate interpreter on purpose: on this
        # host an in-process ``import torch`` poisons the Anaconda MKL NumPy
        # runtime and aborts the whole test process on its next linear-algebra
        # call (OMP Error #15).  Out of process the probe still runs the real
        # ``torch.cuda.is_available()`` path, and the assertions below cover the
        # "no CUDA -> no error" requirement on a CPU-only machine.
        code = (
            "from pose_app.benchmark_timing import cuda_is_available, synchronize_if_cuda\n"
            "available = cuda_is_available()\n"
            "synchronized = synchronize_if_cuda()\n"
            "print('RESULT', isinstance(available, bool), isinstance(synchronized, bool), available)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REALTIME_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        result_lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
        self.assertEqual(len(result_lines), 1, msg=completed.stdout)
        parts = result_lines[0].split()
        self.assertEqual(parts[1], "True")
        self.assertEqual(parts[2], "True")


if __name__ == "__main__":
    unittest.main()
