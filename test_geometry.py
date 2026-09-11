"""问题一验收与逐案例报告（无需pytest）。

生成报告：python test_geometry.py
指定目录：python test_geometry.py --output-dir results/my_run
仅运行某测试：python test_geometry.py --test test_document_two_observations
普通unittest仍可使用，但不导出报告：python -m unittest -v test_geometry

每次求解单独记录；随机测试的100组3点/4点观测共200条记录。
直接半平面用例无检测点，点数填null/CSV空白，而不是伪造检测点。
报告包含cases.csv、report.json、unittest.log，失败时也保存并返回非零退出码。
"""
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from time import perf_counter
import traceback
import unittest
from unittest.mock import Mock, patch

import numpy as np

import geometry_V7 as g
from geometry_test_timing import StageTimer, COLUMNS as TIMING_COLUMNS, NOTE as TIMING_NOTE


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = g.load_config()

    def solve(self, observations):
        return g.solve_q1(observations, config=self.cfg)

    def test_document_two_observations(self):
        result = self.solve([(-800, 0, 0), (0, -800, 90)])
        self.assertEqual(result.status, "POLYGON")
        self.assertEqual(len(result.vertices), 4)
        self.assertAlmostEqual(result.diameter, 39.50834065899, places=8)
        self.assertTrue(result.same_diameter_covers)
        np.testing.assert_allclose(result.circle_center, (0.24381771953,) * 2, atol=1e-8)

    def test_document_three_observations(self):
        result = self.solve([(-800, 0, 0), (400, -400 * math.sqrt(3), 120),
                             (400, 400 * math.sqrt(3), 240)])
        self.assertEqual(result.status, "POLYGON")
        self.assertEqual(len(result.vertices), 6)
        self.assertAlmostEqual(result.diameter, 32.25187208461, places=8)
        self.assertAlmostEqual(result.max_vertex_distance, 16.37030923069, places=8)
        self.assertAlmostEqual(result.coverage_excess, 0.24437318839, places=8)
        self.assertFalse(result.same_diameter_covers)

    def test_forward_not_backward(self):
        rows = g.build_halfplanes([(0, 0, 0)])
        self.assertTrue(np.all(rows[:, :2] @ np.array([100, 0]) <= rows[:, 2]))
        self.assertFalse(np.all(rows[:, :2] @ np.array([-100, 0]) <= rows[:, 2]))

    def test_single_observation_unbounded(self):
        result = self.solve([(0, 0, 359)])
        self.assertEqual(result.status, "UNBOUNDED")
        self.assertEqual(result.diameter, math.inf)
        self.assertIsNone(result.same_diameter_covers)

    def test_empty_observation_intersection(self):
        result = self.solve([(1, 0, 0), (-1, 0, 180)])
        self.assertEqual(result.status, "EMPTY")
        self.assertIsNone(result.diameter)
        self.assertIsNone(result.same_diameter_covers)

    def test_point_and_segment_direct_constraints(self):
        point = g.solve_halfplanes([(1, 0, -2), (-1, 0, 2), (0, 1, -3), (0, -1, 3)])
        self.assertEqual(point.status, "POINT")
        self.assertEqual(point.circle_center, (-2, -3))
        self.assertEqual(point.diameter, 0)
        self.assertTrue(point.same_diameter_covers)
        segment = g.solve_halfplanes([(1, 0, 2), (-1, 0, 0), (0, 1, -3), (0, -1, 3)])
        self.assertEqual(segment.status, "SEGMENT")
        self.assertEqual(segment.diameter, 2)
        self.assertTrue(segment.same_diameter_covers)

    def test_strip_unbounded_with_no_vertices(self):
        self.assertEqual(g.solve_halfplanes([(1, 0, 1), (-1, 0, 1)]).status, "UNBOUNDED")

    def test_contradictory_parallel_constraints(self):
        self.assertEqual(g.solve_halfplanes([(1, 0, 0), (-1, 0, -1)]).status, "EMPTY")

    def test_constant_constraints(self):
        self.assertEqual(g.solve_halfplanes([(0, 0, -1)]).status, "EMPTY")
        self.assertEqual(g.solve_halfplanes([(0, 0, 1)]).status, "UNBOUNDED")

    def test_negative_coordinate_square(self):
        result = g.solve_halfplanes([(1, 0, -2), (-1, 0, 4), (0, 1, -3), (0, -1, 5)])
        self.assertEqual(result.status, "POLYGON")
        self.assertAlmostEqual(result.diameter, math.sqrt(8))
        self.assertEqual(result.circle_center, (-3, -4))

    def test_near_parallel_not_discarded(self):
        # 三角形顶点含(1e12,0)，不能因行列式小而丢失。
        with patch.object(g, "linprog", None):
            result = g.solve_halfplanes([(-1, 0, 0), (0, -1, 0), (1e-12, 1, 1)])
        self.assertEqual(result.status, "POLYGON")
        self.assertAlmostEqual(result.diameter / 1e12, 1)
        self.assertGreater(result.diagnostics["near_parallel_pairs"], 0)

    def test_thin_polygon_not_merged_into_segment(self):
        with patch.object(g, "linprog", None):
            result = g.solve_halfplanes([(1, 0, 1), (-1, 0, 0), (0, 1, 1e-9), (0, -1, 0)])
        self.assertEqual(result.status, "POLYGON")
        self.assertEqual(len(result.vertices), 4)
        self.assertTrue(result.diagnostics["near_degenerate"])

    def test_angle_wrap_and_two_input_styles(self):
        a = self.solve([(-800, 0, -1), (0, -800, 89)])
        b = g.solve_q1([(-800, 0), (0, -800)], [719, 449])
        self.assertEqual(a.status, "POLYGON")
        self.assertAlmostEqual(a.diameter, b.diameter, places=10)

    def test_duplicate_and_permutation(self):
        obs = [(-800, 0, 0), (0, -800, 90)]
        a, b = self.solve(obs), self.solve(obs[::-1] + obs)
        self.assertEqual(a.vertices, b.vertices)
        self.assertEqual(a.diameter, b.diameter)

    def test_translation_and_rotation(self):
        obs = [(-800, 0, 0), (0, -800, 90)]
        base = self.solve(obs)
        translated = self.solve([(x - 4000, y + 2000, t) for x, y, t in obs])
        self.assertAlmostEqual(base.diameter, translated.diameter, places=8)
        np.testing.assert_allclose(np.array(base.vertices) + [-4000, 2000], translated.vertices, atol=1e-8)
        angle = math.radians(37)
        rotated = self.solve([(x * math.cos(angle) - y * math.sin(angle),
                               x * math.sin(angle) + y * math.cos(angle), t + 37) for x, y, t in obs])
        self.assertAlmostEqual(base.diameter, rotated.diameter, places=8)
        self.assertEqual(base.same_diameter_covers, rotated.same_diameter_covers)

    def test_consistent_random_observations(self):
        rng = random.Random(20260911)
        for _ in range(100):
            source = (rng.uniform(-500, 500), rng.uniform(-500, 500))
            obs = []
            for direction in (0, 120, 240, 60):
                alpha = math.radians(direction + rng.uniform(-10, 10))
                radius = rng.uniform(300, 900)
                x, y = (source[0] + radius * math.cos(alpha), source[1] + radius * math.sin(alpha))
                theta = math.degrees(math.atan2(source[1] - y, source[0] - x)) + rng.uniform(-0.9, 0.9)
                obs.append((x, y, theta))
            first, second = self.solve(obs[:3]), self.solve(obs)
            self.assertEqual(first.status, "POLYGON")
            self.assertEqual(second.status, "POLYGON")
            self.assertLessEqual(second.diameter, first.diameter + 1e-7)
            rows = g.build_halfplanes(obs)
            self.assertTrue(np.all(rows[:, :2] @ source - rows[:, 2] <= 1e-7))
            self.assertTrue(np.all(rows[:, :2] @ np.array(second.vertices).T - rows[:, 2, None] <= 1e-7))

    def test_invalid_inputs(self):
        for obs in ([], [(0, 0)], [(math.nan, 0, 0)], [(0, 0, math.inf)]):
            with self.assertRaises(ValueError):
                self.solve(obs)
        for epsilon in (0, -1, 90, math.nan):
            with self.assertRaises(ValueError):
                g.solve_q1([(0, 0, 0)], error_deg=epsilon)
        with self.assertRaises(ValueError):
            g.solve_q1([(0, 0)], [0, 1])

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            replace(self.cfg, lp_primal_feasibility_tol=1e-12)
        original = Path(g.__file__).with_name("config.h").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.h"
            for text in (original.replace("RF_Q1_MIN_OBSERVATIONS 1", "RF_Q1_MIN_OBSERVATIONS 1+0"),
                         original + "\n#define RF_Q1_MIN_OBSERVATIONS 1\n",
                         original.replace("#define RF_Q1_MIN_OBSERVATIONS 1", "")):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    g.load_config(path)

    def test_lp_numerical_failure_not_unbounded(self):
        class Failed:
            status = 4
            message = "numerical failure"
        with patch.object(g, "linprog", return_value=Failed()):
            result = self.solve([(-800, 0, 0), (0, -800, 90)])
        # V7 在 LP 失败后回退到投影边界判定；该正常算例应仍正确求解。
        self.assertEqual(result.status, "POLYGON")
        self.assertAlmostEqual(result.diameter, 39.50834065899, places=8)
        self.assertTrue(result.same_diameter_covers)
        self.assertTrue(any("线性规划求解失败，回退到投影边界判定" in warning
                            for warning in result.diagnostics["warnings"]))

    def test_lp_uses_free_variables(self):
        class Unbounded:
            status = 3
            message = "unbounded"
        with patch.object(g, "linprog", return_value=Unbounded()) as mocked:
            self.solve([(0, 0, 0)])
        self.assertEqual(mocked.call_args.kwargs["bounds"], [(None, None), (None, None)])


# ==================== 报告板块一：可移植的结果序列化 ====================
def _json_value(value):
    """严格JSON：无穷/NaN用字符串，None保留null，不将空集误写为0。"""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: _json_value(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(val) for val in value]
    return value


class CaseRecorder:
    """只计时最外层求解调用，避免solve_q1内部调用solve_halfplanes被重复统计。

    计时包括输入校验、约束生成、状态判断与几何计算；不包括断言、写文件、
    序列化和配置快照。未经预热，首个求解可能含后端初始化时间。
    """
    def __init__(self):
        self.cases = []
        self.active = None
        self.test_name = None
        self.test_case_index = 0

    def wrap(self, name, original):
        def measured(*args, **kwargs):
            if self.active is not None:
                if name == "solve_halfplanes":
                    self.active["constraint_count_input"] = len(args[0])
                return original(*args, **kwargs)
            self.test_case_index += 1
            data = args[0]
            is_q1 = name == "solve_q1"
            cfg = kwargs.get("config") or g.load_config()
            record = {
                "case_id": f"{self.test_name}__{self.test_case_index:03d}",
                "test_name": self.test_name,
                "call_index": self.test_case_index,
                "entrypoint": name,
                "input_kind": "observations" if is_q1 else "halfplanes",
                "observation_count": len(data) if is_q1 else None,
                "unique_detection_point_count": None,
                "constraint_count_input": None if is_q1 else len(data),
                "constraint_count_solver": None,
                "input": _json_value(data),
                "bearings": _json_value(args[1] if len(args) > 1 else kwargs.get("bearings")),
                "error_deg": kwargs.get("error_deg", cfg.bearing_error_deg) if is_q1 else None,
                "configuration": asdict(cfg),
                "backend_mode": ("mocked_linprog" if isinstance(g.linprog, Mock) else
                                 "fraction_only" if g.linprog is None else "scipy_with_fraction_check"),
            }
            if is_q1 and all(isinstance(row, (list, tuple, np.ndarray)) and len(row) >= 2 for row in data):
                record["unique_detection_point_count"] = len({tuple(row[:2]) for row in data})
            self.active = record
            timer = StageTimer()
            timer.__enter__()
            start = perf_counter()
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                record["elapsed_ms"] = (perf_counter() - start) * 1000
                record["status"] = "EXCEPTION"
                record["exception"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            else:
                record["elapsed_ms"] = (perf_counter() - start) * 1000
                record.update(result.to_dict())
                record["vertex_count"] = len(result.vertices)
                record["constraint_count_solver"] = result.diagnostics.get("constraint_count")
                record["backend"] = result.diagnostics.get("classification_backend", "not_run_or_early_return")
                return result
            finally:
                timer.__exit__(None, None, None)
                record.update(timer.report(record["elapsed_ms"]))
                self.active = None
                self.cases.append(record)
        return measured


# ==================== 报告板块二：记录断言结果及失败堆栈 ====================
class ReportingResult(unittest.TextTestResult):
    def __init__(self, *args, recorder, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder
        self.tests = []

    def startTest(self, test):
        super().startTest(test)
        self.recorder.test_name = test._testMethodName
        self.recorder.test_case_index = 0
        self.current = {"name": test._testMethodName, "outcome": "RUNNING"}
        self.started = perf_counter()

    def addSuccess(self, test):
        self.current["outcome"] = "PASS"
        super().addSuccess(test)

    def addFailure(self, test, err):
        self.current.update(outcome="FAIL", traceback="".join(traceback.format_exception(*err)))
        super().addFailure(test, err)

    def addError(self, test, err):
        self.current.update(outcome="ERROR", traceback="".join(traceback.format_exception(*err)))
        super().addError(test, err)

    def addSkip(self, test, reason):
        self.current.update(outcome="SKIP", reason=reason)
        super().addSkip(test, reason)

    def stopTest(self, test):
        self.current["test_elapsed_ms"] = (perf_counter() - self.started) * 1000
        self.current["solver_call_count"] = self.recorder.test_case_index
        self.tests.append(self.current)
        super().stopTest(test)


# ==================== 报告板块三：运行并写入CSV、JSON、日志 ====================
def run_report(output_dir: Path, selected_test: str | None = None) -> bool:
    # 创建新目录，拒绝覆盖已有实验记录。
    output_dir.mkdir(parents=True, exist_ok=False)
    recorder = CaseRecorder()
    suite = (unittest.defaultTestLoader.loadTestsFromName(selected_test, GeometryTests)
             if selected_test else unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests))
    package_versions = {}
    for package in ("numpy", "scipy"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    metadata = {
        "started_at": datetime.now().astimezone().isoformat(),
        "python_executable": sys.executable, "python_version": sys.version,
        "packages": package_versions, "random_seed": 20260911,
        "config": asdict(g.load_config()),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("geometry_V7.py", "config.h", "test_geometry.py")},
        "timing_note": TIMING_NOTE,
        "count_note": "observation_count为输入观测条数；unique_detection_point_count为不同坐标数；"
                      "constraint_count_input为实际传入半平面求解器的约束数；"
                      "constraint_count_solver为归一化去重后约束数，并非最小非冗余约束数；"
                      "null表示不适用或该阶段未到达。",
        "outcome_note": "test_outcome是所属测试方法的整体断言结果；EXCEPTION可能是预期的非法输入；"
                        "mocked_linprog和fraction_only用例不代表真实SciPy耗时。",
    }
    start = perf_counter()
    with (output_dir / "unittest.log").open("w", encoding="utf-8") as log:
        runner = unittest.TextTestRunner(
            stream=log, verbosity=2,
            resultclass=lambda *a, **kw: ReportingResult(*a, recorder=recorder, **kw))
        with patch.object(g, "solve_q1", recorder.wrap("solve_q1", g.solve_q1)), \
             patch.object(g, "solve_halfplanes", recorder.wrap("solve_halfplanes", g.solve_halfplanes)):
            result = runner.run(suite)
    metadata["suite_elapsed_ms"] = (perf_counter() - start) * 1000
    outcomes = {t["name"]: t["outcome"] for t in result.tests}
    for case in recorder.cases:
        case["test_outcome"] = outcomes.get(case["test_name"], "UNKNOWN")
    report = _json_value({"metadata": metadata,
                          "summary": {"successful": result.wasSuccessful(), "test_count": result.testsRun,
                                      "failures": len(result.failures), "errors": len(result.errors),
                                      "solver_call_count": len(recorder.cases)},
                          "tests": result.tests, "cases": recorder.cases})
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    columns = ["case_id", "test_name", "test_outcome", "entrypoint", "input_kind", "observation_count",
               "unique_detection_point_count", "constraint_count_input", "constraint_count_solver",
               "elapsed_ms", "status", "vertex_count", "diameter", "circle_radius", "max_vertex_distance",
               "coverage_excess", "same_diameter_covers", "backend_mode", "backend"]
    columns += TIMING_COLUMNS
    # BOM使Windows Excel直接打开时正确识别UTF-8；JSON保留完整输入/顶点/诊断。
    with (output_dir / "cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["cases"])
    (output_dir / "README.txt").write_text(
        "问题一逐案例测试报告\n" + TIMING_NOTE + "\n\n"
        "cases.csv：每次最外层求解一行，可用Excel打开。\n"
        "report.json：完整输入、配置、顶点、最远点对、圆心、诊断和测试结果。\n"
        "unittest.log：全部测试方法的通过/失败记录及失败堆栈。\n\n"
        "字段说明：\n"
        "case_id：测试方法名+方法内求解序号；不是模拟器正式测试编码。\n"
        "observation_count：输入观测条数（重复观测也计数）。\n"
        "unique_detection_point_count：输入中的不同检测点坐标数。\n"
        "constraint_count_input：实际传入半平面求解器的约束数；合法测向输入通常为2n。\n"
        "constraint_count_solver：归一化并删除完全重复项后的约束数，不是非冗余约束数。\n"
        "elapsed_ms：一次完整求解的墙钟耗时，单位毫秒；不含写报告和断言，无预热。\n"
        "status：几何状态或EXCEPTION（如预期的非法输入）。\n"
        "vertex_count：输出顶点数。\n"
        "diameter：定位区域直径，单位米；Infinity为无界，空白/null表示无有效值。\n"
        "circle_radius/max_vertex_distance/coverage_excess：圆半径/最大顶点距离/超出量，单位米。\n"
        "same_diameter_covers：是否在配置容差下可被同直径圆覆盖。\n"
        "test_outcome：所属测试方法整体断言结果；PASS不等于几何status必须为POLYGON。\n"
        "backend_mode/backend：真实SciPy、纯有理数或模拟求解器；模拟耗时不能用于性能比较。\n\n"
        "随机测试使用固定种子20260911；每组先3点后4点，共100组200行。\n"
        "直接半平面案例没有检测点，因此点数为空；未到达阶段的参数也为空。\n"
        "仅构造约束或检查配置的测试不调用求解器，没有cases行，可在JSON的tests中查看。\n"
        "各行test_outcome以整个测试方法为单位；若中途断言失败，后续案例不会运行。\n"
        "完整精度保存在JSON/CSV中；文件写入使用UTF-8（CSV带BOM）。\n",
        encoding="utf-8")
    print(f"Tests: {result.testsRun}; failures: {len(result.failures)}; errors: {len(result.errors)}")
    print(f"Solver calls: {len(recorder.cases)}; reports: {output_dir.resolve()}")
    return result.wasSuccessful()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="新建结果目录（不得已存在）")
    parser.add_argument("--test", help="只运行指定方法，例如test_document_two_observations")
    args = parser.parse_args()
    if args.test and args.test not in unittest.defaultTestLoader.getTestCaseNames(GeometryTests):
        parser.error("未知测试方法，请使用GeometryTests中已有的test_方法名")
    folder = args.output_dir or (Path(__file__).parent / "results" /
                                datetime.now().strftime("geometry_%Y%m%d_%H%M%S_%f"))
    if folder.exists():
        parser.error("结果目录已经存在，请指定新目录，避免覆盖历史记录")
    sys.exit(0 if run_report(folder, args.test) else 1)
