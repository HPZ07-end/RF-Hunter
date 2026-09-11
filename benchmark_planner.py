"""
benchmark_planner.py

问题二 planner.py 性能与基本正确性测试。

特点：
1. 兼容 solve_q2((x, y, theta)) 和旧版 search_second_point(S, theta)。
2. 测量运行时间、CPU 时间和结果稳定性。
3. 可选验证接收约束、J 值合理性、连续坐标输出。
4. 可选输出 cProfile 热点。
5. 新增候选区域几何指标：AABB 宽高、AABB 面积、采样可行面积。
6. 新增寻找候选区效率指标：几何调用次数、代理评估次数、每次评估时间。
7. 每个命名案例可独立覆盖 PlannerConfig，并验证预期分支。
8. 标准模式覆盖目标圆截断、近切线、多轮情景生成和 hybrid/strict A/B。
9. 检查分层精确校核、降维对抗搜索和最坏情景优先策略是否生效。
10. 直接检查快速几何模块的正常、平行和近平行回退分支。
11. 不写入任何结果文件。

示例：
    python benchmark_planner.py
    python benchmark_planner.py --mode standard --repeats 3 --profile
    python benchmark_planner.py --mode standard --full-ab-check
    python benchmark_planner.py --mode standard --memory
    python benchmark_planner.py --no-region-metrics
    python benchmark_planner.py --region-sample-count 20000
    python benchmark_planner.py --max-seconds 5
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import gc
import inspect
import io
import json
import math
import pstats
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any

import numpy as np

import planner


# ============================================================
# 一、测试案例
# ============================================================

@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    observation: tuple[float, float, float]
    description: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    minimum_scenario_rounds: int | None = None
    expected_fast_geometry_enabled: bool | None = None
    expected_verification_mode: str | None = None
    minimum_strict_fallbacks: int | None = None


def _case(
    name,
    observation,
    description,
    *,
    config_overrides=None,
    minimum_scenario_rounds=None,
    expected_fast_geometry_enabled=None,
    expected_verification_mode=None,
    minimum_strict_fallbacks=None,
):
    return BenchmarkCase(
        name=name,
        observation=tuple(float(value) for value in observation),
        description=description,
        config_overrides=dict(config_overrides or {}),
        minimum_scenario_rounds=minimum_scenario_rounds,
        expected_fast_geometry_enabled=expected_fast_geometry_enabled,
        expected_verification_mode=expected_verification_mode,
        minimum_strict_fallbacks=minimum_strict_fallbacks,
    )


_BASELINE_CASES = [
    _case("baseline_zero", (0, 0, 0), "中心基准场景"),
    _case("rotation_30", (0, 0, 30), "中心场景旋转30度"),
    _case("rotation_135", (0, 0, 135), "中心场景旋转135度"),
    _case("offcenter_210", (600, 400, 210), "非中心第一次检测点"),
    _case("offcenter_350", (-900, 200, 350), "非中心且跨越0度方向"),
]


_STRUCTURAL_CASES = [
    _case(
        "target_clipped",
        (1700, 0, 90),
        "第一次可能源区域被目标圆明显截断",
    ),
    _case(
        "target_near_tangent",
        (1790, 0, 90),
        "第一次测向接近目标圆切线，连续对抗搜索更困难",
    ),
    _case(
        "outside_target",
        (2200, 0, 180),
        "第一次检测点位于目标圆外部并朝向目标区域",
    ),
    _case(
        "scenario_refinement",
        (1700, 0, 90),
        "收紧情景差容差，要求活动(G,e)集合至少扩充一轮",
        config_overrides={"scenario_gap_tolerance_m": 0.01},
        minimum_scenario_rounds=2,
    ),
    _case(
        "fast_geometry_off",
        (0, 0, 0),
        "关闭两检测点快速几何评价器，作为同场景A/B基线",
        config_overrides={"use_fast_geometry": False},
        expected_fast_geometry_enabled=False,
    ),
    _case(
        "strict_verification_baseline",
        (0, 0, 0),
        "全量 geometry_V7 严格校核，作为优先级1-2的同输入A/B基线",
        config_overrides={"verification_mode": "strict"},
        expected_verification_mode="strict",
    ),
]


TEST_CASES = {
    "quick": [
        _case("quick_baseline", (0, 0, 30), "快速冒烟测试"),
        _case(
            "quick_strict_verification",
            (0, 0, 30),
            "快速模式下的全量精确校核A/B基线",
            config_overrides={"verification_mode": "strict"},
            expected_verification_mode="strict",
        ),
    ],
    "standard": _BASELINE_CASES + _STRUCTURAL_CASES,
    "stress": _BASELINE_CASES + _STRUCTURAL_CASES + [
        _case("center_90", (0, 0, 90), "中心场景四分之一周旋转"),
        _case("center_180", (0, 0, 180), "中心场景半周旋转"),
        _case("target_boundary", (1500, 0, 180), "第一次检测点接近目标边界"),
        _case("target_boundary_vertical", (0, -1500, 90), "纵向目标边界场景"),
        _case("outside_target_offaxis", (2500, 400, 190), "目标圆外部非轴对齐场景"),
        _case("clipped_offaxis", (1750, 150, 200), "非轴对齐目标圆截断场景"),
        _case(
            "forced_verification_fallback",
            (0, 0, 0),
            "极小一致性容差，强制覆盖分层校核的严格回退分支",
            config_overrides={
                "verification_disagreement_tolerance_m": 1e-16,
            },
            minimum_strict_fallbacks=1,
        ),
    ],
}


# ============================================================
# 二、配置构造
# ============================================================

def build_config(mode: str):
    """构造 PlannerConfig；quick 模式降低搜索精度以加速冒烟测试。"""
    if not hasattr(planner, "PlannerConfig"):
        return None

    if mode == "quick":
        return planner.PlannerConfig(
            point_sample_count=64,
            point_start_count=3,
            proxy_max_iterations=25,
            exact_max_iterations=10,
            point_initial_step_m=160.0,
            point_tolerance_m=2.0,
            source_sample_count=32,
            source_start_count=4,
            source_max_iterations=20,
            error_sample_count=5,
            error_start_count=3,
            error_max_iterations=12,
            max_scenario_rounds=5,
        )

    return planner.PlannerConfig()


def build_case_config(base_config, case: BenchmarkCase):
    """在全局搜索精度配置上应用单案例覆盖，不修改其他案例。"""
    if not case.config_overrides:
        return base_config
    if base_config is None or not is_dataclass(base_config):
        raise RuntimeError(
            f"案例 {case.name} 需要 PlannerConfig，当前 planner 不支持配置覆盖"
        )
    try:
        return replace(base_config, **case.config_overrides)
    except TypeError as exc:
        raise RuntimeError(
            f"案例 {case.name} 的配置覆盖无效：{case.config_overrides}"
        ) from exc


# ============================================================
# 三、接口调用与结果规范化
# ============================================================

def invoke_planner(observation, config, show_output=False):
    """自动选择新版或旧版入口，屏蔽 planner 内部 stdout。"""
    output_context = (
        contextlib.nullcontext()
        if show_output
        else contextlib.redirect_stdout(io.StringIO())
    )

    with output_context:
        if hasattr(planner, "solve_q2"):
            kwargs = {}
            signature = inspect.signature(planner.solve_q2)
            if config is not None and "config" in signature.parameters:
                kwargs["config"] = config
            return planner.solve_q2(observation, **kwargs)

        S = observation[:2]
        theta1 = observation[2]
        return planner.search_second_point(S, theta1)


def normalize_result(raw_result: Any) -> dict:
    """统一 PlannerResult 或旧式二元组。"""
    if hasattr(raw_result, "second_point"):
        point = raw_result.second_point
        diameter = raw_result.worst_diameter
        status = getattr(raw_result, "status", "UNKNOWN")
        diagnostics = getattr(raw_result, "diagnostics", {}) or {}
        proxy_score = getattr(raw_result, "proxy_score", None)
        worst_source = getattr(raw_result, "worst_source", None)
        worst_error = getattr(raw_result, "worst_second_error_deg", None)
        receive_margin = getattr(raw_result, "receive_margin_m", None)
        active_sources = getattr(raw_result, "active_sources", []) or []
        active_scenarios = getattr(raw_result, "active_scenarios", []) or []
    elif isinstance(raw_result, (tuple, list)) and len(raw_result) == 2:
        point, diameter = raw_result
        status = "LEGACY_RESULT"
        diagnostics = {}
        proxy_score = None
        worst_source = None
        worst_error = None
        receive_margin = None
        active_sources = []
        active_scenarios = []
    else:
        raise AssertionError(
            "planner 返回值应为 PlannerResult 或 (best_point, best_J)"
        )

    if point is not None:
        point = (float(point[0]), float(point[1]))
    if diameter is not None:
        diameter = float(diameter)

    return {
        "status": status,
        "point": point,
        "diameter": diameter,
        "proxy_score": float(proxy_score) if proxy_score is not None else None,
        "worst_source": tuple(worst_source) if worst_source else None,
        "worst_second_error_deg": (
            float(worst_error) if worst_error is not None else None
        ),
        "receive_margin_m": (
            float(receive_margin) if receive_margin is not None else None
        ),
        "active_sources": list(active_sources),
        "active_scenarios": list(active_scenarios),
        "diagnostics": diagnostics,
    }


# ============================================================
# 四、正确性检查
# ============================================================

def validate_basic_result(result: dict):
    point = result["point"]
    diameter = result["diameter"]

    assert point is not None, "没有返回第二检测点"
    assert len(point) == 2, "第二检测点必须是二维坐标"
    assert all(math.isfinite(v) for v in point), "第二检测点包含非有限数值"

    assert diameter is not None, "没有返回定位区域直径"
    assert math.isfinite(diameter), f"定位区域直径不是有限值：{diameter}"
    assert diameter >= 0, f"定位区域直径不能为负数：{diameter}"


def validate_receive_constraint(observation, result, config):
    """使用 planner 返回的 receive_margin_m 检查接收约束。"""
    margin = result.get("receive_margin_m")
    if margin is None:
        return
    tolerance = getattr(config, "receive_tolerance_m", 1e-6) if config else 1e-6
    assert margin >= -tolerance, (
        f"最终第二检测点接收裕量 {margin:.3e} 低于容差 {-tolerance:.3e}"
    )


def validate_continuous_output(result, grid_step=100.0):
    """检查结果是否仍被限制在旧的固定网格上。"""
    point = result["point"]
    if point is None:
        return

    on_grid = all(
        math.isclose(
            coordinate / grid_step,
            round(coordinate / grid_step),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for coordinate in point
    )
    assert not on_grid, (
        f"输出点 {point} 的两个坐标仍同时落在 {grid_step:g} 米网格上"
    )


def _rotate_test_point(point, angle_deg):
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        cosine * point[0] - sine * point[1],
        sine * point[0] + cosine * point[1],
    )


def validate_rotation_invariance(config, show_output=False):
    """检查整体旋转输入后，输出是否按相同角度旋转且 J 保持不变。"""
    base_observation = (600.0, 400.0, 210.0)
    base = normalize_result(
        invoke_planner(base_observation, config, show_output)
    )
    validate_basic_result(base)

    records = []
    for angle in (37.0, 123.0, 251.5):
        rotated_S = _rotate_test_point(base_observation[:2], angle)
        rotated_observation = (
            rotated_S[0],
            rotated_S[1],
            base_observation[2] + angle,
        )
        rotated = normalize_result(
            invoke_planner(rotated_observation, config, show_output)
        )
        validate_basic_result(rotated)

        restored_point = _rotate_test_point(rotated["point"], -angle)
        point_error = math.dist(restored_point, base["point"])
        diameter_error = abs(rotated["diameter"] - base["diameter"])

        assert point_error <= 1e-7, (
            f"旋转 {angle}° 后坐标等变误差过大：{point_error} m"
        )
        assert diameter_error <= 1e-7, (
            f"旋转 {angle}° 后 J 不一致：{diameter_error} m"
        )
        records.append({
            "rotation_deg": angle,
            "point_equivariance_error_m": point_error,
            "diameter_invariance_error_m": diameter_error,
        })

    return records


def validate_fast_geometry_kernel():
    """直接覆盖规划最优解通常不会触发的快速几何正常与回退分支。"""
    try:
        from geometry_V7 import load_config
        from geometry_two_point_fast import TwoBearingFastEvaluator
    except ImportError as exc:
        return [{"status": "SKIPPED", "reason": str(exc)}]

    evaluator = TwoBearingFastEvaluator(
        (0.0, 0.0),
        0.0,
        1.0,
        config=load_config(),
    )
    checks = [
        (
            "regular_crossing",
            (0.0, -1000.0),
            90.0,
            True,
        ),
        (
            "parallel_fallback",
            (1000.0, 0.0),
            0.0,
            False,
        ),
        (
            "near_parallel_fallback",
            (1000.0, 0.0),
            1e-8,
            False,
        ),
    ]
    records = []
    for name, point, theta2, expected_certified in checks:
        result = evaluator.evaluate(point, theta2)
        assert result.certified is expected_certified, (
            f"{name}: certified={result.certified}，"
            f"预期为 {expected_certified}"
        )
        records.append(
            {
                "case": name,
                "certified": result.certified,
                "diameter_m": result.diameter,
                "fallback_reason": result.fallback_reason,
            }
        )
    return records


def verify_returned_J(observation, result, config):
    """轻量检查：返回的最坏直径有限且非负。"""
    diameter = result.get("diameter")
    if diameter is None:
        return
    assert math.isfinite(diameter), f"最坏直径不是有限值：{diameter}"
    assert diameter >= 0, f"最坏直径不能为负：{diameter}"


def validate_priority_1_3_ab(summaries):
    """同一输入比较 hybrid 与 strict，并检查三个优化分支确实生效。"""
    by_name = {summary["case"]: summary for summary in summaries}
    if "baseline_zero" in by_name:
        hybrid = by_name["baseline_zero"]
        strict = by_name.get("strict_verification_baseline")
    else:
        hybrid = by_name.get("quick_baseline")
        strict = by_name.get("quick_strict_verification")

    if hybrid is None or strict is None:
        return None

    j_error = abs(
        hybrid["worst_diameter_m"] - strict["worst_diameter_m"]
    )
    point_error = math.dist(
        hybrid["second_point"], strict["second_point"]
    )
    hybrid_eff = hybrid["efficiency"]
    strict_eff = strict["efficiency"]
    hybrid_exact = hybrid_eff["exact_geometry_calls"]
    strict_exact = strict_eff["exact_geometry_calls"]

    assert j_error <= 1e-4, (
        f"hybrid/strict 的最坏直径差 {j_error:.6g} m 超过 1e-4 m"
    )
    assert point_error <= 1e-7, (
        f"hybrid/strict 的第二检测点相差 {point_error:.6g} m"
    )
    assert hybrid_exact < strict_exact, (
        f"分层校核未减少精确调用：hybrid={hybrid_exact}, strict={strict_exact}"
    )
    assert hybrid_eff["staged_adversary_searches"] >= 1, (
        "hybrid 未进入端点/临界误差降维对抗搜索"
    )
    assert strict_eff["exact_adversarial_searches"] >= 1, (
        "strict 基线未执行全三维精确对抗搜索"
    )
    assert hybrid_eff["scenario_priority_first_evaluations"] >= 1, (
        "活动情景未使用上次最坏情景优先策略"
    )

    hybrid_seconds = hybrid["performance"]["wall_median_s"]
    strict_seconds = strict["performance"]["wall_median_s"]
    return {
        "hybrid_case": hybrid["case"],
        "strict_case": strict["case"],
        "worst_diameter_difference_m": j_error,
        "second_point_difference_m": point_error,
        "hybrid_exact_geometry_calls": hybrid_exact,
        "strict_exact_geometry_calls": strict_exact,
        "exact_call_reduction_ratio": 1.0 - hybrid_exact / strict_exact,
        "hybrid_median_s": hybrid_seconds,
        "strict_median_s": strict_seconds,
        "measured_speedup": (
            strict_seconds / hybrid_seconds if hybrid_seconds > 0 else None
        ),
        "hybrid_strict_fallbacks": hybrid_eff[
            "strict_verification_fallbacks"
        ],
        "hybrid_full_3d_fallbacks": hybrid_eff[
            "staged_full_3d_fallbacks"
        ],
    }


def validate_full_hybrid_strict_ab(config, show_output=False):
    """在五类代表输入上完整比较降维混合流程与旧版严格流程。"""
    if config is None or not is_dataclass(config):
        return [{"status": "SKIPPED", "reason": "PlannerConfig 不可用"}]

    observations = (
        (0.0, 0.0, 0.0),
        (600.0, 400.0, 210.0),
        (1700.0, 0.0, 90.0),
        (1790.0, 0.0, 90.0),
        (2200.0, 0.0, 180.0),
    )
    records = []
    for observation in observations:
        results = {}
        elapsed = {}
        for mode in ("hybrid", "strict"):
            mode_config = replace(config, verification_mode=mode)
            start = time.perf_counter()
            results[mode] = normalize_result(
                invoke_planner(observation, mode_config, show_output)
            )
            elapsed[mode] = time.perf_counter() - start
            validate_basic_result(results[mode])

        j_error = abs(
            results["hybrid"]["diameter"] - results["strict"]["diameter"]
        )
        point_error = math.dist(
            results["hybrid"]["point"], results["strict"]["point"]
        )
        assert j_error <= 1e-4, (
            f"{observation}: hybrid/strict 的 J 相差 {j_error:.6g} m"
        )
        assert point_error <= 1e-7, (
            f"{observation}: hybrid/strict 的 P 相差 {point_error:.6g} m"
        )
        hybrid_exact = results["hybrid"]["diagnostics"].get(
            "exact_geometry_calls"
        )
        strict_exact = results["strict"]["diagnostics"].get(
            "exact_geometry_calls"
        )
        assert hybrid_exact < strict_exact, (
            f"{observation}: hybrid 未减少精确几何调用"
        )
        records.append({
            "observation": observation,
            "worst_diameter_difference_m": j_error,
            "second_point_difference_m": point_error,
            "hybrid_exact_geometry_calls": hybrid_exact,
            "strict_exact_geometry_calls": strict_exact,
            "measured_speedup": elapsed["strict"] / elapsed["hybrid"],
        })
    return records


# ============================================================
# 五、候选区域几何分析
# ============================================================

def compute_candidate_region_metrics(observation, config, sample_count=5000):
    """计算第二检测点候选区域的几何指标。

    使用 planner 内部的 SourceRegion + 接收约束：
    * AABB 范围与面积；
    * 蒙特卡洛采样的真实可行面积（圆交集）。
    """
    if config is None:
        return None

    try:
        from planner import _ContinuousPlanner
        from geometry_V7 import load_config
    except ImportError as exc:
        return {"candidate_region_error": f"无法导入内部结构：{exc}"}

    S = (float(observation[0]), float(observation[1]))
    theta1 = float(observation[2]) % 360.0

    try:
        geometry_config = load_config()
        solver = _ContinuousPlanner(S, theta1, config, geometry_config)

        initial_params = solver.region.initial_parameters()
        sources = [solver.region.point(p) for p in initial_params]
        sources = [s for s in sources if s is not None]

        if not sources:
            return {"candidate_region_error": "初始源情景为空"}

        (xlo, xhi), (ylo, yhi) = solver.active_bounds(sources)
        width = float(xhi - xlo)
        height = float(yhi - ylo)
        bbox_area = width * height

        rng = np.random.default_rng(20260911)
        samples = rng.uniform(
            low=[xlo, ylo], high=[xhi, yhi], size=(sample_count, 2)
        )
        feasible_count = 0
        for px, py in samples:
            if solver.active_feasible((float(px), float(py)), sources):
                feasible_count += 1
        sampled_area = (
            bbox_area * feasible_count / sample_count if sample_count else 0.0
        )

        return {
            "candidate_region_x_lo_m": xlo,
            "candidate_region_x_hi_m": xhi,
            "candidate_region_y_lo_m": ylo,
            "candidate_region_y_hi_m": yhi,
            "candidate_region_width_m": width,
            "candidate_region_height_m": height,
            "candidate_region_bbox_area_m2": bbox_area,
            "candidate_region_sampled_area_m2": sampled_area,
            "candidate_region_sample_count": sample_count,
            "candidate_region_initial_source_count": len(sources),
        }
    except Exception as exc:
        return {"candidate_region_error": f"{type(exc).__name__}: {exc}"}


# ============================================================
# 六、性能测量
# ============================================================

def timed_run(observation, config, show_output):
    gc.collect()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    raw_result = invoke_planner(observation, config, show_output)

    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start

    result = normalize_result(raw_result)
    validate_basic_result(result)
    validate_receive_constraint(observation, result, config)

    return {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "result": result,
    }


def measure_memory(observation, config, show_output):
    """tracemalloc 主要统计 Python 管理的内存。"""
    gc.collect()
    tracemalloc.start()

    try:
        raw_result = invoke_planner(observation, config, show_output)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    result = normalize_result(raw_result)
    validate_basic_result(result)

    return peak_bytes / (1024 ** 2)


def check_repeatability(results, coordinate_tolerance=1e-7):
    """同一输入重复运行时，检查结果是否稳定。"""
    if len(results) < 2:
        return True

    reference = results[0]

    for current in results[1:]:
        if current["status"] != reference["status"]:
            return False

        if not math.isclose(
            current["diameter"],
            reference["diameter"],
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            return False

        if current["point"] is None or reference["point"] is None:
            if current["point"] != reference["point"]:
                return False
            continue

        for actual, expected in zip(current["point"], reference["point"]):
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=coordinate_tolerance,
            ):
                return False

    return True


def summarize_case(
    case_name,
    observation,
    runs,
    peak_memory_mb=None,
    region_metrics=None,
    *,
    description="",
    config_overrides=None,
):
    wall_times = [run["wall_seconds"] for run in runs]
    cpu_times = [run["cpu_seconds"] for run in runs]
    normalized_results = [run["result"] for run in runs]
    representative = normalized_results[-1]
    diagnostics = representative["diagnostics"]

    geometry_calls = diagnostics.get("geometry_calls")
    geometry_cache_hits = diagnostics.get("geometry_cache_hits")
    geometry_requests = (
        geometry_calls + geometry_cache_hits
        if geometry_calls is not None and geometry_cache_hits is not None
        else None
    )
    geometry_cache_hit_rate = (
        geometry_cache_hits / geometry_requests
        if geometry_requests
        else None
    )
    proxy_evaluations = diagnostics.get("proxy_evaluations")
    point_objective_calls = diagnostics.get("point_objective_calls")
    error_searches = diagnostics.get("error_searches")
    source_searches = diagnostics.get("source_searches")
    scenario_rounds = diagnostics.get("scenario_rounds")

    wall_median = statistics.median(wall_times)
    geometry_per_second = (
        geometry_calls / wall_median
        if geometry_calls and wall_median > 0
        else None
    )
    proxy_per_second = (
        proxy_evaluations / wall_median
        if proxy_evaluations and wall_median > 0
        else None
    )

    summary = {
        "case": case_name,
        "description": description,
        "observation": list(observation),
        "config_overrides": dict(config_overrides or {}),
        "status": representative["status"],
        "second_point": (
            list(representative["point"]) if representative["point"] else None
        ),
        "worst_diameter_m": representative["diameter"],
        "worst_source": (
            list(representative["worst_source"])
            if representative["worst_source"]
            else None
        ),
        "worst_second_error_deg": representative["worst_second_error_deg"],
        "receive_margin_m": representative["receive_margin_m"],
        "active_source_count": len(representative["active_sources"]),
        "active_scenario_count": len(representative["active_scenarios"]),
        "repeatable": check_repeatability(normalized_results),
        "performance": {
            "runs": len(runs),
            "wall_min_s": min(wall_times),
            "wall_median_s": wall_median,
            "wall_mean_s": statistics.mean(wall_times),
            "wall_max_s": max(wall_times),
            "wall_stdev_s": (
                statistics.pstdev(wall_times) if len(wall_times) > 1 else 0.0
            ),
            "cpu_median_s": statistics.median(cpu_times),
            "peak_python_memory_mb": peak_memory_mb,
        },
        # 寻找候选区效率
        "efficiency": {
            "verification_mode": diagnostics.get("verification_mode"),
            "verification_top_k": diagnostics.get("verification_top_k"),
            "geometry_calls": geometry_calls,
            "geometry_calls_per_second": geometry_per_second,
            "geometry_cache_hits": geometry_cache_hits,
            "geometry_cache_hit_rate": geometry_cache_hit_rate,
            "fast_geometry_enabled": diagnostics.get(
                "fast_geometry_enabled"
            ),
            "fast_geometry_attempts": diagnostics.get(
                "fast_geometry_attempts"
            ),
            "fast_geometry_successes": diagnostics.get(
                "fast_geometry_successes"
            ),
            "fast_geometry_fallbacks": diagnostics.get(
                "fast_geometry_fallbacks"
            ),
            "fast_geometry_success_rate": diagnostics.get(
                "fast_geometry_success_rate"
            ),
            "exact_geometry_calls": diagnostics.get(
                "exact_geometry_calls"
            ),
            "exact_geometry_cache_hits": diagnostics.get(
                "exact_geometry_cache_hits"
            ),
            "exact_adversarial_searches": diagnostics.get(
                "exact_adversarial_searches"
            ),
            "selective_exact_checks": diagnostics.get(
                "selective_exact_checks"
            ),
            "strict_verification_fallbacks": diagnostics.get(
                "strict_verification_fallbacks"
            ),
            "staged_adversary_searches": diagnostics.get(
                "staged_adversary_searches"
            ),
            "staged_fixed_error_searches": diagnostics.get(
                "staged_fixed_error_searches"
            ),
            "staged_full_3d_fallbacks": diagnostics.get(
                "staged_full_3d_fallbacks"
            ),
            "adversary_screen_evaluations": diagnostics.get(
                "adversary_screen_evaluations"
            ),
            "error_cache_hits": diagnostics.get("error_cache_hits"),
            "cutoff_pruned_error_searches": diagnostics.get(
                "cutoff_pruned_error_searches"
            ),
            "cutoff_pruned_point_evaluations": diagnostics.get(
                "cutoff_pruned_point_evaluations"
            ),
            "skipped_source_evaluations": diagnostics.get(
                "skipped_source_evaluations"
            ),
            "scenario_geometry_evaluations": diagnostics.get(
                "scenario_geometry_evaluations"
            ),
            "cutoff_pruned_scenario_evaluations": diagnostics.get(
                "cutoff_pruned_scenario_evaluations"
            ),
            "skipped_scenario_evaluations": diagnostics.get(
                "skipped_scenario_evaluations"
            ),
            "scenario_priority_updates": diagnostics.get(
                "scenario_priority_updates"
            ),
            "scenario_priority_first_evaluations": diagnostics.get(
                "scenario_priority_first_evaluations"
            ),
            "proxy_evaluations": proxy_evaluations,
            "proxy_evaluations_per_second": proxy_per_second,
            "point_objective_calls": point_objective_calls,
            "error_searches": error_searches,
            "source_searches": source_searches,
            "scenario_rounds": scenario_rounds,
            "active_objective_m": diagnostics.get("active_objective_m"),
            "continuous_adversarial_objective_m": diagnostics.get(
                "continuous_adversarial_objective_m"
            ),
            "max_receive_violation_m": diagnostics.get(
                "max_receive_violation_m"
            ),
        },
    }

    if region_metrics:
        summary["candidate_region"] = region_metrics

    return summary


# ============================================================
# 七、性能热点分析
# ============================================================

def print_profile(observation, config, show_output, line_count=30):
    profiler = cProfile.Profile()

    profiler.enable()
    invoke_planner(observation, config, show_output)
    profiler.disable()

    stream = io.StringIO()
    statistics_report = pstats.Stats(
        profiler, stream=stream
    ).strip_dirs().sort_stats("cumulative")
    statistics_report.print_stats(line_count)

    print("\n========== cProfile 累计耗时热点 ==========")
    print(stream.getvalue())


# ============================================================
# 八、测试入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="测试问题二 planner 的正确性和运行性能"
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "standard", "stress"),
        default="quick",
        help="quick 为快速冒烟测试；standard 使用默认精度；stress 增加案例",
    )
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument(
        "--memory",
        action="store_true",
        help="额外运行一次以估计 Python 峰值内存",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="对第一个测试案例额外执行一次 cProfile",
    )
    parser.add_argument(
        "--verify-j",
        action="store_true",
        help="检查最终返回的 J 是否有限非负",
    )
    parser.add_argument(
        "--require-continuous",
        action="store_true",
        help="要求输出点不能同时落在原 100 米网格上",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="单案例中位运行时间上限，超过时测试失败",
    )
    parser.add_argument(
        "--max-peak-mb",
        type=float,
        default=None,
        help="Python 峰值内存上限，需要同时使用 --memory",
    )
    parser.add_argument(
        "--show-planner-output",
        action="store_true",
        help="显示 planner 自身的过程输出",
    )
    parser.add_argument(
        "--rotation-check",
        action="store_true",
        help="额外运行非中心案例的旋转等变性回归测试",
    )
    parser.add_argument(
        "--full-ab-check",
        action="store_true",
        help="在5类代表输入上额外比较 hybrid 与 strict 的 P、J 和精确调用数",
    )
    parser.add_argument(
        "--no-fast-kernel-check",
        action="store_true",
        help="跳过快速几何模块的正常/近平行回退分支检查",
    )
    parser.add_argument(
        "--no-region-metrics",
        action="store_true",
        help="跳过候选区域几何分析，仅做时间测量",
    )
    parser.add_argument(
        "--region-sample-count",
        type=int,
        default=5000,
        help="候选区域面积采样点数（默认 5000）",
    )
    args = parser.parse_args()

    default_repeats = {"quick": 1, "standard": 3, "stress": 5}
    default_warmups = {"quick": 0, "standard": 1, "stress": 1}

    repeats = (
        args.repeats
        if args.repeats is not None
        else default_repeats[args.mode]
    )
    warmups = (
        args.warmups
        if args.warmups is not None
        else default_warmups[args.mode]
    )

    if repeats <= 0 or warmups < 0:
        parser.error("repeats 必须大于 0，warmups 不能为负数")
    if args.region_sample_count <= 0:
        parser.error("region-sample-count 必须大于 0")

    config = build_config(args.mode)
    cases = TEST_CASES[args.mode]
    summaries = []
    failures = []

    print("========== Planner 性能测试开始 ==========")
    print(f"模式：{args.mode}")
    print(f"案例数：{len(cases)}")
    print(f"每个案例预热：{warmups} 次")
    print(f"每个案例计时：{repeats} 次")
    print(
        "候选区域采样："
        + (
            "关闭"
            if args.no_region_metrics
            else f"{args.region_sample_count} 点/案例"
        )
    )

    if config is not None:
        print("检测到新版 PlannerConfig 接口")
        if is_dataclass(config):
            print("搜索配置：", json.dumps(asdict(config), ensure_ascii=False))
    else:
        print("未检测到 PlannerConfig，将使用旧版默认参数")

    for index, case in enumerate(cases, start=1):
        observation = case.observation
        case_name = case.name
        case_config = build_case_config(config, case)

        print(
            f"\n[{index:02d}:{case_name}] "
            f"S=({observation[0]}, {observation[1]}), "
            f"theta={observation[2]}°"
        )
        print(f"  目的: {case.description}")
        if case.config_overrides:
            print(
                "  单案例配置覆盖: "
                + json.dumps(case.config_overrides, ensure_ascii=False)
            )

        for _ in range(warmups):
            invoke_planner(
                observation,
                case_config,
                args.show_planner_output,
            )

        runs = []
        for repeat_index in range(repeats):
            run = timed_run(
                observation,
                case_config,
                args.show_planner_output,
            )
            runs.append(run)
            print(
                f"  run {repeat_index + 1}: "
                f"{run['wall_seconds']:.6f} s, "
                f"J={run['result']['diameter']:.6f}, "
                f"P={run['result']['point']}"
            )

        representative = runs[-1]["result"]

        if args.require_continuous:
            validate_continuous_output(representative)

        if args.verify_j:
            verify_returned_J(observation, representative, case_config)

        peak_memory_mb = None
        if args.memory:
            peak_memory_mb = measure_memory(
                observation, case_config, args.show_planner_output
            )

        region_metrics = None
        if not args.no_region_metrics:
            region_metrics = compute_candidate_region_metrics(
                observation,
                case_config,
                sample_count=args.region_sample_count,
            )

        summary = summarize_case(
            case_name,
            observation,
            runs,
            peak_memory_mb,
            region_metrics,
            description=case.description,
            config_overrides=case.config_overrides,
        )
        summaries.append(summary)

        # 控制台简洁输出候选区域
        if region_metrics and "candidate_region_error" not in region_metrics:
            print(
                f"  候选区域: "
                f"宽 {region_metrics['candidate_region_width_m']:.2f} m × "
                f"高 {region_metrics['candidate_region_height_m']:.2f} m, "
                f"AABB 面积 {region_metrics['candidate_region_bbox_area_m2']:.0f} m², "
                f"可行面积≈{region_metrics['candidate_region_sampled_area_m2']:.0f} m²"
            )
        elif region_metrics:
            print(
                "  候选区域分析失败: "
                + str(region_metrics.get("candidate_region_error"))
            )

        # 控制台简洁输出效率
        eff = summary["efficiency"]
        cache_rate = eff["geometry_cache_hit_rate"]
        cache_text = f"{cache_rate:.1%}" if cache_rate is not None else "n/a"
        fast_rate = eff["fast_geometry_success_rate"]
        fast_text = f"{fast_rate:.1%}" if fast_rate is not None else "n/a"
        print(
            f"  效率: geometry_calls={eff['geometry_calls']}, "
            f"cache_hit_rate={cache_text}, "
            f"proxy_evals={eff['proxy_evaluations']}, "
            f"point_calls={eff['point_objective_calls']}, "
            f"pruned_points={eff['cutoff_pruned_point_evaluations']}, "
            f"active_scenarios={summary['active_scenario_count']}, "
            f"scenario_evals={eff['scenario_geometry_evaluations']}, "
            f"fast_success={fast_text}, "
            f"exact_calls={eff['exact_geometry_calls']}, "
            f"verify={eff['verification_mode']}, "
            f"selective_exact={eff['selective_exact_checks']}, "
            f"strict_fallbacks={eff['strict_verification_fallbacks']}, "
            f"staged_3d_fallbacks={eff['staged_full_3d_fallbacks']}, "
            f"scenario_rounds={eff['scenario_rounds']}"
        )

        median_seconds = summary["performance"]["wall_median_s"]

        if not summary["repeatable"]:
            failures.append(f"{case_name}: 重复运行结果不稳定")

        actual_rounds = eff["scenario_rounds"]
        if (
            case.minimum_scenario_rounds is not None
            and (
                actual_rounds is None
                or actual_rounds < case.minimum_scenario_rounds
            )
        ):
            failures.append(
                f"{case_name}: 情景轮数 {actual_rounds} 小于预期下限 "
                f"{case.minimum_scenario_rounds}"
            )

        actual_fast_enabled = eff["fast_geometry_enabled"]
        if (
            case.expected_fast_geometry_enabled is not None
            and actual_fast_enabled is not case.expected_fast_geometry_enabled
        ):
            failures.append(
                f"{case_name}: fast_geometry_enabled={actual_fast_enabled}，"
                f"预期为 {case.expected_fast_geometry_enabled}"
            )

        actual_verification_mode = eff["verification_mode"]
        if (
            case.expected_verification_mode is not None
            and actual_verification_mode != case.expected_verification_mode
        ):
            failures.append(
                f"{case_name}: verification_mode={actual_verification_mode}，"
                f"预期为 {case.expected_verification_mode}"
            )

        actual_strict_fallbacks = eff["strict_verification_fallbacks"]
        if (
            case.minimum_strict_fallbacks is not None
            and (
                actual_strict_fallbacks is None
                or actual_strict_fallbacks < case.minimum_strict_fallbacks
            )
        ):
            failures.append(
                f"{case_name}: 严格回退次数 {actual_strict_fallbacks} 小于预期下限 "
                f"{case.minimum_strict_fallbacks}"
            )

        if (
            args.max_seconds is not None
            and median_seconds > args.max_seconds
        ):
            failures.append(
                f"{case_name}: 中位耗时 {median_seconds:.6f}s "
                f"超过上限 {args.max_seconds:.6f}s"
            )

        if (
            args.max_peak_mb is not None
            and peak_memory_mb is not None
            and peak_memory_mb > args.max_peak_mb
        ):
            failures.append(
                f"{case_name}: 峰值内存 {peak_memory_mb:.3f}MB "
                f"超过上限 {args.max_peak_mb:.3f}MB"
            )

    print("\n========== 汇总 ==========")
    print(json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False))

    try:
        priority_ab = validate_priority_1_3_ab(summaries)
        if priority_ab is not None:
            print("\n========== 优先级1-3 A/B检查 ==========")
            print(json.dumps(priority_ab, ensure_ascii=False, indent=2))
    except (AssertionError, TypeError, ZeroDivisionError) as exc:
        failures.append(f"优先级1-3 A/B检查失败: {exc}")

    if args.full_ab_check:
        try:
            full_ab_records = validate_full_hybrid_strict_ab(
                config, args.show_planner_output
            )
            print("\n========== 五类输入完整 hybrid/strict 检查 ==========")
            print(json.dumps(full_ab_records, ensure_ascii=False, indent=2))
        except (AssertionError, TypeError, ValueError, ArithmeticError) as exc:
            failures.append(f"完整 hybrid/strict 检查失败: {exc}")

    if args.rotation_check:
        rotation_records = validate_rotation_invariance(
            config, args.show_planner_output
        )
        print("\n========== 旋转不变性检查 ==========")
        print(json.dumps(rotation_records, ensure_ascii=False, indent=2))

    if not args.no_fast_kernel_check:
        try:
            fast_kernel_records = validate_fast_geometry_kernel()
            print("\n========== 快速几何分支检查 ==========")
            print(
                json.dumps(
                    fast_kernel_records,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except (AssertionError, ValueError, ArithmeticError) as exc:
            failures.append(f"快速几何分支检查失败: {exc}")

    if args.profile:
        profile_case = cases[0]
        print_profile(
            profile_case.observation,
            build_case_config(config, profile_case),
            args.show_planner_output,
        )

    if failures:
        print("\n========== 测试失败 ==========")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)

    print("\n========== 所有测试通过 ==========")


if __name__ == "__main__":
    main()
