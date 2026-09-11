"""
planner.py

问题二：连续鲁棒第二检测点规划器。

数学目标：
    min_P max_{G in Omega_G, e in [-epsilon, epsilon]} D(P, G, e)

其中：
    P       第二检测点，连续二维坐标；
    G       第一次测向后连续可能源区域中的位置；
    e       第二次测向误差；
    D       与 geometry_V7 半平面模型一致的定位区域直径。

算法：
    1. 连续参数化第一次测向后的可能源区域；
    2. 用代理指标 U 对第二检测点进行连续全局预搜索；
    3. 用两检测点专用快速评价器优化有限活动情景 (G,e)；
    4. 在误差端点和临界误差上降维搜索最坏 (G,e)，并低成本三维筛查；
    5. 用 geometry_V7 分层校核高风险候选，异常时自动回退全量严格搜索；
    6. 将新的最坏 (G,e) 加入活动情景集合；
    7. 重复优化，直到最坏情景不能显著恶化当前结果。

说明：
    本程序使用确定性的低差异多起点 + 自适应模式搜索。
    普通四半平面评价走专用浮点快路，高风险候选和最终结果走 geometry_V7；
    strict 模式可恢复全量精确校核与三维对抗搜索。
    外层搜索本身不依赖 SciPy，但仍属于数值连续优化，不构成解析全局最优证明。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Iterator, Sequence

from geometry_V7 import GeometryConfig, load_config, solve_halfplanes
from geometry_two_point_fast import TwoBearingFastEvaluator


Point = tuple[float, float]
Scenario = tuple[Point, float]
AdversarialCandidate = tuple[float, Point, float]


class _CutoffReached(Exception):
    """候选点已经不可能优于当前最优值时，立即结束内层搜索。"""

    def __init__(self, value: float, parameter: float):
        super().__init__()
        self.value = value
        self.parameter = parameter


# ============================================================
# Part 1 配置和输出结构
# ============================================================

@dataclass(frozen=True)
class PlannerConfig:
    """问题二模型常数与连续搜索参数。"""

    # ---------- 题目常数 ----------
    target_radius_m: float = 1800.0
    reception_radius_min_m: float = 1000.0
    reception_radius_max_m: float = 1500.0
    no_bearing_radius_m: float = 5.0
    bearing_error_deg: float = 1.0

    # ---------- 两检测点专用几何快速路径 ----------
    use_fast_geometry: bool = True

    # ---------- 分层精确校核与降维对抗搜索 ----------
    verification_mode: str = "hybrid"
    verification_top_k: int = 6
    verification_disagreement_tolerance_m: float = 1e-5
    adversary_screen_sample_count: int = 24
    adversary_stage_gap_tolerance_m: float = 0.05

    # ---------- 第二检测点连续搜索 ----------
    point_sample_count: int = 128
    point_start_count: int = 6
    proxy_max_iterations: int = 50
    exact_max_iterations: int = 18
    point_initial_step_m: float = 160.0
    point_tolerance_m: float = 1.0

    # ---------- 连续源位置对抗搜索 ----------
    source_sample_count: int = 64
    source_start_count: int = 8
    source_max_iterations: int = 35
    source_parameter_tolerance: float = 1e-4

    # ---------- 连续误差搜索 ----------
    error_sample_count: int = 9
    error_start_count: int = 4
    error_max_iterations: int = 24
    error_tolerance_deg: float = 1e-3

    # ---------- 情景生成 ----------
    max_scenario_rounds: int = 8
    scenario_gap_tolerance_m: float = 0.2
    scenario_merge_tolerance_m: float = 1e-5
    scenario_error_merge_tolerance_deg: float = 1e-6

    # ---------- 数值容差 ----------
    receive_tolerance_m: float = 1e-6
    objective_improvement_tolerance_m: float = 1e-7
    parallel_sine_tolerance: float = 1e-12

    def __post_init__(self):
        if type(self.use_fast_geometry) is not bool:
            raise ValueError("use_fast_geometry 必须是布尔值")
        if self.verification_mode not in {"hybrid", "strict"}:
            raise ValueError("verification_mode 必须是 'hybrid' 或 'strict'")

        positive_numbers = (
            "target_radius_m",
            "reception_radius_min_m",
            "reception_radius_max_m",
            "no_bearing_radius_m",
            "bearing_error_deg",
            "verification_disagreement_tolerance_m",
            "adversary_stage_gap_tolerance_m",
            "point_initial_step_m",
            "point_tolerance_m",
            "source_parameter_tolerance",
            "error_tolerance_deg",
            "scenario_gap_tolerance_m",
            "scenario_merge_tolerance_m",
            "scenario_error_merge_tolerance_deg",
            "receive_tolerance_m",
            "objective_improvement_tolerance_m",
            "parallel_sine_tolerance",
        )

        for name in positive_numbers:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} 必须是有限正数")

        positive_integers = (
            "point_sample_count",
            "verification_top_k",
            "adversary_screen_sample_count",
            "point_start_count",
            "proxy_max_iterations",
            "exact_max_iterations",
            "source_sample_count",
            "source_start_count",
            "source_max_iterations",
            "error_sample_count",
            "error_start_count",
            "error_max_iterations",
            "max_scenario_rounds",
        )

        for name in positive_integers:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")

        if self.reception_radius_min_m > self.reception_radius_max_m:
            raise ValueError("最小有效接收半径不能大于最大有效接收半径")

        if not 0 < self.bearing_error_deg < 90:
            raise ValueError("bearing_error_deg 必须位于 (0, 90) 度")


@dataclass
class PlannerResult:
    """问题二输出，接口形式与 GeometryResult 类似。"""

    status: str
    second_point: Point | None = None
    worst_diameter: float | None = None
    proxy_score: float | None = None
    worst_source: Point | None = None
    worst_second_error_deg: float | None = None
    receive_margin_m: float | None = None
    first_observation: tuple[float, float, float] | None = None
    active_sources: list[Point] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    active_scenarios: list[Scenario] = field(default_factory=list)

    @property
    def best_point(self):
        return self.second_point

    @property
    def best_J(self):
        return self.worst_diameter

    def __iter__(self) -> Iterator:
        """兼容 best_point, best_J = result。"""
        yield self.second_point
        yield self.worst_diameter

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Part 2 输入和基础几何
# ============================================================

def _point(value: Iterable[float], label: str) -> Point:
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 必须包含两个有限数值") from exc

    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{label} 必须包含两个有限数值")

    return values[0], values[1]


def _observation(
    value: Iterable[float],
) -> tuple[float, float, float]:
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "第一次观测必须为 (x, y, bearing_deg)"
        ) from exc

    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(
            "第一次观测必须为三个有限数值 (x, y, bearing_deg)"
        )

    return values[0], values[1], values[2] % 360.0


def distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.hypot(
        float(p1[0]) - float(p2[0]),
        float(p1[1]) - float(p2[1]),
    )


def _angle_difference_deg(a: float, b: float) -> float:
    """
    返回 a-b 的有符号最小角差，范围 [-180, 180)。
    """
    return (a - b + 180.0) % 360.0 - 180.0


def _rotate_point(point: Point, angle_deg: float) -> Point:
    """绕目标圆圆心（全局原点）逆时针旋转二维点。"""
    normalized = float(angle_deg) % 360.0
    if normalized == 0.0:
        return float(point[0]), float(point[1])

    angle = math.radians(normalized)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x = cosine * point[0] - sine * point[1]
    y = sine * point[0] + cosine * point[1]

    # 清除旋转原点或坐标轴时出现的无意义 -0.0 / 1e-14。
    scale = max(1.0, abs(point[0]), abs(point[1]))
    tolerance = 1e-14 * scale
    if abs(x) <= tolerance:
        x = 0.0
    if abs(y) <= tolerance:
        y = 0.0
    return float(x), float(y)


def _restore_global_result(
    result: PlannerResult,
    original_observation: tuple[float, float, float],
    rotation_deg: float,
    canonical_first_point: Point,
) -> PlannerResult:
    """将规范坐标系中的所有位置型输出旋转回用户坐标系。"""
    if result.second_point is not None:
        result.second_point = _rotate_point(result.second_point, rotation_deg)
    if result.worst_source is not None:
        result.worst_source = _rotate_point(result.worst_source, rotation_deg)
    result.active_sources = [
        _rotate_point(source, rotation_deg) for source in result.active_sources
    ]
    result.active_scenarios = [
        (_rotate_point(source, rotation_deg), error)
        for source, error in result.active_scenarios
    ]
    result.first_observation = original_observation

    history = result.diagnostics.get("scenario_history", [])
    for record in history:
        point = record.get("point")
        if point is not None:
            record["point"] = _rotate_point(
                (float(point[0]), float(point[1])), rotation_deg
            )
        for key in ("worst_source", "receive_source"):
            source = record.get(key)
            if source is not None:
                record[key] = _rotate_point(
                    (float(source[0]), float(source[1])), rotation_deg
                )

    result.diagnostics["coordinate_frame"] = {
        "method": "rotate_about_target_center_to_zero_bearing",
        "rotation_to_canonical_deg": -rotation_deg,
        "rotation_to_global_deg": rotation_deg,
        "canonical_first_point": canonical_first_point,
        "canonical_bearing_deg": 0.0,
        "position_outputs_restored_to_global": True,
    }
    return result


def _bearing_halfplanes(
    x: float,
    y: float,
    theta_deg: float,
    error_deg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """构造一次测向对应的两个半平面，与 geometry_V7 完全同义。"""
    lower = math.radians(theta_deg - error_deg)
    upper = math.radians(theta_deg + error_deg)
    rows = []
    for A, B in (
        (math.sin(lower), -math.cos(lower)),
        (-math.sin(upper), math.cos(upper)),
    ):
        rows.append((A, B, math.fsum((A * x, B * y))))
    return rows[0], rows[1]


# ============================================================
# Part 3 连续可能源区域
# ============================================================

@dataclass(frozen=True)
class SourceRegion:
    """
    使用两个归一化参数表示连续可能源位置：

        q_angle in [0,1]
        q_radius in [0,1]

    q_angle 对应第一次示向度误差区间；
    q_radius 对应该方向射线与目标圆、接收距离区间的交集。
    """

    S: Point
    theta1_deg: float
    config: PlannerConfig

    def angle_from_parameter(self, q_angle: float) -> float:
        q_angle = min(1.0, max(0.0, float(q_angle)))
        error = self.config.bearing_error_deg

        return self.theta1_deg - error + 2.0 * error * q_angle

    def radial_interval(
        self,
        q_angle: float,
    ) -> tuple[float, float] | None:
        """
        求给定射线与目标圆的交段，再与 (5,1500] 米相交。
        """
        angle_deg = self.angle_from_parameter(q_angle)
        angle_rad = math.radians(angle_deg)

        ux = math.cos(angle_rad)
        uy = math.sin(angle_rad)

        sx, sy = self.S
        target_radius = self.config.target_radius_m

        # |S + r*u|^2 <= R^2
        b = sx * ux + sy * uy
        c = sx * sx + sy * sy - target_radius * target_radius
        discriminant = b * b - c

        if discriminant < 0:
            return None

        root = math.sqrt(max(0.0, discriminant))
        circle_lower = -b - root
        circle_upper = -b + root

        lower = max(
            math.nextafter(
                self.config.no_bearing_radius_m,
                math.inf,
            ),
            circle_lower,
            0.0,
        )

        upper = min(
            self.config.reception_radius_max_m,
            circle_upper,
        )

        if upper < lower:
            return None

        return lower, upper

    def point(
        self,
        parameters: Sequence[float],
    ) -> Point | None:
        if len(parameters) != 2:
            raise ValueError("源位置参数必须为二维")

        q_angle = min(1.0, max(0.0, float(parameters[0])))
        q_radius = min(1.0, max(0.0, float(parameters[1])))

        interval = self.radial_interval(q_angle)
        if interval is None:
            return None

        lower, upper = interval
        radius = lower + q_radius * (upper - lower)

        angle_rad = math.radians(
            self.angle_from_parameter(q_angle)
        )

        return (
            self.S[0] + radius * math.cos(angle_rad),
            self.S[1] + radius * math.sin(angle_rad),
        )

    def initial_parameters(self) -> list[tuple[float, float]]:
        """
        初始活动情景只用于启动约束生成，不是固定 F1 网格。
        """
        values = []

        for q_angle in (0.0, 0.5, 1.0):
            for q_radius in (0.0, 0.5, 1.0):
                parameters = (q_angle, q_radius)

                if self.point(parameters) is not None:
                    values.append(parameters)

        return values


# ============================================================
# Part 4 通用连续模式搜索
# ============================================================

_HALTON_BASES = (2, 3, 5, 7, 11)


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base

    while index:
        index, digit = divmod(index, base)
        result += digit * factor
        factor /= base

    return result


def _halton_points(
    dimension: int,
    count: int,
) -> list[tuple[float, ...]]:
    if dimension > len(_HALTON_BASES):
        raise ValueError("当前低差异序列维数不足")

    return [
        tuple(
            _radical_inverse(index, _HALTON_BASES[axis])
            for axis in range(dimension)
        )
        for index in range(1, count + 1)
    ]


def _is_better(
    candidate: float,
    current: float,
    maximize: bool,
    tolerance: float = 0.0,
) -> bool:
    if maximize:
        return candidate > current + tolerance

    return candidate < current - tolerance


def _pattern_search(
    start: Sequence[float],
    objective: Callable[[tuple[float, ...]], float],
    bounds: Sequence[tuple[float, float]],
    *,
    maximize: bool,
    max_iterations: int,
    tolerances: Sequence[float],
    initial_steps: Sequence[float] | None = None,
    improvement_tolerance: float = 0.0,
) -> tuple[tuple[float, ...], float, int]:
    """
    连续坐标模式搜索。

    每次只在当前点附近试探，失败后缩小步长，因此最终坐标
    不受预先规定的固定网格限制。
    """
    dimension = len(bounds)

    point = tuple(
        min(high, max(low, float(value)))
        for value, (low, high) in zip(start, bounds)
    )

    if initial_steps is None:
        steps = [
            0.25 * max(high - low, tolerance)
            for (low, high), tolerance in zip(bounds, tolerances)
        ]
    else:
        steps = list(initial_steps)

    value = objective(point)
    evaluations = 1

    for _ in range(max_iterations):
        if all(
            step <= tolerance
            for step, tolerance in zip(steps, tolerances)
        ):
            break

        # 最大化时一旦发现正无穷，已经无需继续。
        if maximize and value == math.inf:
            break

        best_point = point
        best_value = value

        for axis in range(dimension):
            for sign in (-1.0, 1.0):
                trial = list(point)
                low, high = bounds[axis]

                trial[axis] = min(
                    high,
                    max(low, trial[axis] + sign * steps[axis]),
                )
                trial = tuple(trial)

                if trial == point:
                    continue

                trial_value = objective(trial)
                evaluations += 1

                if _is_better(
                    trial_value,
                    best_value,
                    maximize,
                    improvement_tolerance,
                ):
                    best_point = trial
                    best_value = trial_value

        if best_point == point:
            steps = [step * 0.5 for step in steps]
        else:
            point = best_point
            value = best_value

    return point, value, evaluations


def _pattern_search_min_with_cutoff(
    start: Sequence[float],
    full_objective: Callable[[tuple[float, ...]], float],
    cutoff_objective: Callable[[tuple[float, ...], float], tuple[float, bool]],
    bounds: Sequence[tuple[float, float]],
    *,
    max_iterations: int,
    tolerances: Sequence[float],
    initial_steps: Sequence[float],
    improvement_tolerance: float = 0.0,
) -> tuple[tuple[float, ...], float, int]:
    """带安全截止值的最小化模式搜索。

    ``cutoff_objective`` 可在目标下界达到当前最优值时返回
    ``complete=False``。这种候选不可能改善结果，无需算完其余源与误差。
    """
    point = tuple(
        min(high, max(low, float(value)))
        for value, (low, high) in zip(start, bounds)
    )
    steps = list(initial_steps)
    value = full_objective(point)
    evaluations = 1

    for _ in range(max_iterations):
        if all(
            step <= tolerance
            for step, tolerance in zip(steps, tolerances)
        ):
            break

        best_point = point
        best_value = value

        if len(bounds) == 2:
            diagonal = 1.0 / math.sqrt(2.0)
            directions = (
                (1.0, 0.0),
                (-1.0, 0.0),
                (0.0, 1.0),
                (0.0, -1.0),
                (diagonal, diagonal),
                (diagonal, -diagonal),
                (-diagonal, diagonal),
                (-diagonal, -diagonal),
            )
        else:
            directions = tuple(
                tuple(sign if i == axis else 0.0 for i in range(len(bounds)))
                for axis in range(len(bounds))
                for sign in (-1.0, 1.0)
            )

        for direction in directions:
            trial = tuple(
                min(high, max(low, coordinate + delta * step))
                for coordinate, delta, step, (low, high) in zip(
                    point, direction, steps, bounds
                )
            )
            if trial == point:
                continue

            trial_value, complete = cutoff_objective(trial, best_value)
            evaluations += 1
            if not complete:
                continue
            if trial_value < best_value - improvement_tolerance:
                best_point = trial
                best_value = trial_value

        if best_point == point:
            steps = [step * 0.5 for step in steps]
        else:
            point = best_point
            value = best_value

    return point, value, evaluations


def _global_pattern_search(
    objective: Callable[[tuple[float, ...]], float],
    bounds: Sequence[tuple[float, float]],
    *,
    maximize: bool,
    sample_count: int,
    start_count: int,
    max_iterations: int,
    tolerances: Sequence[float],
    extra_starts: Iterable[Sequence[float]] = (),
    improvement_tolerance: float = 0.0,
) -> tuple[tuple[float, ...], float, int]:
    """
    低差异多起点 + 连续局部模式搜索。
    """
    dimension = len(bounds)

    normalized_starts = list(
        _halton_points(dimension, sample_count)
    )
    normalized_starts.extend(
        tuple(float(v) for v in point)
        for point in extra_starts
    )

    starts = []
    for normalized in normalized_starts:
        if len(normalized) != dimension:
            continue

        actual = tuple(
            low + min(1.0, max(0.0, q)) * (high - low)
            for q, (low, high) in zip(normalized, bounds)
        )
        starts.append(actual)

    scored = []
    evaluations = 0

    for point in starts:
        value = objective(point)
        evaluations += 1

        if maximize:
            if value != -math.inf and not math.isnan(value):
                scored.append((point, value))
        else:
            if math.isfinite(value):
                scored.append((point, value))

    if not scored:
        raise ArithmeticError("连续搜索没有找到有效初始点")

    scored.sort(
        key=lambda item: item[1],
        reverse=maximize,
    )

    best_point, best_value = scored[0]

    for start, _ in scored[:start_count]:
        point, value, local_evaluations = _pattern_search(
            start,
            objective,
            bounds,
            maximize=maximize,
            max_iterations=max_iterations,
            tolerances=tolerances,
            improvement_tolerance=improvement_tolerance,
        )

        evaluations += local_evaluations

        if _is_better(
            value,
            best_value,
            maximize,
            improvement_tolerance,
        ):
            best_point = point
            best_value = value

    return best_point, best_value, evaluations


# ============================================================
# Part 5 U 指标
# ============================================================

def calculate_U(
    P: Iterable[float],
    S: Iterable[float],
    G: Iterable[float],
    parallel_tolerance: float = 1e-12,
) -> float:
    P = _point(P, "P")
    S = _point(S, "S")
    G = _point(G, "G")

    d1 = distance(S, G)
    d2 = distance(P, G)

    if d1 == 0 or d2 == 0:
        return math.inf

    v1x = G[0] - S[0]
    v1y = G[1] - S[1]
    v2x = G[0] - P[0]
    v2y = G[1] - P[1]

    cross = abs(v1x * v2y - v1y * v2x)
    sin_phi = cross / (d1 * d2)

    if sin_phi < parallel_tolerance:
        return math.inf

    return math.sqrt(d1 * d2) / sin_phi


def candidate_U_score(
    P: Iterable[float],
    S: Iterable[float],
    sources: Sequence[Point],
) -> float:
    return max(calculate_U(P, S, G) for G in sources)


# ============================================================
# Part 6 连续鲁棒求解器
# ============================================================

class _ContinuousPlanner:
    def __init__(
        self,
        S: Point,
        theta1: float,
        planner_config: PlannerConfig,
        geometry_config: GeometryConfig,
    ):
        self.S = S
        self.theta1 = theta1
        self.cfg = planner_config
        self.geometry_cfg = geometry_config
        self.region = SourceRegion(S, theta1, planner_config)

        self.geometry_calls = 0
        self.error_searches = 0
        self.source_searches = 0
        self.point_objective_calls = 0
        self.geometry_cache_hits = 0
        self.exact_geometry_cache_hits = 0
        self.error_cache_hits = 0
        self.cutoff_pruned_error_searches = 0
        self.cutoff_pruned_point_evaluations = 0
        self.skipped_source_evaluations = 0
        self.scenario_geometry_evaluations = 0
        self.cutoff_pruned_scenario_evaluations = 0
        self.skipped_scenario_evaluations = 0
        self.fast_geometry_attempts = 0
        self.fast_geometry_successes = 0
        self.fast_geometry_fallbacks = 0
        self.exact_geometry_calls = 0
        self.exact_adversarial_searches = 0
        self.selective_exact_checks = 0
        self.strict_verification_fallbacks = 0
        self.staged_adversary_searches = 0
        self.staged_fixed_error_searches = 0
        self.staged_full_3d_fallbacks = 0
        self.adversary_screen_evaluations = 0
        self.scenario_priority_updates = 0
        self.scenario_priority_first_evaluations = 0

        self._diameter_cache: dict[tuple, float] = {}
        self._exact_diameter_cache: dict[tuple, float] = {}
        self._error_cache: dict[tuple, tuple[float, float]] = {}
        self._error_warm_start: dict[tuple[float, float], float] = {}
        self._preferred_scenario_key: tuple | None = None
        self._first_halfplanes = _bearing_halfplanes(
            self.S[0],
            self.S[1],
            self.theta1,
            self.cfg.bearing_error_deg,
        )
        self._fast_geometry = TwoBearingFastEvaluator(
            self.S,
            self.theta1,
            self.cfg.bearing_error_deg,
            config=self.geometry_cfg,
        )

    # --------------------------------------------------------
    # 几何评价
    # --------------------------------------------------------

    @staticmethod
    def _geometry_key(P: Point, theta2: float) -> tuple:
        return (
            round(P[0], 9),
            round(P[1], 9),
            round(theta2 % 360.0, 10),
        )

    @staticmethod
    def _geometry_result_value(result) -> float:
        if result.status in {
            "UNBOUNDED",
            "EMPTY",
            "NUMERICAL_ISSUE",
        }:
            return math.inf
        return float(result.diameter)

    def _exact_diameter_from_theta2(
        self,
        P: Point,
        theta2: float,
        *,
        count_total_call: bool = True,
    ) -> float:
        """只使用 geometry_V7；缓存与快速路径完全分离。"""
        theta2 = theta2 % 360.0
        key = self._geometry_key(P, theta2)
        if key in self._exact_diameter_cache:
            self.geometry_cache_hits += 1
            self.exact_geometry_cache_hits += 1
            return self._exact_diameter_cache[key]

        if count_total_call:
            self.geometry_calls += 1
        second_halfplanes = _bearing_halfplanes(
            P[0], P[1], theta2, self.cfg.bearing_error_deg
        )
        result = solve_halfplanes(
            self._first_halfplanes + second_halfplanes,
            config=self.geometry_cfg,
        )
        self.exact_geometry_calls += 1
        value = self._geometry_result_value(result)
        self._exact_diameter_cache[key] = value
        return value

    def _theta2_for_scenario(
        self,
        P: Point,
        G: Point,
        second_error_deg: float,
    ) -> float | None:
        if distance(P, G) <= self.cfg.no_bearing_radius_m:
            return None
        true_angle = math.degrees(
            math.atan2(G[1] - P[1], G[0] - P[0])
        )
        return (true_angle + second_error_deg) % 360.0

    def _diameter(
        self,
        P: Point,
        G: Point,
        second_error_deg: float,
    ) -> float:
        """
        给定 P、真实 G 和第二次误差，计算交会区域直径。
        """
        theta2 = self._theta2_for_scenario(P, G, second_error_deg)
        if theta2 is None:
            # 此处无法获得第二次示向度，因此不能按两次测向评价。
            return math.inf

        # 对固定的第一次观测，问题一的结果只由 P 和 theta2 决定，
        # 不由产生 theta2 的具体 G 决定。删除 G 可显著增加跨情景缓存命中。
        key = self._geometry_key(P, theta2)

        if key in self._diameter_cache:
            self.geometry_cache_hits += 1
            return self._diameter_cache[key]

        self.geometry_calls += 1
        if self.cfg.use_fast_geometry:
            self.fast_geometry_attempts += 1
            fast_result = self._fast_geometry.evaluate(P, theta2)
            if fast_result.certified:
                self.fast_geometry_successes += 1
                value = float(fast_result.diameter)
            else:
                self.fast_geometry_fallbacks += 1
                value = self._exact_diameter_from_theta2(
                    P,
                    theta2,
                    count_total_call=False,
                )
        else:
            value = self._exact_diameter_from_theta2(
                P,
                theta2,
                count_total_call=False,
            )

        self._diameter_cache[key] = value
        return value

    def _exact_diameter(
        self,
        P: Point,
        G: Point,
        second_error_deg: float,
    ) -> float:
        """最终校核专用入口，明确绕过快速路径及其缓存。"""
        theta2 = self._theta2_for_scenario(P, G, second_error_deg)
        if theta2 is None:
            return math.inf
        return self._exact_diameter_from_theta2(P, theta2)

    # --------------------------------------------------------
    # P3：连续搜索第二次测向误差
    # --------------------------------------------------------

    def _critical_error_parameters(
        self,
        P: Point,
        G: Point,
    ) -> list[tuple[float]]:
        """
        加入可能使两次测向方向平行或反向平行的临界误差。
        """
        true_angle = math.degrees(
            math.atan2(G[1] - P[1], G[0] - P[0])
        )
        error_bound = self.cfg.bearing_error_deg

        critical_errors = [
            _angle_difference_deg(self.theta1, true_angle),
            _angle_difference_deg(
                self.theta1 + 180.0,
                true_angle,
            ),
        ]

        parameters = []

        for error in critical_errors:
            if -error_bound <= error <= error_bound:
                q = (error + error_bound) / (2.0 * error_bound)
                parameters.append((q,))

        return parameters

    def worst_error_for_source(
        self,
        P: Point,
        G: Point,
        cutoff: float = math.inf,
    ) -> tuple[float, float, bool]:
        """
        连续求解：
            max_{e in [-epsilon, epsilon]} D(P,G,e)

        complete=False 表示搜索在发现直径已达到 cutoff 后安全提前结束。
        """
        cache_key = (
            round(P[0], 8),
            round(P[1], 8),
            round(G[0], 8),
            round(G[1], 8),
        )

        if cache_key in self._error_cache:
            self.error_cache_hits += 1
            value, error = self._error_cache[cache_key]
            return value, error, True

        error_bound = self.cfg.bearing_error_deg
        source_key = round(G[0], 7), round(G[1], 7)

        extra_starts = [
            (0.0,),
            (1.0,),
        ]
        warm_error = self._error_warm_start.get(source_key)
        if warm_error is not None:
            extra_starts.append(
                ((warm_error + error_bound) / (2.0 * error_bound),)
            )
        extra_starts.extend(self._critical_error_parameters(P, G))
        extra_starts.append((0.5,))

        # 先测试端点、上一位置的最坏误差和几何临界误差。
        # 任一点达到 cutoff，就已能证明当前 P 不可能更优。
        seen_parameters = set()
        ordered_starts = []
        for item in extra_starts:
            q = min(1.0, max(0.0, float(item[0])))
            key = round(q, 14)
            if key not in seen_parameters:
                seen_parameters.add(key)
                ordered_starts.append((q,))

        if math.isfinite(cutoff):
            for (q_error,) in ordered_starts:
                error = -error_bound + 2.0 * error_bound * q_error
                value = self._diameter(P, G, error)
                if value >= cutoff:
                    self.cutoff_pruned_error_searches += 1
                    return value, error, False

        def objective(parameters):
            q_error = parameters[0]
            error = -error_bound + 2.0 * error_bound * q_error
            value = self._diameter(P, G, error)
            if math.isfinite(cutoff) and value >= cutoff:
                raise _CutoffReached(value, error)
            return value

        try:
            parameters, value, _ = _global_pattern_search(
                objective,
                bounds=((0.0, 1.0),),
                maximize=True,
                sample_count=self.cfg.error_sample_count,
                start_count=self.cfg.error_start_count,
                max_iterations=self.cfg.error_max_iterations,
                tolerances=(
                    self.cfg.error_tolerance_deg
                    / (2.0 * error_bound),
                ),
                extra_starts=ordered_starts,
            )
        except _CutoffReached as reached:
            self.cutoff_pruned_error_searches += 1
            return reached.value, reached.parameter, False
        self.error_searches += 1

        worst_error = (
            -error_bound
            + 2.0 * error_bound * parameters[0]
        )

        result = value, worst_error
        self._error_cache[cache_key] = result
        self._error_warm_start[source_key] = worst_error
        return value, worst_error, True

    def _scenario_key(self, scenario: Scenario) -> tuple:
        G, error = scenario
        return (
            round(G[0], 8),
            round(G[1], 8),
            round(error, 8),
        )

    def active_scenario_objective(
        self,
        P: Point,
        scenarios: Sequence[Scenario],
        cutoff: float = math.inf,
    ) -> tuple[float, Point | None, float | None, bool]:
        """
        在有限活动情景 (G,e) 上计算真实目标。

        这里不再针对每个试探点 P、每个 G 重做连续误差搜索。
        上一次完整评价中的最坏情景优先，不再为每个 P 排序全部情景。
        """
        worst_value = -math.inf
        worst_source = None
        worst_error = None

        preferred_index = None
        if self._preferred_scenario_key is not None:
            for index, scenario in enumerate(scenarios):
                if self._scenario_key(scenario) == self._preferred_scenario_key:
                    preferred_index = index
                    break

        if preferred_index is None:
            evaluation_order = range(len(scenarios))
        else:
            self.scenario_priority_first_evaluations += 1
            evaluation_order = (
                preferred_index,
                *(index for index in range(len(scenarios)) if index != preferred_index),
            )

        for order_index, scenario_index in enumerate(evaluation_order):
            G, error = scenarios[scenario_index]
            value = self._diameter(P, G, error)
            self.scenario_geometry_evaluations += 1

            if value > worst_value:
                worst_value = value
                worst_source = G
                worst_error = error

            if worst_value >= cutoff:
                self._preferred_scenario_key = self._scenario_key((G, error))
                self.scenario_priority_updates += 1
                self.cutoff_pruned_scenario_evaluations += 1
                skipped = len(scenarios) - order_index - 1
                self.skipped_scenario_evaluations += skipped
                return worst_value, worst_source, worst_error, False

            if worst_value == math.inf:
                break

        if worst_source is not None and worst_error is not None:
            self._preferred_scenario_key = self._scenario_key(
                (worst_source, worst_error)
            )
            self.scenario_priority_updates += 1
        return worst_value, worst_source, worst_error, True

    def exact_active_scenario_objective(
        self,
        P: Point,
        scenarios: Sequence[Scenario],
    ) -> tuple[float, Point | None, float | None]:
        """用 geometry_V7 重新评价全部活动情景，供每轮收敛校核使用。"""
        worst_value = -math.inf
        worst_source = None
        worst_error = None
        for G, error in scenarios:
            value = self._exact_diameter(P, G, error)
            if value > worst_value:
                worst_value = value
                worst_source = G
                worst_error = error
        return worst_value, worst_source, worst_error

    def verify_active_scenarios(
        self,
        P: Point,
        scenarios: Sequence[Scenario],
    ) -> tuple[float, Point | None, float | None, dict]:
        """分层校核活动集；严格模式保留旧版全量 geometry_V7 行为。"""
        if self.cfg.verification_mode == "strict":
            value, source, error = self.exact_active_scenario_objective(
                P, scenarios
            )
            return value, source, error, {
                "method": "strict_all_active_scenarios",
                "verified_count": len(scenarios),
                "strict_fallback": False,
            }

        scored = sorted(
            (
                (self._diameter(P, G, error), G, error)
                for G, error in scenarios
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        verify_count = min(self.cfg.verification_top_k, len(scored))
        verified = []
        maximum_disagreement = 0.0
        for fast_value, G, error in scored[:verify_count]:
            exact_value = self._exact_diameter(P, G, error)
            self.selective_exact_checks += 1
            if math.isfinite(fast_value) and math.isfinite(exact_value):
                maximum_disagreement = max(
                    maximum_disagreement,
                    abs(fast_value - exact_value),
                )
            elif fast_value != exact_value:
                maximum_disagreement = math.inf
            verified.append((exact_value, G, error))

        verified.sort(key=lambda item: item[0], reverse=True)
        strict_fallback = (
            maximum_disagreement
            > self.cfg.verification_disagreement_tolerance_m
        )
        if len(scored) > verify_count and verified:
            unverified_upper = scored[verify_count][0]
            strict_fallback = strict_fallback or (
                unverified_upper
                > verified[0][0]
                + self.cfg.verification_disagreement_tolerance_m
            )

        if strict_fallback:
            self.strict_verification_fallbacks += 1
            value, source, error = self.exact_active_scenario_objective(
                P, scenarios
            )
            return value, source, error, {
                "method": "hybrid_then_strict_active_fallback",
                "verified_count": len(scenarios),
                "strict_fallback": True,
                "maximum_fast_exact_disagreement_m": maximum_disagreement,
            }

        value, source, error = verified[0]
        return value, source, error, {
            "method": "hybrid_top_k_active",
            "verified_count": verify_count,
            "strict_fallback": False,
            "maximum_fast_exact_disagreement_m": maximum_disagreement,
        }

    # --------------------------------------------------------
    # P2：连续搜索最坏源位置
    # --------------------------------------------------------

    def _source_extra_starts(self):
        return self.region.initial_parameters()

    def worst_receive_violation(
        self,
        P: Point,
    ) -> tuple[float, Point]:
        """
        连续求解：
            max_G [d(P,G)-max(1000,d(S,G))]

        返回值 <= 0 表示满足全部源位置的接收约束。
        """

        def objective(parameters):
            G = self.region.point(parameters)

            if G is None:
                return -math.inf

            d1 = distance(self.S, G)
            d2 = distance(P, G)

            return d2 - max(
                self.cfg.reception_radius_min_m,
                d1,
            )

        parameters, value, _ = _global_pattern_search(
            objective,
            bounds=((0.0, 1.0), (0.0, 1.0)),
            maximize=True,
            sample_count=self.cfg.source_sample_count,
            start_count=self.cfg.source_start_count,
            max_iterations=self.cfg.source_max_iterations,
            tolerances=(
                self.cfg.source_parameter_tolerance,
                self.cfg.source_parameter_tolerance,
            ),
            extra_starts=self._source_extra_starts(),
        )
        self.source_searches += 1

        G = self.region.point(parameters)
        if G is None:
            raise ArithmeticError("最坏接收源位置映射失败")

        return value, G

    def worst_localization(
        self,
        P: Point,
        *,
        exact_geometry: bool = False,
    ) -> tuple[float, Point, float]:
        """
        联合连续求解：
            max_{G in Omega_G, e in [-epsilon,epsilon]}
                D(P,G,e)
        """
        error_bound = self.cfg.bearing_error_deg

        def objective(parameters):
            G = self.region.point(parameters[:2])

            if G is None:
                return -math.inf

            q_error = parameters[2]
            error = -error_bound + 2.0 * error_bound * q_error

            if exact_geometry:
                return self._exact_diameter(P, G, error)
            return self._diameter(P, G, error)

        extra_starts = []

        for source_parameters in self._source_extra_starts():
            G = self.region.point(source_parameters)

            if G is None:
                continue

            error_parameters = [
                (0.0,),
                (0.5,),
                (1.0,),
            ]
            error_parameters.extend(
                self._critical_error_parameters(P, G)
            )

            for error_parameter in error_parameters:
                extra_starts.append(
                    (
                        source_parameters[0],
                        source_parameters[1],
                        error_parameter[0],
                    )
                )

        parameters, value, _ = _global_pattern_search(
            objective,
            bounds=(
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
            ),
            maximize=True,
            sample_count=self.cfg.source_sample_count,
            start_count=self.cfg.source_start_count,
            max_iterations=self.cfg.source_max_iterations,
            tolerances=(
                self.cfg.source_parameter_tolerance,
                self.cfg.source_parameter_tolerance,
                self.cfg.error_tolerance_deg
                / (2.0 * error_bound),
            ),
            extra_starts=extra_starts,
        )
        self.source_searches += 1
        if exact_geometry:
            self.exact_adversarial_searches += 1

        G = self.region.point(parameters[:2])
        if G is None:
            raise ArithmeticError("最坏定位源位置映射失败")

        error = (
            -error_bound
            + 2.0 * error_bound * parameters[2]
        )

        return value, G, error

    def _fixed_error_source_search(
        self,
        P: Point,
        error: float,
        *,
        sample_count: int,
        start_count: int,
        max_iterations: int,
        extra_starts: Iterable[Sequence[float]] = (),
    ) -> tuple[float, Point, tuple[float, float]]:
        """固定误差后只对二维源位置做连续最大化。"""

        def objective(parameters):
            G = self.region.point(parameters)
            if G is None:
                return -math.inf
            return self._diameter(P, G, error)

        starts = list(self._source_extra_starts())
        starts.extend(extra_starts)
        parameters, value, _ = _global_pattern_search(
            objective,
            bounds=((0.0, 1.0), (0.0, 1.0)),
            maximize=True,
            sample_count=sample_count,
            start_count=start_count,
            max_iterations=max_iterations,
            tolerances=(
                self.cfg.source_parameter_tolerance,
                self.cfg.source_parameter_tolerance,
            ),
            extra_starts=starts,
        )
        self.source_searches += 1
        self.staged_fixed_error_searches += 1
        G = self.region.point(parameters)
        if G is None:
            raise ArithmeticError("固定误差源位置搜索映射失败")
        return value, G, (parameters[0], parameters[1])

    @staticmethod
    def _deduplicate_adversarial_candidates(
        candidates: Iterable[AdversarialCandidate],
    ) -> list[AdversarialCandidate]:
        best_by_key: dict[tuple, AdversarialCandidate] = {}
        for value, G, error in candidates:
            key = (
                round(G[0], 7),
                round(G[1], 7),
                round(error, 7),
            )
            previous = best_by_key.get(key)
            if previous is None or value > previous[0]:
                best_by_key[key] = (value, G, error)
        return sorted(
            best_by_key.values(),
            key=lambda item: item[0],
            reverse=True,
        )

    def staged_worst_localization(
        self,
        P: Point,
    ) -> tuple[list[AdversarialCandidate], dict]:
        """端点/零误差二维搜索 + 一维误差细化 + 低成本三维筛查。"""
        self.staged_adversary_searches += 1
        error_bound = self.cfg.bearing_error_deg
        initial_sample_count = max(12, self.cfg.source_sample_count // 4)
        initial_start_count = max(2, self.cfg.source_start_count // 4)
        initial_iterations = max(10, self.cfg.source_max_iterations // 2)
        candidates: list[AdversarialCandidate] = []
        refined_starts = []

        for error in (-error_bound, 0.0, error_bound):
            value, G, parameters = self._fixed_error_source_search(
                P,
                error,
                sample_count=initial_sample_count,
                start_count=initial_start_count,
                max_iterations=initial_iterations,
            )
            candidates.append((value, G, error))

            refined_value, refined_error, complete = (
                self.worst_error_for_source(P, G)
            )
            if not complete:
                raise ArithmeticError("无截止误差搜索不应提前结束")
            candidates.append((refined_value, G, refined_error))
            refined_starts.append(
                (refined_value, G, refined_error, parameters)
            )

        # 只对初筛最好的两个源—误差组合交替再优化一次源位置和误差。
        refined_starts.sort(key=lambda item: item[0], reverse=True)
        second_sample_count = max(10, self.cfg.source_sample_count // 6)
        second_iterations = max(8, self.cfg.source_max_iterations // 3)
        for _, _, error, parameters in refined_starts[:2]:
            value, G, _ = self._fixed_error_source_search(
                P,
                error,
                sample_count=second_sample_count,
                start_count=2,
                max_iterations=second_iterations,
                extra_starts=(parameters,),
            )
            candidates.append((value, G, error))
            refined_value, refined_error, complete = (
                self.worst_error_for_source(P, G)
            )
            if not complete:
                raise ArithmeticError("无截止误差搜索不应提前结束")
            candidates.append((refined_value, G, refined_error))

        staged_reference = max(value for value, _, _ in candidates)

        # 三维低差异筛查只负责发现明显遗漏；不做局部搜索。
        screen_candidates = []
        for parameters in _halton_points(
            3, self.cfg.adversary_screen_sample_count
        ):
            G = self.region.point(parameters[:2])
            if G is None:
                continue
            error = -error_bound + 2.0 * error_bound * parameters[2]
            value = self._diameter(P, G, error)
            screen_candidates.append((value, G, error))
            self.adversary_screen_evaluations += 1
        screen_candidates.sort(key=lambda item: item[0], reverse=True)
        candidates.extend(screen_candidates[: self.cfg.verification_top_k])
        candidates = self._deduplicate_adversarial_candidates(candidates)

        full_3d_fallback = False
        if screen_candidates:
            screen_best = screen_candidates[0][0]
            if screen_best > staged_reference + self.cfg.adversary_stage_gap_tolerance_m:
                full_3d_fallback = True

        if full_3d_fallback:
            self.staged_full_3d_fallbacks += 1
            value, G, error = self.worst_localization(
                P,
                exact_geometry=False,
            )
            candidates.append((value, G, error))
            candidates = self._deduplicate_adversarial_candidates(candidates)

        return candidates, {
            "method": "fixed_error_2d_plus_error_refinement",
            "candidate_count": len(candidates),
            "screen_count": len(screen_candidates),
            "full_3d_fallback": full_3d_fallback,
        }

    def verify_adversarial_candidates(
        self,
        P: Point,
        candidates: Sequence[AdversarialCandidate],
    ) -> tuple[float, Point, float, dict]:
        """用 geometry_V7 校核分层搜索给出的前 K 个最坏候选。"""
        verify_count = min(self.cfg.verification_top_k, len(candidates))
        verified = []
        maximum_disagreement = 0.0
        for fast_value, G, error in candidates[:verify_count]:
            exact_value = self._exact_diameter(P, G, error)
            self.selective_exact_checks += 1
            if math.isfinite(fast_value) and math.isfinite(exact_value):
                maximum_disagreement = max(
                    maximum_disagreement,
                    abs(fast_value - exact_value),
                )
            elif fast_value != exact_value:
                maximum_disagreement = math.inf
            verified.append((exact_value, G, error))
        verified.sort(key=lambda item: item[0], reverse=True)

        strict_fallback = (
            not verified
            or maximum_disagreement
            > self.cfg.verification_disagreement_tolerance_m
        )
        if strict_fallback:
            self.strict_verification_fallbacks += 1
            value, G, error = self.worst_localization(
                P,
                exact_geometry=True,
            )
            return value, G, error, {
                "method": "hybrid_then_strict_adversary_fallback",
                "verified_count": verify_count,
                "strict_fallback": True,
                "maximum_fast_exact_disagreement_m": maximum_disagreement,
            }

        value, G, error = verified[0]
        return value, G, error, {
            "method": "hybrid_top_k_adversary",
            "verified_count": verify_count,
            "strict_fallback": False,
            "maximum_fast_exact_disagreement_m": maximum_disagreement,
        }

    # --------------------------------------------------------
    # 活动情景约束
    # --------------------------------------------------------

    def source_receive_margin(
        self,
        P: Point,
        G: Point,
    ) -> float:
        d1 = distance(self.S, G)
        d2 = distance(P, G)

        return max(
            self.cfg.reception_radius_min_m,
            d1,
        ) - d2

    def active_feasible(
        self,
        P: Point,
        sources: Sequence[Point],
    ) -> bool:
        tolerance = self.cfg.receive_tolerance_m

        return all(
            self.source_receive_margin(P, G) >= -tolerance
            for G in sources
        )

    def active_bounds(
        self,
        sources: Sequence[Point],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        每个源情景给出一个接收圆。这里返回这些圆交集的
        轴对齐包围盒，真正圆约束仍由 active_feasible 检查。
        """
        lower_x = -math.inf
        upper_x = math.inf
        lower_y = -math.inf
        upper_y = math.inf

        for G in sources:
            radius = max(
                self.cfg.reception_radius_min_m,
                distance(self.S, G),
            )

            lower_x = max(lower_x, G[0] - radius)
            upper_x = min(upper_x, G[0] + radius)
            lower_y = max(lower_y, G[1] - radius)
            upper_y = min(upper_y, G[1] + radius)

        if lower_x > upper_x or lower_y > upper_y:
            raise ArithmeticError("活动接收约束交集为空")

        return (
            (lower_x, upper_x),
            (lower_y, upper_y),
        )

    def active_radial_limit(
        self,
        angle_rad: float,
        sources: Sequence[Point],
    ) -> float:
        """从第一次检测点沿给定方向，在活动接收圆交集中可走的最远距离。"""
        ux = math.cos(angle_rad)
        uy = math.sin(angle_rad)
        limit = math.inf

        for G in sources:
            vx = G[0] - self.S[0]
            vy = G[1] - self.S[1]
            d1 = math.hypot(vx, vy)
            radius = max(self.cfg.reception_radius_min_m, d1)
            projection = ux * vx + uy * vy
            discriminant = projection * projection + radius * radius - d1 * d1
            upper = projection + math.sqrt(max(0.0, discriminant))
            limit = min(limit, upper)

        if not math.isfinite(limit) or limit < 0.0:
            raise ArithmeticError("无法构造活动接收区域的径向边界")
        return limit

    def point_from_radial_parameters(
        self,
        parameters: Sequence[float],
        sources: Sequence[Point],
    ) -> Point:
        """用方向和相对径向距离参数化活动接收区域，自动贴合弯曲边界。"""
        q_angle = min(1.0, max(0.0, float(parameters[0])))
        q_radius = min(1.0, max(0.0, float(parameters[1])))
        angle = -math.pi + 2.0 * math.pi * q_angle
        radial_limit = self.active_radial_limit(angle, sources)
        radius = q_radius * radial_limit
        return (
            self.S[0] + radius * math.cos(angle),
            self.S[1] + radius * math.sin(angle),
        )

    def radial_parameters_from_point(
        self,
        P: Point,
        sources: Sequence[Point],
    ) -> tuple[float, float]:
        dx = P[0] - self.S[0]
        dy = P[1] - self.S[1]
        radius = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) if radius > 0.0 else 0.0
        q_angle = (angle + math.pi) / (2.0 * math.pi)
        radial_limit = self.active_radial_limit(angle, sources)
        q_radius = radius / radial_limit if radial_limit > 0.0 else 0.0
        return (
            min(1.0, max(0.0, q_angle)),
            min(1.0, max(0.0, q_radius)),
        )

    # --------------------------------------------------------
    # P1：连续优化第二检测点
    # --------------------------------------------------------

    def proxy_search(
        self,
        sources: Sequence[Point],
    ) -> tuple[Point, float, int]:
        """
        使用 U 指标进行连续多起点预搜索。
        """
        bounds = self.active_bounds(sources)

        def objective(parameters):
            P = float(parameters[0]), float(parameters[1])

            if not self.active_feasible(P, sources):
                return math.inf

            return candidate_U_score(P, self.S, sources)

        bearing_rad = math.radians(self.theta1)
        perpendicular = (
            -math.sin(bearing_rad),
            math.cos(bearing_rad),
        )

        extra_actual_points = [self.S]

        for sign in (-1.0, 1.0):
            for radius in (250.0, 500.0, 750.0, 1000.0):
                extra_actual_points.append(
                    (
                        self.S[0]
                        + sign * radius * perpendicular[0],
                        self.S[1]
                        + sign * radius * perpendicular[1],
                    )
                )

        extra_normalized = []

        for P in extra_actual_points:
            normalized = []

            for coordinate, (low, high) in zip(P, bounds):
                if high == low:
                    normalized.append(0.5)
                else:
                    normalized.append(
                        (coordinate - low) / (high - low)
                    )

            if all(0.0 <= q <= 1.0 for q in normalized):
                extra_normalized.append(tuple(normalized))

        point, value, evaluations = _global_pattern_search(
            objective,
            bounds=bounds,
            maximize=False,
            sample_count=self.cfg.point_sample_count,
            start_count=self.cfg.point_start_count,
            max_iterations=self.cfg.proxy_max_iterations,
            tolerances=(
                self.cfg.point_tolerance_m,
                self.cfg.point_tolerance_m,
            ),
            extra_starts=extra_normalized,
            improvement_tolerance=(
                self.cfg.objective_improvement_tolerance_m
            ),
        )

        return (float(point[0]), float(point[1])), value, evaluations

    def refine_point(
        self,
        start: Point,
        sources: Sequence[Point],
        scenarios: Sequence[Scenario],
    ) -> tuple[Point, float, int]:
        """
        在活动 (G,e) 情景上直接连续最小化真实 J。

        sources 仅定义鲁棒接收可行域；scenarios 定义目标函数。
        """
        # 使用“方向 + 相对可行半径”，而不是世界坐标 x/y。
        # q_radius=1 会沿任意方向自动落在接收圆交集边界上，
        # 因而搜索可以沿弯曲边界移动，不会因旋转坐标轴而改变结果。
        bounds = ((0.0, 1.0), (0.0, 1.0))
        start_parameters = self.radial_parameters_from_point(start, sources)
        start_radius = max(distance(self.S, start), 1.0)
        start_angle = -math.pi + 2.0 * math.pi * start_parameters[0]
        radial_limit = max(self.active_radial_limit(start_angle, sources), 1.0)

        angular_tolerance = min(
            0.01,
            self.cfg.point_tolerance_m / (2.0 * math.pi * start_radius),
        )
        angular_step = min(
            0.125,
            self.cfg.point_initial_step_m / (2.0 * math.pi * start_radius),
        )
        radial_tolerance = min(
            0.01,
            self.cfg.point_tolerance_m / radial_limit,
        )
        radial_step = min(
            0.25,
            self.cfg.point_initial_step_m / radial_limit,
        )

        def full_objective(parameters):
            self.point_objective_calls += 1
            P = self.point_from_radial_parameters(parameters, sources)

            value, _, _, complete = self.active_scenario_objective(
                P, scenarios
            )
            if not complete:
                raise ArithmeticError("无截止值评价不应提前结束")
            return value

        def cutoff_objective(parameters, cutoff):
            self.point_objective_calls += 1
            P = self.point_from_radial_parameters(parameters, sources)

            value, _, _, complete = self.active_scenario_objective(
                P, scenarios, cutoff=cutoff
            )
            if not complete:
                self.cutoff_pruned_point_evaluations += 1
            return value, complete

        point, value, evaluations = _pattern_search_min_with_cutoff(
            start_parameters,
            full_objective,
            cutoff_objective,
            bounds,
            max_iterations=self.cfg.exact_max_iterations,
            tolerances=(angular_tolerance, radial_tolerance),
            initial_steps=(angular_step, radial_step),
            improvement_tolerance=(
                self.cfg.objective_improvement_tolerance_m
            ),
        )

        best_point = self.point_from_radial_parameters(point, sources)
        return best_point, value, evaluations

    # --------------------------------------------------------
    # P0：连续鲁棒最小—最大主流程
    # --------------------------------------------------------

    def solve(self) -> PlannerResult:
        initial_parameters = self.region.initial_parameters()

        active_sources = [
            self.region.point(parameters)
            for parameters in initial_parameters
        ]
        active_sources = [
            G for G in active_sources if G is not None
        ]

        if not active_sources:
            return PlannerResult(
                status="INFEASIBLE_FIRST_OBSERVATION",
                first_observation=(
                    self.S[0],
                    self.S[1],
                    self.theta1,
                ),
                diagnostics={
                    "reason": "第一次测向角域与目标圆、接收距离没有交集"
                },
            )

        active_sources = self._unique_sources(active_sources)

        best_point, proxy_score, proxy_evaluations = (
            self.proxy_search(active_sources)
        )

        # 初始活动目标采用少量代表性误差：两个端点、零误差，以及
        # 代理解处可能造成两次测向平行的临界误差。之后由连续联合
        # 对抗搜索按需补充真正的最坏 (G,e)，而不是固定误差网格。
        active_scenarios: list[Scenario] = []
        for source in active_sources:
            self._append_new_scenarios(
                active_scenarios,
                self._seed_scenarios(source, best_point),
            )

        status = "MAX_SCENARIO_ROUNDS"
        last_active_J = math.inf
        last_worst_J = math.inf
        last_worst_source = None
        last_worst_error = None
        last_receive_violation = math.inf
        scenario_history = []

        for round_index in range(
            1,
            self.cfg.max_scenario_rounds + 1,
        ):
            best_point, fast_active_J, point_evaluations = (
                self.refine_point(
                    best_point,
                    active_sources,
                    active_scenarios,
                )
            )

            receive_violation, receive_source = (
                self.worst_receive_violation(best_point)
            )

            # hybrid 只精确校核高风险候选；strict 保留旧版全量精确流程。
            active_J, active_source, active_error, active_verification = (
                self.verify_active_scenarios(
                    best_point,
                    active_scenarios,
                )
            )

            if self.cfg.verification_mode == "strict":
                adversarial_J, adversarial_source, adversarial_error = (
                    self.worst_localization(
                        best_point,
                        exact_geometry=True,
                    )
                )
                adversarial_verification = {
                    "method": "strict_full_3d_exact_adversary",
                    "strict_fallback": False,
                }
                adversarial_stage = None
            else:
                adversarial_candidates, adversarial_stage = (
                    self.staged_worst_localization(best_point)
                )
                (
                    adversarial_J,
                    adversarial_source,
                    adversarial_error,
                    adversarial_verification,
                ) = self.verify_adversarial_candidates(
                    best_point,
                    adversarial_candidates,
                )
            if active_J > adversarial_J:
                worst_J = active_J
                worst_source = active_source
                worst_error = active_error
            else:
                worst_J = adversarial_J
                worst_source = adversarial_source
                worst_error = adversarial_error

            objective_gap = worst_J - active_J

            scenario_record = {
                "round": round_index,
                "active_source_count": len(active_sources),
                "active_scenario_count": len(active_scenarios),
                "point": best_point,
                "fast_active_J": fast_active_J,
                "active_J": active_J,
                "continuous_worst_J": worst_J,
                "objective_gap_m": objective_gap,
                "max_receive_violation_m": receive_violation,
                "worst_source": worst_source,
                "worst_error_deg": worst_error,
                "receive_source": receive_source,
                "point_evaluations": point_evaluations,
                "active_verification": active_verification,
                "adversarial_stage": adversarial_stage,
                "adversarial_verification": adversarial_verification,
            }
            scenario_history.append(scenario_record)

            last_active_J = active_J
            last_worst_J = worst_J
            last_worst_source = worst_source
            last_worst_error = worst_error
            last_receive_violation = receive_violation

            new_sources = []
            new_scenarios: list[Scenario] = []

            if receive_violation > self.cfg.receive_tolerance_m:
                new_sources.append(receive_source)
                new_scenarios.extend(
                    self._seed_scenarios(receive_source, best_point)
                )

            if (
                objective_gap
                > self.cfg.scenario_gap_tolerance_m
            ):
                new_sources.append(worst_source)
                new_scenarios.append((worst_source, worst_error))

            source_added = self._append_new_sources(
                active_sources,
                new_sources,
            )
            scenario_added = self._append_new_scenarios(
                active_scenarios,
                new_scenarios,
            )

            receive_converged = (
                receive_violation
                <= self.cfg.receive_tolerance_m
            )
            objective_converged = (
                objective_gap
                <= self.cfg.scenario_gap_tolerance_m
            )

            if receive_converged and objective_converged:
                status = "SUCCESS_APPROXIMATE"
                break

            if not source_added and not scenario_added:
                status = "ADVERSARY_STALLED"
                break

        final_receive_margin = -last_receive_violation

        return PlannerResult(
            status=status,
            second_point=best_point,
            worst_diameter=last_worst_J,
            proxy_score=proxy_score,
            worst_source=last_worst_source,
            worst_second_error_deg=last_worst_error,
            receive_margin_m=final_receive_margin,
            first_observation=(
                self.S[0],
                self.S[1],
                self.theta1,
            ),
            active_sources=list(active_sources),
            active_scenarios=list(active_scenarios),
            diagnostics={
                "model": (
                    "continuous robust min-max with "
                    "adaptive adversarial (G,e) scenarios"
                ),
                "fixed_candidate_grid_used": False,
                "fixed_source_grid_used": False,
                "fixed_error_grid_used": False,
                "outer_search_coordinates": (
                    "bearing-relative direction plus feasible radial fraction"
                ),
                "point_tolerance_m": (
                    self.cfg.point_tolerance_m
                ),
                "source_parameter_tolerance": (
                    self.cfg.source_parameter_tolerance
                ),
                "error_tolerance_deg": (
                    self.cfg.error_tolerance_deg
                ),
                "scenario_gap_tolerance_m": (
                    self.cfg.scenario_gap_tolerance_m
                ),
                "verification_mode": self.cfg.verification_mode,
                "verification_top_k": self.cfg.verification_top_k,
                "verification_disagreement_tolerance_m": (
                    self.cfg.verification_disagreement_tolerance_m
                ),
                "scenario_rounds": len(scenario_history),
                "scenario_history": scenario_history,
                "active_source_count": len(active_sources),
                "active_scenario_count": len(active_scenarios),
                "active_objective_m": last_active_J,
                "continuous_adversarial_objective_m": (
                    last_worst_J
                ),
                "max_receive_violation_m": (
                    last_receive_violation
                ),
                "proxy_evaluations": proxy_evaluations,
                "point_objective_calls": (
                    self.point_objective_calls
                ),
                "error_searches": self.error_searches,
                "source_searches": self.source_searches,
                "geometry_calls": self.geometry_calls,
                "geometry_cache_hits": self.geometry_cache_hits,
                "fast_geometry_enabled": self.cfg.use_fast_geometry,
                "fast_geometry_attempts": self.fast_geometry_attempts,
                "fast_geometry_successes": self.fast_geometry_successes,
                "fast_geometry_fallbacks": self.fast_geometry_fallbacks,
                "fast_geometry_success_rate": (
                    self.fast_geometry_successes
                    / self.fast_geometry_attempts
                    if self.fast_geometry_attempts
                    else None
                ),
                "exact_geometry_calls": self.exact_geometry_calls,
                "exact_geometry_cache_hits": (
                    self.exact_geometry_cache_hits
                ),
                "exact_adversarial_searches": (
                    self.exact_adversarial_searches
                ),
                "selective_exact_checks": self.selective_exact_checks,
                "strict_verification_fallbacks": (
                    self.strict_verification_fallbacks
                ),
                "staged_adversary_searches": (
                    self.staged_adversary_searches
                ),
                "staged_fixed_error_searches": (
                    self.staged_fixed_error_searches
                ),
                "staged_full_3d_fallbacks": (
                    self.staged_full_3d_fallbacks
                ),
                "adversary_screen_evaluations": (
                    self.adversary_screen_evaluations
                ),
                "error_cache_hits": self.error_cache_hits,
                "cutoff_pruned_error_searches": (
                    self.cutoff_pruned_error_searches
                ),
                "cutoff_pruned_point_evaluations": (
                    self.cutoff_pruned_point_evaluations
                ),
                "skipped_source_evaluations": (
                    self.skipped_source_evaluations
                ),
                "scenario_geometry_evaluations": (
                    self.scenario_geometry_evaluations
                ),
                "cutoff_pruned_scenario_evaluations": (
                    self.cutoff_pruned_scenario_evaluations
                ),
                "skipped_scenario_evaluations": (
                    self.skipped_scenario_evaluations
                ),
                "scenario_priority_updates": (
                    self.scenario_priority_updates
                ),
                "scenario_priority_first_evaluations": (
                    self.scenario_priority_first_evaluations
                ),
                "active_objective_method": (
                    "last-worst-first active scenarios without per-point sorting"
                ),
                "geometry_backend_policy": (
                    "hybrid top-k geometry_V7 verification with automatic strict "
                    "fallback; strict mode preserves full exact checks"
                ),
                "geometry_input_reuse": (
                    "第一次观测半平面预构造；缓存键仅使用 P 与 theta2"
                ),
                "global_optimality_note": (
                    "连续搜索采用有限精度的确定性多起点"
                    "模式搜索，不构成解析全局最优证明"
                ),
            },
        )

    def _same_source(
        self,
        first: Point,
        second: Point,
    ) -> bool:
        return (
            distance(first, second)
            <= self.cfg.scenario_merge_tolerance_m
        )

    def _unique_sources(
        self,
        sources: Iterable[Point],
    ) -> list[Point]:
        unique = []

        for source in sources:
            if not any(
                self._same_source(source, existing)
                for existing in unique
            ):
                unique.append(source)

        return unique

    def _append_new_sources(
        self,
        active_sources: list[Point],
        new_sources: Iterable[Point],
    ) -> bool:
        added = False

        for source in new_sources:
            if source is None:
                continue

            if any(
                self._same_source(source, existing)
                for existing in active_sources
            ):
                continue

            active_sources.append(source)
            added = True

        return added

    def _seed_scenarios(
        self,
        source: Point,
        reference_point: Point | None = None,
    ) -> list[Scenario]:
        """为新源生成少量非网格化的必要初始误差情景。"""
        error_bound = self.cfg.bearing_error_deg
        errors = [-error_bound, 0.0, error_bound]

        if reference_point is not None:
            for (q_error,) in self._critical_error_parameters(
                reference_point, source
            ):
                errors.append(
                    -error_bound + 2.0 * error_bound * q_error
                )

        scenarios: list[Scenario] = []
        for error in errors:
            scenario = (source, float(error))
            if not any(
                self._same_scenario(scenario, existing)
                for existing in scenarios
            ):
                scenarios.append(scenario)
        return scenarios

    def _same_scenario(
        self,
        first: Scenario,
        second: Scenario,
    ) -> bool:
        return (
            self._same_source(first[0], second[0])
            and abs(first[1] - second[1])
            <= self.cfg.scenario_error_merge_tolerance_deg
        )

    def _append_new_scenarios(
        self,
        active_scenarios: list[Scenario],
        new_scenarios: Iterable[Scenario],
    ) -> bool:
        added = False

        for scenario in new_scenarios:
            if scenario is None:
                continue

            source, error = scenario
            if source is None or error is None or not math.isfinite(error):
                continue

            normalized = (
                (float(source[0]), float(source[1])),
                float(error),
            )
            if any(
                self._same_scenario(normalized, existing)
                for existing in active_scenarios
            ):
                continue

            active_scenarios.append(normalized)
            added = True

        return added


# ============================================================
# Part 7 公共接口
# ============================================================

def solve_q2(
    observation: Iterable[float],
    *,
    config: PlannerConfig | None = None,
    geometry_config: GeometryConfig | None = None,
) -> PlannerResult:
    """
    问题二公共入口。

    参数：
        observation:
            (x, y, bearing_deg)

            x、y 为第一次检测点坐标，单位为米；
            bearing_deg 为第一次示向度，单位为度。

        config:
            问题二搜索配置。

        geometry_config:
            geometry_V7 的配置。未传入时只加载一次 config.h。

    返回：
        PlannerResult
    """
    x, y, theta1 = _observation(observation)

    planner_config = (
        config if config is not None else PlannerConfig()
    )
    q1_config = (
        geometry_config
        if geometry_config is not None
        else load_config()
    )

    # 目标区域是以全局原点为圆心的圆，整体旋转不改变任何距离、
    # 夹角或接收条件。先旋转到 theta1=0 的唯一规范坐标系，避免
    # Halton 起点和坐标轴模式搜索对世界坐标方向产生偏好。
    original_observation = (x, y, theta1)
    canonical_first_point = _rotate_point((x, y), -theta1)

    solver = _ContinuousPlanner(
        S=canonical_first_point,
        theta1=0.0,
        planner_config=planner_config,
        geometry_config=q1_config,
    )

    result = solver.solve()
    return _restore_global_result(
        result,
        original_observation=original_observation,
        rotation_deg=theta1,
        canonical_first_point=canonical_first_point,
    )


def search_second_point(
    S: Iterable[float],
    theta1: float,
    *,
    config: PlannerConfig | None = None,
    geometry_config: GeometryConfig | None = None,
) -> PlannerResult:
    """
    兼容原有接口。

    示例：
        result = search_second_point((0, 0), 30)
        best_point, best_J = result
    """
    S = _point(S, "S")

    if (
        isinstance(theta1, bool)
        or not isinstance(theta1, (int, float))
        or not math.isfinite(theta1)
    ):
        raise ValueError("theta1 必须是有限数值")

    return solve_q2(
        (S[0], S[1], float(theta1)),
        config=config,
        geometry_config=geometry_config,
    )
