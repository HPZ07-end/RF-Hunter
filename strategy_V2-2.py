"""问题三基线+P2：在所有已到达位置尝试顺便测量。

同目录依赖（原文件内容无需修改，仅统一文件名）：
    client.py, geometry_V7.py, geometry_two_point_fast.py, planner.py
运行（模拟器已登录、已启动问题三测试且接口就绪）：
    python strategy_V2_2.py --robot-id 你的参赛队号
关闭插入对照：
    python strategy_V2_2.py --robot-id 你的参赛队号 --no-insert
参数对照：
    python strategy_V2_2.py --robot-id 你的参赛队号 --rho 1123 --insert-threshold 120
清除失败上限：
    python strategy_V2_2.py --robot-id 你的参赛队号 --max-clear-failures 3
可选既有问题一配置：--geometry-config config.h
可选问题二配置：--planner-config planner_config.json（PlannerConfig 字段的JSON对象）

输出 runs/q3_时间戳/{summary.json, operations.jsonl, events.jsonl, client.log}。
正式测试加密日志必须从模拟器界面另行导出，本程序不能获取案例编码或隐藏总数。
所有采样仅用于下一测向点评分；清除证书使用全部历史角域+先验方框的完整外包顶点。
首次选点调用原 solve_q2；后续适配采用有限源/误差/候选采样，不冒称连续全局最优。
默认配置是可运行工程参数；有限次重规划失败会报告未完成，绝不会当成清除成功。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import combinations
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
import time
import traceback

from client import SimulatorClient, ClientError, TransportError
from geometry_V7 import GeometryConfig, build_halfplanes, solve_halfplanes, load_config
from planner import PlannerConfig, SourceRegion, solve_q2

Point = tuple[float, float]
GOOD_REGIONS = {"POLYGON", "SEGMENT", "POINT"}
TERMINAL = {"CLEARED", "ABSENT"}
BOX = [(1., 0., 1800.), (-1., 0., 1800.), (0., 1., 1800.), (0., -1., 1800.)]


class IncompleteRun(RuntimeError):
    """算法/证书异常；保留未完成状态，并报告具体原因。"""


class BudgetStop(IncompleteRun):
    """即将耗尽现实或虚拟时间；不是任务完成。"""


@dataclass(frozen=True)
class Q3Config:
    # ---------- 七点覆盖搜索 ----------
    rho: float = 1200.0

    # ---------- 搜索阶段可靠清除插入 ----------
    enable_insert: bool = True
    insert_threshold_s: float = 120.0

    # ---------- 清除证书 ----------
    clear_margin_m: float = 0.01

    # ---------- V2.1：覆盖点顺路补测 ----------
    max_opportunistic_measurements: int = 2
    opportunistic_receive_m: float = 1100.0

    # ---------- V2.1：测向候选硬约束 ----------
    repeated_point_m: float = 50.0
    min_cross_angle_deg: float = 45.0

    # ---------- V2.1：定位近优集合 ----------
    # near_limit = best_score * (1 + near_optimal_rel)
    #              + near_optimal_abs_m
    # 默认即 best_score * 1.15
    near_optimal_abs_m: float = 0.0
    near_optimal_rel: float = 0.15

    # ---------- 无进展与重规划 ----------
    no_progress_trigger: int = 3
    max_replans: int = 3
    # 单个频道允许的清除失败总次数。失败后先原地重新测向；
    # 达到上限时停止整局，避免在异常模型/协议状态下无限循环。
    max_clear_failures_per_channel: int = 3
    progress_abs_m: float = 0.01
    progress_rel: float = 0.001

    # ---------- 规划预算 ----------
    planning_timeout_s: float = 45.0
    real_reserve_s: float = 10.0

    # ---------- 有限情景采样 ----------
    source_samples: int = 24
    candidate_count: int = 32
    error_samples: int = 3

    def __post_init__(self):
        coverage_lower = (
            900.0 * math.sqrt(3.0)
            - math.sqrt(190000.0)
        )
        coverage_upper = 1000.0 * math.sqrt(3.0)

        if (
            not math.isfinite(self.rho)
            or not coverage_lower <= self.rho <= coverage_upper
        ):
            raise ValueError(
                "rho 必须位于保证覆盖区间 "
                f"[{coverage_lower}, {coverage_upper}] 内"
            )

        if type(self.enable_insert) is not bool:
            raise ValueError("enable_insert 必须是布尔值")

        if (
            not math.isfinite(self.clear_margin_m)
            or not 0.0 <= self.clear_margin_m < 20.0
        ):
            raise ValueError(
                "clear_margin_m 必须是 [0,20) 内的有限数值"
            )

        positive_integer_fields = (
            "no_progress_trigger",
            "max_replans",
            "max_clear_failures_per_channel",
            "source_samples",
            "candidate_count",
            "error_samples",
        )
        for key in positive_integer_fields:
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} 必须是正整数")

        if (
            type(self.max_opportunistic_measurements) is not int
            or self.max_opportunistic_measurements < 0
        ):
            raise ValueError(
                "max_opportunistic_measurements 必须是非负整数"
            )

        if self.source_samples < 3:
            raise ValueError("source_samples 至少为3")

        if self.error_samples < 2:
            raise ValueError(
                "error_samples 至少为2，以包含误差区间端点"
            )

        positive_number_fields = (
            "planning_timeout_s",
            "real_reserve_s",
            "repeated_point_m",
            "opportunistic_receive_m",
        )
        for key in positive_number_fields:
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} 必须是有限正数")

        nonnegative_number_fields = (
            "insert_threshold_s",
            "near_optimal_abs_m",
            "near_optimal_rel",
            "progress_abs_m",
            "progress_rel",
        )
        for key in nonnegative_number_fields:
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{key} 必须是有限非负数")

        if not 0.0 < self.min_cross_angle_deg <= 90.0:
            raise ValueError(
                "min_cross_angle_deg 必须位于 (0,90] 度"
            )

        if not 5.0 < self.opportunistic_receive_m <= 1500.0:
            raise ValueError(
                "opportunistic_receive_m 必须位于 (5,1500] 米"
            )

        if self.near_optimal_rel > 1.0:
            raise ValueError(
                "near_optimal_rel 不应大于1；"
                "v2推荐值为0.15"
            )

        if self.progress_rel > 1.0:
            raise ValueError("progress_rel 不应大于1")
        
@dataclass
class ChannelRecord:
    channel_id: int
    status: str = "UNKNOWN"
    bearing_observations: list = field(default_factory=list)
    no_signal_observations: list = field(default_factory=list)
    near_signal_observations: list = field(default_factory=list)
    outer_polygon: list = field(default_factory=list)
    region_status: str | None = None
    diameter: float | None = None
    clearance_center: Point | None = None
    clearance_radius: float | None = None
    certificate_source: str | None = None
    absent_reason: str | None = None
    last_action: str | None = None
    failure_count: int = 0
    clear_failure_count: int = 0
    failed_clear_attempts: list = field(default_factory=list)
    invalidated_localization_rounds: list = field(default_factory=list)
    recovery_reset_pending: bool = False
    no_progress_count: int = 0
    replan_level: int = 0
    progress_history: list = field(default_factory=list)
    revision: int = 0
    q2_attempted: bool = False
    # 只缓存耗时的Q2首次建议；多历史建议随机器人位置重评近优候选。
    q2_proposal: dict | None = None


def coverage_points(rho=1200.0):
    return [(0., 0.)] + [(rho * math.cos(k * math.pi / 3),
                         rho * math.sin(k * math.pi / 3)) for k in range(6)]


def halfplanes(record):
    rows = list(BOX)
    if record.bearing_observations:
        obs = [(*o["position"], o["bearing_deg"]) for o in record.bearing_observations]
        rows.extend(map(tuple, build_halfplanes(obs, error_deg=1.0).tolist()))
    return rows


def minimum_enclosing_circle(vertices):
    """枚举1/2/3点支持圆；所有候选重新计算最大顶点距离，返回实测外包半径。

    近共线三点圆可能病态，跳过后仍有逐顶点检查的保守候选；绝不低估半径。
    最坏 O(h^4)，仅用于小规模凸多边形。不是直径中点圆的替代命名。
    """
    pts = list(dict.fromkeys(tuple(map(float, p)) for p in vertices))
    if not pts:
        raise IncompleteRun("空顶点集不能出具清除证书")
    candidates = [(p, 0.) for p in pts]
    for a, b in combinations(pts, 2):
        q = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        candidates.append((q, math.dist(a, b) / 2))
    for a, b, c in combinations(pts, 3):
        ux, uy = b[0]-a[0], b[1]-a[1]
        vx, vy = c[0]-a[0], c[1]-a[1]
        cross = ux*vy-uy*vx
        if abs(cross) <= 1e-12 * max(1., math.hypot(ux, uy)*math.hypot(vx, vy)):
            continue
        u2, v2 = ux*ux+uy*uy, vx*vx+vy*vy
        q = (a[0]+(u2*vy-v2*uy)/(2*cross), a[1]+(ux*v2-vx*u2)/(2*cross))
        if all(math.isfinite(v) for v in q):
            candidates.append((q, math.dist(q, a)))
    best = None
    for q, r in candidates:
        actual = max(math.dist(q, p) for p in pts)
        if actual <= r + 1e-7 * max(1., r):
            candidate = (actual, q)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        # 理论上端点圆或三点圆应存在；若数值失效，报告而不是造证书。
        raise IncompleteRun("最小包围圆支持圆枚举失败")
    r, q = best
    return q, max(math.dist(q, p) for p in pts)


def update_region(record, cfg, geometry_cfg):
    result = solve_halfplanes(halfplanes(record), config=geometry_cfg)
    record.region_status = result.status
    if result.status not in GOOD_REGIONS or not result.vertices:
        raise IncompleteRun(f"频道{record.channel_id}几何状态{result.status}: {result.diagnostics}")
    if record.diameter is not None and result.diameter > record.diameter + 1e-5:
        raise IncompleteRun(f"频道{record.channel_id}增加历史约束后直径异常增大")
    record.outer_polygon = result.vertices
    record.diameter = result.diameter
    q, radius = minimum_enclosing_circle(result.vertices)
    record.clearance_center, record.clearance_radius = q, radius
    record.certificate_source = "all_history_box_mec" if radius <= 20-cfg.clear_margin_m else None
    record.status = "READY" if record.certificate_source else "FOUND"


def radical_inverse(index, base):
    result, fraction = 0., 1./base
    while index:
        index, digit = divmod(index, base)
        result += digit*fraction
        fraction /= base
    return result


def sample_sources(record, cfg, planner_cfg):
    """从外包多边形及Q2的SourceRegion生成样本，并检查全部物理约束。

    无信号只留账，不裁剪几何；正常测向点构成成功接收点集合 H。
    """
    rows = halfplanes(record)
    target = cfg.source_samples * (2 ** record.replan_level)
    samples = []

    def add(p):
        p = tuple(map(float, p))
        if math.hypot(*p) > 1800 + 1e-7:
            return
        if any(a*p[0]+b*p[1]-c > 1e-7 for a, b, c in rows):
            return
        if any(not 5.0 < math.dist(p, o["position"]) <= 1500+1e-7
               for o in record.bearing_observations):
            return
        if not any(math.dist(p, q) < 1e-6 for q in samples):
            samples.append(p)

    vertices = record.outer_polygon
    center = tuple(sum(p[k] for p in vertices)/len(vertices) for k in (0, 1))
    add(center)
    for p in vertices:
        add(p)
    # 凸多边形三角扇采样，避免很窄的多观测区域被包围盒拒绝采样漏掉。
    for i in range(1, target*12+1):
        a = vertices[(i-1) % len(vertices)]
        b = vertices[i % len(vertices)]
        u, v = math.sqrt(radical_inverse(i, 2)), radical_inverse(i, 3)
        add(((1-u)*center[0]+u*((1-v)*a[0]+v*b[0]),
             (1-u)*center[1]+u*((1-v)*a[1]+v*b[1])))
        if len(samples) >= target:
            break
    # 复用问题二第一次角域+圆盘的参数化；对所有历史再筛选。
    first = record.bearing_observations[0]
    region = SourceRegion(tuple(first["position"]), first["bearing_deg"], planner_cfg)
    for i in range(1, target*24+1):
        if len(samples) >= target:
            break
        p = region.point((radical_inverse(i, 2), radical_inverse(i, 3)))
        if p is not None:
            add(p)
    if not samples:
        raise IncompleteRun(f"频道{record.channel_id}无满足全部历史/目标圆/接收距离的源样本")
    return samples

def source_estimate(record):
    """使用当前规划区域中心作为源位置估计，不读取隐藏真值。"""
    if record.clearance_center is not None:
        return tuple(map(float, record.clearance_center))
    if record.outer_polygon:
        return tuple(
            sum(float(p[k]) for p in record.outer_polygon) / len(record.outer_polygon)
            for k in (0, 1)
        )
    raise IncompleteRun(f"频道{record.channel_id}尚无可用源位置估计")


def folded_cross_angle_deg(first_point, source, candidate):
    """两条测向直线的锐角交会角，统一折算到 [0,90]。"""
    ax = first_point[0] - source[0]
    ay = first_point[1] - source[1]
    bx = candidate[0] - source[0]
    by = candidate[1] - source[1]

    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0

    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    angle = math.degrees(math.acos(cosine))
    return min(angle, 180.0 - angle)


def candidate_cross_angle(record, candidate, estimate=None):
    """取候选点相对全部历史正常测向点能够形成的最大交会角。"""
    if not record.bearing_observations:
        return 0.0

    estimate = source_estimate(record) if estimate is None else estimate
    return max(
        folded_cross_angle_deg(obs["position"], estimate, candidate)
        for obs in record.bearing_observations
    )


def candidate_min_history_distance(record, candidate):
    """只对历史正常测向点执行 50 m 硬约束。"""
    if not record.bearing_observations:
        return math.inf
    return min(
        math.dist(candidate, obs["position"])
        for obs in record.bearing_observations
    )


def point_already_tested(record, point, tolerance=1e-6):
    observations = (
        record.bearing_observations
        + record.no_signal_observations
        + record.near_signal_observations
    )
    return any(math.dist(point, obs["position"]) <= tolerance for obs in observations)

def deduplicate_vertices(vertices, tolerance=1e-7):
    result = []

    for vertex in vertices:
        point = tuple(map(float, vertex))
        if not any(
            math.dist(point, old) <= tolerance
            for old in result
        ):
            result.append(point)

    return result


def clip_vertices_by_halfplane(
    vertices,
    halfplane,
    tolerance=1e-7,
):
    """用 A*x+B*y<=C 裁剪点、线段或凸多边形。"""
    points = deduplicate_vertices(vertices, tolerance)
    if not points:
        return []

    a, b, c = map(float, halfplane)

    def residual(point):
        return a * point[0] + b * point[1] - c

    if len(points) == 1:
        return (
            points
            if residual(points[0]) <= tolerance
            else []
        )

    clipped = []

    for start, end in zip(
        points,
        points[1:] + points[:1],
    ):
        start_value = residual(start)
        end_value = residual(end)
        start_inside = start_value <= tolerance
        end_inside = end_value <= tolerance

        if start_inside and end_inside:
            clipped.append(end)
            continue

        denominator = start_value - end_value

        if start_inside and not end_inside:
            if abs(denominator) > 1e-15:
                ratio = start_value / denominator
                clipped.append(
                    (
                        start[0] + ratio * (end[0] - start[0]),
                        start[1] + ratio * (end[1] - start[1]),
                    )
                )
            continue

        if not start_inside and end_inside:
            if abs(denominator) > 1e-15:
                ratio = start_value / denominator
                clipped.append(
                    (
                        start[0] + ratio * (end[0] - start[0]),
                        start[1] + ratio * (end[1] - start[1]),
                    )
                )
            clipped.append(end)

    return deduplicate_vertices(clipped, tolerance)


def lightweight_future_diameter(
    current_vertices,
    point,
    reported_bearing_deg,
    bearing_error_deg,
):
    """仅用于候选预测，不用于真实区域和清除证书。"""
    vertices = [
        tuple(map(float, vertex))
        for vertex in current_vertices
    ]

    new_halfplanes = build_halfplanes(
        [(*point, reported_bearing_deg)],
        error_deg=bearing_error_deg,
    ).tolist()

    for halfplane in new_halfplanes:
        vertices = clip_vertices_by_halfplane(
            vertices,
            halfplane,
        )
        if not vertices:
            return None

    if len(vertices) == 1:
        return 0.0

    return max(
        math.dist(first, second)
        for first, second in combinations(vertices, 2)
    )

def score_candidate_geometry(
    record,
    point,
    sources,
    cfg,
    planner_cfg,
    geometry_cfg,
):
    """使用轻量凸集裁剪完成规划评分。

    geometry_cfg 保留在签名中以维持调用关系；
    本函数不再调用严格 solve_halfplanes()。
    """
    del geometry_cfg

    bearing_error = planner_cfg.bearing_error_deg
    errors = [
        -bearing_error
        + 2.0 * bearing_error * index / (cfg.error_samples - 1)
        for index in range(cfg.error_samples)
    ]

    worst_diameter = 0.0
    fallback_counts = {}

    for source in sources:
        distance_to_source = math.dist(point, source)
        reception_lower_bound = max(
            planner_cfg.reception_radius_min_m,
            *(
                math.dist(source, observation["position"])
                for observation in record.bearing_observations
            ),
        )

        if distance_to_source > reception_lower_bound + 1e-7:
            worst_diameter = max(
                worst_diameter,
                record.diameter,
            )
            continue

        if distance_to_source <= planner_cfg.no_bearing_radius_m:
            worst_diameter = max(
                worst_diameter,
                min(
                    2.0 * planner_cfg.no_bearing_radius_m,
                    record.diameter,
                ),
            )
            continue

        true_angle = math.degrees(
            math.atan2(
                source[1] - point[1],
                source[0] - point[0],
            )
        ) % 360.0

        for error in errors:
            predicted_diameter = lightweight_future_diameter(
                record.outer_polygon,
                point,
                true_angle + error,
                bearing_error,
            )

            if predicted_diameter is None:
                fallback_counts["LIGHTWEIGHT_EMPTY"] = (
                    fallback_counts.get(
                        "LIGHTWEIGHT_EMPTY",
                        0,
                    )
                    + 1
                )
                predicted_diameter = record.diameter

            # 新增约束理论上不能使区域变大。
            if predicted_diameter > record.diameter + 1e-5:
                fallback_counts["LIGHTWEIGHT_INCREASE"] = (
                    fallback_counts.get(
                        "LIGHTWEIGHT_INCREASE",
                        0,
                    )
                    + 1
                )
                predicted_diameter = record.diameter

            worst_diameter = max(
                worst_diameter,
                predicted_diameter,
            )

    return worst_diameter, fallback_counts


def build_static_plan(
    record,
    cfg,
    planner_cfg,
    geometry_cfg,
):
    """生成只依赖频道观测状态的静态候选评分。

    不包含机器人当前位置，因此可在多次TSP重规划之间复用。
    """
    sources = sample_sources(record, cfg, planner_cfg)
    estimate = tuple(
        sum(source[axis] for source in sources) / len(sources)
        for axis in (0, 1)
    )

    candidates = []

    def add_candidate(point, origin):
        point = tuple(map(float, point))
        if not all(
            math.isfinite(value) and abs(value) <= 2_000_000
            for value in point
        ):
            return

        if any(
            math.dist(point, old["point"]) <= 1e-6
            for old in candidates
        ):
            return

        candidates.append(
            {
                "point": point,
                "origin": origin,
            }
        )

    if record.q2_proposal:
        add_candidate(
            record.q2_proposal["point"],
            "original_q2",
        )

    for radius in (400.0, 600.0, 800.0):
        for angle_deg in range(0, 360, 45):
            angle = math.radians(angle_deg)
            add_candidate(
                (
                    estimate[0] + radius * math.cos(angle),
                    estimate[1] + radius * math.sin(angle),
                ),
                "fixed_radial",
            )

    for point in coverage_points(cfg.rho):
        add_candidate(point, "coverage_point")

    count = cfg.candidate_count * (2 ** record.replan_level)
    for index in range(count):
        angle = 2.0 * math.pi * (
            index / count
            + record.replan_level * 0.137
        )
        radius = (400.0, 600.0, 800.0)[index % 3]
        add_candidate(
            (
                estimate[0] + radius * math.cos(angle),
                estimate[1] + radius * math.sin(angle),
            ),
            "dense_radial",
        )

    rejection_counts = {
        "near_history_position": 0,
        "cross_angle": 0,
        "receive_impossible": 0,
    }
    legal_candidates = []

    for candidate in candidates:
        point = candidate["point"]

        min_history_distance = candidate_min_history_distance(
            record,
            point,
        )
        if min_history_distance < cfg.repeated_point_m:
            rejection_counts["near_history_position"] += 1
            continue

        cross_angle = candidate_cross_angle(
            record,
            point,
            estimate,
        )
        if cross_angle < cfg.min_cross_angle_deg:
            rejection_counts["cross_angle"] += 1
            continue

        if all(
            math.dist(point, source)
            > planner_cfg.reception_radius_max_m + 1e-7
            for source in sources
        ):
            rejection_counts["receive_impossible"] += 1
            continue

        legal_candidates.append(
            {
                **candidate,
                "cross_angle_deg": cross_angle,
                "min_history_distance_m": min_history_distance,
            }
        )

    if not legal_candidates:
        raise IncompleteRun(
            f"频道{record.channel_id}无满足"
            f"{cfg.repeated_point_m}米距离和"
            f"{cfg.min_cross_angle_deg}度交会角的候选"
        )

    scored_candidates = []
    prediction_fallbacks = {}

    for candidate in legal_candidates:
        predicted_diameter, fallbacks = score_candidate_geometry(
            record,
            candidate["point"],
            sources,
            cfg,
            planner_cfg,
            geometry_cfg,
        )

        for status, count in fallbacks.items():
            prediction_fallbacks[status] = (
                prediction_fallbacks.get(status, 0) + count
            )

        scored_candidates.append(
            {
                **candidate,
                "predicted_diameter": predicted_diameter,
            }
        )

    return {
        "method": "q3_cached_static_geometry",
        "estimate": estimate,
        "source_count": len(sources),
        "candidate_count": len(candidates),
        "legal_candidate_count": len(legal_candidates),
        "scored_candidates": scored_candidates,
        "candidate_rejections": rejection_counts,
        "prediction_fallback_counts": prediction_fallbacks,
        "score_scope": "all_bearings_and_prior_box",
    }


def select_dynamic_proposal(
    static_plan,
    robot_position,
    cfg,
):
    """在缓存的几何评分上，仅重新计算当前位置相关的路线代价。"""
    available = []
    near_current_rejections = 0

    for candidate in static_plan["scored_candidates"]:
        point = candidate["point"]

        if math.dist(point, robot_position) < cfg.repeated_point_m:
            near_current_rejections += 1
            continue

        route_cost = (
            math.dist(robot_position, point)
            + math.dist(point, static_plan["estimate"])
        )
        available.append(
            {
                **candidate,
                "route_cost_m": route_cost,
            }
        )

    if not available:
        raise IncompleteRun(
            "所有缓存候选均距离当前位置不足"
            f"{cfg.repeated_point_m}米"
        )

    best_score = min(
        candidate["predicted_diameter"]
        for candidate in available
    )
    near_limit = (
        best_score * (1.0 + cfg.near_optimal_rel)
        + cfg.near_optimal_abs_m
    )
    near_optimal = [
        candidate
        for candidate in available
        if candidate["predicted_diameter"]
        <= near_limit + 1e-9
    ]

    selected = min(
        near_optimal,
        key=lambda candidate: (
            candidate["route_cost_m"],
            candidate["predicted_diameter"],
            candidate["point"],
        ),
    )

    rejection_counts = dict(
        static_plan["candidate_rejections"]
    )
    rejection_counts["near_current_position"] = (
        near_current_rejections
    )

    return {
        "point": selected["point"],
        "score": selected["predicted_diameter"],
        "method": "q3_cached_near_optimal_route_aware",
        "candidate_origin": selected["origin"],
        "source_count": static_plan["source_count"],
        "candidate_count": static_plan["candidate_count"],
        "legal_candidate_count": static_plan[
            "legal_candidate_count"
        ],
        "available_candidate_count": len(available),
        "near_optimal_candidate_count": len(near_optimal),
        "best_sampled_score": best_score,
        "near_optimal_limit": near_limit,
        "near_optimal_factor": 1.0 + cfg.near_optimal_rel,
        "route_cost_m": selected["route_cost_m"],
        "cross_angle_deg": selected["cross_angle_deg"],
        "min_history_distance_m": selected[
            "min_history_distance_m"
        ],
        "candidate_rejections": rejection_counts,
        "prediction_fallback_counts": static_plan[
            "prediction_fallback_counts"
        ],
        "score_scope": static_plan["score_scope"],
    }


def plan_history(
    record,
    robot_position,
    cfg,
    planner_cfg,
    geometry_cfg,
):
    """兼容原有内部调用；直接调用时仍返回最终选点。"""
    static_plan = build_static_plan(
        record,
        cfg,
        planner_cfg,
        geometry_cfg,
    )
    return select_dynamic_proposal(
        static_plan,
        robot_position,
        cfg,
    )


def _planning_worker(
    connection,
    record,
    position,
    cfg,
    planner_cfg,
    geometry_cfg,
):
    """子进程只计算某频道的静态候选几何评分。"""
    try:
        q2_seed_error = None

        if (
            len(record.bearing_observations) == 1
            and not record.q2_attempted
            and record.q2_proposal is None
        ):
            try:
                observation = record.bearing_observations[0]
                result = solve_q2(
                    (
                        *observation["position"],
                        observation["bearing_deg"],
                    ),
                    config=planner_cfg,
                    geometry_config=geometry_cfg,
                )

                if (
                    result.second_point is not None
                    and result.worst_diameter is not None
                    and math.isfinite(result.worst_diameter)
                    and all(
                        math.isfinite(value)
                        and abs(value) <= 2_000_000
                        for value in result.second_point
                    )
                ):
                    record.q2_proposal = {
                        "point": tuple(result.second_point),
                        "score": result.worst_diameter,
                        "method": "original_q2_seed",
                        "planner_status": result.status,
                        "receive_margin_m": result.receive_margin_m,
                    }
                else:
                    q2_seed_error = (
                        "问题二未返回有限有效候选："
                        f"{result.status}"
                    )
            except Exception as error:
                q2_seed_error = repr(error)

        static_plan = build_static_plan(
            record,
            cfg,
            planner_cfg,
            geometry_cfg,
        )
        static_plan["q2_seed"] = record.q2_proposal
        static_plan["q2_seed_error"] = q2_seed_error
        connection.send((True, static_plan))

    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def bounded_plan(record, position, cfg, planner_cfg, geometry_cfg, budget_s):
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_planning_worker,
                              args=(child, record, position, cfg, planner_cfg, geometry_cfg))
    process.start()
    child.close()
    try:
        if not parent.poll(max(0., min(cfg.planning_timeout_s, budget_s))):
            raise IncompleteRun(f"频道{record.channel_id}单次选点超时；未获得可用建议")
        try:
            ok, result = parent.recv()
        except EOFError as error:
            raise IncompleteRun("选点子进程异常结束") from error
        if not ok:
            raise IncompleteRun(result)
        return result
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()


def insertion_cost(position, clearance, next_point):
    extra = math.dist(position, clearance)+math.dist(clearance, next_point)-math.dist(position, next_point)
    if extra < -1e-6:
        raise IncompleteRun("插入增量显著为负")
    return max(0., extra)/5 + 5.  # clear不切频道

def action_operation_s(action, current_channel):
    if action["action_type"] == "MEASURE":
        return 5.0 + float(action["channel"] != current_channel)
    # 光学定位3秒 + 清除2秒；clear不改变频道。
    return 5.0


def open_route_distance(start, route):
    if not route:
        return 0.0

    total = math.dist(start, route[0]["point"])
    for first, second in zip(route, route[1:]):
        total += math.dist(first["point"], second["point"])
    return total


def open_route_objective(start, route, current_channel):
    if not route:
        return 0.0
    # 将首个行动的操作时间换算为等效移动距离。
    return (
        open_route_distance(start, route)
        + 5.0 * action_operation_s(route[0], current_channel)
    )


def dynamic_open_tsp(actions, start, current_channel):
    """最近邻构造 + 2-opt；开放路径不返回原点。"""
    if not actions:
        return []

    remaining = list(actions)
    route = []
    position = tuple(start)

    while remaining:
        selected = min(
            remaining,
            key=lambda action: (
                math.dist(position, action["point"])
                + (
                    5.0 * action_operation_s(
                        action,
                        current_channel,
                    )
                    if not route
                    else 0.0
                ),
                action["channel"],
            ),
        )
        route.append(selected)
        remaining.remove(selected)
        position = selected["point"]

    best_objective = open_route_objective(
        start,
        route,
        current_channel,
    )

    improved = True
    while improved:
        improved = False
        for left in range(len(route) - 1):
            for right in range(left + 1, len(route)):
                candidate = (
                    route[:left]
                    + list(reversed(route[left:right + 1]))
                    + route[right + 1:]
                )
                objective = open_route_objective(
                    start,
                    candidate,
                    current_channel,
                )
                if objective < best_objective - 1e-9:
                    route = candidate
                    best_objective = objective
                    improved = True
                    break
            if improved:
                break

    return route

def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class Q3Runner:
    """主状态机。client可注入本地测试替身；在线决策只读取公开接口返回值。"""
    def __init__(self, client, *, config=None, planner_config=None, geometry_config=None,
                 output_dir=None, plan_function=bounded_plan):
        self.client = client
        self.cfg = config or Q3Config()
        self.pcfg = planner_config or PlannerConfig()
        self.gcfg = geometry_config or GeometryConfig()
        # 模型常数必须与问题三固定物理约束一致，允许调整搜索数值参数。
        fixed = {"target_radius_m":1800., "reception_radius_min_m":1000.,
                 "reception_radius_max_m":1500., "no_bearing_radius_m":5., "bearing_error_deg":1.}
        if any(getattr(self.pcfg, k) != v for k, v in fixed.items()) or self.gcfg.bearing_error_deg != 1.:
            raise ValueError("问题一/二配置的题目物理常数必须与问题三一致")
        self.plan_function = plan_function
        # 每个频道只保留最新观测版本的静态规划缓存。
        self._static_plan_cache = {}
        self.records = {c: ChannelRecord(c) for c in range(1, 21)}
        self.points = coverage_points(self.cfg.rho)
        self.scan_done = [[False]*20 for _ in self.points]
        self.next_coverage_index = 0
        self.operations, self.events, self.trajectory = [], [], []
        self.phase = "initialization"
        self.total_distance_m = 0.
        self.counts = {
            "measure": 0,
            "switch": 0,
            "optical": 0,
            "clear_success": 0,
            "clear_failure": 0,
            "insertion": 0,
            "unknown_measure": 0,
            "opportunistic_measure": 0,
            "opportunistic_arrivals_considered": 0,
            "opportunistic_arrivals_used": 0,
            "active_measure": 0,
            "recovery_measure": 0,
            "candidate_repeat_rejected": 0,
            "candidate_angle_rejected": 0,
            "tsp_replans": 0,
            "planning_cache_hit": 0,
            "planning_cache_miss": 0,
        }

        self.phase_virtual_s = {"search":0., "localization":0.}
        self.entered_at = None
        self.end_at = None
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=False)
        self._streams = {}
        try:
            if self.output_dir:
                for name in ("operations", "events"):
                    self._streams[name] = (self.output_dir / f"{name}.jsonl").open("w", encoding="utf-8")
        except Exception:
            self.close()
            raise

    @property
    def discovered_count(self):
        return sum(bool(r.bearing_observations or r.near_signal_observations) for r in self.records.values())

    @property
    def cleared_count(self):
        return sum(r.status == "CLEARED" for r in self.records.values())

    def close(self):
        for stream in self._streams.values():
            stream.close()

    def _append(self, kind, entry):
        entry = json_safe(entry)
        (self.operations if kind == "operations" else self.events).append(entry)
        if kind in self._streams:
            self._streams[kind].write(json.dumps(entry, ensure_ascii=False, allow_nan=False)+"\n")
            self._streams[kind].flush()

    def event(self, kind, **data):
        self._append("events", {"event":kind, "phase":self.phase,
                     "virtual_time_s":self.client.virtual_time_s, **data})

    def check_budget(self, move_to=None, action="measure", channel=None):
        remaining = self.client.remaining_real_time_s
        if remaining is not None and remaining <= self.cfg.real_reserve_s:
            raise BudgetStop("现实预算不足，保留退出余量；任务未完成")
        if self.client.virtual_time_s is not None:
            cost = 0.
            if move_to is not None:
                cost = math.dist(self.client.position, move_to)/5 + 5
                if action == "measure" and channel != self.client.current_channel:
                    cost += 1
            if self.client.virtual_time_s+cost >= self.client.max_virtual_duration_s:
                raise BudgetStop("虚拟时间不足；任务未完成")

    def action(self, method, position=None, channel=None):
        self.check_budget(position, method, channel)
        before_position = self.client.position
        before_channel = self.client.current_channel
        before_time = self.client.virtual_time_s or 0.
        started = time.monotonic()
        args = () if position is None else (*position, channel)
        try:
            result = getattr(self.client, method)(*args)
        except Exception as error:
            self._append("operations", {"method":method, "position":position, "channel":channel,
                 "phase":self.phase, "accepted":getattr(error, "response", {}).get("accepted"), "execution_confirmed":False,
                 "error":repr(error), "pending_request_id":self.client.pending_request_id,
                 "http_status":getattr(error, "status", None), "response":getattr(error, "response", None)})
            raise
        delta = result["virtual_time_s"]-before_time
        distance = math.dist(before_position, position) if position is not None else 0.
        switched = int(method == "measure" and channel != before_channel)
        expected = 0.
        if method in ("measure", "clear"):
            self.total_distance_m += distance
            if method == "measure":
                self.counts["measure"] += 1
                self.counts["switch"] += switched
                expected = distance/5 + 5 + switched
            else:
                self.counts["optical"] += 1
                success = result["clear_result"] == "success"
                self.counts["clear_success"] += int(success)
                self.counts["clear_failure"] += int(not success)
                expected = distance/5 + (5 if success else 3)
            self.phase_virtual_s[self.phase] += delta
            self.trajectory.append({"from":before_position, "to":position, "phase":self.phase,
                                    "method":method, "channel":channel, "virtual_time_s":result["virtual_time_s"]})
        self._append("operations", {"method":method, "position":position, "channel":channel,
            "phase":self.phase, "accepted":True, "execution_confirmed":True, "response":result,
            "distance_m":distance, "switch_count":switched, "virtual_delta_s":delta,
            "expected_delta_s":expected, "request_elapsed_s":time.monotonic()-started})
        if abs(delta-expected) > 2e-5:
            self.event("time_accounting_mismatch", actual=delta, expected=expected)
        return result

    def measure(
        self,
        record,
        point,
        coverage_index=None,
        measurement_kind="active",
    ):
        if measurement_kind not in {
            "unknown",
            "opportunistic",
            "active",
            "recovery",
        }:
            raise ValueError("未知测向任务类型")

        old_diameter = record.diameter
        old_radius = record.clearance_radius
        result = self.action(
            "measure",
            point,
            record.channel_id,
        )

        self.counts[f"{measurement_kind}_measure"] += 1

        if coverage_index is not None:
            self.scan_done[coverage_index][
                record.channel_id - 1
            ] = True

        datum = {
            "position": tuple(point),
            "virtual_time_s": result["virtual_time_s"],
            "real_timestamp_ms": result["real_timestamp_ms"],
            "measurement_kind": measurement_kind,
        }
        record.last_action = "measure"
        record.revision += 1
        code = result["measure_result"]

        # 清除失败已经否定了上一轮定位证书。旧角域仅暂时用于选择恢复
        # 测向点；一旦重新获得 direction/near，就以该反馈开启全新的定位轮，
        # 防止旧的矛盾角域再次生成同一个错误清除点。
        if (
            record.recovery_reset_pending
            and code in {"direction", "near"}
        ):
            record.bearing_observations = []
            record.near_signal_observations = []
            record.outer_polygon = []
            record.region_status = None
            record.diameter = None
            record.clearance_center = None
            record.clearance_radius = None
            record.certificate_source = None
            record.q2_attempted = False
            record.q2_proposal = None
            record.recovery_reset_pending = False
            old_diameter = None
            old_radius = None
            self.event(
                "localization_round_reset",
                channel=record.channel_id,
                triggering_result=code,
                measurement_kind=measurement_kind,
                invalidated_round_count=len(
                    record.invalidated_localization_rounds
                ),
            )

        if code == "direction":
            record.bearing_observations.append(
                {
                    **datum,
                    "bearing_deg": result["svd_deg"],
                }
            )
            record.status = "FOUND"
            update_region(record, self.cfg, self.gcfg)

        elif code == "near":
            record.near_signal_observations.append(datum)
            record.status = "READY"
            record.clearance_center = tuple(point)
            record.clearance_radius = 5.0
            record.certificate_source = "near_feedback"

        elif code == "no_signal":
            record.no_signal_observations.append(datum)

        else:
            raise IncompleteRun(f"未知检测结果 {code}")

        improved = (
            record.status == "READY"
            or (
                code == "direction"
                and (
                    old_diameter is None
                    or old_diameter - record.diameter
                    > max(
                        self.cfg.progress_abs_m,
                        self.cfg.progress_rel * old_diameter,
                    )
                )
            )
        )
        record.no_progress_count = (
            0 if improved else record.no_progress_count + 1
        )
        record.progress_history.append(
            {
                "result": code,
                "measurement_kind": measurement_kind,
                "diameter_before": old_diameter,
                "diameter": record.diameter,
                "radius_before": old_radius,
                "clearance_radius": record.clearance_radius,
                "status": record.status,
                "improved": improved,
                "virtual_time_s": result["virtual_time_s"],
                "outer_polygon": record.outer_polygon,
                "clearance_center": record.clearance_center,
            }
        )
        self.event(
            "measurement_update",
            channel=record.channel_id,
            result=code,
            measurement_kind=measurement_kind,
            status=record.status,
            diameter_before=old_diameter,
            diameter_after=record.diameter,
            radius_before=old_radius,
            radius_after=record.clearance_radius,
        )

    def clear(self, record, insertion=False):
        if record.status != "READY" or not record.certificate_source:
            raise IncompleteRun("禁止无可靠证书清除")
        if record.certificate_source == "all_history_box_mec":
            radius = max(math.dist(record.clearance_center, p) for p in record.outer_polygon)
            if radius > 20-self.cfg.clear_margin_m:
                raise IncompleteRun("执行前逐顶点证书复核失败")
        clear_point = tuple(record.clearance_center)
        result = self.action("clear", clear_point, record.channel_id)
        record.last_action = "clear"
        if insertion:
            self.counts["insertion"] += 1
        if result["clear_result"] == "success":
            record.status = "CLEARED"
            self.event("cleared", channel=record.channel_id, insertion=insertion,
                       certificate_source=record.certificate_source)
            self.perform_opportunistic_measurements(
                clear_point,
                arrival_kind="clear_success",
                primary_channel=record.channel_id,
            )
            return True

        # 清除失败说明当前证书与模拟器反馈不一致。不能标记成功，也不能原样
        # 重复清除；先撤销证书，再利用机器狗已经到达失败点的条件原地重测。
        failed_point = clear_point
        record.failure_count += 1
        record.clear_failure_count += 1
        record.failed_clear_attempts.append(
            {
                "position": failed_point,
                "virtual_time_s": result["virtual_time_s"],
                "real_timestamp_ms": result["real_timestamp_ms"],
                "clear_result": result["clear_result"],
                "certificate_source": record.certificate_source,
                "certificate_radius": record.clearance_radius,
            }
        )
        record.invalidated_localization_rounds.append(
            {
                "failed_point": failed_point,
                "failed_virtual_time_s": result["virtual_time_s"],
                "bearing_observations": list(
                    record.bearing_observations
                ),
                "near_signal_observations": list(
                    record.near_signal_observations
                ),
                "outer_polygon": list(record.outer_polygon),
                "region_status": record.region_status,
                "diameter": record.diameter,
                "clearance_center": record.clearance_center,
                "clearance_radius": record.clearance_radius,
                "certificate_source": record.certificate_source,
            }
        )
        self.event(
            "clear_failed",
            channel=record.channel_id,
            insertion=insertion,
            failure_count=record.clear_failure_count,
            failed_point=failed_point,
            certificate_source=record.certificate_source,
            certificate_radius=record.clearance_radius,
        )

        if (
            record.clear_failure_count
            >= self.cfg.max_clear_failures_per_channel
        ):
            raise IncompleteRun(
                f"频道{record.channel_id}连续定位清除失败达到"
                f"{self.cfg.max_clear_failures_per_channel}次；停止以避免死循环"
            )

        record.status = "FOUND"
        record.certificate_source = None
        record.clearance_center = None
        record.clearance_radius = None
        record.no_progress_count = 0
        record.q2_attempted = True
        record.q2_proposal = None
        record.recovery_reset_pending = True
        record.revision += 1
        # 扩大后续候选采样，但不能使该频道仅因清除失败立刻超过重规划上限。
        record.replan_level = min(
            record.replan_level + 1,
            self.cfg.max_replans,
        )

        self.event(
            "clear_relocalization_started",
            channel=record.channel_id,
            point=failed_point,
            failure_count=record.clear_failure_count,
        )
        self.measure(
            record,
            failed_point,
            measurement_kind="recovery",
        )
        self.perform_opportunistic_measurements(
            failed_point,
            arrival_kind="clear_failure_recovery",
            primary_channel=record.channel_id,
        )
        return False

    def mark_absent(self):
        for record in self.records.values():
            if record.status != "UNKNOWN":
                continue
            if self.discovered_count >= 16:
                reason = "16_distinct_sources_discovered"
            elif all(row[record.channel_id-1] for row in self.scan_done):
                reason = "seven_coverage_points_all_no_signal"
            else:
                continue
            record.status, record.absent_reason = "ABSENT", reason
            self.event("absent", channel=record.channel_id, reason=reason)

    def predict_opportunistic_radius(self, record, point):
        """估计在覆盖点补测后的最坏最小包围圆半径。"""
        if (
            record.status != "FOUND"
            or record.clearance_radius is None
            or not record.outer_polygon
        ):
            return None

        estimate = source_estimate(record)
        true_angle = math.degrees(
            math.atan2(
                estimate[1] - point[1],
                estimate[0] - point[0],
            )
        ) % 360

        predicted_radii = []
        rows = halfplanes(record)

        for measurement_error in (-1.0, 0.0, 1.0):
            reported_angle = true_angle + measurement_error
            future = rows + list(
                map(
                    tuple,
                    build_halfplanes(
                        [(*point, reported_angle)],
                        error_deg=1.0,
                    ).tolist(),
                )
            )
            result = solve_halfplanes(
                future,
                config=self.gcfg,
            )

            if result.status not in GOOD_REGIONS or not result.vertices:
                return record.clearance_radius

            _, radius = minimum_enclosing_circle(result.vertices)
            predicted_radii.append(radius)

        return max(predicted_radii)


    def opportunistic_candidates(
        self,
        point,
        *,
        arrival_kind,
        coverage_index=None,
        primary_channel=None,
    ):
        candidates = []

        for channel, record in self.records.items():
            if record.status != "FOUND":
                continue
            if point_already_tested(record, point):
                continue

            min_distance = candidate_min_history_distance(
                record,
                point,
            )
            if min_distance < self.cfg.repeated_point_m:
                self.counts["candidate_repeat_rejected"] += 1
                self.event(
                    "opportunistic_candidate_rejected",
                    channel=channel,
                    coverage_index=coverage_index,
                    arrival_kind=arrival_kind,
                    primary_channel=primary_channel,
                    reason="near_history_position",
                    min_history_distance_m=min_distance,
                )
                continue

            estimate = source_estimate(record)
            cross_angle = candidate_cross_angle(
                record,
                point,
                estimate,
            )
            if cross_angle < self.cfg.min_cross_angle_deg:
                self.counts["candidate_angle_rejected"] += 1
                self.event(
                    "opportunistic_candidate_rejected",
                    channel=channel,
                    coverage_index=coverage_index,
                    arrival_kind=arrival_kind,
                    primary_channel=primary_channel,
                    reason="cross_angle",
                    cross_angle_deg=cross_angle,
                )
                continue

            estimated_distance = math.dist(point, estimate)
            if estimated_distance > self.cfg.opportunistic_receive_m:
                continue

            predicted_radius = self.predict_opportunistic_radius(
                record,
                point,
            )
            if predicted_radius is None:
                continue

            reduction = max(
                0.0,
                record.clearance_radius - predicted_radius,
            )
            operation_s = (
                5.0
                + float(channel != self.client.current_channel)
            )
            benefit = reduction / operation_s

            if reduction <= 0:
                continue

            candidates.append(
                {
                    "channel": channel,
                    "record": record,
                    "point": tuple(point),
                    "benefit": benefit,
                    "predicted_radius": predicted_radius,
                    "radius_reduction": reduction,
                    "cross_angle_deg": cross_angle,
                    "min_history_distance_m": min_distance,
                    "estimated_receive_distance_m": estimated_distance,
                    "operation_s": operation_s,
                }
            )

        return candidates


    def perform_opportunistic_measurements(
        self,
        point,
        *,
        arrival_kind,
        coverage_index=None,
        primary_channel=None,
    ):
        """利用一次已经发生的到达，为其他FOUND频道原地补测。"""
        self.counts["opportunistic_arrivals_considered"] += 1
        completed = 0

        while (
            completed
            < self.cfg.max_opportunistic_measurements
        ):
            candidates = self.opportunistic_candidates(
                point,
                arrival_kind=arrival_kind,
                coverage_index=coverage_index,
                primary_channel=primary_channel,
            )
            if not candidates:
                break

            selected = max(
                candidates,
                key=lambda item: (
                    item["benefit"],
                    item["radius_reduction"],
                    -item["channel"],
                ),
            )
            self.event(
                "opportunistic_selected",
                coverage_index=coverage_index,
                arrival_kind=arrival_kind,
                primary_channel=primary_channel,
                channel=selected["channel"],
                benefit=selected["benefit"],
                predicted_radius=selected["predicted_radius"],
                radius_reduction=selected["radius_reduction"],
                cross_angle_deg=selected["cross_angle_deg"],
                min_history_distance_m=selected[
                    "min_history_distance_m"
                ],
                candidate_count=len(candidates),
            )

            self.measure(
                selected["record"],
                point,
                measurement_kind="opportunistic",
            )
            completed += 1

        if completed:
            self.counts["opportunistic_arrivals_used"] += 1
        self.event(
            "opportunistic_arrival_completed",
            point=tuple(point),
            arrival_kind=arrival_kind,
            coverage_index=coverage_index,
            primary_channel=primary_channel,
            measurement_count=completed,
        )
        return completed

    def search(self):
        self.phase = "search"

        for index, point in enumerate(self.points):
            self.next_coverage_index = index

            unknown = [
                channel
                for channel, record in self.records.items()
                if record.status == "UNKNOWN"
            ]
            if not unknown:
                break

            current = self.client.current_channel
            order = (
                ([current] if current in unknown else [])
                + [
                    channel
                    for channel in unknown
                    if channel != current
                ]
            )

            # 1. 保证覆盖所需的 UNKNOWN 扫描。
            for channel in order:
                self.measure(
                    self.records[channel],
                    point,
                    coverage_index=index,
                    measurement_kind="unknown",
                )
                if self.discovered_count == 16:
                    self.mark_absent()
                    break

            # 2. 对此前发现、但尚未 READY 的频道顺路补测。
            self.perform_opportunistic_measurements(
                point,
                arrival_kind="coverage_scan",
                coverage_index=index,
            )

            self.next_coverage_index = index + 1

            if self.discovered_count == 16:
                break

            # 3. 保留 v1 原有的可靠清除插入规则。
            if index < 6 and self.cfg.enable_insert:
                choices = [
                    (
                        insertion_cost(
                            self.client.position,
                            record.clearance_center,
                            self.points[index + 1],
                        ),
                        channel,
                        record,
                    )
                    for channel, record in self.records.items()
                    if record.status == "READY"
                ]

                if choices:
                    cost, _, record = min(
                        choices,
                        key=lambda item: (
                            item[0],
                            item[1],
                        ),
                    )
                    self.event(
                        "insertion_decision",
                        channel=record.channel_id,
                        cost_s=cost,
                        accepted=(
                            cost
                            <= self.cfg.insert_threshold_s
                        ),
                    )
                    if cost <= self.cfg.insert_threshold_s:
                        self.clear(record, insertion=True)

        self.mark_absent()

    def planning_cache_key(self, record):
        """机器人位置不属于缓存键，因为它只影响动态路线代价。"""
        return (
            record.revision,
            record.replan_level,
            self.cfg.rho,
            self.cfg.source_samples,
            self.cfg.candidate_count,
            self.cfg.error_samples,
            self.cfg.repeated_point_m,
            self.cfg.min_cross_angle_deg,
            self.cfg.near_optimal_abs_m,
            self.cfg.near_optimal_rel,
        )

    def proposal(self, record):
        self.check_budget()

        cache_key = self.planning_cache_key(record)
        cache_entry = self._static_plan_cache.get(
            record.channel_id
        )
        cache_hit = (
            cache_entry is not None
            and cache_entry["key"] == cache_key
        )

        if cache_hit:
            self.counts["planning_cache_hit"] += 1
            static_plan = cache_entry["static_plan"]

        else:
            self.counts["planning_cache_miss"] += 1

            remaining = self.client.remaining_real_time_s
            budget = (
                self.cfg.planning_timeout_s
                if remaining is None
                else remaining - self.cfg.real_reserve_s
            )
            static_plan = self.plan_function(
                record,
                self.client.position,
                self.cfg,
                self.pcfg,
                self.gcfg,
                budget,
            )

            if static_plan.get("q2_seed") is not None:
                record.q2_proposal = static_plan["q2_seed"]

            self._static_plan_cache[record.channel_id] = {
                "key": cache_key,
                "static_plan": static_plan,
            }

            rejections = static_plan.get(
                "candidate_rejections",
                {},
            )
            self.counts["candidate_repeat_rejected"] += (
                rejections.get("near_history_position", 0)
            )
            self.counts["candidate_angle_rejected"] += (
                rejections.get("cross_angle", 0)
            )

        result = select_dynamic_proposal(
            static_plan,
            self.client.position,
            self.cfg,
        )

        self.counts["candidate_repeat_rejected"] += (
            result["candidate_rejections"].get(
                "near_current_position",
                0,
            )
        )

        self.event(
            "planned",
            channel=record.channel_id,
            planning_cache_hit=cache_hit,
            cache_revision=record.revision,
            cache_replan_level=record.replan_level,
            **result,
        )
        self.check_budget()
        return result

    def replan(self, record, reason):
        record.failure_count += 1
        record.replan_level += 1
        record.no_progress_count = 0
        # 原Q2已尝试但失败/无信号时，启用全历史候选适配，避免原地重复同一建议。
        record.q2_attempted = True
        self.event("replan", channel=record.channel_id, level=record.replan_level, reason=reason)

    def localize(self):
        """每次为全部未完成源生成行动点，求开放TSP并只执行首个行动。"""
        self.phase = "localization"

        while any(
            record.status in ("FOUND", "READY")
            for record in self.records.values()
        ):
            self.check_budget()
            actions = []

            for channel, record in self.records.items():
                if record.status not in ("FOUND", "READY"):
                    continue

                # READY 已有可靠证书，不应因此前重规划次数过多而丢弃。
                if (
                    record.status == "FOUND"
                    and record.replan_level > self.cfg.max_replans
                ):
                    continue

                if record.status == "READY":
                    actions.append(
                        {
                            "channel": channel,
                            "action_type": "CLEAR",
                            "point": tuple(record.clearance_center),
                            "proposal": None,
                        }
                    )
                    continue

                try:
                    proposal = self.proposal(record)
                except BudgetStop:
                    raise
                except IncompleteRun as error:
                    self.replan(record, str(error))
                    continue

                actions.append(
                    {
                        "channel": channel,
                        "action_type": "MEASURE",
                        "point": tuple(proposal["point"]),
                        "proposal": proposal,
                    }
                )

            if not actions:
                pending = [
                    channel
                    for channel, record in self.records.items()
                    if record.status in ("FOUND", "READY")
                ]
                if any(
                    self.records[channel].status == "FOUND"
                    and self.records[channel].replan_level
                    <= self.cfg.max_replans
                    for channel in pending
                ):
                    continue

                raise IncompleteRun(
                    f"重规划预算耗尽，频道{pending}仍未完成；"
                    "没有将它们标为成功"
                )

            route = dynamic_open_tsp(
                actions,
                self.client.position,
                self.client.current_channel,
            )
            self.counts["tsp_replans"] += 1

            route_distance = open_route_distance(
                self.client.position,
                route,
            )
            route_objective = open_route_objective(
                self.client.position,
                route,
                self.client.current_channel,
            )
            self.event(
                "open_tsp_planned",
                route=[
                    {
                        "channel": action["channel"],
                        "action_type": action["action_type"],
                        "point": action["point"],
                    }
                    for action in route
                ],
                route_distance_m=route_distance,
                route_objective_equivalent_m=route_objective,
            )

            # 动态规划只执行第一个行动。
            selected = route[0]
            channel = selected["channel"]
            record = self.records[channel]

            self.event(
                "target_selected",
                channel=channel,
                action_type=selected["action_type"],
                point=selected["point"],
                tsp_route_distance_m=route_distance,
            )

            if selected["action_type"] == "CLEAR":
                self.clear(record)
                continue

            record.q2_attempted = True
            self.measure(
                record,
                selected["point"],
                measurement_kind="active",
            )
            self.perform_opportunistic_measurements(
                selected["point"],
                arrival_kind="active_measurement",
                primary_channel=channel,
            )

            if (
                record.status == "FOUND"
                and record.no_progress_count
                >= self.cfg.no_progress_trigger
            ):
                self.replan(
                    record,
                    "连续测向无足够直径缩减/无信号",
                )

    def complete(self):
        return self.cleared_count == 16 or all(r.status in TERMINAL for r in self.records.values())

    def attempt_safe_exit(self, phase, trigger):
        """在动作执行状态确定、会话仍有效时尝试主动退出。

        若客户端存在待确认请求，则不能发送不同动作；此时只记录跳过原因，
        交由原请求的幂等重试/模拟器超时机制处理。
        """
        if self.entered_at is None:
            return False

        pending_request_id = self.client.pending_request_id
        remaining = self.client.remaining_real_time_s
        virtual = self.client.virtual_time_s
        max_virtual = self.client.max_virtual_duration_s

        if pending_request_id is not None:
            self.event(
                "safe_exit_skipped",
                trigger=trigger,
                reason="pending_request",
                pending_request_id=pending_request_id,
            )
            return False
        if remaining is None or remaining <= 0:
            self.event(
                "safe_exit_skipped",
                trigger=trigger,
                reason="real_session_expired_or_unknown",
                remaining_real_time_s=remaining,
            )
            return False
        if (
            virtual is None
            or max_virtual is None
            or virtual >= max_virtual
        ):
            self.event(
                "safe_exit_skipped",
                trigger=trigger,
                reason="virtual_session_expired_or_unknown",
                virtual_time_s=virtual,
                max_virtual_duration_s=max_virtual,
            )
            return False

        self.phase = phase
        started = time.monotonic()
        try:
            response = self.client.exit()
            self._append(
                "operations",
                {
                    "method": "exit",
                    "position": None,
                    "channel": None,
                    "phase": self.phase,
                    "accepted": True,
                    "execution_confirmed": True,
                    "response": response,
                    "distance_m": 0.0,
                    "switch_count": 0,
                    "virtual_delta_s": 0.0,
                    "expected_delta_s": 0.0,
                    "request_elapsed_s": time.monotonic() - started,
                    "safe_exit_trigger": trigger,
                },
            )
            self.event("safe_exit_succeeded", trigger=trigger)
            return True
        except Exception as exit_error:
            self.event(
                "safe_exit_failed",
                trigger=trigger,
                reason=repr(exit_error),
                pending_request_id=self.client.pending_request_id,
            )
            return False

    def run(self):
        outcome, reason = "incomplete", "尚未完成"
        exit_confirmed = False
        try:
            self.action("enter")
            self.entered_at = time.monotonic()
            self.trajectory.append({"to":(0.,0.), "phase":"enter", "virtual_time_s":0.})
            self.search()
            self.localize()
            if not self.complete():
                raise IncompleteRun("存在非终态频道，不能正常宣告任务完成")
            # 完成后退出，不再计入定位动作；exit前保留通信预算。
            self.phase = "finish"
            self.action("exit")
            exit_confirmed = True
            outcome, reason = "success", "全部频道终态，或确认清除16源；主动退出已确认"
        except (IncompleteRun, ClientError, KeyboardInterrupt) as error:
            reason = f"{type(error).__name__}: {error}"
            self.event("run_incomplete", reason=reason)
            # 通信异常/结果未知时不能发不同动作，也不能用exit查询结束原因。
            # 本地算法异常且会话预算尚存时，主动结束本次不完整测试，节省等待时间。
            if not isinstance(error, ClientError):
                exit_confirmed = self.attempt_safe_exit(
                    "incomplete_exit",
                    type(error).__name__,
                )
        except Exception as error:
            reason = f"unexpected_error: {error}"
            self.event("unexpected_error", traceback=traceback.format_exc())
            # 未预期的本地异常也应尽量主动结束；存在待确认请求时不能发送exit。
            exit_confirmed = self.attempt_safe_exit(
                "unexpected_exit",
                type(error).__name__,
            )
        finally:
            self.end_at = time.monotonic()
            summary = self.summary(outcome, reason, exit_confirmed)
            if self.output_dir:
                (self.output_dir / "summary.json").write_text(
                    json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            self.close()
        return summary

    def summary(self, outcome, reason, exit_confirmed):
        virtual = self.client.virtual_time_s
        account = self.total_distance_m/5 + self.counts["switch"] + 5*self.counts["measure"] + 3*self.counts["optical"] + 2*self.counts["clear_success"]
        return {"outcome":outcome, "reason":reason, "all_targets_resolved":self.complete(),
            "exit_confirmed":exit_confirmed, "discovered_count":self.discovered_count,
            "cleared_count":self.cleared_count, "total_source_count":None,
            "clearance_ratio":None, "truth_note":"总数/清除比例须在演练结束后由界面真值补算，在线不读取",
            "virtual_time_s":virtual, "accounted_virtual_time_s":account,
            "average_clear_time_s":virtual/self.cleared_count if self.cleared_count and virtual is not None else None,
            "program_elapsed_s":self.end_at-self.entered_at if self.entered_at is not None else None,
            "program_time_note":"本地enter响应后至流程结束的近似耗时；正式计时以模拟器为准",
            "total_distance_m":self.total_distance_m, "counts":self.counts,
            "phase_virtual_s":self.phase_virtual_s, "next_coverage_index":self.next_coverage_index,
            "coverage_points":self.points, "scan_done":self.scan_done,
            "coverage_proof_endpoint_distances_m":[math.sqrt(r*r+self.cfg.rho**2-math.sqrt(3)*r*self.cfg.rho) for r in (1000,1800)],
            "trajectory":self.trajectory, "channels":{c:asdict(r) for c,r in self.records.items()},
            "q3_config":asdict(self.cfg), "planner_config":asdict(self.pcfg), "geometry_config":asdict(self.gcfg),
            "formal_log_note":"正式加密日志请在模拟器界面导出，并保留原文件名"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--rho", type=float, default=1200.)
    parser.add_argument("--no-insert", action="store_true")
    parser.add_argument("--insert-threshold", type=float, default=60.)
    parser.add_argument("--clear-margin", type=float, default=0.01)
    parser.add_argument("--planning-timeout", type=float, default=45.)
    parser.add_argument("--max-clear-failures", type=int, default=3)
    parser.add_argument("--geometry-config", type=Path)
    parser.add_argument("--planner-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = Q3Config(rho=args.rho, enable_insert=not args.no_insert,
                      insert_threshold_s=args.insert_threshold, clear_margin_m=args.clear_margin,
                      planning_timeout_s=args.planning_timeout,
                      max_clear_failures_per_channel=args.max_clear_failures)
    gcfg = load_config(args.geometry_config) if args.geometry_config else GeometryConfig()
    pcfg = PlannerConfig(**json.loads(args.planner_config.read_text(encoding="utf-8-sig"))) if args.planner_config else PlannerConfig()
    out = args.output_dir or Path("runs") / datetime.now().strftime("q3_%Y%m%d_%H%M%S_%f")
    client = SimulatorClient(args.robot_id, args.base_url)
    runner = Q3Runner(client, config=config, planner_config=pcfg, geometry_config=gcfg, output_dir=out)
    logging.basicConfig(filename=out/"client.log", level=logging.INFO,
                        encoding="utf-8", format="%(asctime)s %(levelname)s %(message)s")
    print(f"问题三基线+P2启动，输出目录：{out}", flush=True)
    summary = runner.run()
    print(json.dumps({k:summary[k] for k in ("outcome","reason","discovered_count","cleared_count",
                     "virtual_time_s","average_clear_time_s","program_elapsed_s")}, ensure_ascii=False, indent=2))
    return 0 if summary["outcome"] == "success" else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
