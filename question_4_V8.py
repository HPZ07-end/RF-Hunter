"""问题四 V8 独立终版：认证搜索、并行定位与全局价值调度。

本文件已完整包含问题四终版运行所需的策略实现，不再导入或依赖
任何早期问题四策略文件。运行时仅依赖同目录中的公共模块：

    client.py       - SimulatorClient 通信接口
    geometry_V7.py  - 示向角域半平面与严格几何求交
    planner.py      - 问题二的第二检测点规划器（仅作候选种子）

在线运行：
    python question_4_V8.py --robot-id 你的参赛队号

离线自检：
    python question_4_V8.py --self-test

终版策略保留完整的25点双环认证节点，并综合实现动态搜索、搜索途中
机会定位、READY源全局清除路线、插入代价顺路清除、单示向安全重捕获、
局部闭环、恢复路径重排、概率校准和严格几何清除证书。
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import combinations
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
import random
import time
import traceback
from typing import Iterable, Sequence

from client import ClientError, SimulatorClient
from geometry_V7 import (
    GeometryConfig,
    build_halfplanes,
    load_config,
    solve_halfplanes,
)
from planner import PlannerConfig, solve_q2


Point = tuple[float, float]
Triangle = tuple[Point, Point, Point]
GOOD_REGIONS = {"POLYGON", "SEGMENT", "POINT"}
TERMINAL_STATES = {"CLEARED", "ABSENT"}
BOX_HALFPLANES = (
    (1.0, 0.0, 1800.0),
    (-1.0, 0.0, 1800.0),
    (0.0, 1.0, 1800.0),
    (0.0, -1.0, 1800.0),
)


class IncompleteRun(RuntimeError):
    """策略或证书无法继续；绝不把未完成伪装成成功。"""


class BudgetStop(IncompleteRun):
    """现实或虚拟时间不足。"""


@dataclass(frozen=True)
class Q4Config:
    # 题目常数
    target_radius_m: float = 1800.0
    reception_radius_min_m: float = 1000.0
    reception_radius_max_m: float = 1500.0
    bearing_error_deg: float = 1.0
    strong_signal_radius_m: float = 5.0
    clearance_radius_m: float = 20.0
    dog_speed_m_per_s: float = 5.0
    channel_switch_time_s: float = 1.0
    detection_time_s: float = 5.0
    optical_time_s: float = 3.0
    clearance_time_s: float = 2.0
    channel_min: int = 1
    channel_max: int = 20
    source_count_max: int = 16

    # 问题四保证参数
    global_grid_side_m: float = 1500.0
    recovery_grid_side_m: float = 4.9
    coarse_recovery_grid_side_m: float = 500.0
    fine_recovery_max_radius_m: float = 120.0
    fine_recovery_max_nodes: int = 4000
    clear_margin_m: float = 0.01
    geometry_fallback_margin_m: float = 0.05

    # 主动定位启发式
    no_progress_trigger: int = 3
    progress_abs_m: float = 0.01
    progress_rel: float = 0.001
    source_samples: int = 32
    error_samples: int = 3
    orientation_samples: int = 24
    max_scored_candidates: int = 256
    repeated_point_m: float = 5.0
    candidate_local_radius_m: float = 250.0
    no_signal_side_width_deg: float = 30.0
    enable_q2_seed: bool = True
    q2_timeout_s: float = 30.0

    # 全局搜索途中顺带定位已发现频道
    enable_parallel_search_localization: bool = True
    opportunistic_max_per_node: int = 2
    opportunistic_max_per_channel: int = 3
    opportunistic_lookahead_nodes: int = 3
    opportunistic_lookahead_ratio: float = 0.85
    opportunistic_min_receive_score: float = 0.35
    opportunistic_min_possible_fraction: float = 0.50
    opportunistic_min_relative_reduction: float = 0.03
    opportunistic_min_absolute_reduction_m: float = 2.0

    # 搜索结束后的全局多频道调度
    enable_value_time_scheduler: bool = False
    scheduler_active_shortlist: int = 4
    scheduler_completion_value_m: float = 1000.0
    scheduler_recovery_value_m: float = 250.0
    scheduler_wait_weight: float = 0.05
    scheduler_wait_cap: int = 20

    # 清除失败与预算
    max_clear_failures_per_channel: int = 3
    real_reserve_s: float = 10.0
    recovery_batch_size: int = 20
    min_request_interval_s: float = 0.03

    def __post_init__(self):
        positive = (
            "target_radius_m",
            "reception_radius_min_m",
            "reception_radius_max_m",
            "bearing_error_deg",
            "strong_signal_radius_m",
            "clearance_radius_m",
            "dog_speed_m_per_s",
            "channel_switch_time_s",
            "detection_time_s",
            "optical_time_s",
            "clearance_time_s",
            "global_grid_side_m",
            "recovery_grid_side_m",
            "coarse_recovery_grid_side_m",
            "fine_recovery_max_radius_m",
            "geometry_fallback_margin_m",
            "source_samples",
            "error_samples",
            "orientation_samples",
            "repeated_point_m",
            "candidate_local_radius_m",
            "q2_timeout_s",
            "real_reserve_s",
            "opportunistic_min_absolute_reduction_m",
            "scheduler_completion_value_m",
            "scheduler_recovery_value_m",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} 必须是数值")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")

        integers = (
            "channel_min",
            "channel_max",
            "source_count_max",
            "no_progress_trigger",
            "source_samples",
            "error_samples",
            "orientation_samples",
            "max_scored_candidates",
            "fine_recovery_max_nodes",
            "recovery_batch_size",
            "max_clear_failures_per_channel",
            "opportunistic_max_per_node",
            "opportunistic_max_per_channel",
            "opportunistic_lookahead_nodes",
            "scheduler_active_shortlist",
            "scheduler_wait_cap",
        )
        for name in integers:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须是正整数")

        if self.channel_min != 1 or self.channel_max != 20:
            raise ValueError("当前客户端协议要求频道范围固定为 1..20")
        if self.source_count_max != 16:
            raise ValueError("题目干扰源数量上界必须为16")
        # 数学上界：等边三角形内任意点到最近顶点距离 ≤ L/√3
        # 要求 ≤ 保证接收半径 1000m，故 L ≤ 1000√3 ≈ 1732.05m
        max_valid_side = self.reception_radius_min_m * math.sqrt(3.0)
        if self.global_grid_side_m > max_valid_side:
            raise ValueError(
                f"三角网格边长不能超过 {max_valid_side:.2f} 米，"
                "否则无法保证源点到最近网格顶点的距离≤1000米"
            )
        if self.recovery_grid_side_m >= self.strong_signal_radius_m:
            raise ValueError("恢复网格边长必须严格小于近距离强信号半径")
        if self.coarse_recovery_grid_side_m > self.reception_radius_min_m:
            raise ValueError("粗重捕获网格边长不能超过保证接收半径")
        if not 0.0 <= self.clear_margin_m < self.clearance_radius_m:
            raise ValueError("clear_margin_m 必须位于 [0, clearance_radius_m)")
        if not 0.0 < self.bearing_error_deg < 90.0:
            raise ValueError("bearing_error_deg 必须位于 (0,90)")
        if self.error_samples < 2:
            raise ValueError("error_samples 至少为2，以覆盖误差区间端点")
        if type(self.enable_q2_seed) is not bool:
            raise ValueError("enable_q2_seed 必须是布尔值")
        if type(self.enable_parallel_search_localization) is not bool:
            raise ValueError("enable_parallel_search_localization 必须是布尔值")
        if type(self.enable_value_time_scheduler) is not bool:
            raise ValueError("enable_value_time_scheduler 必须是布尔值")
        if not 0.0 < self.no_signal_side_width_deg <= 180.0:
            raise ValueError("no_signal_side_width_deg 必须位于 (0,180]")
        unit_interval_parameters = (
            "opportunistic_lookahead_ratio",
            "opportunistic_min_receive_score",
            "opportunistic_min_possible_fraction",
            "opportunistic_min_relative_reduction",
        )
        for name in unit_interval_parameters:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} 必须是数值")
            if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0,1]")
        if (
            not math.isfinite(self.min_request_interval_s)
            or self.min_request_interval_s < 0.0
        ):
            raise ValueError("min_request_interval_s 必须是有限非负数")
        if (
            isinstance(self.scheduler_wait_weight, bool)
            or not isinstance(self.scheduler_wait_weight, (int, float))
            or not math.isfinite(float(self.scheduler_wait_weight))
            or self.scheduler_wait_weight < 0.0
        ):
            raise ValueError("scheduler_wait_weight 必须是有限非负数")
        if not 0.0 <= self.progress_rel <= 1.0 or self.progress_abs_m < 0.0:
            raise ValueError("进展阈值非法")


@dataclass
class ChannelRecord:
    channel_id: int
    status: str = "UNKNOWN"
    bearing_observations: list[dict] = field(default_factory=list)
    no_signal_observations: list[dict] = field(default_factory=list)
    near_signal_observations: list[dict] = field(default_factory=list)
    global_nodes_tested: set[int] = field(default_factory=set)
    outer_polygon: list[Point] = field(default_factory=list)
    region_status: str | None = None
    diameter: float | None = None
    diameter_pair: tuple[Point, Point] | None = None
    clearance_center: Point | None = None
    clearance_radius: float | None = None
    certificate_source: str | None = None
    absent_reason: str | None = None
    no_progress_count: int = 0
    clear_failure_count: int = 0
    q2_attempted: bool = False
    q2_seed: Point | None = None
    recovery_active: bool = False
    recovery_mode: str | None = None
    recovery_generation: int = 0
    recovery_total_nodes: int = 0
    recovery_visited_nodes: int = 0
    recovery_reset_pending: bool = False
    geometry_certifiable: bool = True
    geometry_fallback_count: int = 0
    geometry_diagnostics: list[dict] = field(default_factory=list)
    localization_turn_count: int = 0
    scheduler_wait_count: int = 0
    opportunistic_measure_count: int = 0
    progress_history: list[dict] = field(default_factory=list)
    invalidated_rounds: list[dict] = field(default_factory=list)

    def all_observations(self) -> list[dict]:
        return (
            self.bearing_observations
            + self.no_signal_observations
            + self.near_signal_observations
        )


@dataclass
class GridPlan:
    side_m: float
    points: list[Point]
    route: list[Point]
    triangle_count: int
    route_distance_m: float
    triangles: list[Triangle] = field(default_factory=list)


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_segment_distance(point: Point, a: Point, b: Point) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    denominator = vx * vx + vy * vy
    if denominator == 0.0:
        return math.dist(point, a)
    ratio = ((point[0] - a[0]) * vx + (point[1] - a[1]) * vy) / denominator
    ratio = min(1.0, max(0.0, ratio))
    projection = (a[0] + ratio * vx, a[1] + ratio * vy)
    return math.dist(point, projection)


def point_in_triangle(point: Point, triangle: Triangle, tolerance: float = 1e-8) -> bool:
    a, b, c = triangle
    values = (_cross(a, b, point), _cross(b, c, point), _cross(c, a, point))
    return all(v >= -tolerance for v in values) or all(v <= tolerance for v in values)


def point_triangle_distance(point: Point, triangle: Triangle) -> float:
    if point_in_triangle(point, triangle):
        return 0.0
    a, b, c = triangle
    return min(
        _point_segment_distance(point, a, b),
        _point_segment_distance(point, b, c),
        _point_segment_distance(point, c, a),
    )


def point_in_convex_region(point: Point, vertices: Sequence[Point], tolerance: float = 1e-8) -> bool:
    if not vertices:
        return False
    if len(vertices) == 1:
        return math.dist(point, vertices[0]) <= tolerance
    if len(vertices) == 2:
        return _point_segment_distance(point, vertices[0], vertices[1]) <= tolerance
    values = [
        _cross(a, b, point)
        for a, b in zip(vertices, list(vertices[1:]) + [vertices[0]])
    ]
    return all(v >= -tolerance for v in values) or all(v <= tolerance for v in values)


def _orientation_sign(value: float, tolerance: float = 1e-8) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = _orientation_sign(_cross(a, b, c))
    o2 = _orientation_sign(_cross(a, b, d))
    o3 = _orientation_sign(_cross(c, d, a))
    o4 = _orientation_sign(_cross(c, d, b))
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return any(
        _point_segment_distance(point, first, second) <= 1e-8
        for point, first, second, orientation in (
            (c, a, b, o1),
            (d, a, b, o2),
            (a, c, d, o3),
            (b, c, d, o4),
        )
        if orientation == 0
    )


def triangle_intersects_region(triangle: Triangle, vertices: Sequence[Point]) -> bool:
    if any(point_in_convex_region(point, vertices) for point in triangle):
        return True
    if any(point_in_triangle(point, triangle) for point in vertices):
        return True

    tri_edges = list(zip(triangle, triangle[1:] + triangle[:1]))
    if len(vertices) == 1:
        return False
    region_edges = (
        [(vertices[0], vertices[1])]
        if len(vertices) == 2
        else list(zip(vertices, list(vertices[1:]) + [vertices[0]]))
    )
    return any(
        segments_intersect(a, b, c, d)
        for a, b in tri_edges
        for c, d in region_edges
    )


def lattice_point(i: int, j: int, side_m: float) -> Point:
    return (
        side_m * (i + 0.5 * j),
        side_m * math.sqrt(3.0) * 0.5 * j,
    )


def _triangle_indices(i: int, j: int):
    yield ((i, j), (i + 1, j), (i, j + 1))
    yield ((i + 1, j), (i + 1, j + 1), (i, j + 1))


def open_route_distance(start: Point, route: Sequence[Point]) -> float:
    total = 0.0
    current = tuple(start)
    for point in route:
        total += math.dist(current, point)
        current = point
    return total


def nearest_neighbor_route(points: Iterable[Point], start: Point = (0.0, 0.0)) -> list[Point]:
    remaining = sorted(set(tuple(map(float, p)) for p in points))
    route: list[Point] = []
    current = tuple(start)
    while remaining:
        selected = min(remaining, key=lambda p: (math.dist(current, p), p[1], p[0]))
        route.append(selected)
        remaining.remove(selected)
        current = selected
    return route


def two_opt_open(route: Sequence[Point], start: Point = (0.0, 0.0)) -> list[Point]:
    route = list(route)
    if len(route) < 4:
        return route
    best = open_route_distance(start, route)
    improved = True
    while improved:
        improved = False
        # 若第一个点就是起点，保持它不动，避免无意义旋转。
        left_start = 1 if math.dist(start, route[0]) <= 1e-9 else 0
        for left in range(left_start, len(route) - 1):
            for right in range(left + 1, len(route)):
                candidate = route[:left] + list(reversed(route[left:right + 1])) + route[right + 1:]
                distance = open_route_distance(start, candidate)
                if distance < best - 1e-8:
                    route, best = candidate, distance
                    improved = True
                    break
            if improved:
                break
    return route


def serpentine_lattice_route(
    indexed_points: dict[tuple[int, int], Point],
    start: Point,
) -> list[Point]:
    """局部细网格使用线性复杂度蛇形路线，避免大点集的 O(n^2) 最近邻。"""
    rows: dict[int, list[tuple[int, Point]]] = {}
    for (i, j), point in indexed_points.items():
        rows.setdefault(j, []).append((i, point))

    candidates: list[list[Point]] = []
    for reverse_rows in (False, True):
        ordered_rows = sorted(rows, reverse=reverse_rows)
        for first_reverse in (False, True):
            result: list[Point] = []
            for row_index, j in enumerate(ordered_rows):
                reverse = first_reverse ^ bool(row_index % 2)
                result.extend(point for _, point in sorted(rows[j], reverse=reverse))
            candidates.append(result)
    return min(candidates, key=lambda route: open_route_distance(start, route))


def build_global_grid(config: Q4Config) -> GridPlan:
    side = config.global_grid_side_m
    height = side * math.sqrt(3.0) / 2.0
    reach = config.target_radius_m + side
    j_min = math.floor(-reach / height) - 2
    j_max = math.ceil(reach / height) + 2
    vertices: dict[tuple[int, int], Point] = {}
    triangles: list[Triangle] = []

    for j in range(j_min, j_max + 1):
        i_min = math.floor(-reach / side - 0.5 * j) - 2
        i_max = math.ceil(reach / side - 0.5 * j) + 2
        for i in range(i_min, i_max + 1):
            for indices in _triangle_indices(i, j):
                triangle = tuple(lattice_point(a, b, side) for a, b in indices)
                if point_triangle_distance((0.0, 0.0), triangle) <= config.target_radius_m + 1e-8:
                    triangles.append(triangle)
                    for index, point in zip(indices, triangle):
                        vertices[index] = point

    points = list(vertices.values())
    route = two_opt_open(nearest_neighbor_route(points), (0.0, 0.0))
    return GridPlan(
        side_m=side,
        points=points,
        route=route,
        triangle_count=len(triangles),
        route_distance_m=open_route_distance((0.0, 0.0), route),
        triangles=triangles,
    )


def build_recovery_grid(vertices: Sequence[Point], side_m: float, start: Point) -> GridPlan:
    if not vertices:
        raise IncompleteRun("空定位范围不能建立恢复网格")
    min_x = min(p[0] for p in vertices) - side_m
    max_x = max(p[0] for p in vertices) + side_m
    min_y = min(p[1] for p in vertices) - side_m
    max_y = max(p[1] for p in vertices) + side_m
    height = side_m * math.sqrt(3.0) / 2.0
    j_min = math.floor(min_y / height) - 2
    j_max = math.ceil(max_y / height) + 2
    selected: dict[tuple[int, int], Point] = {}
    triangle_count = 0

    for j in range(j_min, j_max + 1):
        i_min = math.floor(min_x / side_m - 0.5 * j) - 2
        i_max = math.ceil(max_x / side_m - 0.5 * j) + 2
        for i in range(i_min, i_max + 1):
            for indices in _triangle_indices(i, j):
                triangle = tuple(lattice_point(a, b, side_m) for a, b in indices)
                if triangle_intersects_region(triangle, vertices):
                    triangle_count += 1
                    for index, point in zip(indices, triangle):
                        selected[index] = point

    route = serpentine_lattice_route(selected, start)
    return GridPlan(
        side_m=side_m,
        points=list(selected.values()),
        route=route,
        triangle_count=triangle_count,
        route_distance_m=open_route_distance(start, route),
    )


def deduplicate_vertices(vertices: Iterable[Point], tolerance: float = 1e-7) -> list[Point]:
    result: list[Point] = []
    for raw in vertices:
        point = tuple(map(float, raw))
        if not any(math.dist(point, old) <= tolerance for old in result):
            result.append(point)
    return result


def clip_vertices_by_halfplane(
    vertices: Sequence[Point],
    halfplane: Sequence[float],
    tolerance: float = 1e-7,
) -> list[Point]:
    points = deduplicate_vertices(vertices, tolerance)
    if not points:
        return []
    a, b, c = map(float, halfplane)

    def residual(point: Point) -> float:
        return a * point[0] + b * point[1] - c

    if len(points) == 1:
        return points if residual(points[0]) <= tolerance else []

    clipped: list[Point] = []
    for start, end in zip(points, points[1:] + points[:1]):
        start_value, end_value = residual(start), residual(end)
        start_inside = start_value <= tolerance
        end_inside = end_value <= tolerance
        denominator = start_value - end_value

        if start_inside and end_inside:
            clipped.append(end)
        elif start_inside and not end_inside and abs(denominator) > 1e-15:
            ratio = start_value / denominator
            clipped.append((
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            ))
        elif not start_inside and end_inside:
            if abs(denominator) > 1e-15:
                ratio = start_value / denominator
                clipped.append((
                    start[0] + ratio * (end[0] - start[0]),
                    start[1] + ratio * (end[1] - start[1]),
                ))
            clipped.append(end)
    return deduplicate_vertices(clipped, tolerance)


def minimum_enclosing_circle(vertices: Sequence[Point]) -> tuple[Point, float]:
    """枚举1/2/3点支持圆，并以全部顶点复核，绝不低估外包半径。"""
    points = deduplicate_vertices(vertices)
    if not points:
        raise IncompleteRun("空顶点集不能生成清除证书")

    candidates: list[tuple[Point, float]] = [(point, 0.0) for point in points]
    for a, b in combinations(points, 2):
        center = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        candidates.append((center, math.dist(a, b) / 2.0))
    for a, b, c in combinations(points, 3):
        ux, uy = b[0] - a[0], b[1] - a[1]
        vx, vy = c[0] - a[0], c[1] - a[1]
        determinant = ux * vy - uy * vx
        scale = max(1.0, math.hypot(ux, uy) * math.hypot(vx, vy))
        if abs(determinant) <= 1e-12 * scale:
            continue
        u2, v2 = ux * ux + uy * uy, vx * vx + vy * vy
        center = (
            a[0] + (u2 * vy - v2 * uy) / (2.0 * determinant),
            a[1] + (ux * v2 - vx * u2) / (2.0 * determinant),
        )
        if all(math.isfinite(value) for value in center):
            candidates.append((center, math.dist(center, a)))

    best: tuple[float, Point] | None = None
    for center, proposed_radius in candidates:
        actual_radius = max(math.dist(center, point) for point in points)
        if actual_radius <= proposed_radius + 1e-7 * max(1.0, proposed_radius):
            item = (actual_radius, center)
            if best is None or item < best:
                best = item
    if best is None:
        raise IncompleteRun("最小包围圆数值求解失败")
    return best[1], max(math.dist(best[1], point) for point in points)


def record_halfplanes(record: ChannelRecord, bearing_error_deg: float) -> list[tuple[float, float, float]]:
    rows = list(BOX_HALFPLANES)
    if record.bearing_observations:
        observations = [
            (*item["position"], item["bearing_deg"])
            for item in record.bearing_observations
        ]
        rows.extend(
            tuple(map(float, row))
            for row in build_halfplanes(observations, error_deg=bearing_error_deg).tolist()
        )
    return rows


def polygon_diameter(vertices: Sequence[Point]) -> tuple[float, tuple[Point, Point]]:
    points = deduplicate_vertices(vertices)
    if not points:
        raise IncompleteRun("空多边形没有直径")
    if len(points) == 1:
        return 0.0, (points[0], points[0])
    distance, first, second = max(
        (math.dist(a, b), a, b)
        for a, b in combinations(points, 2)
    )
    return distance, (first, second)


def conservative_fallback_polygon(record: ChannelRecord, config: Q4Config) -> list[Point]:
    """以向外扩张的浮点半平面裁剪，供规划使用但不签发清除证书。"""
    radius = config.target_radius_m
    vertices: list[Point] = [
        (-radius, -radius),
        (radius, -radius),
        (radius, radius),
        (-radius, radius),
    ]
    rows = record_halfplanes(record, config.bearing_error_deg)[len(BOX_HALFPLANES):]
    for a, b, c in rows:
        norm = math.hypot(a, b)
        if norm <= 0.0 or not math.isfinite(norm):
            continue
        expanded = (a, b, c + config.geometry_fallback_margin_m * norm)
        vertices = clip_vertices_by_halfplane(vertices, expanded, tolerance=1e-6)
        if not vertices:
            break
    return vertices


def update_region(
    record: ChannelRecord,
    config: Q4Config,
    geometry_config: GeometryConfig,
) -> dict:
    previous_vertices = list(record.outer_polygon)
    previous_diameter = record.diameter
    rows = record_halfplanes(record, config.bearing_error_deg)
    result = solve_halfplanes(rows, config=geometry_config)
    strict_success = result.status in GOOD_REGIONS and bool(result.vertices)
    diagnostics = {
        "strict_status": result.status,
        "strict_diagnostics": result.diagnostics,
        "fallback_used": not strict_success,
    }

    if strict_success:
        vertices = [tuple(map(float, point)) for point in result.vertices]
        diameter = float(result.diameter)
        diameter_pair = (
            polygon_diameter(vertices)[1]
            if result.diameter_pair is None
            else (tuple(result.diameter_pair[0]), tuple(result.diameter_pair[1]))
        )
        record.region_status = result.status
        record.geometry_certifiable = True
    else:
        record.geometry_fallback_count += 1
        record.geometry_certifiable = False
        vertices = conservative_fallback_polygon(record, config)
        if not vertices:
            # 最保守的最终回退：忽略造成数值退化的新收缩，保留上一有效外包。
            if previous_vertices:
                vertices = previous_vertices
                diagnostics["fallback_mode"] = "keep_previous_outer_polygon"
            else:
                radius = config.target_radius_m
                vertices = [
                    (-radius, -radius),
                    (radius, -radius),
                    (radius, radius),
                    (-radius, radius),
                ]
                diagnostics["fallback_mode"] = "full_prior_box"
        else:
            diagnostics["fallback_mode"] = "expanded_float_halfplane_clip"
        diameter, diameter_pair = polygon_diameter(vertices)
        record.region_status = "FALLBACK_POLYGON"

    # 新约束不应放大区域；若严格结果出现异常，退回上一外包。
    if (
        previous_diameter is not None
        and diameter > previous_diameter + 1e-4
        and previous_vertices
    ):
        vertices = previous_vertices
        diameter, diameter_pair = polygon_diameter(vertices)
        record.geometry_certifiable = False
        record.region_status = "FALLBACK_PREVIOUS"
        diagnostics["monotonicity_fallback"] = True

    record.outer_polygon = vertices
    record.diameter = diameter
    record.diameter_pair = diameter_pair
    try:
        center, circle_radius = minimum_enclosing_circle(vertices)
    except IncompleteRun:
        min_x, max_x = min(p[0] for p in vertices), max(p[0] for p in vertices)
        min_y, max_y = min(p[1] for p in vertices), max(p[1] for p in vertices)
        center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        circle_radius = max(math.dist(center, point) for point in vertices)
        record.geometry_certifiable = False
        diagnostics["circle_fallback"] = "bounding_box_center"
    record.clearance_center = center
    record.clearance_radius = circle_radius
    record.geometry_diagnostics.append(diagnostics)

    threshold = config.clearance_radius_m - config.clear_margin_m
    if record.geometry_certifiable and circle_radius <= threshold:
        record.status = "READY"
        record.certificate_source = "strict_all_bearings_box_mec"
    else:
        record.status = "FOUND"
        record.certificate_source = None
    return diagnostics


def radical_inverse(index: int, base: int) -> float:
    value, fraction = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * fraction
        fraction /= base
    return value


def sample_source_positions(record: ChannelRecord, config: Q4Config) -> list[Point]:
    """仅为候选点评分生成离散位置样本；no_signal 不参与位置裁剪。"""
    if not record.outer_polygon or not record.bearing_observations:
        raise IncompleteRun(f"频道{record.channel_id}尚无可采样的定位范围")
    rows = record_halfplanes(record, config.bearing_error_deg)
    target = config.source_samples
    result: list[Point] = []

    def add(raw: Point) -> None:
        point = tuple(map(float, raw))
        if math.hypot(*point) > config.target_radius_m + 1e-7:
            return
        if any(a * point[0] + b * point[1] - c > 1e-7 for a, b, c in rows):
            return
        # direction 表明该次距离大于近距离阈值且不超过实际接收半径上界。
        if any(
            not config.strong_signal_radius_m < math.dist(point, item["position"])
            <= config.reception_radius_max_m + 1e-7
            for item in record.bearing_observations
        ):
            return
        if not any(math.dist(point, old) <= 1e-6 for old in result):
            result.append(point)

    vertices = record.outer_polygon
    center = (
        sum(point[0] for point in vertices) / len(vertices),
        sum(point[1] for point in vertices) / len(vertices),
    )
    add(center)
    for point in vertices:
        add(point)

    # 凸多边形三角扇低差异采样。
    if len(vertices) >= 2:
        for index in range(1, target * 64 + 1):
            a = vertices[(index - 1) % len(vertices)]
            b = vertices[index % len(vertices)]
            u = math.sqrt(radical_inverse(index, 2))
            v = radical_inverse(index, 3)
            add((
                (1.0 - u) * center[0] + u * ((1.0 - v) * a[0] + v * b[0]),
                (1.0 - u) * center[1] + u * ((1.0 - v) * a[1] + v * b[1]),
            ))
            if len(result) >= target:
                break

    # 从成功示向位置沿角域补样，避免细长角域在面积采样中漏掉。
    for observation_index, observation in enumerate(record.bearing_observations):
        for index in range(1, target * 48 + 1):
            if len(result) >= target:
                break
            offset_index = index + observation_index * target * 48
            radius = (
                config.strong_signal_radius_m
                + 1e-4
                + (config.reception_radius_max_m - config.strong_signal_radius_m - 1e-4)
                * radical_inverse(offset_index, 2)
            )
            angle = math.radians(
                observation["bearing_deg"]
                + config.bearing_error_deg * (2.0 * radical_inverse(offset_index, 3) - 1.0)
            )
            add((
                observation["position"][0] + radius * math.cos(angle),
                observation["position"][1] + radius * math.sin(angle),
            ))

    if not result:
        raise IncompleteRun(f"频道{record.channel_id}未生成有效源位置样本")
    return result[:target]


def lightweight_future_geometry(
    current_vertices: Sequence[Point],
    point: Point,
    reported_bearing_deg: float,
    bearing_error_deg: float,
) -> tuple[float, float] | None:
    vertices = [tuple(map(float, item)) for item in current_vertices]
    new_rows = build_halfplanes(
        [(*point, reported_bearing_deg)],
        error_deg=bearing_error_deg,
    ).tolist()
    for row in new_rows:
        vertices = clip_vertices_by_halfplane(vertices, row)
        if not vertices:
            return None
    diameter = (
        0.0
        if len(vertices) == 1
        else max(math.dist(a, b) for a, b in combinations(vertices, 2))
    )
    _, radius = minimum_enclosing_circle(vertices)
    return diameter, radius


def angle_difference_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def no_signal_side_penalty(record: ChannelRecord, estimate: Point, candidate: Point, config: Q4Config) -> int:
    candidate_angle = math.degrees(
        math.atan2(candidate[1] - estimate[1], candidate[0] - estimate[0])
    ) % 360.0
    penalty = 0
    for observation in record.no_signal_observations:
        if observation.get("measurement_kind") not in {"active", "recovery"}:
            continue
        point = observation["position"]
        angle = math.degrees(math.atan2(point[1] - estimate[1], point[0] - estimate[0])) % 360.0
        penalty += angle_difference_deg(candidate_angle, angle) <= config.no_signal_side_width_deg
    return penalty


def candidate_points(
    record: ChannelRecord,
    sources: Sequence[Point],
    current_position: Point,
    config: Q4Config,
    q2_seed: Point | None,
) -> tuple[Point, list[dict]]:
    estimate = (
        sum(point[0] for point in sources) / len(sources),
        sum(point[1] for point in sources) / len(sources),
    )
    candidates: list[dict] = []

    def add(point: Point, origin: str) -> None:
        point = tuple(map(float, point))
        if not all(math.isfinite(value) and abs(value) <= 2_000_000 for value in point):
            return
        if math.dist(point, current_position) < config.repeated_point_m:
            return
        if any(
            math.dist(point, observation["position"]) < config.repeated_point_m
            for observation in record.all_observations()
        ):
            return
        if any(math.dist(point, item["point"]) <= 1e-6 for item in candidates):
            return
        candidates.append({"point": point, "origin": origin})

    if q2_seed is not None:
        add(q2_seed, "q2_seed")

    # 修复一：源位置样本本身必须进入候选集。它们覆盖成功示向角域内
    # 从近端到远端的可能位置，不再只在错误侧的小半径扇区中移动。
    for source in sources:
        add(source, "source_sample")

    # 对每个成功示向点，将样本距离的分位点投影回示向中心线，并加入
    # 观测点到样本之间的内插点。内插点比越过样本继续前进更不易进入背面。
    for observation in record.bearing_observations:
        distances = sorted(math.dist(observation["position"], source) for source in sources)
        if distances:
            for fraction in (0.15, 0.35, 0.55, 0.75, 0.95):
                index = min(len(distances) - 1, round(fraction * (len(distances) - 1)))
                distance = distances[index]
                angle = math.radians(observation["bearing_deg"])
                add((
                    observation["position"][0] + distance * math.cos(angle),
                    observation["position"][1] + distance * math.sin(angle),
                ), "bearing_distance_quantile")
        for source in sources:
            for fraction in (0.35, 0.60, 0.85):
                add((
                    observation["position"][0]
                    + fraction * (source[0] - observation["position"][0]),
                    observation["position"][1]
                    + fraction * (source[1] - observation["position"][1]),
                ), "observation_source_interpolation")

    if record.diameter_pair is not None and record.diameter is not None:
        a, b = record.diameter_pair
        midpoint = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        length = max(record.diameter, 1e-9)
        tangent = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
        normal = (-tangent[1], tangent[0])
        for scale in (record.diameter / 4.0, record.diameter / 2.0, record.diameter):
            for angle_deg in range(0, 360, 45):
                angle = math.radians(angle_deg)
                add((
                    midpoint[0] + scale * (math.cos(angle) * tangent[0] + math.sin(angle) * normal[0]),
                    midpoint[1] + scale * (math.cos(angle) * tangent[1] + math.sin(angle) * normal[1]),
                ), "diameter_frame")

    # 当前估计附近的小尺度候选，防止长条区域的 D/4 仍然过远。
    for radius in (config.candidate_local_radius_m, 2.0 * config.candidate_local_radius_m):
        for angle_deg in range(0, 360, 45):
            angle = math.radians(angle_deg)
            add((estimate[0] + radius * math.cos(angle), estimate[1] + radius * math.sin(angle)),
                "local_radial")

    # 保留成对侧向候选，但不再使用报告中导致失败的 ±75° 小半径扇区。
    for observation in record.bearing_observations:
        for step in (150.0, 300.0, 500.0):
            for offset_deg in (-45.0, -20.0, 0.0, 20.0, 45.0):
                angle = math.radians(observation["bearing_deg"] + offset_deg)
                add((
                    observation["position"][0] + step * math.cos(angle),
                    observation["position"][1] + step * math.sin(angle),
                ), "successful_anchor")

    if not candidates:
        raise IncompleteRun(f"频道{record.channel_id}没有未检测候选点")

    # 完整几何评分较贵。优先保留源样本、距离分位点和内插点，再按靠近
    # 任一可能源及移动成本截断；不再让大量外圈几何候选淹没关键候选。
    if len(candidates) > config.max_scored_candidates:
        origin_priority = {
            "q2_seed": 0,
            "source_sample": 1,
            "bearing_distance_quantile": 2,
            "diameter_frame": 3,
            "observation_source_interpolation": 4,
            "successful_anchor": 5,
            "local_radial": 6,
        }
        candidates = sorted(
            candidates,
            key=lambda item: (
                origin_priority.get(item["origin"], 9),
                min(math.dist(item["point"], source) for source in sources),
                math.dist(current_position, item["point"]),
                item["point"],
            ),
        )[:config.max_scored_candidates]
    return estimate, candidates


def sampled_directional_receive_fraction(
    source: Point,
    candidate: Point,
    successful_observations: Sequence[dict],
    orientation_samples: int,
) -> float:
    """在与全部历史成功接收相容的定向朝向中，估计候选点前向占比。"""
    feasible = 0
    receives = 0
    for index in range(orientation_samples):
        angle = 2.0 * math.pi * index / orientation_samples
        direction = (math.cos(angle), math.sin(angle))
        if not all(
            direction[0] * (observation["position"][0] - source[0])
            + direction[1] * (observation["position"][1] - source[1])
            >= -1e-9
            for observation in successful_observations
        ):
            continue
        feasible += 1
        if (
            direction[0] * (candidate[0] - source[0])
            + direction[1] * (candidate[1] - source[1])
            >= -1e-9
        ):
            receives += 1
    return receives / feasible if feasible else 0.0


def sampled_radius_receive_fraction(
    source: Point,
    candidate: Point,
    successful_observations: Sequence[dict],
    config: Q4Config,
) -> float:
    """按与历史成功接收相容的 R 区间，给出距离接收的保守线性权重。"""
    lower = max(
        config.reception_radius_min_m,
        *(math.dist(source, observation["position"]) for observation in successful_observations),
    )
    upper = config.reception_radius_max_m
    distance = math.dist(source, candidate)
    if distance <= lower + 1e-7:
        return 1.0
    if distance > upper + 1e-7:
        return 0.0
    if upper <= lower + 1e-9:
        return 0.0
    return max(0.0, min(1.0, (upper - distance) / (upper - lower)))


def score_candidate(
    record: ChannelRecord,
    candidate: dict,
    sources: Sequence[Point],
    estimate: Point,
    current_position: Point,
    current_channel: int,
    config: Q4Config,
) -> dict | None:
    point = candidate["point"]
    current_radius = float(record.clearance_radius)
    current_diameter = float(record.diameter)
    receivable = [
        source
        for source in sources
        if math.dist(point, source) <= config.reception_radius_max_m + 1e-7
    ]
    if not receivable:
        return None

    worst_radius = 0.0
    worst_diameter = 0.0
    errors = [
        -config.bearing_error_deg
        + 2.0 * config.bearing_error_deg * index / (config.error_samples - 1)
        for index in range(config.error_samples)
    ]

    for source in receivable:
        distance = math.dist(point, source)
        if distance <= config.strong_signal_radius_m + 1e-9:
            worst_radius = max(worst_radius, min(current_radius, config.strong_signal_radius_m))
            worst_diameter = max(worst_diameter, min(current_diameter, 2.0 * config.strong_signal_radius_m))
            continue

        true_bearing = math.degrees(
            math.atan2(source[1] - point[1], source[0] - point[0])
        ) % 360.0
        for error in errors:
            try:
                future = lightweight_future_geometry(
                    record.outer_polygon,
                    point,
                    true_bearing + error,
                    config.bearing_error_deg,
                )
            except IncompleteRun:
                future = None
            if future is None:
                future_diameter, future_radius = current_diameter, current_radius
            else:
                future_diameter, future_radius = future
            worst_diameter = max(worst_diameter, min(current_diameter, future_diameter))
            worst_radius = max(worst_radius, min(current_radius, future_radius))

    time_cost = (
        math.dist(current_position, point) / config.dog_speed_m_per_s
        + config.detection_time_s
        + (config.channel_switch_time_s if current_channel != record.channel_id else 0.0)
    )
    radius_reduction = max(0.0, current_radius - worst_radius)
    diameter_reduction = max(0.0, current_diameter - worst_diameter)
    possible_fraction = len(receivable) / len(sources)
    guaranteed_distance_fraction = sum(
        math.dist(point, source) <= config.reception_radius_min_m + 1e-7
        for source in sources
    ) / len(sources)
    per_source_receive = []
    for source in sources:
        directional_fraction = sampled_directional_receive_fraction(
            source,
            point,
            record.bearing_observations,
            config.orientation_samples,
        )
        radius_fraction = sampled_radius_receive_fraction(
            source,
            point,
            record.bearing_observations,
            config,
        )
        per_source_receive.append(directional_fraction * radius_fraction)
    average_directional_receive = sum(per_source_receive) / len(per_source_receive)
    worst_directional_receive = min(per_source_receive)
    robust_receive_score = 0.75 * average_directional_receive + 0.25 * worst_directional_receive
    side_penalty = no_signal_side_penalty(record, estimate, point, config)
    conditional_efficiency = radius_reduction / max(time_cost, 1e-9)
    weighted_score = (
        conditional_efficiency
        * (0.10 + 0.90 * robust_receive_score)
        / (1.0 + 0.25 * side_penalty)
    )
    return {
        **candidate,
        "predicted_worst_radius": worst_radius,
        "predicted_worst_diameter": worst_diameter,
        "predicted_clearable": worst_radius <= config.clearance_radius_m - config.clear_margin_m,
        "radius_reduction": radius_reduction,
        "diameter_reduction": diameter_reduction,
        "possible_receive_fraction": possible_fraction,
        "guaranteed_distance_fraction": guaranteed_distance_fraction,
        "average_directional_receive_fraction": average_directional_receive,
        "worst_directional_receive_fraction": worst_directional_receive,
        "robust_receive_score": robust_receive_score,
        "side_penalty": side_penalty,
        "time_cost_s": time_cost,
        "weighted_score": weighted_score,
    }


def _q2_worker(connection, observation, planner_config, geometry_config) -> None:
    try:
        result = solve_q2(
            observation,
            config=planner_config,
            geometry_config=geometry_config,
        )
        connection.send((True, result.to_dict()))
    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def bounded_q2_seed(
    observation: tuple[float, float, float],
    planner_config: PlannerConfig,
    geometry_config: GeometryConfig,
    timeout_s: float,
) -> tuple[Point | None, dict]:
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_q2_worker,
        args=(child, observation, planner_config, geometry_config),
    )
    process.start()
    child.close()
    diagnostics: dict = {}
    try:
        if not parent.poll(timeout_s):
            diagnostics["error"] = "q2_timeout"
            return None, diagnostics
        try:
            ok, payload = parent.recv()
        except EOFError:
            return None, {"error": "q2_worker_eof"}
        if not ok:
            return None, {"error": payload}
        diagnostics = payload
        point = payload.get("second_point")
        if point is None:
            return None, diagnostics
        point = tuple(map(float, point))
        if not all(math.isfinite(value) and abs(value) <= 2_000_000 for value in point):
            return None, {**diagnostics, "error": "q2_invalid_point"}
        return point, diagnostics
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class _CoreRunner:
    """问题四主状态机；client 可注入离线测试替身。"""

    def __init__(
        self,
        client,
        *,
        config: Q4Config | None = None,
        planner_config: PlannerConfig | None = None,
        geometry_config: GeometryConfig | None = None,
        output_dir: Path | None = None,
    ):
        self.client = client
        self.cfg = config or Q4Config()
        self.pcfg = planner_config or PlannerConfig()
        self.gcfg = geometry_config or GeometryConfig()

        fixed_planner = {
            "target_radius_m": self.cfg.target_radius_m,
            "reception_radius_min_m": self.cfg.reception_radius_min_m,
            "reception_radius_max_m": self.cfg.reception_radius_max_m,
            "no_bearing_radius_m": self.cfg.strong_signal_radius_m,
            "bearing_error_deg": self.cfg.bearing_error_deg,
        }
        if any(getattr(self.pcfg, name) != value for name, value in fixed_planner.items()):
            raise ValueError("PlannerConfig 的题目物理常数与问题四配置不一致")
        if self.gcfg.bearing_error_deg != self.cfg.bearing_error_deg:
            raise ValueError("GeometryConfig 的示向误差与问题四配置不一致")

        self.records = {
            channel: ChannelRecord(channel)
            for channel in range(self.cfg.channel_min, self.cfg.channel_max + 1)
        }
        self.global_grid = build_global_grid(self.cfg)
        self.search_points = self.global_grid.route
        self.search_complete = False
        self.next_search_index = 0
        self.phase = "initialization"
        self.entered_at: float | None = None
        self.ended_at: float | None = None
        self.total_distance_m = 0.0
        self.operations: list[dict] = []
        self.events: list[dict] = []
        self.trajectory: list[dict] = []
        self._recovery_queues: dict[int, deque[Point]] = {}
        self._active_plan_cache: dict[int, dict] = {}
        self._last_request_started = 0.0
        self.counts = {
            "measure": 0,
            "switch": 0,
            "unknown_measure": 0,
            "active_measure": 0,
            "recovery_measure": 0,
            "opportunistic_measure": 0,
            "opportunistic_direction": 0,
            "opportunistic_no_signal": 0,
            "opportunistic_near": 0,
            "opportunistic_evaluation": 0,
            "opportunistic_deferred_by_lookahead": 0,
            "opportunistic_ready_queued": 0,
            "clear_success": 0,
            "clear_failure": 0,
            "q2_attempt": 0,
            "q2_success": 0,
            "active_plan": 0,
            "scheduler_round": 0,
            "scheduler_plan_evaluation": 0,
            "scheduler_plan_cache_hit": 0,
            "scheduler_selected_plan_refinement": 0,
            "scheduler_active_selected": 0,
            "scheduler_clear_selected": 0,
            "scheduler_recovery_selected": 0,
            "recovery_grid_build": 0,
            "coarse_recovery_build": 0,
            "fine_recovery_build": 0,
            "geometry_fallback": 0,
            "request_throttle_sleep": 0,
        }

        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._streams: dict[str, object] = {}
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            try:
                for name in ("operations", "events"):
                    self._streams[name] = (self.output_dir / f"{name}.jsonl").open(
                        "w", encoding="utf-8"
                    )
            except Exception:
                self.close_streams()
                raise

    @property
    def discovered_count(self) -> int:
        return sum(
            bool(record.bearing_observations or record.near_signal_observations)
            for record in self.records.values()
        )

    @property
    def cleared_count(self) -> int:
        return sum(record.status == "CLEARED" for record in self.records.values())

    def close_streams(self) -> None:
        for stream in self._streams.values():
            stream.close()

    def _append(self, stream_name: str, entry: dict) -> None:
        entry = json_safe(entry)
        target = self.operations if stream_name == "operations" else self.events
        target.append(entry)
        if stream_name in self._streams:
            self._streams[stream_name].write(
                json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"
            )
            self._streams[stream_name].flush()

    def event(self, kind: str, **data) -> None:
        self._append("events", {
            "event": kind,
            "phase": self.phase,
            "virtual_time_s": self.client.virtual_time_s,
            **data,
        })

    def check_budget(self, move_to: Point | None = None, method: str = "measure", channel: int | None = None) -> None:
        remaining_real = self.client.remaining_real_time_s
        if remaining_real is not None and remaining_real <= self.cfg.real_reserve_s:
            raise BudgetStop("现实时间不足，保留安全退出余量")
        if self.client.virtual_time_s is None or self.client.max_virtual_duration_s is None:
            return

        cost = 0.0
        if move_to is not None:
            cost += math.dist(self.client.position, move_to) / self.cfg.dog_speed_m_per_s
            if method == "measure":
                cost += self.cfg.detection_time_s
                if channel != self.client.current_channel:
                    cost += self.cfg.channel_switch_time_s
            elif method == "clear":
                cost += self.cfg.optical_time_s + self.cfg.clearance_time_s
        if self.client.virtual_time_s + cost >= self.client.max_virtual_duration_s:
            raise BudgetStop("虚拟时间不足，不能保证下一动作完成")

    def action(self, method: str, position: Point | None = None, channel: int | None = None):
        self.check_budget(position, method, channel)
        before_position = self.client.position
        before_channel = self.client.current_channel
        before_time = self.client.virtual_time_s or 0.0
        # 修复高频请求导致模拟器主动断开。节流不改变虚拟时间，只消耗少量现实时间。
        elapsed = time.monotonic() - self._last_request_started
        delay = self.cfg.min_request_interval_s - elapsed
        if delay > 0.0:
            time.sleep(delay)
            self.counts["request_throttle_sleep"] += 1
        started = time.monotonic()
        self._last_request_started = started
        arguments = () if position is None else (*position, channel)
        try:
            response = getattr(self.client, method)(*arguments)
        except Exception as error:
            self._append("operations", {
                "method": method,
                "position": position,
                "channel": channel,
                "phase": self.phase,
                "execution_confirmed": False,
                "error": repr(error),
                "pending_request_id": self.client.pending_request_id,
                "response": getattr(error, "response", None),
            })
            raise

        distance = math.dist(before_position, position) if position is not None else 0.0
        switched = int(method == "measure" and channel != before_channel)
        delta = response["virtual_time_s"] - before_time
        if method in {"measure", "clear"}:
            self.total_distance_m += distance
            self.trajectory.append({
                "from": before_position,
                "to": position,
                "method": method,
                "channel": channel,
                "phase": self.phase,
                "virtual_time_s": response["virtual_time_s"],
            })
        if method == "measure":
            self.counts["measure"] += 1
            self.counts["switch"] += switched
        elif method == "clear":
            success = response["clear_result"] == "success"
            self.counts["clear_success"] += int(success)
            self.counts["clear_failure"] += int(not success)

        self._append("operations", {
            "method": method,
            "position": position,
            "channel": channel,
            "phase": self.phase,
            "execution_confirmed": True,
            "response": response,
            "distance_m": distance,
            "switch_count": switched,
            "virtual_delta_s": delta,
            "request_elapsed_s": time.monotonic() - started,
        })
        return response

    def _reset_invalidated_round(self, record: ChannelRecord, triggering_result: str) -> None:
        self._active_plan_cache.pop(record.channel_id, None)
        record.bearing_observations = []
        record.near_signal_observations = []
        record.outer_polygon = []
        record.region_status = None
        record.diameter = None
        record.diameter_pair = None
        record.clearance_center = None
        record.clearance_radius = None
        record.certificate_source = None
        record.q2_attempted = False
        record.q2_seed = None
        record.recovery_reset_pending = False
        record.recovery_active = False
        record.recovery_mode = None
        record.geometry_certifiable = True
        self._recovery_queues.pop(record.channel_id, None)
        self.event(
            "localization_round_reset",
            channel=record.channel_id,
            triggering_result=triggering_result,
        )

    def measure(
        self,
        record: ChannelRecord,
        point: Point,
        *,
        measurement_kind: str,
        global_index: int | None = None,
    ) -> str:
        if measurement_kind not in {"unknown", "active", "recovery", "opportunistic"}:
            raise ValueError("measurement_kind 非法")
        # 本频道几何状态即将改变；此前基于旧区域生成的主动测向方案立即失效。
        self._active_plan_cache.pop(record.channel_id, None)
        old_diameter = record.diameter
        old_radius = record.clearance_radius
        response = self.action("measure", point, record.channel_id)
        self.counts[f"{measurement_kind}_measure"] += 1
        if global_index is not None:
            record.global_nodes_tested.add(global_index)

        result = response["measure_result"]
        if measurement_kind == "opportunistic":
            self.counts[f"opportunistic_{result}"] += 1
        datum = {
            "position": tuple(map(float, point)),
            "measurement_kind": measurement_kind,
            "virtual_time_s": response["virtual_time_s"],
            "real_timestamp_ms": response["real_timestamp_ms"],
        }
        if record.recovery_reset_pending and result in {"direction", "near"}:
            self._reset_invalidated_round(record, result)
            old_diameter = None
            old_radius = None

        geometry_update = None
        if result == "direction":
            record.bearing_observations.append({
                **datum,
                "bearing_deg": float(response["svd_deg"]),
            })
            geometry_update = update_region(record, self.cfg, self.gcfg)
            if geometry_update["fallback_used"]:
                self.counts["geometry_fallback"] += 1
            if measurement_kind == "recovery":
                record.recovery_active = False
                record.recovery_mode = None
                self._recovery_queues.pop(record.channel_id, None)
        elif result == "near":
            record.near_signal_observations.append(datum)
            record.status = "READY"
            record.clearance_center = tuple(map(float, point))
            record.clearance_radius = self.cfg.strong_signal_radius_m
            record.certificate_source = "near_feedback"
            record.recovery_active = False
            record.recovery_mode = None
            self._recovery_queues.pop(record.channel_id, None)
        elif result == "no_signal":
            # 第四问的安全不变量：不据此裁剪连续位置或删除目标。
            record.no_signal_observations.append(datum)
        else:
            raise IncompleteRun(f"模拟器返回未知 measure_result={result!r}")

        threshold = (
            max(self.cfg.progress_abs_m, self.cfg.progress_rel * old_radius)
            if old_radius is not None
            else self.cfg.progress_abs_m
        )
        improved = (
            record.status == "READY"
            or (
                result == "direction"
                and (
                    old_radius is None
                    or old_radius - record.clearance_radius > threshold
                )
            )
        )
        record.no_progress_count = 0 if improved else record.no_progress_count + 1
        record.progress_history.append({
            "result": result,
            "measurement_kind": measurement_kind,
            "diameter_before": old_diameter,
            "diameter_after": record.diameter,
            "radius_before": old_radius,
            "radius_after": record.clearance_radius,
            "improved": improved,
            "status": record.status,
        })
        self.event(
            "measurement_update",
            channel=record.channel_id,
            result=result,
            measurement_kind=measurement_kind,
            status=record.status,
            diameter_before=old_diameter,
            diameter_after=record.diameter,
            radius_before=old_radius,
            radius_after=record.clearance_radius,
            no_signal_position_pruned=False,
            geometry_update=geometry_update,
        )
        return result

    def clear(self, record: ChannelRecord) -> bool:
        self._active_plan_cache.pop(record.channel_id, None)
        if record.status != "READY" or record.certificate_source is None:
            raise IncompleteRun(f"频道{record.channel_id}没有可靠清除证书")
        if record.certificate_source == "strict_all_bearings_box_mec":
            if not record.outer_polygon or record.clearance_center is None:
                raise IncompleteRun("MEC 清除证书缺少连续外包范围")
            verified_radius = max(
                math.dist(record.clearance_center, point)
                for point in record.outer_polygon
            )
            if verified_radius > self.cfg.clearance_radius_m - self.cfg.clear_margin_m:
                raise IncompleteRun("执行前MEC逐顶点复核失败")

        clear_point = tuple(record.clearance_center)
        response = self.action("clear", clear_point, record.channel_id)
        if response["clear_result"] == "success":
            record.status = "CLEARED"
            self._recovery_queues.pop(record.channel_id, None)
            self.event(
                "cleared",
                channel=record.channel_id,
                certificate_source=record.certificate_source,
            )
            return True

        record.clear_failure_count += 1
        record.invalidated_rounds.append({
            "failed_point": clear_point,
            "certificate_source": record.certificate_source,
            "clearance_radius": record.clearance_radius,
            "bearing_observations": list(record.bearing_observations),
            "outer_polygon": list(record.outer_polygon),
        })
        self.event(
            "clear_failed",
            channel=record.channel_id,
            failure_count=record.clear_failure_count,
            failed_point=clear_point,
        )
        if record.clear_failure_count >= self.cfg.max_clear_failures_per_channel:
            raise IncompleteRun(
                f"频道{record.channel_id}清除失败达到"
                f"{self.cfg.max_clear_failures_per_channel}次"
            )

        record.status = "FOUND"
        record.certificate_source = None
        record.clearance_center = None
        record.clearance_radius = None
        record.no_progress_count = self.cfg.no_progress_trigger
        record.recovery_reset_pending = True
        record.recovery_active = False
        record.recovery_mode = None
        self._recovery_queues.pop(record.channel_id, None)
        return False

    def mark_absent(self) -> None:
        for record in self.records.values():
            if record.status != "UNKNOWN":
                continue
            if self.discovered_count >= self.cfg.source_count_max:
                reason = "source_count_upper_bound_reached"
            elif self.search_complete and len(record.global_nodes_tested) == len(self.search_points):
                reason = "all_guaranteed_triangular_grid_nodes_no_signal"
            else:
                continue
            record.status = "ABSENT"
            record.absent_reason = reason
            self.event("absent", channel=record.channel_id, reason=reason)

    def _score_route_waypoint(
        self,
        record: ChannelRecord,
        point: Point,
        sources: Sequence[Point],
        *,
        resume_unknown_search: bool,
    ) -> dict | None:
        """评价一个本来就会经过的全局网格点，不把网格间移动重复计入成本。"""
        if record.status != "FOUND":
            return None
        if record.diameter is None or record.clearance_radius is None:
            return None
        if self._point_was_tested(record, point):
            return None

        estimate = (
            sum(source[0] for source in sources) / len(sources),
            sum(source[1] for source in sources) / len(sources),
        )
        scored = score_candidate(
            record,
            {"point": tuple(point), "origin": "global_route_waypoint"},
            sources,
            estimate,
            tuple(point),
            self.client.current_channel,
            self.cfg,
        )
        self.counts["opportunistic_evaluation"] += 1
        if scored is None:
            return None

        # 补测后若还要继续搜索未知频道，通常还需切回一个未知频道，
        # 因而把这 1 秒也计入增量成本，避免高估顺带补测的收益。
        resume_switch_s = (
            self.cfg.channel_switch_time_s if resume_unknown_search else 0.0
        )
        effective_time = scored["time_cost_s"] + resume_switch_s
        receive_factor = 0.10 + 0.90 * scored["robust_receive_score"]
        opportunity_score = (
            scored["radius_reduction"]
            * receive_factor
            / max(effective_time, 1e-9)
            / (1.0 + 0.25 * scored["side_penalty"])
        )
        minimum_reduction = max(
            self.cfg.opportunistic_min_absolute_reduction_m,
            self.cfg.opportunistic_min_relative_reduction
            * float(record.clearance_radius),
        )
        eligible = (
            scored["robust_receive_score"]
            >= self.cfg.opportunistic_min_receive_score
            and scored["possible_receive_fraction"]
            >= self.cfg.opportunistic_min_possible_fraction
            and (
                scored["predicted_clearable"]
                or scored["radius_reduction"] >= minimum_reduction
            )
        )
        return {
            **scored,
            "eligible": eligible,
            "minimum_required_reduction_m": minimum_reduction,
            "resume_switch_cost_s": resume_switch_s,
            "effective_incremental_time_s": effective_time,
            "opportunity_score": opportunity_score,
        }

    def plan_opportunistic_measurements(
        self,
        global_index: int,
        point: Point,
    ) -> list[dict]:
        """选择当前网格点值得顺带补测的少量已发现频道。"""
        if not self.cfg.enable_parallel_search_localization:
            return []

        resume_unknown_search = (
            self.discovered_count < self.cfg.source_count_max
            and any(record.status == "UNKNOWN" for record in self.records.values())
        )
        window_end = min(
            len(self.search_points),
            global_index + self.cfg.opportunistic_lookahead_nodes + 1,
        )
        future_points = self.search_points[global_index + 1:window_end]
        proposals: list[dict] = []

        for record in self.records.values():
            if record.status != "FOUND":
                continue
            if (
                record.opportunistic_measure_count
                >= self.cfg.opportunistic_max_per_channel
            ):
                continue
            try:
                sources = sample_source_positions(record, self.cfg)
            except IncompleteRun as error:
                self.event(
                    "opportunistic_scoring_skipped",
                    channel=record.channel_id,
                    reason=str(error),
                )
                continue

            current = self._score_route_waypoint(
                record,
                point,
                sources,
                resume_unknown_search=resume_unknown_search,
            )
            if current is None or not current["eligible"]:
                continue

            future_scores = [
                score
                for future_point in future_points
                if (
                    score := self._score_route_waypoint(
                        record,
                        future_point,
                        sources,
                        resume_unknown_search=resume_unknown_search,
                    )
                ) is not None
                and score["eligible"]
            ]
            best_future = max(
                future_scores,
                key=lambda item: item["opportunity_score"],
                default=None,
            )
            if (
                best_future is not None
                and not current["predicted_clearable"]
                and current["opportunity_score"]
                < self.cfg.opportunistic_lookahead_ratio
                * best_future["opportunity_score"]
            ):
                self.counts["opportunistic_deferred_by_lookahead"] += 1
                continue

            proposals.append({
                **current,
                "channel": record.channel_id,
                "best_future_opportunity_score": (
                    best_future["opportunity_score"]
                    if best_future is not None
                    else None
                ),
            })

        selected = sorted(
            proposals,
            key=lambda item: (
                item["predicted_clearable"],
                item["opportunity_score"],
                item["robust_receive_score"],
                item["radius_reduction"],
                -item["effective_incremental_time_s"],
                -item["channel"],
            ),
            reverse=True,
        )[:self.cfg.opportunistic_max_per_node]
        if selected:
            self.event(
                "opportunistic_batch_planned",
                global_index=global_index,
                point=point,
                candidate_channel_count=len(proposals),
                selected=[
                    {
                        "channel": item["channel"],
                        "opportunity_score": item["opportunity_score"],
                        "robust_receive_score": item["robust_receive_score"],
                        "radius_reduction": item["radius_reduction"],
                        "predicted_clearable": item["predicted_clearable"],
                    }
                    for item in selected
                ],
            )
        return selected

    def opportunistic_localization_at_waypoint(
        self,
        global_index: int,
        point: Point,
    ) -> None:
        """在不离开当前全局网格点的前提下，对已发现频道补充测向。"""
        for proposal in self.plan_opportunistic_measurements(global_index, point):
            record = self.records[proposal["channel"]]
            if record.status != "FOUND":
                continue
            result = self.measure(
                record,
                point,
                measurement_kind="opportunistic",
            )
            record.opportunistic_measure_count += 1
            self.event(
                "opportunistic_measurement_completed",
                channel=record.channel_id,
                global_index=global_index,
                point=point,
                result=result,
                predicted_opportunity_score=proposal["opportunity_score"],
                predicted_receive_score=proposal["robust_receive_score"],
                predicted_radius_reduction=proposal["radius_reduction"],
                status_after=record.status,
            )
            if result == "near":
                # near 的清除点就是当前网格点，不产生路线偏离，因此立即清除。
                self.clear(record)
            elif record.status == "READY":
                # MEC 中心可能偏离全局路线；仅登记，留给搜索结束后的调度处理。
                self.counts["opportunistic_ready_queued"] += 1
                self.event(
                    "opportunistic_ready_queued",
                    channel=record.channel_id,
                    clearance_center=record.clearance_center,
                    clearance_radius=record.clearance_radius,
                    certificate_source=record.certificate_source,
                )

    def search(self) -> None:
        self.phase = "global_search"
        for index, point in enumerate(self.search_points):
            self.next_search_index = index
            unknown = [
                channel
                for channel, record in self.records.items()
                if record.status == "UNKNOWN"
            ]
            if not unknown:
                break
            current = self.client.current_channel
            order = ([current] if current in unknown else []) + [
                channel for channel in unknown if channel != current
            ]
            for channel in order:
                record = self.records[channel]
                result = self.measure(
                    record,
                    point,
                    measurement_kind="unknown",
                    global_index=index,
                )
                # near 点本身就是可靠清除点，立即清除后仍停在该搜索点。
                if result == "near":
                    self.clear(record)
                if self.discovered_count >= self.cfg.source_count_max:
                    self.mark_absent()
                    break

            # 已完成本节点的未知频道检测后，再利用同一位置顺带定位 FOUND 频道。
            # 此步骤不改变全局网格及“未知频道完整检测”的发现保证。
            self.opportunistic_localization_at_waypoint(index, point)
            self.next_search_index = index + 1
            if self.discovered_count >= self.cfg.source_count_max:
                break

        self.search_complete = self.next_search_index >= len(self.search_points)
        self.mark_absent()
        if not self.search_complete and self.discovered_count < self.cfg.source_count_max:
            raise IncompleteRun("全局保证搜索未完成，不能判定剩余频道不存在")

    def q2_candidate(self, record: ChannelRecord) -> Point | None:
        if not self.cfg.enable_q2_seed or record.q2_attempted:
            return record.q2_seed
        record.q2_attempted = True
        if len(record.bearing_observations) != 1:
            return record.q2_seed
        observation = record.bearing_observations[0]
        self.counts["q2_attempt"] += 1
        point, diagnostics = bounded_q2_seed(
            (*observation["position"], observation["bearing_deg"]),
            self.pcfg,
            self.gcfg,
            self.cfg.q2_timeout_s,
        )
        if point is not None:
            self.counts["q2_success"] += 1
            record.q2_seed = point
        self.event(
            "q2_seed",
            channel=record.channel_id,
            point=point,
            diagnostics=diagnostics,
        )
        return point

    def _record_active_plan(
        self,
        record: ChannelRecord,
        proposal: dict,
        **scheduler_data,
    ) -> None:
        """只为真正被执行的主动方案计数，候选比较不污染 active_plan 统计。"""
        self.counts["active_plan"] += 1
        self.event(
            "active_planned",
            channel=record.channel_id,
            **proposal,
            **scheduler_data,
        )

    def plan_active_measurement(
        self,
        record: ChannelRecord,
        *,
        record_plan: bool = True,
        allow_new_q2_seed: bool = True,
    ) -> dict:
        sources = sample_source_positions(record, self.cfg)
        # 全局跨频道比较不能同时触发多个可能耗时的 q2 子进程；只有最终选中
        # 的频道才允许首次调用问题二规划器。已有 q2_seed 仍可直接复用。
        q2_seed = (
            self.q2_candidate(record)
            if allow_new_q2_seed
            else record.q2_seed
        )
        estimate, candidates = candidate_points(
            record,
            sources,
            self.client.position,
            self.cfg,
            q2_seed,
        )
        scored = [
            result
            for candidate in candidates
            if (result := score_candidate(
                record,
                candidate,
                sources,
                estimate,
                self.client.position,
                self.client.current_channel,
                self.cfg,
            )) is not None
        ]
        if not scored:
            raise IncompleteRun(f"频道{record.channel_id}没有可能接收信号的主动候选")

        positive = [item for item in scored if item["radius_reduction"] > 1e-9]
        if positive:
            selected = max(
                positive,
                key=lambda item: (
                    item["weighted_score"],
                    item["robust_receive_score"],
                    item["predicted_clearable"],
                    item["radius_reduction"],
                    item["guaranteed_distance_fraction"],
                    -item["side_penalty"],
                    -item["time_cost_s"],
                    item["point"],
                ),
            )
            selection_stage = "conditional_direction_gain"
        else:
            # 有些测量只约束朝向或所有离散几何收益为零；仍轮换到未失败侧。
            selected = max(
                scored,
                key=lambda item: (
                    item["robust_receive_score"],
                    item["guaranteed_distance_fraction"],
                    item["possible_receive_fraction"],
                    -item["side_penalty"],
                    -item["time_cost_s"],
                    item["point"],
                ),
            )
            selection_stage = "side_diversity_fallback"

        proposal = {
            **selected,
            "selection_stage": selection_stage,
            "candidate_count": len(candidates),
            "scored_candidate_count": len(scored),
            "source_sample_count": len(sources),
            "estimate": estimate,
            "score_scope": (
                "source_position_and_orientation_sampled_receive_score; "
                "geometry_gain_conditional_on_direction; no_signal_keeps_position_region"
            ),
        }
        if record_plan:
            self._record_active_plan(record, proposal)
        return proposal

    def _point_was_tested(self, record: ChannelRecord, point: Point) -> bool:
        return any(
            math.dist(point, observation["position"]) <= 1e-6
            for observation in record.all_observations()
        )

    def start_recovery(self, record: ChannelRecord) -> None:
        self._active_plan_cache.pop(record.channel_id, None)
        if not record.outer_polygon:
            raise IncompleteRun(f"频道{record.channel_id}没有定位范围，无法启动恢复网格")

        # 修复二：大楔形先用 500 米网格重新获得示向；只有位置范围已小且
        # 细网格节点受控时，才使用 <5 米网格直接触发 near。
        fine_candidate = (
            record.clearance_radius is not None
            and record.clearance_radius <= self.cfg.fine_recovery_max_radius_m
        )
        plan = None
        mode = "coarse_reacquire"
        if fine_candidate:
            candidate_plan = build_recovery_grid(
                record.outer_polygon,
                self.cfg.recovery_grid_side_m,
                self.client.position,
            )
            untested_count = sum(
                not self._point_was_tested(record, point)
                for point in candidate_plan.route
            )
            if untested_count <= self.cfg.fine_recovery_max_nodes:
                plan = candidate_plan
                mode = "fine_near"
        if plan is None:
            plan = build_recovery_grid(
                record.outer_polygon,
                self.cfg.coarse_recovery_grid_side_m,
                self.client.position,
            )
        route = [point for point in plan.route if not self._point_was_tested(record, point)]
        if not route:
            raise IncompleteRun(f"频道{record.channel_id}恢复网格没有未检测节点")
        self._recovery_queues[record.channel_id] = deque(route)
        record.recovery_active = True
        record.recovery_mode = mode
        record.recovery_generation += 1
        record.recovery_total_nodes += len(route)
        self.counts["recovery_grid_build"] += 1
        self.counts[
            "fine_recovery_build" if mode == "fine_near" else "coarse_recovery_build"
        ] += 1
        self.event(
            "recovery_grid_started",
            channel=record.channel_id,
            generation=record.recovery_generation,
            point_count=len(route),
            triangle_count=plan.triangle_count,
            route_distance_m=open_route_distance(self.client.position, route),
            side_m=plan.side_m,
            recovery_mode=mode,
            fine_node_limit=self.cfg.fine_recovery_max_nodes,
        )

    def recovery_step(self, record: ChannelRecord) -> str:
        queue = self._recovery_queues.get(record.channel_id)
        if not record.recovery_active or queue is None:
            self.start_recovery(record)
            queue = self._recovery_queues[record.channel_id]
        if not queue:
            raise IncompleteRun(
                f"频道{record.channel_id}已检测完整个局部细网格仍未进入近距离分支"
            )
        point = queue.popleft()
        record.recovery_visited_nodes += 1
        return self.measure(record, point, measurement_kind="recovery")

    def process_target(self, record: ChannelRecord, action_limit: int) -> None:
        actions = 0
        while record.status in {"FOUND", "READY"} and actions < action_limit:
            self.check_budget()
            if record.status == "READY":
                self.clear(record)
                return
            if record.recovery_active or record.no_progress_count >= self.cfg.no_progress_trigger:
                result = self.recovery_step(record)
                actions += 1
                # 一旦重新获得示向或 near，就让出本轮并重新规划，不沿旧网格惯性扫描。
                if result != "no_signal":
                    return
                continue
            try:
                proposal = self.plan_active_measurement(record)
            except IncompleteRun as error:
                record.no_progress_count = self.cfg.no_progress_trigger
                self.event(
                    "active_plan_failed_start_recovery",
                    channel=record.channel_id,
                    reason=str(error),
                )
                continue
            self.measure(record, proposal["point"], measurement_kind="active")
            return

    def _localization_task_point(self, record: ChannelRecord) -> Point:
        """给尚未生成精确行动方案的频道提供轻量级预筛选位置。"""
        if record.clearance_center is not None:
            return tuple(record.clearance_center)
        if record.bearing_observations:
            return tuple(record.bearing_observations[-1]["position"])
        return self.client.position

    def _scheduler_age_factor(self, record: ChannelRecord) -> float:
        # 只作为软防饿死项，不改变“价值/耗时”为第一优先级的原则。
        return 1.0 + self.cfg.scheduler_wait_weight * min(
            record.scheduler_wait_count,
            self.cfg.scheduler_wait_cap,
        )

    def _clear_scheduler_task(self, record: ChannelRecord) -> dict:
        point = tuple(record.clearance_center)
        travel_distance = math.dist(self.client.position, point)
        time_cost = (
            travel_distance / self.cfg.dog_speed_m_per_s
            + self.cfg.optical_time_s
            + self.cfg.clearance_time_s
        )
        value = self.cfg.scheduler_completion_value_m
        return {
            "record": record,
            "action_type": "clear",
            "point": point,
            "point_is_exact": True,
            "travel_distance_m": travel_distance,
            "time_cost_s": time_cost,
            "expected_value_m": value,
            "base_score": value / max(time_cost, 1e-9),
            "scheduler_score": (
                value / max(time_cost, 1e-9) * self._scheduler_age_factor(record)
            ),
            "predicted_clearable": True,
        }

    def _active_scheduler_task(self, record: ChannelRecord, proposal: dict) -> dict:
        point = tuple(proposal["point"])
        travel_distance = math.dist(self.client.position, point)
        time_cost = (
            travel_distance / self.cfg.dog_speed_m_per_s
            + self.cfg.detection_time_s
            + (
                self.cfg.channel_switch_time_s
                if self.client.current_channel != record.channel_id
                else 0.0
            )
        )
        receive_factor = 0.10 + 0.90 * proposal["robust_receive_score"]
        geometric_value = (
            proposal["radius_reduction"]
            * receive_factor
            / (1.0 + 0.25 * proposal["side_penalty"])
        )
        completion_bonus = (
            self.cfg.scheduler_completion_value_m
            if proposal["predicted_clearable"]
            else 0.0
        )
        expected_value = geometric_value + completion_bonus
        base_score = expected_value / max(time_cost, 1e-9)
        adjusted_proposal = {
            **proposal,
            "planning_time_cost_s": proposal["time_cost_s"],
            "time_cost_s": time_cost,
            "weighted_score": geometric_value / max(time_cost, 1e-9),
        }
        return {
            "record": record,
            "action_type": "active",
            "point": point,
            "point_is_exact": True,
            "travel_distance_m": travel_distance,
            "time_cost_s": time_cost,
            "expected_value_m": expected_value,
            "geometric_value_m": geometric_value,
            "completion_bonus_m": completion_bonus,
            "base_score": base_score,
            "scheduler_score": base_score * self._scheduler_age_factor(record),
            "predicted_clearable": proposal["predicted_clearable"],
            "proposal": adjusted_proposal,
        }

    def _recovery_scheduler_task(self, record: ChannelRecord) -> dict:
        queue = self._recovery_queues.get(record.channel_id)
        point_is_exact = bool(record.recovery_active and queue)
        point = tuple(queue[0]) if point_is_exact else self._localization_task_point(record)
        travel_distance = math.dist(self.client.position, point)
        time_cost = (
            travel_distance / self.cfg.dog_speed_m_per_s
            + self.cfg.detection_time_s
            + (
                self.cfg.channel_switch_time_s
                if self.client.current_channel != record.channel_id
                else 0.0
            )
        )
        expected_value = self.cfg.scheduler_recovery_value_m
        base_score = expected_value / max(time_cost, 1e-9)
        return {
            "record": record,
            "action_type": "recovery",
            "point": point,
            "point_is_exact": point_is_exact,
            "travel_distance_m": travel_distance,
            "time_cost_s": time_cost,
            "expected_value_m": expected_value,
            "base_score": base_score,
            "scheduler_score": base_score * self._scheduler_age_factor(record),
            "predicted_clearable": False,
        }

    def _build_localization_tasks(self, pending: Sequence[ChannelRecord]) -> list[dict]:
        """构造跨频道的下一动作集合；昂贵主动规划只对短名单执行并缓存。"""
        tasks = [
            self._clear_scheduler_task(record)
            for record in pending
            if record.status == "READY"
        ]
        found = [record for record in pending if record.status == "FOUND"]
        recovery_records = [
            record
            for record in found
            if record.recovery_active
            or record.no_progress_count >= self.cfg.no_progress_trigger
        ]
        tasks.extend(self._recovery_scheduler_task(record) for record in recovery_records)

        recovery_channels = {record.channel_id for record in recovery_records}
        active_records = [
            record for record in found if record.channel_id not in recovery_channels
        ]
        active_channels = {record.channel_id for record in active_records}
        for channel in list(self._active_plan_cache):
            if channel not in active_channels:
                self._active_plan_cache.pop(channel, None)

        cached = [
            record for record in active_records
            if record.channel_id in self._active_plan_cache
        ]
        uncached = [
            record for record in active_records
            if record.channel_id not in self._active_plan_cache
        ]
        uncached.sort(
            key=lambda record: (
                math.dist(
                    self.client.position,
                    self._localization_task_point(record),
                ) / self._scheduler_age_factor(record),
                record.channel_id,
            )
        )
        new_plan_slots = max(0, self.cfg.scheduler_active_shortlist - len(cached))
        plan_records = cached + uncached[:new_plan_slots]

        for record in plan_records:
            proposal = self._active_plan_cache.get(record.channel_id)
            if proposal is None:
                try:
                    proposal = self.plan_active_measurement(
                        record,
                        record_plan=False,
                        allow_new_q2_seed=False,
                    )
                except IncompleteRun as error:
                    record.no_progress_count = self.cfg.no_progress_trigger
                    self.event(
                        "active_plan_failed_start_recovery",
                        channel=record.channel_id,
                        reason=str(error),
                        scheduling_stage="global_task_comparison",
                    )
                    tasks.append(self._recovery_scheduler_task(record))
                    continue
                self._active_plan_cache[record.channel_id] = proposal
                self.counts["scheduler_plan_evaluation"] += 1
            else:
                self.counts["scheduler_plan_cache_hit"] += 1
            tasks.append(self._active_scheduler_task(record, proposal))
        return tasks

    @staticmethod
    def _scheduler_selection_key(task: dict) -> tuple:
        # channel 仅用于完全同分时给出确定性结果，不表达公平轮转优先级。
        return (
            task["scheduler_score"],
            task["predicted_clearable"],
            -task["time_cost_s"],
            -task["record"].channel_id,
        )

    def _execute_localization_task(self, task: dict) -> None:
        record = task["record"]
        action_type = task["action_type"]
        if action_type == "clear":
            self.counts["scheduler_clear_selected"] += 1
            self.clear(record)
            return
        if action_type == "active":
            self.counts["scheduler_active_selected"] += 1
            proposal = task["proposal"]
            if (
                self.cfg.enable_q2_seed
                and not record.q2_attempted
                and len(record.bearing_observations) == 1
            ):
                preliminary_point = proposal["point"]
                refined = self.plan_active_measurement(
                    record,
                    record_plan=False,
                    allow_new_q2_seed=True,
                )
                task = self._active_scheduler_task(record, refined)
                proposal = task["proposal"]
                self.counts["scheduler_selected_plan_refinement"] += 1
                self.event(
                    "scheduler_selected_plan_refined",
                    channel=record.channel_id,
                    preliminary_point=preliminary_point,
                    refined_point=proposal["point"],
                    q2_seed=record.q2_seed,
                    refined_scheduler_score=task["scheduler_score"],
                )
            self._active_plan_cache.pop(record.channel_id, None)
            self._record_active_plan(
                record,
                proposal,
                scheduling_basis="global_expected_value_per_incremental_time",
                scheduler_score=task["scheduler_score"],
                scheduler_expected_value_m=task["expected_value_m"],
                scheduler_travel_distance_m=task["travel_distance_m"],
            )
            self.measure(record, task["point"], measurement_kind="active")
            return
        if action_type == "recovery":
            self.counts["scheduler_recovery_selected"] += 1
            self._active_plan_cache.pop(record.channel_id, None)
            self.recovery_step(record)
            return
        raise RuntimeError(f"未知定位任务类型：{action_type!r}")

    def _localize_value_time(self) -> None:
        self.phase = "localization"
        while True:
            pending = [
                record
                for record in self.records.values()
                if record.status in {"FOUND", "READY"}
            ]
            if not pending:
                return
            self.check_budget()
            tasks = self._build_localization_tasks(pending)
            if not tasks:
                raise IncompleteRun("全局定位调度器没有生成可执行任务")
            selected_task = max(tasks, key=self._scheduler_selection_key)
            selected = selected_task["record"]
            self.counts["scheduler_round"] += 1
            self.event(
                "scheduler_target_selected",
                channel=selected.channel_id,
                status=selected.status,
                action_type=selected_task["action_type"],
                task_point=selected_task["point"],
                task_point_is_exact=selected_task["point_is_exact"],
                scheduler_score=selected_task["scheduler_score"],
                base_score=selected_task["base_score"],
                expected_value_m=selected_task["expected_value_m"],
                time_cost_s=selected_task["time_cost_s"],
                travel_distance_m=selected_task["travel_distance_m"],
                predicted_clearable=selected_task["predicted_clearable"],
                scheduler_wait_count=selected.scheduler_wait_count,
                localization_turn_count=selected.localization_turn_count,
                selection_basis="expected_value_per_incremental_time_with_soft_aging",
                candidate_summary=[
                    {
                        "channel": task["record"].channel_id,
                        "action_type": task["action_type"],
                        "scheduler_score": task["scheduler_score"],
                        "time_cost_s": task["time_cost_s"],
                        "travel_distance_m": task["travel_distance_m"],
                    }
                    for task in sorted(
                        tasks,
                        key=self._scheduler_selection_key,
                        reverse=True,
                    )[:8]
                ],
            )

            self._execute_localization_task(selected_task)
            selected.localization_turn_count += 1
            for record in pending:
                if record.channel_id == selected.channel_id:
                    record.scheduler_wait_count = 0
                elif record.status in {"FOUND", "READY"}:
                    record.scheduler_wait_count += 1

    def _localize_legacy_round_robin(self) -> None:
        """保留旧调度器，仅供同一代码与 client 接口下做 A/B 对照。"""
        self.phase = "localization"
        while True:
            pending = [
                record
                for record in self.records.values()
                if record.status in {"FOUND", "READY"}
            ]
            if not pending:
                return
            selected = min(
                pending,
                key=lambda record: (
                    record.status != "READY",
                    record.localization_turn_count,
                    math.dist(
                        self.client.position,
                        self._localization_task_point(record),
                    ),
                    record.channel_id,
                ),
            )
            self.event(
                "target_selected",
                channel=selected.channel_id,
                status=selected.status,
                task_point=self._localization_task_point(selected),
                localization_turn_count=selected.localization_turn_count,
                selection_basis="legacy_round_count_first",
            )
            action_limit = (
                self.cfg.recovery_batch_size
                if selected.recovery_active
                or selected.no_progress_count >= self.cfg.no_progress_trigger
                else 1
            )
            self.process_target(selected, action_limit)
            selected.localization_turn_count += 1

    def localize(self) -> None:
        if self.cfg.enable_value_time_scheduler:
            self._localize_value_time()
        else:
            self._localize_legacy_round_robin()

    def complete(self) -> bool:
        return (
            self.cleared_count == self.cfg.source_count_max
            or all(record.status in TERMINAL_STATES for record in self.records.values())
        )

    def attempt_safe_exit(self, trigger: str) -> bool:
        if self.entered_at is None or self.client.pending_request_id is not None:
            return False
        if self.client.remaining_real_time_s is None or self.client.remaining_real_time_s <= 0:
            return False
        if (
            self.client.virtual_time_s is None
            or self.client.max_virtual_duration_s is None
            or self.client.virtual_time_s >= self.client.max_virtual_duration_s
        ):
            return False
        self.phase = "incomplete_exit"
        try:
            self.action("exit")
            self.event("safe_exit_succeeded", trigger=trigger)
            return True
        except Exception as error:
            self.event("safe_exit_failed", trigger=trigger, reason=repr(error))
            return False

    def run(self) -> dict:
        outcome, reason, exit_confirmed = "incomplete", "尚未完成", False
        try:
            self.action("enter")
            self.entered_at = time.monotonic()
            self.trajectory.append({"to": (0.0, 0.0), "phase": "enter", "virtual_time_s": 0.0})
            self.search()
            self.localize()
            if not self.complete():
                raise IncompleteRun("仍有非终态频道，不能宣告完成")
            self.phase = "finish"
            self.action("exit")
            exit_confirmed = True
            outcome = "success"
            reason = "全部频道已清除或证明不存在；或已成功清除16个源"
        except (IncompleteRun, ClientError, KeyboardInterrupt) as error:
            reason = f"{type(error).__name__}: {error}"
            self.event("run_incomplete", reason=reason)
            if not isinstance(error, ClientError):
                exit_confirmed = self.attempt_safe_exit(type(error).__name__)
        except Exception as error:
            reason = f"unexpected_error: {error}"
            self.event("unexpected_error", traceback=traceback.format_exc())
            exit_confirmed = self.attempt_safe_exit(type(error).__name__)
        finally:
            self.ended_at = time.monotonic()
            summary = self.summary(outcome, reason, exit_confirmed)
            if self.output_dir is not None:
                (self.output_dir / "summary.json").write_text(
                    json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8",
                )
            self.close_streams()
        return summary

    def summary(self, outcome: str, reason: str, exit_confirmed: bool) -> dict:
        virtual_time = self.client.virtual_time_s
        accounted = (
            self.total_distance_m / self.cfg.dog_speed_m_per_s
            + self.counts["switch"] * self.cfg.channel_switch_time_s
            + self.counts["measure"] * self.cfg.detection_time_s
            + self.counts["clear_success"]
            * (self.cfg.optical_time_s + self.cfg.clearance_time_s)
            + self.counts["clear_failure"] * self.cfg.optical_time_s
        )
        return {
            "outcome": outcome,
            "reason": reason,
            "all_targets_resolved": self.complete(),
            "exit_confirmed": exit_confirmed,
            "discovered_count": self.discovered_count,
            "cleared_count": self.cleared_count,
            "virtual_time_s": virtual_time,
            "accounted_virtual_time_s": accounted,
            "total_distance_m": self.total_distance_m,
            "program_elapsed_s": (
                self.ended_at - self.entered_at
                if self.entered_at is not None and self.ended_at is not None
                else None
            ),
            "counts": self.counts,
            "search_complete": self.search_complete,
            "next_search_index": self.next_search_index,
            "global_grid": {
                "side_m": self.global_grid.side_m,
                "point_count": len(self.global_grid.points),
                "triangle_count": self.global_grid.triangle_count,
                "route_distance_m": self.global_grid.route_distance_m,
                "proof": "每个源所在相交等边三角形的三个顶点均在1000米保证距离内，且至少一个顶点位于前向半平面",
            },
            "trajectory": self.trajectory,
            "channels": {
                channel: asdict(record)
                for channel, record in self.records.items()
            },
            "q4_config": asdict(self.cfg),
            "planner_config": asdict(self.pcfg),
            "geometry_config": asdict(self.gcfg),
            "model_note": (
                "全局保证搜索途中对已发现频道执行受控补测；"
                "no_signal 不裁剪连续位置；主动选择联合采样位置与方向；"
                "严格几何失败时的浮点外包不签发MEC清除证书"
            ),
            "formal_log_note": "正式加密日志仍需从模拟器界面导出",
        }

@dataclass(frozen=True)
class _CertifiedSearchTuning:
    # 优先级0：双环搜索
    ring_point_count: int = 12
    inner_ring_radius_m: float = 980.0
    outer_tangent_margin_m: float = 2.0

    # 优先级1：信息驱动主动定位/恢复
    active_receive_floor: float = 0.55
    active_move_cap_m: float = 800.0
    recovery_receive_floor: float = 0.42
    recovery_move_cap_m: float = 700.0
    recovery_information_probe_limit: int = 5
    recovery_fallback_side_m: float = 500.0
    orientation_particle_count: int = 36

    # 主动方案缓存失效距离
    active_cache_replan_distance_m: float = 350.0

    # 优先级2：联合路径调度
    scheduler_horizon: int = 3
    scheduler_beam_width: int = 48
    scheduler_task_limit: int = 12
    scheduler_commit_ratio: float = 0.80
    scheduler_recovery_commit_ratio: float = 0.65
    scheduler_cluster_radius_m: float = 900.0
    scheduler_cluster_bonus: float = 0.06

    def __post_init__(self):
        if self.ring_point_count < 12:
            raise ValueError("外环至少需要12点，才能用≤1000米边覆盖1800米圆盘")
        if not 0.0 < self.inner_ring_radius_m < 1000.0:
            raise ValueError("inner_ring_radius_m 必须位于 (0,1000)")
        if self.outer_tangent_margin_m < 0.0:
            raise ValueError("outer_tangent_margin_m 不能为负数")
        for name in (
            "active_receive_floor",
            "recovery_receive_floor",
            "scheduler_commit_ratio",
            "scheduler_recovery_commit_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} 必须位于 (0,1]")
        for name in (
            "active_move_cap_m",
            "recovery_move_cap_m",
            "recovery_fallback_side_m",
            "active_cache_replan_distance_m",
            "scheduler_cluster_radius_m",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} 必须为正数")


def triangle_max_edge(triangle: Triangle) -> float:
    a, b, c = triangle
    return max(
        math.dist(a, b),
        math.dist(b, c),
        math.dist(c, a),
    )


def origin_segment_distance(a: Point, b: Point) -> float:
    vx = b[0] - a[0]
    vy = b[1] - a[1]
    denominator = vx * vx + vy * vy
    if denominator <= 1e-15:
        return math.hypot(*a)

    ratio = -(a[0] * vx + a[1] * vy) / denominator
    ratio = min(1.0, max(0.0, ratio))
    return math.hypot(
        a[0] + ratio * vx,
        a[1] + ratio * vy,
    )


def polar_point(radius: float, angle: float) -> Point:
    return (
        radius * math.cos(angle),
        radius * math.sin(angle),
    )


def build_dual_ring_search(
    config: Q4Config,
    tuning: _CertifiedSearchTuning,
) -> tuple[GridPlan, dict]:
    """构造25点双环认证搜索。

    默认结构：
        原点：1点
        内环：半径980米，12点
        外环：边到原点距离1802米，12点，并相对内环旋转15度

    证明：
    1. 外环正多边形完整包住1800米目标圆。
    2. 内扇区与环间三角形覆盖整个外环多边形。
    3. 每个认证三角形的三条边均不超过1000米。
    4. 任意源位于某个认证三角形内。
    5. 三个顶点均处于源的1000米接收范围。
    6. 任意过源的发射前半平面至少包含一个顶点。
    """

    count = tuning.ring_point_count
    step = 2.0 * math.pi / count
    inner_radius = tuning.inner_ring_radius_m

    target_support_radius = (
        config.target_radius_m + tuning.outer_tangent_margin_m
    )
    outer_radius = target_support_radius / math.cos(math.pi / count)

    origin: Point = (0.0, 0.0)

    inner = [
        polar_point(inner_radius, index * step)
        for index in range(count)
    ]
    outer = [
        polar_point(outer_radius, (index + 0.5) * step)
        for index in range(count)
    ]

    triangles: list[Triangle] = []

    # 原点到内环的三角扇。
    for index in range(count):
        triangles.append((
            origin,
            inner[index],
            inner[(index + 1) % count],
        ))

    # 内外环之间的交错三角剖分。
    for index in range(count):
        next_index = (index + 1) % count
        triangles.append((
            inner[index],
            outer[index],
            inner[next_index],
        ))
        triangles.append((
            inner[next_index],
            outer[index],
            outer[next_index],
        ))

    max_edge = max(triangle_max_edge(item) for item in triangles)
    if max_edge > config.reception_radius_min_m + 1e-7:
        raise ValueError(
            f"双环证书最大边为{max_edge:.3f}米，超过"
            f"{config.reception_radius_min_m:.3f}米"
        )

    outer_min_distance = min(
        origin_segment_distance(
            outer[index],
            outer[(index + 1) % count],
        )
        for index in range(count)
    )
    if outer_min_distance < config.target_radius_m - 1e-7:
        raise ValueError("双环外边界没有完全包住目标圆盘")

    # 开放路线：
    # 原点 -> 顺时针访问内环 -> 从最近点进入外环 -> 反向访问外环。
    # 两个环都不闭合，分别省去一条环边。
    route = [origin]
    route.extend(inner)
    route.extend(reversed(outer))

    points = [origin, *inner, *outer]
    route_distance = open_route_distance(origin, route)

    plan = GridPlan(
        side_m=max_edge,
        points=points,
        route=route,
        triangle_count=len(triangles),
        route_distance_m=route_distance,
        triangles=triangles,
    )
    diagnostics = {
        "type": "dual_ring_certified_triangulation",
        "point_count": len(points),
        "triangle_count": len(triangles),
        "inner_radius_m": inner_radius,
        "outer_radius_m": outer_radius,
        "outer_boundary_min_distance_m": outer_min_distance,
        "max_certificate_edge_m": max_edge,
        "route_distance_m": route_distance,
    }
    return plan, diagnostics


class _CertifiedSearchRunner(_CoreRunner):
    def __init__(
        self,
        client,
        *,
        config: Q4Config,
        tuning: _CertifiedSearchTuning | None = None,
        planner_config=None,
        geometry_config=None,
        output_dir: Path | None = None,
    ):
        self.tuning = tuning or _CertifiedSearchTuning()

        super().__init__(
            client,
            config=config,
            planner_config=planner_config,
            geometry_config=geometry_config,
            output_dir=output_dir,
        )

        # super 只进行了内存初始化，尚未向客户端发出任何动作。
        # 在 enter/search 前用双环计划替换基础执行层网格。
        self.global_grid, self.dual_ring_diagnostics = (
            build_dual_ring_search(self.cfg, self.tuning)
        )
        self.search_points = list(self.global_grid.route)

        self._information_attempts: dict[int, int] = {}
        self._last_selected_channel: int | None = None

        self.counts.update({
            "information_recovery_plan": 0,
            "information_recovery_measure": 0,
            "information_recovery_fallback": 0,
            "active_plan_rejected_by_receive_floor": 0,
            "active_plan_rejected_by_move_cap": 0,
            "scheduler_horizon_round": 0,
            "scheduler_commit_selected": 0,
        })

    # ------------------------------------------------------------------
    # 优先级0：正确的不存在证明
    # ------------------------------------------------------------------

    def mark_absent(self) -> None:
        for record in self.records.values():
            if record.status != "UNKNOWN":
                continue

            if self.discovered_count >= self.cfg.source_count_max:
                reason = "source_count_upper_bound_reached"
            elif (
                self.search_complete
                and len(record.global_nodes_tested)
                == len(self.search_points)
            ):
                reason = "all_certified_dual_ring_nodes_no_signal"
            else:
                continue

            record.status = "ABSENT"
            record.absent_reason = reason
            self.event(
                "absent",
                channel=record.channel_id,
                reason=reason,
            )

    # ------------------------------------------------------------------
    # 优先级1：位置、朝向和接收半径联合粒子
    # ------------------------------------------------------------------

    @staticmethod
    def _particle_front(
        direction: Point | None,
        source: Point,
        receiver: Point,
    ) -> bool:
        if direction is None:
            # None 表示全向源粒子。
            return True
        return (
            direction[0] * (receiver[0] - source[0])
            + direction[1] * (receiver[1] - source[1])
            >= -1e-8
        )

    def _joint_hypotheses(
        self,
        record: ChannelRecord,
        sources: Sequence[Point],
    ) -> list[dict]:
        """生成仅供规划使用的联合假设。

        no_signal 只淘汰规划粒子，不修改连续位置外包，
        因而不会被用于ABSENT或MEC清除证书。
        """

        hypotheses: list[dict] = []
        source_groups: list[list[dict]] = []

        no_signals = list(record.no_signal_observations)
        positive = list(record.bearing_observations)

        for source in sources:
            lower_radius = max(
                self.cfg.reception_radius_min_m,
                *(
                    math.dist(source, obs["position"])
                    for obs in positive
                ),
            )
            if lower_radius > self.cfg.reception_radius_max_m + 1e-7:
                continue

            radii = sorted(set([
                lower_radius,
                (lower_radius + self.cfg.reception_radius_max_m) / 2.0,
                self.cfg.reception_radius_max_m,
            ]))

            omnidirectional: list[dict] = []
            directional: list[dict] = []

            direction_values: list[Point | None] = [None]
            direction_values.extend(
                (
                    math.cos(2.0 * math.pi * index /
                             self.tuning.orientation_particle_count),
                    math.sin(2.0 * math.pi * index /
                             self.tuning.orientation_particle_count),
                )
                for index in range(self.tuning.orientation_particle_count)
            )

            for radius in radii:
                for direction in direction_values:
                    if any(
                        math.dist(source, obs["position"]) > radius + 1e-7
                        or not self._particle_front(
                            direction,
                            source,
                            obs["position"],
                        )
                        for obs in positive
                    ):
                        continue

                    # 如果该假设在历史no_signal点理应能接收到，
                    # 则该联合假设与日志不相容。
                    inconsistent = any(
                        math.dist(source, obs["position"]) <= radius + 1e-7
                        and self._particle_front(
                            direction,
                            source,
                            obs["position"],
                        )
                        for obs in no_signals
                    )
                    if inconsistent:
                        continue

                    item = {
                        "source": tuple(source),
                        "radius": float(radius),
                        "direction": direction,
                    }
                    if direction is None:
                        omnidirectional.append(item)
                    else:
                        directional.append(item)

            local: list[dict] = []

            # 全向和定向先各占一半先验；若一类已被观测完全排除，
            # 则将权重转给仍然相容的一类。
            present_classes = sum(bool(group) for group in (
                omnidirectional,
                directional,
            ))
            if present_classes == 0:
                continue

            class_weight = 1.0 / present_classes
            for group in (omnidirectional, directional):
                if not group:
                    continue
                item_weight = class_weight / len(group)
                for item in group:
                    local.append({
                        **item,
                        "weight": item_weight,
                    })

            source_groups.append(local)

        if not source_groups:
            return []

        source_weight = 1.0 / len(source_groups)
        for group in source_groups:
            for item in group:
                hypotheses.append({
                    **item,
                    "weight": item["weight"] * source_weight,
                })

        total_weight = sum(item["weight"] for item in hypotheses)
        if total_weight <= 0.0:
            return []

        for item in hypotheses:
            item["weight"] /= total_weight
        return hypotheses

    def _joint_receive_probability(
        self,
        point: Point,
        hypotheses: Sequence[dict],
    ) -> float:
        probability = 0.0
        for item in hypotheses:
            source = item["source"]
            if math.dist(point, source) > item["radius"] + 1e-7:
                continue
            if not self._particle_front(
                item["direction"],
                source,
                point,
            ):
                continue
            probability += item["weight"]
        return min(1.0, max(0.0, probability))

    def _extra_anchor_candidates(
        self,
        record: ChannelRecord,
    ) -> list[dict]:
        if not record.bearing_observations:
            return []

        observation = record.bearing_observations[-1]
        result: list[dict] = []

        for step in (200.0, 350.0, 550.0, 750.0):
            for offset in (-30.0, -15.0, 0.0, 15.0, 30.0):
                angle = math.radians(
                    observation["bearing_deg"] + offset
                )
                point = (
                    observation["position"][0] + step * math.cos(angle),
                    observation["position"][1] + step * math.sin(angle),
                )
                if self._point_was_tested(record, point):
                    continue
                result.append({
                    "point": point,
                    "origin": "certified_success_anchor",
                })
        return result

    def _select_information_measurement(
        self,
        record: ChannelRecord,
        *,
        allow_new_q2_seed: bool,
        receive_floor: float,
        move_cap_m: float,
        strict_limits: bool,
        purpose: str,
    ) -> dict:
        sources = sample_source_positions(record, self.cfg)
        hypotheses = self._joint_hypotheses(record, sources)

        q2_seed = (
            self.q2_candidate(record)
            if allow_new_q2_seed
            else record.q2_seed
        )
        estimate, candidates = candidate_points(
            record,
            sources,
            self.client.position,
            self.cfg,
            q2_seed,
        )
        candidates.extend(self._extra_anchor_candidates(record))

        # 去重。
        unique: list[dict] = []
        for candidate in candidates:
            point = tuple(candidate["point"])
            if self._point_was_tested(record, point):
                continue
            if any(
                math.dist(point, old["point"]) <= 1e-6
                for old in unique
            ):
                continue
            unique.append({
                **candidate,
                "point": point,
            })

        scored: list[dict] = []

        for candidate in unique:
            point = candidate["point"]
            travel_distance = math.dist(
                self.client.position,
                point,
            )

            conditional = score_candidate(
                record,
                candidate,
                sources,
                estimate,
                self.client.position,
                self.client.current_channel,
                self.cfg,
            )
            if conditional is None:
                continue

            joint_receive = (
                self._joint_receive_probability(point, hypotheses)
                if hypotheses
                else conditional["robust_receive_score"]
            )

            current_radius = float(record.clearance_radius)
            exploration_value = min(current_radius, 300.0) * 0.08
            completion_bonus = (
                800.0
                if conditional["predicted_clearable"]
                else 0.0
            )

            expected_value = joint_receive * (
                conditional["radius_reduction"]
                + exploration_value
                + completion_bonus
            )
            time_cost = conditional["time_cost_s"]

            information_score = (
                expected_value
                / max(time_cost, 1e-9)
                / (1.0 + 0.30 * conditional["side_penalty"])
            )

            scored.append({
                **conditional,
                "legacy_robust_receive_score": (
                    conditional["robust_receive_score"]
                ),
                # 让基础执行层后续调度器使用联合模型概率。
                "robust_receive_score": joint_receive,
                "joint_receive_probability": joint_receive,
                "information_expected_value_m": expected_value,
                "information_score": information_score,
                "weighted_score": information_score,
                "travel_distance_m": travel_distance,
                "planned_from": tuple(self.client.position),
                "purpose": purpose,
            })

        if not scored:
            raise IncompleteRun(
                f"频道{record.channel_id}没有可评分候选"
            )

        eligible = [
            item for item in scored
            if (
                item["joint_receive_probability"] >= receive_floor
                and item["travel_distance_m"] <= move_cap_m
            )
        ]

        self.counts["active_plan_rejected_by_receive_floor"] += sum(
            item["joint_receive_probability"] < receive_floor
            for item in scored
        )
        self.counts["active_plan_rejected_by_move_cap"] += sum(
            item["travel_distance_m"] > move_cap_m
            for item in scored
        )

        if not eligible and not strict_limits:
            # 主动定位允许有限放宽，但仍避免回到无约束远距离跳跃。
            eligible = [
                item for item in scored
                if (
                    item["joint_receive_probability"]
                    >= max(0.25, receive_floor * 0.65)
                    and item["travel_distance_m"]
                    <= move_cap_m * 1.5
                )
            ]

        if not eligible and strict_limits:
            raise IncompleteRun(
                f"频道{record.channel_id}没有满足恢复接收率/移动上限的候选"
            )

        if not eligible:
            eligible = scored

        selected = max(
            eligible,
            key=lambda item: (
                item["information_score"],
                item["joint_receive_probability"],
                item["predicted_clearable"],
                item["radius_reduction"],
                -item["travel_distance_m"],
                -item["time_cost_s"],
            ),
        )

        return {
            **selected,
            "selection_stage": purpose,
            "candidate_count": len(unique),
            "scored_candidate_count": len(scored),
            "eligible_candidate_count": len(eligible),
            "source_sample_count": len(sources),
            "hypothesis_count": len(hypotheses),
            "estimate": estimate,
            "receive_floor": receive_floor,
            "move_cap_m": move_cap_m,
            "score_scope": (
                "joint_source_orientation_radius_posterior;"
                "no_signal_used_for_planning_only;"
                "strict_geometry_used_for_clear_certificate"
            ),
        }

    def plan_active_measurement(
        self,
        record: ChannelRecord,
        *,
        record_plan: bool = True,
        allow_new_q2_seed: bool = True,
    ) -> dict:
        proposal = self._select_information_measurement(
            record,
            allow_new_q2_seed=allow_new_q2_seed,
            receive_floor=self.tuning.active_receive_floor,
            move_cap_m=self.tuning.active_move_cap_m,
            strict_limits=False,
            purpose="certified_receive_constrained_active",
        )
        if record_plan:
            self._record_active_plan(record, proposal)
        return proposal

    def _build_information_recovery_point(
        self,
        record: ChannelRecord,
    ) -> Point:
        proposal = self._select_information_measurement(
            record,
            allow_new_q2_seed=False,
            receive_floor=self.tuning.recovery_receive_floor,
            move_cap_m=self.tuning.recovery_move_cap_m,
            strict_limits=True,
            purpose="certified_information_recovery",
        )
        self.event(
            "information_recovery_planned",
            channel=record.channel_id,
            point=proposal["point"],
            joint_receive_probability=(
                proposal["joint_receive_probability"]
            ),
            information_score=proposal["information_score"],
            predicted_radius_reduction=proposal["radius_reduction"],
            travel_distance_m=proposal["travel_distance_m"],
        )
        return tuple(proposal["point"])

    def _start_grid_fallback(
        self,
        record: ChannelRecord,
        reason: str,
    ) -> None:
        plan = build_recovery_grid(
            record.outer_polygon,
            self.tuning.recovery_fallback_side_m,
            self.client.position,
        )
        route = [
            point for point in plan.route
            if not self._point_was_tested(record, point)
        ]
        if not route:
            raise IncompleteRun(
                f"频道{record.channel_id}恢复兜底网格没有未检测点"
            )

        self._recovery_queues[record.channel_id] = deque(route)
        record.recovery_active = True
        record.recovery_mode = "coarse_grid_last_resort"
        record.recovery_generation += 1
        record.recovery_total_nodes += len(route)

        self.counts["recovery_grid_build"] += 1
        self.counts["coarse_recovery_build"] += 1
        self.counts["information_recovery_fallback"] += 1

        self.event(
            "recovery_grid_started",
            channel=record.channel_id,
            generation=record.recovery_generation,
            point_count=len(route),
            triangle_count=plan.triangle_count,
            route_distance_m=open_route_distance(
                self.client.position,
                route,
            ),
            side_m=plan.side_m,
            recovery_mode=record.recovery_mode,
            fallback_reason=reason,
        )

    def start_recovery(self, record: ChannelRecord) -> None:
        self._active_plan_cache.pop(record.channel_id, None)

        existing = self._recovery_queues.get(record.channel_id)
        if record.recovery_active and existing:
            return

        attempts = self._information_attempts.get(
            record.channel_id,
            0,
        )

        if attempts >= self.tuning.recovery_information_probe_limit:
            self._start_grid_fallback(
                record,
                "information_probe_limit_reached",
            )
            return

        try:
            point = self._build_information_recovery_point(record)
        except IncompleteRun as error:
            self._start_grid_fallback(
                record,
                f"information_planning_failed: {error}",
            )
            return

        self._recovery_queues[record.channel_id] = deque([point])
        record.recovery_active = True
        record.recovery_mode = "information_reacquire"
        record.recovery_generation += 1
        record.recovery_total_nodes += 1

        self.counts["information_recovery_plan"] += 1
        self.event(
            "recovery_grid_started",
            channel=record.channel_id,
            generation=record.recovery_generation,
            point_count=1,
            triangle_count=0,
            route_distance_m=math.dist(
                self.client.position,
                point,
            ),
            side_m=None,
            recovery_mode=record.recovery_mode,
            information_attempt=attempts + 1,
        )

    def recovery_step(self, record: ChannelRecord) -> str:
        queue = self._recovery_queues.get(record.channel_id)
        if not record.recovery_active or not queue:
            self.start_recovery(record)
            queue = self._recovery_queues[record.channel_id]

        if not queue:
            raise IncompleteRun(
                f"频道{record.channel_id}没有恢复候选"
            )

        mode = record.recovery_mode
        point = queue.popleft()
        record.recovery_visited_nodes += 1

        result = self.measure(
            record,
            point,
            measurement_kind="recovery",
        )

        if mode == "information_reacquire":
            self.counts["information_recovery_measure"] += 1
            if result == "no_signal":
                self._information_attempts[record.channel_id] = (
                    self._information_attempts.get(
                        record.channel_id,
                        0,
                    ) + 1
                )
            else:
                self._information_attempts[record.channel_id] = 0
        elif result != "no_signal":
            self._information_attempts[record.channel_id] = 0

        return result

    def _recovery_scheduler_task(
        self,
        record: ChannelRecord,
    ) -> dict:
        queue = self._recovery_queues.get(record.channel_id)
        if not record.recovery_active or not queue:
            self.start_recovery(record)
        return super()._recovery_scheduler_task(record)

    # ------------------------------------------------------------------
    # 优先级2：缓存失效 + 三步联合路径调度
    # ------------------------------------------------------------------

    def _build_localization_tasks(
        self,
        pending: Sequence[ChannelRecord],
    ) -> list[dict]:
        # 基础执行层只更新缓存方案的移动时间，却不更新候选点本身。
        # 机器人移动较远后必须重新规划候选。
        for channel, proposal in list(
            self._active_plan_cache.items()
        ):
            planned_from = proposal.get("planned_from")
            if planned_from is None:
                self._active_plan_cache.pop(channel, None)
                continue
            if (
                math.dist(self.client.position, planned_from)
                > self.tuning.active_cache_replan_distance_m
            ):
                self._active_plan_cache.pop(channel, None)

        return super()._build_localization_tasks(pending)

    def _future_task_cost(
        self,
        task: dict,
        position: Point,
        current_channel: int | None,
    ) -> tuple[float, int | None]:
        point = tuple(task["point"])
        travel_s = math.dist(position, point) / self.cfg.dog_speed_m_per_s

        if task["action_type"] == "clear":
            return (
                travel_s
                + self.cfg.optical_time_s
                + self.cfg.clearance_time_s,
                current_channel,
            )

        switch_s = (
            self.cfg.channel_switch_time_s
            if current_channel != task["record"].channel_id
            else 0.0
        )
        return (
            travel_s + self.cfg.detection_time_s + switch_s,
            task["record"].channel_id,
        )

    def _select_horizon_task(
        self,
        tasks: Sequence[dict],
    ) -> tuple[dict, list[dict]]:
        candidates = sorted(
            tasks,
            key=self._scheduler_selection_key,
            reverse=True,
        )[:self.tuning.scheduler_task_limit]

        horizon = min(
            self.tuning.scheduler_horizon,
            len(candidates),
        )

        states = [{
            "sequence": [],
            "used": frozenset(),
            "position": tuple(self.client.position),
            "channel": self.client.current_channel,
            "value": 0.0,
            "time": 0.0,
        }]

        for _ in range(horizon):
            expanded = []

            for state in states:
                for index, task in enumerate(candidates):
                    if index in state["used"]:
                        continue

                    cost, next_channel = self._future_task_cost(
                        task,
                        state["position"],
                        state["channel"],
                    )

                    value = max(
                        0.0,
                        float(task["expected_value_m"]),
                    )
                    value *= self._scheduler_age_factor(
                        task["record"]
                    )

                    if state["sequence"]:
                        distance = math.dist(
                            state["position"],
                            task["point"],
                        )
                        if distance <= self.tuning.scheduler_cluster_radius_m:
                            value *= (
                                1.0
                                + self.tuning.scheduler_cluster_bonus
                            )

                    expanded.append({
                        "sequence": [
                            *state["sequence"],
                            task,
                        ],
                        "used": state["used"] | {index},
                        "position": tuple(task["point"]),
                        "channel": next_channel,
                        "value": state["value"] + value,
                        "time": state["time"] + cost,
                    })

            if not expanded:
                break

            expanded.sort(
                key=lambda state: (
                    state["value"] / max(state["time"], 1e-9),
                    state["value"],
                    -state["time"],
                ),
                reverse=True,
            )
            states = expanded[:self.tuning.scheduler_beam_width]

        best_state = max(
            states,
            key=lambda state: (
                state["value"] / max(state["time"], 1e-9),
                state["value"],
                -state["time"],
            ),
        )
        selected = best_state["sequence"][0]

        # 频道驻留：当前频道下一动作仍有接近最优的价值时，
        # 避免为了小幅分数差跨越地图。
        if self._last_selected_channel is not None:
            stay_tasks = [
                task for task in tasks
                if (
                    task["record"].channel_id
                    == self._last_selected_channel
                )
            ]
            if stay_tasks:
                stay = max(
                    stay_tasks,
                    key=self._scheduler_selection_key,
                )
                ratio = (
                    self.tuning.scheduler_recovery_commit_ratio
                    if stay["action_type"] == "recovery"
                    else self.tuning.scheduler_commit_ratio
                )
                if (
                    stay["scheduler_score"]
                    >= ratio * selected["scheduler_score"]
                ):
                    selected = stay
                    self.counts["scheduler_commit_selected"] += 1

        return selected, best_state["sequence"]

    def _localize_value_time(self) -> None:
        self.phase = "localization"

        while True:
            pending = [
                record for record in self.records.values()
                if record.status in {"FOUND", "READY"}
            ]
            if not pending:
                return

            self.check_budget()
            tasks = self._build_localization_tasks(pending)
            if not tasks:
                raise IncompleteRun(
                    "认证搜索层联合调度器没有生成可执行任务"
                )

            selected_task, planned_sequence = (
                self._select_horizon_task(tasks)
            )
            selected = selected_task["record"]

            self.counts["scheduler_round"] += 1
            self.counts["scheduler_horizon_round"] += 1

            self.event(
                "scheduler_target_selected",
                channel=selected.channel_id,
                status=selected.status,
                action_type=selected_task["action_type"],
                task_point=selected_task["point"],
                task_point_is_exact=(
                    selected_task["point_is_exact"]
                ),
                scheduler_score=selected_task["scheduler_score"],
                base_score=selected_task["base_score"],
                expected_value_m=selected_task["expected_value_m"],
                time_cost_s=selected_task["time_cost_s"],
                travel_distance_m=(
                    selected_task["travel_distance_m"]
                ),
                predicted_clearable=(
                    selected_task["predicted_clearable"]
                ),
                scheduler_wait_count=(
                    selected.scheduler_wait_count
                ),
                localization_turn_count=(
                    selected.localization_turn_count
                ),
                selection_basis=(
                    "certified_three_step_beam_search_with_channel_commit"
                ),
                planned_sequence=[
                    {
                        "channel": task["record"].channel_id,
                        "action_type": task["action_type"],
                        "point": task["point"],
                    }
                    for task in planned_sequence
                ],
            )

            self._execute_localization_task(selected_task)
            self._last_selected_channel = selected.channel_id
            selected.localization_turn_count += 1

            for record in pending:
                if record.channel_id == selected.channel_id:
                    record.scheduler_wait_count = 0
                elif record.status in {"FOUND", "READY"}:
                    record.scheduler_wait_count += 1

    def summary(
        self,
        outcome: str,
        reason: str,
        exit_confirmed: bool,
    ) -> dict:
        result = super().summary(
            outcome,
            reason,
            exit_confirmed,
        )
        result["global_grid"] = {
            **self.dual_ring_diagnostics,
            "proof": (
                "1800米目标圆被外环正多边形完整包含；"
                "双环三角剖分最大边不超过1000米；"
                "任意过源的定向前半平面至少包含所在"
                "证书三角形的一个顶点"
            ),
        }
        result["certified_tuning"] = asdict(self.tuning)
        result["model_note"] = (
            "双环认证搜索；no_signal仅更新规划粒子，"
            "不裁剪连续位置证书；信息重捕获失败后才使用"
            "500米网格；定位阶段采用短视野联合路线调度"
        )
        return result


def _run_certified_search_self_test() -> dict:
    config = Q4Config(
        global_grid_side_m=1000.0,
        enable_q2_seed=False,
        min_request_interval_s=0.0,
    )
    tuning = _CertifiedSearchTuning()
    plan, diagnostics = build_dual_ring_search(
        config,
        tuning,
    )

    assert len(plan.points) == 25
    assert len(plan.triangles) == 36
    assert (
        diagnostics["max_certificate_edge_m"]
        <= config.reception_radius_min_m + 1e-7
    )
    assert (
        diagnostics["outer_boundary_min_distance_m"]
        >= config.target_radius_m
    )

    # 随机检查位置、距离和任意发射朝向。
    rng = random.Random(20260912)
    for _ in range(5000):
        radius = (
            config.target_radius_m
            * math.sqrt(rng.random())
        )
        angle = 2.0 * math.pi * rng.random()
        source = (
            radius * math.cos(angle),
            radius * math.sin(angle),
        )

        containing = [
            triangle for triangle in plan.triangles
            if point_in_triangle(source, triangle)
        ]
        assert containing, source

        direction_angle = 2.0 * math.pi * rng.random()
        direction = (
            math.cos(direction_angle),
            math.sin(direction_angle),
        )

        assert any(
            all(
                math.dist(source, vertex)
                <= config.reception_radius_min_m + 1e-6
                for vertex in triangle
            )
            and any(
                direction[0] * (vertex[0] - source[0])
                + direction[1] * (vertex[1] - source[1])
                >= -1e-7
                for vertex in triangle
            )
            for triangle in containing
        )

    return {
        "status": "ok",
        **diagnostics,
        "random_directional_coverage_cases": 5000,
    }

@dataclass(frozen=True)
class _AdaptiveSearchTuning(_CertifiedSearchTuning):
    # 优先级3：动态认证节点排序
    dynamic_route_candidate_limit: int = 8
    dynamic_route_found_limit: int = 6
    dynamic_route_regret_ratio: float = 0.03
    dynamic_route_regret_floor_m: float = 180.0
    dynamic_route_regret_penalty: float = 0.25
    dynamic_ready_bonus_m: float = 500.0

    # 优先级4：顺路清除
    search_inline_clear_radius_m: float = 180.0
    localization_inline_clear_radius_m: float = 300.0

    # 优先级5：机会补测
    opportunistic_receive_floor: float = 0.48
    opportunistic_max_per_node: int = 3
    opportunistic_max_per_channel: int = 5
    opportunistic_min_reduction_m: float = 0.50
    opportunistic_min_relative_reduction: float = 0.005
    opportunistic_ready_bonus_m: float = 800.0

    # 主动定位自适应约束
    adaptive_receive_penalty: float = 0.18
    adaptive_receive_ceiling: float = 0.82
    adaptive_active_min_move_cap_m: float = 450.0
    adaptive_recovery_min_move_cap_m: float = 350.0

    def __post_init__(self):
        super().__post_init__()

        for name in (
            "dynamic_route_candidate_limit",
            "dynamic_route_found_limit",
            "opportunistic_max_per_node",
            "opportunistic_max_per_channel",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正整数")

        for name in (
            "dynamic_route_regret_ratio",
            "opportunistic_receive_floor",
            "opportunistic_min_relative_reduction",
            "adaptive_receive_penalty",
            "adaptive_receive_ceiling",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0,1]")


class _AdaptiveSearchRunner(_CertifiedSearchRunner):
    def __init__(
        self,
        client,
        *,
        config: Q4Config,
        tuning: _AdaptiveSearchTuning | None = None,
        planner_config=None,
        geometry_config=None,
        output_dir: Path | None = None,
    ):
        self.tuning = tuning or _AdaptiveSearchTuning()

        super().__init__(
            client,
            config=config,
            tuning=self.tuning,
            planner_config=planner_config,
            geometry_config=geometry_config,
            output_dir=output_dir,
        )

        self.dynamic_search_visit_order: list[int] = []
        self.dynamic_search_route_distance_m = 0.0

        self.counts.update({
            "dynamic_search_node_selection": 0,
            "dynamic_search_localization_aware_selection": 0,
            "dynamic_search_total_regret_m": 0.0,
            "adaptive_opportunistic_planned": 0,
            "adaptive_inline_clear_search": 0,
            "adaptive_inline_clear_localization": 0,
            "adaptive_active_plan": 0,
            "adaptive_recovery_plan": 0,
        })

    # ================================================================
    # 优先级5：根据历史成功率动态调整接收概率下限
    # ================================================================

    def _adaptive_receive_floor(
        self,
        record: ChannelRecord,
        base_floor: float,
    ) -> float:
        relevant = [
            item for item in record.progress_history
            if item.get("measurement_kind") in {
                "active",
                "recovery",
                "opportunistic",
            }
        ]
        if not relevant:
            return base_floor

        failures = sum(
            item.get("result") == "no_signal"
            for item in relevant
        )
        failure_rate = failures / len(relevant)

        return min(
            self.tuning.adaptive_receive_ceiling,
            base_floor
            + self.tuning.adaptive_receive_penalty * failure_rate,
        )

    def _adaptive_move_cap(
        self,
        record: ChannelRecord,
        maximum: float,
        minimum: float,
    ) -> float:
        radius = record.clearance_radius
        if radius is None:
            return maximum

        # 范围已经较小时，避免为了少量几何收益再次远距离移动。
        return min(
            maximum,
            max(minimum, 0.80 * float(radius)),
        )

    def plan_active_measurement(
        self,
        record: ChannelRecord,
        *,
        record_plan: bool = True,
        allow_new_q2_seed: bool = True,
    ) -> dict:
        receive_floor = self._adaptive_receive_floor(
            record,
            self.tuning.active_receive_floor,
        )
        move_cap = self._adaptive_move_cap(
            record,
            self.tuning.active_move_cap_m,
            self.tuning.adaptive_active_min_move_cap_m,
        )

        proposal = self._select_information_measurement(
            record,
            allow_new_q2_seed=allow_new_q2_seed,
            receive_floor=receive_floor,
            move_cap_m=move_cap,
            strict_limits=False,
            purpose="adaptive_receive_constrained_active",
        )
        proposal["adaptive_receive_floor"] = receive_floor
        proposal["adaptive_move_cap_m"] = move_cap

        self.counts["adaptive_active_plan"] += 1
        if record_plan:
            self._record_active_plan(record, proposal)
        return proposal

    def _build_information_recovery_point(
        self,
        record: ChannelRecord,
    ) -> Point:
        receive_floor = self._adaptive_receive_floor(
            record,
            self.tuning.recovery_receive_floor,
        )
        move_cap = self._adaptive_move_cap(
            record,
            self.tuning.recovery_move_cap_m,
            self.tuning.adaptive_recovery_min_move_cap_m,
        )

        proposal = self._select_information_measurement(
            record,
            allow_new_q2_seed=False,
            receive_floor=receive_floor,
            move_cap_m=move_cap,
            strict_limits=True,
            purpose="adaptive_information_recovery",
        )

        self.counts["adaptive_recovery_plan"] += 1
        self.event(
            "information_recovery_planned",
            channel=record.channel_id,
            point=proposal["point"],
            joint_receive_probability=(
                proposal["joint_receive_probability"]
            ),
            information_score=proposal["information_score"],
            predicted_radius_reduction=proposal["radius_reduction"],
            travel_distance_m=proposal["travel_distance_m"],
            adaptive_receive_floor=receive_floor,
            adaptive_move_cap_m=move_cap,
        )
        return tuple(proposal["point"])

    # ================================================================
    # 优先级3：评价认证节点对所有FOUND源的定位价值
    # ================================================================

    @staticmethod
    def _line_separation_value(
        new_angle_rad: float,
        old_angle_rad: float,
    ) -> float:
        # bearing是一条射线，但定位交叉角以模π后的正弦衡量。
        return abs(math.sin(new_angle_rad - old_angle_rad))

    def _lightweight_node_localization_value(
        self,
        record: ChannelRecord,
        point: Point,
    ) -> float:
        if record.status != "FOUND":
            return 0.0
        if self._point_was_tested(record, point):
            return 0.0
        if not record.bearing_observations:
            return 0.0

        try:
            sources = sample_source_positions(
                record,
                self.cfg,
            )
        except IncompleteRun:
            return 0.0

        hypotheses = self._joint_hypotheses(
            record,
            sources,
        )
        receive_probability = (
            self._joint_receive_probability(
                point,
                hypotheses,
            )
            if hypotheses
            else 0.35
        )

        floor = self._adaptive_receive_floor(
            record,
            self.tuning.opportunistic_receive_floor,
        )
        if receive_probability < floor:
            return 0.0

        old_angles = [
            math.radians(item["bearing_deg"])
            for item in record.bearing_observations
        ]

        separation_values = []
        for source in sources:
            new_angle = math.atan2(
                source[1] - point[1],
                source[0] - point[0],
            )
            separation_values.append(
                max(
                    self._line_separation_value(
                        new_angle,
                        old_angle,
                    )
                    for old_angle in old_angles
                )
            )

        if not separation_values:
            return 0.0

        average_separation = (
            sum(separation_values)
            / len(separation_values)
        )
        radius = float(record.clearance_radius or 0.0)

        value = (
            receive_probability
            * average_separation
            * min(radius, 1000.0)
        )

        if radius <= 120.0:
            value += (
                self.tuning.dynamic_ready_bonus_m
                * receive_probability
                * average_separation
            )
        return value

    def _node_total_localization_value(
        self,
        point: Point,
    ) -> float:
        found = [
            record for record in self.records.values()
            if record.status == "FOUND"
        ]
        found.sort(
            key=lambda record: (
                record.clearance_radius
                if record.clearance_radius is not None
                else math.inf,
                -len(record.bearing_observations),
                record.channel_id,
            )
        )

        return sum(
            self._lightweight_node_localization_value(
                record,
                point,
            )
            for record in found[
                :self.tuning.dynamic_route_found_limit
            ]
        )

    def _nearest_neighbor_tail_distance(
        self,
        start: Point,
        indices: Sequence[int],
    ) -> float:
        if not indices:
            return 0.0

        route = nearest_neighbor_route(
            [self.search_points[index] for index in indices],
            start,
        )
        return open_route_distance(start, route)

    def _select_next_search_node(
        self,
        remaining: set[int],
    ) -> tuple[int, dict]:
        current = tuple(self.client.position)

        ordered_by_distance = sorted(
            remaining,
            key=lambda index: (
                math.dist(current, self.search_points[index]),
                index,
            ),
        )
        candidates = ordered_by_distance[
            :self.tuning.dynamic_route_candidate_limit
        ]

        baseline_index = candidates[0]
        baseline_tail = self._nearest_neighbor_tail_distance(
            current,
            list(remaining),
        )

        allowed_regret = max(
            self.tuning.dynamic_route_regret_floor_m,
            self.tuning.dynamic_route_regret_ratio
            * baseline_tail,
        )

        evaluations = []
        for index in candidates:
            point = self.search_points[index]
            tail_indices = [
                other for other in remaining
                if other != index
            ]
            route_distance = (
                math.dist(current, point)
                + self._nearest_neighbor_tail_distance(
                    point,
                    tail_indices,
                )
            )
            regret = max(
                0.0,
                route_distance - baseline_tail,
            )
            if regret > allowed_regret + 1e-7:
                continue

            localization_value = (
                self._node_total_localization_value(point)
            )
            combined_score = (
                localization_value
                - self.tuning.dynamic_route_regret_penalty
                * regret
            )

            evaluations.append({
                "index": index,
                "point": point,
                "route_distance_m": route_distance,
                "route_regret_m": regret,
                "localization_value": localization_value,
                "combined_score": combined_score,
            })

        if not evaluations:
            evaluations.append({
                "index": baseline_index,
                "point": self.search_points[baseline_index],
                "route_distance_m": baseline_tail,
                "route_regret_m": 0.0,
                "localization_value": 0.0,
                "combined_score": 0.0,
            })

        selected = max(
            evaluations,
            key=lambda item: (
                item["combined_score"],
                item["localization_value"],
                -item["route_distance_m"],
                -item["index"],
            ),
        )

        if selected["localization_value"] > 0.0:
            self.counts[
                "dynamic_search_localization_aware_selection"
            ] += 1

        self.counts["dynamic_search_node_selection"] += 1
        self.counts["dynamic_search_total_regret_m"] += (
            selected["route_regret_m"]
        )
        return selected["index"], selected

    # ================================================================
    # 优先级5：使用联合粒子重新评价机会补测
    # ================================================================

    def _score_adaptive_opportunity(
        self,
        record: ChannelRecord,
        point: Point,
    ) -> dict | None:
        if record.status != "FOUND":
            return None
        if self._point_was_tested(record, point):
            return None
        if (
            record.opportunistic_measure_count
            >= self.tuning.opportunistic_max_per_channel
        ):
            return None

        try:
            sources = sample_source_positions(
                record,
                self.cfg,
            )
        except IncompleteRun:
            return None

        estimate = (
            sum(item[0] for item in sources) / len(sources),
            sum(item[1] for item in sources) / len(sources),
        )
        candidate = {
            "point": tuple(point),
            "origin": "adaptive_dynamic_certified_node",
        }

        conditional = score_candidate(
            record,
            candidate,
            sources,
            estimate,
            tuple(point),
            self.client.current_channel,
            self.cfg,
        )
        self.counts["opportunistic_evaluation"] += 1

        if conditional is None:
            return None

        hypotheses = self._joint_hypotheses(
            record,
            sources,
        )
        receive_probability = (
            self._joint_receive_probability(
                point,
                hypotheses,
            )
            if hypotheses
            else conditional["robust_receive_score"]
        )

        receive_floor = self._adaptive_receive_floor(
            record,
            self.tuning.opportunistic_receive_floor,
        )
        minimum_reduction = max(
            self.tuning.opportunistic_min_reduction_m,
            self.tuning.opportunistic_min_relative_reduction
            * float(record.clearance_radius),
        )

        if receive_probability < receive_floor:
            return None
        if (
            not conditional["predicted_clearable"]
            and conditional["radius_reduction"]
            < minimum_reduction
        ):
            return None

        incremental_time = (
            self.cfg.detection_time_s
            + (
                self.cfg.channel_switch_time_s
                if self.client.current_channel
                != record.channel_id
                else 0.0
            )
        )
        expected_value = receive_probability * (
            conditional["radius_reduction"]
            + (
                self.tuning.opportunistic_ready_bonus_m
                if conditional["predicted_clearable"]
                else 0.0
            )
        )
        opportunity_score = (
            expected_value
            / max(incremental_time, 1e-9)
            / (1.0 + 0.25 * conditional["side_penalty"])
        )

        return {
            **conditional,
            "channel": record.channel_id,
            "legacy_robust_receive_score": (
                conditional["robust_receive_score"]
            ),
            "robust_receive_score": receive_probability,
            "joint_receive_probability": receive_probability,
            "receive_floor": receive_floor,
            "expected_value_m": expected_value,
            "opportunity_score": opportunity_score,
            "effective_incremental_time_s": incremental_time,
            "minimum_required_reduction_m": minimum_reduction,
        }

    def plan_opportunistic_measurements(
        self,
        global_index: int,
        point: Point,
    ) -> list[dict]:
        if not self.cfg.enable_parallel_search_localization:
            return []

        proposals = []
        for record in self.records.values():
            proposal = self._score_adaptive_opportunity(
                record,
                point,
            )
            if proposal is not None:
                proposals.append(proposal)

        selected = sorted(
            proposals,
            key=lambda item: (
                item["predicted_clearable"],
                item["opportunity_score"],
                item["joint_receive_probability"],
                item["radius_reduction"],
                -item["channel"],
            ),
            reverse=True,
        )[:self.tuning.opportunistic_max_per_node]

        if selected:
            self.counts["adaptive_opportunistic_planned"] += len(
                selected
            )
            self.event(
                "opportunistic_batch_planned",
                global_index=global_index,
                point=point,
                strategy="adaptive_joint_posterior",
                candidate_channel_count=len(proposals),
                selected=[
                    {
                        "channel": item["channel"],
                        "opportunity_score": (
                            item["opportunity_score"]
                        ),
                        "joint_receive_probability": (
                            item[
                                "joint_receive_probability"
                            ]
                        ),
                        "radius_reduction": (
                            item["radius_reduction"]
                        ),
                        "predicted_clearable": (
                            item["predicted_clearable"]
                        ),
                    }
                    for item in selected
                ],
            )
        return selected

    # ================================================================
    # 优先级4：当前位置附近的READY源立即顺路清除
    # ================================================================

    def _clear_ready_nearby(
        self,
        maximum_distance_m: float,
        phase_name: str,
    ) -> int:
        cleared = 0

        while True:
            nearby = [
                record for record in self.records.values()
                if (
                    record.status == "READY"
                    and record.clearance_center is not None
                    and math.dist(
                        self.client.position,
                        record.clearance_center,
                    ) <= maximum_distance_m
                )
            ]
            if not nearby:
                return cleared

            selected = min(
                nearby,
                key=lambda record: (
                    math.dist(
                        self.client.position,
                        record.clearance_center,
                    ),
                    record.channel_id,
                ),
            )
            if not self.clear(selected):
                return cleared

            cleared += 1
            counter = (
                "adaptive_inline_clear_search"
                if phase_name == "search"
                else "adaptive_inline_clear_localization"
            )
            self.counts[counter] += 1
            self.event(
                "adaptive_inline_clear_completed",
                channel=selected.channel_id,
                source_phase=phase_name,
                maximum_distance_m=maximum_distance_m,
            )

    def _execute_localization_task(self, task: dict) -> None:
        super()._execute_localization_task(task)

        self._clear_ready_nearby(
            self.tuning.localization_inline_clear_radius_m,
            "localization",
        )

    # ================================================================
    # 优先级3：动态搜索主循环
    # ================================================================

    def search(self) -> None:
        self.phase = "global_search"
        remaining = set(range(len(self.search_points)))

        while remaining:
            unknown = [
                channel
                for channel, record in self.records.items()
                if record.status == "UNKNOWN"
            ]
            if not unknown:
                break

            index, diagnostics = (
                self._select_next_search_node(remaining)
            )
            point = self.search_points[index]
            remaining.remove(index)

            self.dynamic_search_route_distance_m += math.dist(
                self.client.position,
                point,
            )
            self.dynamic_search_visit_order.append(index)
            self.next_search_index = len(
                self.dynamic_search_visit_order
            )

            self.event(
                "dynamic_search_node_selected",
                global_index=index,
                visit_number=self.next_search_index,
                point=point,
                remaining_node_count=len(remaining),
                route_regret_m=(
                    diagnostics["route_regret_m"]
                ),
                localization_value=(
                    diagnostics["localization_value"]
                ),
                combined_score=(
                    diagnostics["combined_score"]
                ),
            )

            current_channel = self.client.current_channel
            order = (
                [current_channel]
                if current_channel in unknown
                else []
            )
            order.extend(
                channel for channel in unknown
                if channel != current_channel
            )

            for channel in order:
                record = self.records[channel]
                result = self.measure(
                    record,
                    point,
                    measurement_kind="unknown",
                    global_index=index,
                )

                if result == "near":
                    self.clear(record)

                if (
                    self.discovered_count
                    >= self.cfg.source_count_max
                ):
                    self.mark_absent()
                    break

            # 使用刚刚更新后的FOUND集合，在当前位置进行高命中补测。
            self.opportunistic_localization_at_waypoint(
                index,
                point,
            )

            self._clear_ready_nearby(
                self.tuning.search_inline_clear_radius_m,
                "search",
            )

            if (
                self.discovered_count
                >= self.cfg.source_count_max
            ):
                break

        self.search_complete = not remaining
        self.mark_absent()

        if (
            not self.search_complete
            and self.discovered_count
            < self.cfg.source_count_max
        ):
            raise IncompleteRun(
                "动态认证搜索未完成，不能判定剩余频道不存在"
            )

    def summary(
        self,
        outcome: str,
        reason: str,
        exit_confirmed: bool,
    ) -> dict:
        result = super().summary(
            outcome,
            reason,
            exit_confirmed,
        )

        result["adaptive_tuning"] = asdict(self.tuning)
        result["dynamic_search"] = {
            "visit_order": self.dynamic_search_visit_order,
            "visited_node_count": len(
                self.dynamic_search_visit_order
            ),
            "actual_route_distance_m": (
                self.dynamic_search_route_distance_m
            ),
            "certificate_preserved": (
                self.search_complete
                or self.discovered_count
                >= self.cfg.source_count_max
            ),
        }
        result["model_note"] = (
            "认证搜索层双环证书、信息恢复和联合调度保持不变；"
            "自适应搜索层只动态重排尚未访问的认证节点，不删除节点；"
            "机会补测使用联合位置-朝向-半径概率；"
            "附近READY源顺路清除；主动测量门槛根据历史"
            "no_signal比例自适应"
        )
        return result


def _run_adaptive_search_self_test() -> dict:
    result = _run_certified_search_self_test()
    tuning = _AdaptiveSearchTuning()

    assert tuning.opportunistic_max_per_node > 0
    assert tuning.opportunistic_max_per_channel > 0
    assert (
        tuning.search_inline_clear_radius_m
        < tuning.localization_inline_clear_radius_m
    )

    return {
        **result,
        "adaptive_status": "ok",
        "adaptive_dynamic_route": True,
        "adaptive_inline_clear": True,
        "adaptive_measurement": True,
    }

@dataclass(frozen=True)
class _PrioritySearchTuning(_AdaptiveSearchTuning):
    # ================================================================
    # 优先级1：UNKNOWN预期发现收益
    # ================================================================

    # 自适应搜索层 默认为8；增至12，但仍受路线后悔值约束。
    dynamic_route_candidate_limit: int = 12

    # UNKNOWN 源的确定性准蒙特卡洛位置、朝向样本数。
    unknown_position_samples: int = 72
    unknown_orientation_samples: int = 12

    # 未知源为定向源的规划先验。
    # 该参数只影响节点顺序，不参与不存在证明。
    unknown_directional_prior: float = 0.50

    # 发现一个频道后，省去其后续UNKNOWN测量的价值系数。
    unknown_discovery_value_scale: float = 0.40

    # 接近16个源时，对“提前结束整个认证搜索”的奖励权重。
    unknown_early_stop_weight: float = 0.45

    # 防止概率启发式完全压过路线距离约束。
    unknown_value_cap_m: float = 3200.0

    # ================================================================
    # 优先级2：READY源全局清除路线
    # ================================================================

    ready_route_start_limit: int = 16
    ready_route_two_opt_passes: int = 10

    # 全局路线允许首个清除点比最近点多出的最大首段距离。
    ready_route_first_leg_slack_m: float = 350.0

    # ================================================================
    # 优先级3：基于插入代价的顺路清除
    # ================================================================

    # dist(A,C)+dist(C,B)-dist(A,B) 的允许上限。
    search_inline_detour_cap_m: float = 240.0
    localization_inline_detour_cap_m: float = 420.0

    # 即使插入代价很小，也避免跨越过远距离直接清除。
    inline_max_direct_distance_m: float = 1800.0

    # 一次搜索/定位动作结束后最多连续顺路清除的数量。
    inline_clear_batch_limit: int = 3

    def __post_init__(self):
        super().__post_init__()

        for name in (
            "unknown_position_samples",
            "unknown_orientation_samples",
            "ready_route_start_limit",
            "ready_route_two_opt_passes",
            "inline_clear_batch_limit",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} 必须是整数")
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正整数")

        for name in (
            "unknown_directional_prior",
            "unknown_discovery_value_scale",
            "unknown_early_stop_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} 必须是数值")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} 必须位于[0,1]")

        for name in (
            "unknown_value_cap_m",
            "ready_route_first_leg_slack_m",
            "search_inline_detour_cap_m",
            "localization_inline_detour_cap_m",
            "inline_max_direct_distance_m",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} 必须是数值")
            if not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} 必须是有限正数")


class _PrioritySearchRunner(_AdaptiveSearchRunner):
    def __init__(
        self,
        client,
        *,
        config: Q4Config,
        tuning: _PrioritySearchTuning | None = None,
        planner_config=None,
        geometry_config=None,
        output_dir: Path | None = None,
    ):
        self.tuning = tuning or _PrioritySearchTuning()

        super().__init__(
            client,
            config=config,
            tuning=self.tuning,
            planner_config=planner_config,
            geometry_config=geometry_config,
            output_dir=output_dir,
        )

        # UNKNOWN概率评价缓存。
        self._unknown_probability_cache: dict[
            tuple[int, int],
            float,
        ] = {}
        self._unknown_omni_masks: list[int] | None = None
        self._unknown_directional_masks: list[int] | None = None

        # 顺路清除可以指定清除后必须访问的认证节点。
        self._forced_search_index: int | None = None

        self.counts.update({
            "priority_unknown_aware_selection": 0,
            "priority_unknown_value_selected_m": 0.0,
            "priority_expected_unknown_hits_selected": 0.0,
            "priority_early_stop_probability_sum": 0.0,
            "priority_forced_search_anchor": 0,
            "priority_ready_route_plan": 0,
            "priority_ready_route_first_override": 0,
            "priority_ready_route_improvement_m": 0.0,
            "priority_inline_clear_search": 0,
            "priority_inline_clear_localization": 0,
            "priority_inline_rejected_by_detour": 0,
        })

    # ================================================================
    # 优先级1：UNKNOWN源的后验接收概率
    # ================================================================

    def _build_unknown_signal_masks(self) -> None:
        """预计算每个未知源粒子能在哪些认证节点被接收。"""
        if self._unknown_omni_masks is not None:
            return

        position_count = self.tuning.unknown_position_samples
        orientation_count = self.tuning.unknown_orientation_samples

        positions: list[Point] = [(0.0, 0.0)]

        remaining_count = max(1, position_count - 1)
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))

        for index in range(remaining_count):
            fraction = (index + 0.5) / remaining_count
            radius = (
                self.cfg.target_radius_m
                * math.sqrt(fraction)
            )
            angle = index * golden_angle
            positions.append((
                radius * math.cos(angle),
                radius * math.sin(angle),
            ))

        if len(positions) > position_count:
            positions = positions[:position_count]

        radius_min = self.cfg.reception_radius_min_m
        radius_max = self.cfg.reception_radius_max_m
        reception_radii = (
            radius_min,
            0.5 * (radius_min + radius_max),
            radius_max,
        )

        orientations = [
            (
                math.cos(2.0 * math.pi * index / orientation_count),
                math.sin(2.0 * math.pi * index / orientation_count),
            )
            for index in range(orientation_count)
        ]

        omni_masks: list[int] = []
        directional_masks: list[int] = []

        for source in positions:
            for reception_radius in reception_radii:
                omni_mask = 0

                for node_index, receiver in enumerate(
                    self.search_points
                ):
                    if (
                        math.dist(source, receiver)
                        <= reception_radius + 1e-9
                    ):
                        omni_mask |= 1 << node_index

                omni_masks.append(omni_mask)

                for direction in orientations:
                    directional_mask = 0

                    for node_index, receiver in enumerate(
                        self.search_points
                    ):
                        if (
                            math.dist(source, receiver)
                            > reception_radius + 1e-9
                        ):
                            continue

                        front = (
                            direction[0]
                            * (receiver[0] - source[0])
                            + direction[1]
                            * (receiver[1] - source[1])
                            >= -1e-9
                        )
                        if front:
                            directional_mask |= 1 << node_index

                    directional_masks.append(
                        directional_mask
                    )

        self._unknown_omni_masks = omni_masks
        self._unknown_directional_masks = (
            directional_masks
        )

    @staticmethod
    def _tested_node_mask(
        tested_indices: Sequence[int] | set[int],
    ) -> int:
        mask = 0
        for index in tested_indices:
            mask |= 1 << int(index)
        return mask

    @staticmethod
    def _posterior_class_statistics(
        signal_masks: Sequence[int],
        tested_mask: int,
        candidate_bit: int,
    ) -> tuple[float, float]:
        """返回P(历史无信号)和P(历史无信号且候选点有信号)。"""
        total = len(signal_masks)
        if total <= 0:
            return 0.0, 0.0

        survivor_count = 0
        candidate_hit_count = 0

        for mask in signal_masks:
            if mask & tested_mask:
                continue

            survivor_count += 1
            if mask & candidate_bit:
                candidate_hit_count += 1

        return (
            survivor_count / total,
            candidate_hit_count / total,
        )

    def _unknown_receive_probability(
        self,
        record: ChannelRecord,
        candidate_index: int,
    ) -> float:
        """计算在历史UNKNOWN无信号条件下，候选节点的接收概率。"""
        self._build_unknown_signal_masks()

        tested_mask = self._tested_node_mask(
            record.global_nodes_tested
        )
        cache_key = (tested_mask, candidate_index)

        cached = self._unknown_probability_cache.get(
            cache_key
        )
        if cached is not None:
            return cached

        candidate_bit = 1 << candidate_index

        omni_survival, omni_joint_hit = (
            self._posterior_class_statistics(
                self._unknown_omni_masks or [],
                tested_mask,
                candidate_bit,
            )
        )
        directional_survival, directional_joint_hit = (
            self._posterior_class_statistics(
                self._unknown_directional_masks or [],
                tested_mask,
                candidate_bit,
            )
        )

        directional_prior = (
            self.tuning.unknown_directional_prior
        )
        omni_prior = 1.0 - directional_prior

        posterior_denominator = (
            omni_prior * omni_survival
            + directional_prior * directional_survival
        )
        posterior_numerator = (
            omni_prior * omni_joint_hit
            + directional_prior * directional_joint_hit
        )

        if posterior_denominator <= 1e-12:
            probability = 0.0
        else:
            probability = (
                posterior_numerator
                / posterior_denominator
            )

        probability = min(1.0, max(0.0, probability))
        self._unknown_probability_cache[
            cache_key
        ] = probability
        return probability

    @staticmethod
    def _poisson_binomial_tail(
        probabilities: Sequence[float],
        required_hits: int,
    ) -> float:
        """独立伯努利近似下，至少命中required_hits个频道的概率。"""
        if required_hits <= 0:
            return 1.0
        if required_hits > len(probabilities):
            return 0.0

        distribution = [1.0] + [0.0] * len(
            probabilities
        )

        processed = 0
        for raw_probability in probabilities:
            probability = min(
                1.0,
                max(0.0, float(raw_probability)),
            )
            processed += 1

            for hit_count in range(
                processed,
                0,
                -1,
            ):
                distribution[hit_count] = (
                    distribution[hit_count]
                    * (1.0 - probability)
                    + distribution[hit_count - 1]
                    * probability
                )

            distribution[0] *= 1.0 - probability

        return min(
            1.0,
            max(
                0.0,
                sum(distribution[required_hits:]),
            ),
        )

    def _unknown_node_value(
        self,
        candidate_index: int,
        *,
        future_node_count: int,
        early_stop_value_m: float,
    ) -> dict:
        unknown_records = [
            record
            for record in self.records.values()
            if record.status == "UNKNOWN"
        ]

        if not unknown_records:
            return {
                "value_m": 0.0,
                "expected_hits": 0.0,
                "early_stop_probability": 0.0,
            }

        maximum_sources_remaining = max(
            0,
            self.cfg.source_count_max
            - self.discovered_count,
        )

        # 源存在概率只作为排序先验。即使该先验不准确，
        # 也不会导致节点被删除或UNKNOWN被错误判为ABSENT。
        source_presence_prior = min(
            1.0,
            maximum_sources_remaining
            / max(1, len(unknown_records)),
        )

        detection_probabilities = [
            source_presence_prior
            * self._unknown_receive_probability(
                record,
                candidate_index,
            )
            for record in unknown_records
        ]

        expected_hits = sum(detection_probabilities)

        # 每提前发现一个频道，可以省去它在后续节点的检测和切频。
        measurement_equivalent_m = (
            self.cfg.detection_time_s
            + self.cfg.channel_switch_time_s
        ) * self.cfg.dog_speed_m_per_s

        ordinary_value_m = (
            expected_hits
            * future_node_count
            * measurement_equivalent_m
            * self.tuning.unknown_discovery_value_scale
        )

        required_hits = (
            self.cfg.source_count_max
            - self.discovered_count
        )
        early_stop_probability = (
            self._poisson_binomial_tail(
                detection_probabilities,
                required_hits,
            )
        )

        early_stop_bonus_m = (
            early_stop_probability
            * max(0.0, early_stop_value_m)
            * self.tuning.unknown_early_stop_weight
        )

        total_value_m = min(
            self.tuning.unknown_value_cap_m,
            ordinary_value_m + early_stop_bonus_m,
        )

        return {
            "value_m": total_value_m,
            "ordinary_value_m": ordinary_value_m,
            "early_stop_bonus_m": early_stop_bonus_m,
            "expected_hits": expected_hits,
            "early_stop_probability": (
                early_stop_probability
            ),
            "source_presence_prior": (
                source_presence_prior
            ),
        }

    # ================================================================
    # 优先级1：联合UNKNOWN发现收益的认证节点选择
    # ================================================================

    def _search_candidate_metrics(
        self,
        candidate_index: int,
        remaining: set[int],
        current: Point,
        baseline_tail_m: float,
    ) -> dict:
        point = self.search_points[candidate_index]

        tail_indices = [
            index
            for index in remaining
            if index != candidate_index
        ]

        first_leg_m = math.dist(current, point)
        route_distance_m = (
            first_leg_m
            + self._nearest_neighbor_tail_distance(
                point,
                tail_indices,
            )
        )
        route_regret_m = max(
            0.0,
            route_distance_m - baseline_tail_m,
        )

        localization_value_m = (
            self._node_total_localization_value(point)
        )

        unknown_count = sum(
            record.status == "UNKNOWN"
            for record in self.records.values()
        )
        future_node_count = max(
            0,
            len(remaining) - 1,
        )

        measurement_equivalent_m = (
            self.cfg.detection_time_s
            + self.cfg.channel_switch_time_s
        ) * self.cfg.dog_speed_m_per_s

        remaining_route_m = max(
            0.0,
            route_distance_m - first_leg_m,
        )

        # 达到16个后能够额外省去：
        # 1. 剩余认证节点移动；
        # 2. 剩余UNKNOWN频道的认证测量。
        early_stop_value_m = (
            remaining_route_m
            + future_node_count
            * unknown_count
            * measurement_equivalent_m
        )

        unknown_metrics = self._unknown_node_value(
            candidate_index,
            future_node_count=future_node_count,
            early_stop_value_m=early_stop_value_m,
        )

        combined_score = (
            localization_value_m
            + unknown_metrics["value_m"]
            - self.tuning.dynamic_route_regret_penalty
            * route_regret_m
        )

        return {
            "index": candidate_index,
            "point": point,
            "route_distance_m": route_distance_m,
            "route_regret_m": route_regret_m,
            "localization_value": localization_value_m,
            "unknown_discovery_value": (
                unknown_metrics["value_m"]
            ),
            "unknown_expected_hits": (
                unknown_metrics["expected_hits"]
            ),
            "unknown_early_stop_probability": (
                unknown_metrics[
                    "early_stop_probability"
                ]
            ),
            "unknown_source_presence_prior": (
                unknown_metrics[
                    "source_presence_prior"
                ]
            ),
            "combined_score": combined_score,
        }

    def _normal_search_evaluations(
        self,
        remaining: set[int],
    ) -> list[dict]:
        current = tuple(self.client.position)

        ordered_by_distance = sorted(
            remaining,
            key=lambda index: (
                math.dist(
                    current,
                    self.search_points[index],
                ),
                index,
            ),
        )
        candidate_indices = ordered_by_distance[
            :self.tuning.dynamic_route_candidate_limit
        ]

        baseline_index = candidate_indices[0]
        baseline_tail_m = (
            self._nearest_neighbor_tail_distance(
                current,
                list(remaining),
            )
        )

        allowed_regret_m = max(
            self.tuning.dynamic_route_regret_floor_m,
            self.tuning.dynamic_route_regret_ratio
            * baseline_tail_m,
        )

        evaluations: list[dict] = []

        for candidate_index in candidate_indices:
            metrics = self._search_candidate_metrics(
                candidate_index,
                remaining,
                current,
                baseline_tail_m,
            )

            if (
                metrics["route_regret_m"]
                > allowed_regret_m + 1e-7
            ):
                continue

            evaluations.append(metrics)

        if not evaluations:
            evaluations.append(
                self._search_candidate_metrics(
                    baseline_index,
                    remaining,
                    current,
                    baseline_tail_m,
                )
            )

        return evaluations

    @staticmethod
    def _best_search_evaluation(
        evaluations: Sequence[dict],
    ) -> dict:
        return max(
            evaluations,
            key=lambda item: (
                item["combined_score"],
                item["unknown_discovery_value"],
                item["localization_value"],
                item["unknown_expected_hits"],
                -item["route_distance_m"],
                -item["index"],
            ),
        )

    def _peek_next_search_index(
        self,
        remaining: set[int],
    ) -> int:
        if (
            self._forced_search_index is not None
            and self._forced_search_index in remaining
        ):
            return self._forced_search_index

        evaluations = self._normal_search_evaluations(
            remaining
        )
        return self._best_search_evaluation(
            evaluations
        )["index"]

    def _select_next_search_node(
        self,
        remaining: set[int],
    ) -> tuple[int, dict]:
        current = tuple(self.client.position)

        forced_index = self._forced_search_index
        forced = (
            forced_index is not None
            and forced_index in remaining
        )

        if forced:
            baseline_tail_m = (
                self._nearest_neighbor_tail_distance(
                    current,
                    list(remaining),
                )
            )
            selected = self._search_candidate_metrics(
                int(forced_index),
                remaining,
                current,
                baseline_tail_m,
            )
            selected["forced_by_inline_clear"] = True

            self._forced_search_index = None
            self.counts["priority_forced_search_anchor"] += 1
        else:
            self._forced_search_index = None
            evaluations = (
                self._normal_search_evaluations(
                    remaining
                )
            )
            selected = self._best_search_evaluation(
                evaluations
            )
            selected["forced_by_inline_clear"] = False

        if selected["localization_value"] > 0.0:
            self.counts[
                "dynamic_search_localization_aware_selection"
            ] += 1

        if selected["unknown_discovery_value"] > 0.0:
            self.counts[
                "priority_unknown_aware_selection"
            ] += 1

        self.counts[
            "dynamic_search_node_selection"
        ] += 1
        self.counts[
            "dynamic_search_total_regret_m"
        ] += selected["route_regret_m"]
        self.counts[
            "priority_unknown_value_selected_m"
        ] += selected["unknown_discovery_value"]
        self.counts[
            "priority_expected_unknown_hits_selected"
        ] += selected["unknown_expected_hits"]
        self.counts[
            "priority_early_stop_probability_sum"
        ] += selected[
            "unknown_early_stop_probability"
        ]

        self.event(
            "priority_search_node_value",
            global_index=selected["index"],
            point=selected["point"],
            forced_by_inline_clear=(
                selected["forced_by_inline_clear"]
            ),
            localization_value_m=(
                selected["localization_value"]
            ),
            unknown_discovery_value_m=(
                selected["unknown_discovery_value"]
            ),
            unknown_expected_hits=(
                selected["unknown_expected_hits"]
            ),
            unknown_early_stop_probability=(
                selected[
                    "unknown_early_stop_probability"
                ]
            ),
            route_regret_m=(
                selected["route_regret_m"]
            ),
            combined_score=selected["combined_score"],
        )

        return selected["index"], selected

    # ================================================================
    # 优先级2：READY源的全局开放路线
    # ================================================================

    @staticmethod
    def _ready_center(
        record: ChannelRecord,
    ) -> Point:
        if record.clearance_center is None:
            raise IncompleteRun(
                f"频道{record.channel_id}缺少清除中心"
            )
        return tuple(record.clearance_center)

    def _ready_path_distance(
        self,
        start: Point,
        route: Sequence[ChannelRecord],
    ) -> float:
        total = 0.0
        current = tuple(start)

        for record in route:
            point = self._ready_center(record)
            total += math.dist(current, point)
            current = point

        return total

    def _nearest_ready_route(
        self,
        records: Sequence[ChannelRecord],
        start: Point,
        *,
        forced_first: ChannelRecord | None = None,
    ) -> list[ChannelRecord]:
        remaining = {
            record.channel_id: record
            for record in records
        }
        route: list[ChannelRecord] = []
        current = tuple(start)

        if forced_first is not None:
            selected = remaining.pop(
                forced_first.channel_id
            )
            route.append(selected)
            current = self._ready_center(selected)

        while remaining:
            selected = min(
                remaining.values(),
                key=lambda record: (
                    math.dist(
                        current,
                        self._ready_center(record),
                    ),
                    record.channel_id,
                ),
            )
            remaining.pop(selected.channel_id)
            route.append(selected)
            current = self._ready_center(selected)

        return route

    def _two_opt_ready_route(
        self,
        route: Sequence[ChannelRecord],
        start: Point,
    ) -> list[ChannelRecord]:
        result = list(route)
        if len(result) < 3:
            return result

        for _ in range(
            self.tuning.ready_route_two_opt_passes
        ):
            best_gain = 1e-7
            best_pair: tuple[int, int] | None = None

            for left in range(len(result) - 1):
                previous_point = (
                    tuple(start)
                    if left == 0
                    else self._ready_center(
                        result[left - 1]
                    )
                )
                left_point = self._ready_center(
                    result[left]
                )

                for right in range(
                    left + 1,
                    len(result),
                ):
                    right_point = self._ready_center(
                        result[right]
                    )

                    before = math.dist(
                        previous_point,
                        left_point,
                    )
                    after = math.dist(
                        previous_point,
                        right_point,
                    )

                    if right + 1 < len(result):
                        next_point = self._ready_center(
                            result[right + 1]
                        )
                        before += math.dist(
                            right_point,
                            next_point,
                        )
                        after += math.dist(
                            left_point,
                            next_point,
                        )

                    gain = before - after
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (left, right)

            if best_pair is None:
                break

            left, right = best_pair
            result[left:right + 1] = reversed(
                result[left:right + 1]
            )

        return result

    def _plan_ready_route(
        self,
        records: Sequence[ChannelRecord],
    ) -> tuple[
        list[ChannelRecord],
        float,
        float,
    ]:
        records = sorted(
            records,
            key=lambda record: record.channel_id,
        )
        start = tuple(self.client.position)

        if not records:
            return [], 0.0, 0.0

        baseline_route = self._nearest_ready_route(
            records,
            start,
        )
        baseline_distance_m = (
            self._ready_path_distance(
                start,
                baseline_route,
            )
        )

        best_route = self._two_opt_ready_route(
            baseline_route,
            start,
        )
        best_distance_m = self._ready_path_distance(
            start,
            best_route,
        )

        nearest_first_leg_m = min(
            math.dist(
                start,
                self._ready_center(record),
            )
            for record in records
        )

        start_candidates = [
            record
            for record in sorted(
                records,
                key=lambda record: (
                    math.dist(
                        start,
                        self._ready_center(record),
                    ),
                    record.channel_id,
                ),
            )
            if (
                math.dist(
                    start,
                    self._ready_center(record),
                )
                <= (
                    nearest_first_leg_m
                    + self.tuning.ready_route_first_leg_slack_m
                )
            )
        ][:self.tuning.ready_route_start_limit]

        for first in start_candidates:
            candidate_route = self._nearest_ready_route(
                records,
                start,
                forced_first=first,
            )
            candidate_route = (
                self._two_opt_ready_route(
                    candidate_route,
                    start,
                )
            )
            candidate_distance_m = (
                self._ready_path_distance(
                    start,
                    candidate_route,
                )
            )

            candidate_key = tuple(
                record.channel_id
                for record in candidate_route
            )
            best_key = tuple(
                record.channel_id
                for record in best_route
            )

            if (
                candidate_distance_m
                < best_distance_m - 1e-7
                or (
                    abs(
                        candidate_distance_m
                        - best_distance_m
                    )
                    <= 1e-7
                    and candidate_key < best_key
                )
            ):
                best_route = candidate_route
                best_distance_m = (
                    candidate_distance_m
                )

        return (
            best_route,
            best_distance_m,
            baseline_distance_m,
        )

    def _select_horizon_task(
        self,
        tasks: Sequence[dict],
    ) -> tuple[dict, list[dict]]:
        selected, original_sequence = (
            super()._select_horizon_task(tasks)
        )

        clear_tasks = [
            task
            for task in tasks
            if task["action_type"] == "clear"
        ]

        # 如果本轮调度器认为应该清除，则对所有READY源统一优化
        # 清除顺序；不强行打断高价值主动定位任务。
        if (
            selected["action_type"] != "clear"
            or len(clear_tasks) < 2
        ):
            return selected, original_sequence

        task_by_channel = {
            task["record"].channel_id: task
            for task in clear_tasks
        }

        ready_route, optimized_distance_m, baseline_distance_m = (
            self._plan_ready_route(
                [
                    task["record"]
                    for task in clear_tasks
                ]
            )
        )

        if not ready_route:
            return selected, original_sequence

        optimized_first = task_by_channel[
            ready_route[0].channel_id
        ]

        improvement_m = max(
            0.0,
            baseline_distance_m
            - optimized_distance_m,
        )

        self.counts["priority_ready_route_plan"] += 1
        self.counts[
            "priority_ready_route_improvement_m"
        ] += improvement_m

        if (
            optimized_first["record"].channel_id
            != selected["record"].channel_id
        ):
            self.counts[
                "priority_ready_route_first_override"
            ] += 1

        horizon = max(
            1,
            self.tuning.scheduler_horizon,
        )
        planned_sequence = [
            task_by_channel[record.channel_id]
            for record in ready_route[:horizon]
        ]

        used = {
            (
                task["action_type"],
                task["record"].channel_id,
            )
            for task in planned_sequence
        }

        for task in original_sequence:
            key = (
                task["action_type"],
                task["record"].channel_id,
            )
            if key in used:
                continue
            planned_sequence.append(task)
            used.add(key)
            if len(planned_sequence) >= horizon:
                break

        self.event(
            "priority_ready_route_selected",
            ready_count=len(ready_route),
            route_channels=[
                record.channel_id
                for record in ready_route
            ],
            optimized_distance_m=(
                optimized_distance_m
            ),
            nearest_neighbor_distance_m=(
                baseline_distance_m
            ),
            estimated_improvement_m=improvement_m,
            selected_channel=(
                optimized_first["record"].channel_id
            ),
        )

        return optimized_first, planned_sequence

    # ================================================================
    # 优先级3：基于路线插入代价的顺路清除
    # ================================================================

    @staticmethod
    def _insertion_detour(
        current: Point,
        inserted: Point,
        anchor: Point,
    ) -> float:
        return max(
            0.0,
            math.dist(current, inserted)
            + math.dist(inserted, anchor)
            - math.dist(current, anchor),
        )

    def _remaining_search_indices(self) -> set[int]:
        visited = set(
            self.dynamic_search_visit_order
        )
        return set(range(len(self.search_points))) - visited

    def _next_search_anchor(
        self,
    ) -> tuple[int, Point] | None:
        if (
            self.discovered_count
            >= self.cfg.source_count_max
        ):
            return None

        if not any(
            record.status == "UNKNOWN"
            for record in self.records.values()
        ):
            return None

        remaining = self._remaining_search_indices()
        if not remaining:
            return None

        index = self._peek_next_search_index(
            remaining
        )
        return index, self.search_points[index]

    def _localization_anchor_points(
        self,
    ) -> list[Point]:
        anchors: list[Point] = []

        for record in self.records.values():
            if record.status != "FOUND":
                continue

            cached = self._active_plan_cache.get(
                record.channel_id
            )
            if cached is not None and cached.get(
                "point"
            ) is not None:
                anchors.append(tuple(cached["point"]))
                continue

            queue = self._recovery_queues.get(
                record.channel_id
            )
            if record.recovery_active and queue:
                anchors.append(tuple(queue[0]))
                continue

            anchors.append(
                tuple(
                    self._localization_task_point(
                        record
                    )
                )
            )

        current = tuple(self.client.position)
        anchors.sort(
            key=lambda point: math.dist(
                current,
                point,
            )
        )
        return anchors[
            :self.tuning.dynamic_route_candidate_limit
        ]

    def _clear_ready_nearby(
        self,
        maximum_distance_m: float,
        phase_name: str,
    ) -> int:
        """用插入代价替代自适应搜索层的固定欧氏距离阈值。"""
        cleared = 0

        while (
            cleared
            < self.tuning.inline_clear_batch_limit
        ):
            ready_records = [
                record
                for record in self.records.values()
                if (
                    record.status == "READY"
                    and record.clearance_center
                    is not None
                )
            ]
            if not ready_records:
                return cleared

            ready_route, _, _ = (
                self._plan_ready_route(
                    ready_records
                )
            )
            route_rank = {
                record.channel_id: index
                for index, record in enumerate(
                    ready_route
                )
            }

            current = tuple(self.client.position)
            selected: ChannelRecord | None = None
            selected_detour_m = math.inf
            selected_anchor: Point | None = None
            selected_anchor_index: int | None = None

            if phase_name == "search":
                search_anchor = (
                    self._next_search_anchor()
                )
                if search_anchor is None:
                    return cleared

                anchor_index, anchor_point = (
                    search_anchor
                )

                choices = []
                for record in ready_records:
                    center = self._ready_center(record)
                    direct_distance_m = math.dist(
                        current,
                        center,
                    )

                    if (
                        direct_distance_m
                        > self.tuning.inline_max_direct_distance_m
                    ):
                        continue

                    detour_m = self._insertion_detour(
                        current,
                        center,
                        anchor_point,
                    )
                    if (
                        detour_m
                        > self.tuning.search_inline_detour_cap_m
                    ):
                        continue

                    choices.append((
                        detour_m,
                        route_rank.get(
                            record.channel_id,
                            10**9,
                        ),
                        direct_distance_m,
                        record.channel_id,
                        record,
                    ))

                if choices:
                    (
                        selected_detour_m,
                        _,
                        _,
                        _,
                        selected,
                    ) = min(choices)
                    selected_anchor = anchor_point
                    selected_anchor_index = anchor_index
                else:
                    self.counts[
                        "priority_inline_rejected_by_detour"
                    ] += 1
                    return cleared

            else:
                anchors = (
                    self._localization_anchor_points()
                )

                # 如果没有FOUND任务，所有剩余READY源都是必做任务，
                # 直接按照全局开放路线清除。
                if not anchors:
                    selected = ready_route[0]
                    selected_detour_m = 0.0
                else:
                    choices = []

                    for record in ready_records:
                        center = self._ready_center(
                            record
                        )
                        direct_distance_m = math.dist(
                            current,
                            center,
                        )

                        if (
                            direct_distance_m
                            > self.tuning.inline_max_direct_distance_m
                        ):
                            continue

                        best_anchor = min(
                            anchors,
                            key=lambda anchor: (
                                self._insertion_detour(
                                    current,
                                    center,
                                    anchor,
                                ),
                                math.dist(center, anchor),
                            ),
                        )
                        detour_m = (
                            self._insertion_detour(
                                current,
                                center,
                                best_anchor,
                            )
                        )

                        if (
                            detour_m
                            > self.tuning.localization_inline_detour_cap_m
                        ):
                            continue

                        choices.append((
                            detour_m,
                            route_rank.get(
                                record.channel_id,
                                10**9,
                            ),
                            direct_distance_m,
                            record.channel_id,
                            best_anchor,
                            record,
                        ))

                    if choices:
                        (
                            selected_detour_m,
                            _,
                            _,
                            _,
                            selected_anchor,
                            selected,
                        ) = min(choices)
                    else:
                        self.counts[
                            "priority_inline_rejected_by_detour"
                        ] += 1
                        return cleared

            if selected is None:
                return cleared

            if phase_name == "search":
                self._forced_search_index = (
                    selected_anchor_index
                )

            if not self.clear(selected):
                return cleared

            cleared += 1

            legacy_counter = (
                "adaptive_inline_clear_search"
                if phase_name == "search"
                else "adaptive_inline_clear_localization"
            )
            new_counter = (
                "priority_inline_clear_search"
                if phase_name == "search"
                else "priority_inline_clear_localization"
            )
            self.counts[legacy_counter] += 1
            self.counts[new_counter] += 1

            self.event(
                "priority_inline_clear_completed",
                channel=selected.channel_id,
                source_phase=phase_name,
                decision_basis="route_insertion_cost",
                insertion_detour_m=(
                    selected_detour_m
                ),
                detour_cap_m=(
                    self.tuning.search_inline_detour_cap_m
                    if phase_name == "search"
                    else self.tuning.localization_inline_detour_cap_m
                ),
                direct_distance_limit_m=(
                    self.tuning.inline_max_direct_distance_m
                ),
                legacy_radius_argument_m=(
                    maximum_distance_m
                ),
                next_anchor=selected_anchor,
                next_search_index=(
                    selected_anchor_index
                ),
            )

        return cleared

    # ================================================================
    # 汇总
    # ================================================================

    def summary(
        self,
        outcome: str,
        reason: str,
        exit_confirmed: bool,
    ) -> dict:
        result = super().summary(
            outcome,
            reason,
            exit_confirmed,
        )

        result["priority_tuning"] = asdict(self.tuning)
        result["priority_optimization"] = {
            "unknown_probability_cache_entries": len(
                self._unknown_probability_cache
            ),
            "unknown_aware_node_selection_count": (
                self.counts[
                    "priority_unknown_aware_selection"
                ]
            ),
            "unknown_value_selected_m": (
                self.counts[
                    "priority_unknown_value_selected_m"
                ]
            ),
            "expected_unknown_hits_selected": (
                self.counts[
                    "priority_expected_unknown_hits_selected"
                ]
            ),
            "ready_route_plan_count": (
                self.counts[
                    "priority_ready_route_plan"
                ]
            ),
            "ready_route_estimated_improvement_m": (
                self.counts[
                    "priority_ready_route_improvement_m"
                ]
            ),
            "inline_clear_search_count": (
                self.counts[
                    "priority_inline_clear_search"
                ]
            ),
            "inline_clear_localization_count": (
                self.counts[
                    "priority_inline_clear_localization"
                ]
            ),
            "certificate_preserved": (
                self.search_complete
                or self.discovered_count
                >= self.cfg.source_count_max
            ),
        }

        result["model_note"] = (
            result.get("model_note", "")
            + "；优先调度层将UNKNOWN后验发现收益加入认证节点排序，"
            "但不减少双环认证节点；READY清除采用多起点最近邻"
            "和2-opt开放路线；顺路清除使用路线插入代价，"
            "不再以固定欧氏半径作为主要判据"
        )
        return result


def _run_priority_search_self_test() -> dict:
    result = _run_adaptive_search_self_test()
    tuning = _PrioritySearchTuning()

    assert tuning.dynamic_route_candidate_limit >= 8
    assert tuning.unknown_position_samples > 0
    assert tuning.unknown_orientation_samples > 0
    assert 0.0 <= tuning.unknown_directional_prior <= 1.0
    assert (
        tuning.search_inline_detour_cap_m
        < tuning.localization_inline_detour_cap_m
    )
    assert tuning.inline_clear_batch_limit > 0

    # 插入代价基本性质检查。
    straight = _PrioritySearchRunner._insertion_detour(
        (0.0, 0.0),
        (5.0, 0.0),
        (10.0, 0.0),
    )
    off_route = _PrioritySearchRunner._insertion_detour(
        (0.0, 0.0),
        (5.0, 5.0),
        (10.0, 0.0),
    )
    assert abs(straight) <= 1e-9
    assert off_route > 0.0

    probability = (
        _PrioritySearchRunner._poisson_binomial_tail(
            [0.5, 0.5],
            1,
        )
    )
    assert abs(probability - 0.75) <= 1e-9

    return {
        **result,
        "priority_status": "ok",
        "priority_unknown_discovery_ordering": True,
        "priority_global_ready_route": True,
        "priority_insertion_cost_inline_clear": True,
        "priority_certificate_nodes_removed": False,
    }

@dataclass(frozen=True)
class _RecoveryTuning(_PrioritySearchTuning):
    # ================================================================
    # 优先级1：单示向安全重捕获
    # ================================================================

    # 第一次从最近成功示向点沿示向中心线前进。
    single_bearing_first_step_m: float = 90.0

    # 第一次安全重捕获无信号时，尝试更短距离，
    # 处理源距离很近而第一次移动越过源的情况。
    single_bearing_second_step_m: float = 35.0

    # 单示向安全探针最多尝试两次；再失败则进入恢复逻辑。
    single_bearing_probe_limit: int = 2

    # 安全重捕获是完成该频道的必要步骤，调度时给予完成价值。
    single_bearing_completion_value_m: float = 900.0

    # 安全中心线探针用于“高置信失败”检测的参考概率。
    single_bearing_reference_probability: float = 0.80

    # ================================================================
    # 优先级2：恢复点失效和局部恢复事务
    # ================================================================

    # 认证搜索层原来最多尝试5次信息恢复；安全恢复层缩短为3次。
    recovery_information_probe_limit: int = 3

    # 一旦调度器选中信息恢复，最多连续执行的局部探针数量。
    recovery_burst_limit: int = 3

    # 连续恢复时，下一探针不得离当前位置过远。
    recovery_burst_leg_cap_m: float = 350.0

    # 当前恢复点的距离超过规划移动上限的这个比例时失效。
    recovery_stale_cap_ratio: float = 1.05

    # 提高近距离信息恢复任务的调度价值，
    # 避免生成恢复点后先横跨地图清除远端源。
    recovery_locality_value_m: float = 700.0

    # ================================================================
    # 优先级3：联合概率校准和快速降级
    # ================================================================

    # 将认证搜索层的36个朝向粒子提高到72个，减小半平面边界误差。
    orientation_particle_count: int = 72

    # 位置组接收概率的保守分位数。
    posterior_receive_quantile: float = 0.20

    # 最终概率 = 均值和低分位概率的加权结果。
    posterior_robust_blend: float = 0.45

    # 经验可靠度的Beta型先验强度；初始可靠度为1。
    model_reliability_prior_strength: float = 3.0

    # 多次失败后仍保留的最低模型可靠度。
    model_reliability_floor: float = 0.35

    # 自适应搜索层原来在失败后提高门槛；安全恢复层不再提高，
    # 仅允许在模型不可靠时小幅降低最低门槛。
    calibrated_receive_floor_min: float = 0.32

    # 预测不低于该值却返回no_signal，记为一次高置信失败。
    surprise_no_signal_probability: float = 0.72

    # 连续达到该次数后快速进入确定性恢复。
    surprise_no_signal_limit: int = 2

    def __post_init__(self):
        super().__post_init__()

        for name in (
            "single_bearing_probe_limit",
            "recovery_burst_limit",
            "surprise_no_signal_limit",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name}必须是整数")
            if getattr(self, name) <= 0:
                raise ValueError(f"{name}必须为正整数")

        for name in (
            "single_bearing_first_step_m",
            "single_bearing_second_step_m",
            "single_bearing_completion_value_m",
            "recovery_burst_leg_cap_m",
            "recovery_stale_cap_ratio",
            "recovery_locality_value_m",
            "model_reliability_prior_strength",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name}必须是数值")
            if not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name}必须是有限正数")

        for name in (
            "single_bearing_reference_probability",
            "posterior_receive_quantile",
            "posterior_robust_blend",
            "model_reliability_floor",
            "calibrated_receive_floor_min",
            "surprise_no_signal_probability",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name}必须是数值")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name}必须位于[0,1]")


class _RecoveryRunner(_PrioritySearchRunner):
    def __init__(
        self,
        client,
        *,
        config: Q4Config,
        tuning: _RecoveryTuning | None = None,
        planner_config=None,
        geometry_config=None,
        output_dir: Path | None = None,
    ):
        self.tuning = tuning or _RecoveryTuning()

        super().__init__(
            client,
            config=config,
            tuning=self.tuning,
            planner_config=planner_config,
            geometry_config=geometry_config,
            output_dir=output_dir,
        )

        # 当前真正准备执行的主动测量预测。
        self._active_execution_prediction: dict[
            int,
            dict,
        ] = {}

        # information_reacquire恢复点的生成位置、预测概率和移动上限。
        self._recovery_plan_meta: dict[
            int,
            dict,
        ] = {}

        # 连续高置信预测失败次数。
        self._surprise_misses: dict[int, int] = {}

        self.counts.update({
            "recovery_single_bearing_plan": 0,
            "recovery_single_bearing_measure": 0,
            "recovery_single_bearing_positive": 0,
            "recovery_single_bearing_no_signal": 0,
            "recovery_likelihood_weighted_hypothesis_build": 0,
            "recovery_surprise_no_signal": 0,
            "recovery_surprise_fast_fallback": 0,
            "recovery_plan_recorded": 0,
            "recovery_stale_recovery_replanned": 0,
            "recovery_burst_round": 0,
            "recovery_burst_extra_measure": 0,
            "recovery_locality_boost": 0,
        })

    # ================================================================
    # 优先级3：联合粒子按存活似然加权
    # ================================================================

    def _joint_hypotheses(
        self,
        record: ChannelRecord,
        sources: Sequence[Point],
    ) -> list[dict]:
        """构建位置、半径和朝向联合后验。

        与认证搜索层的关键区别：

        认证搜索层会先对每个位置样本内部重新归一化，导致只有极少
        朝向存活的位置仍可能获得与大量朝向存活位置相同的总权重。

        安全恢复层先给所有原始粒子分配先验权重，再删除与观测冲突的
        粒子，最后只做一次全局归一化。因此位置样本的后验质量
        与其真实存活似然一致。

        no_signal仍然只影响规划粒子，不参与ABSENT证明和MEC证书。
        """
        if not sources:
            return []

        positive = list(record.bearing_observations)
        no_signals = list(record.no_signal_observations)

        orientation_count = (
            self.tuning.orientation_particle_count
        )
        orientations = [
            (
                math.cos(
                    2.0 * math.pi * index
                    / orientation_count
                ),
                math.sin(
                    2.0 * math.pi * index
                    / orientation_count
                ),
            )
            for index in range(orientation_count)
        ]

        source_prior = 1.0 / len(sources)
        hypotheses: list[dict] = []

        for source_index, source_raw in enumerate(sources):
            source = tuple(source_raw)

            positive_distances = [
                math.dist(
                    source,
                    observation["position"],
                )
                for observation in positive
            ]
            lower_radius = max(
                [
                    self.cfg.reception_radius_min_m,
                    *positive_distances,
                ]
            )

            if (
                lower_radius
                > self.cfg.reception_radius_max_m
                + 1e-7
            ):
                continue

            radii = sorted(set([
                float(lower_radius),
                float(
                    0.5
                    * (
                        lower_radius
                        + self.cfg.reception_radius_max_m
                    )
                ),
                float(
                    self.cfg.reception_radius_max_m
                ),
            ]))

            radius_count = len(radii)
            if radius_count <= 0:
                continue

            omni_prior = (
                source_prior
                * 0.5
                / radius_count
            )
            directional_prior = (
                source_prior
                * 0.5
                / radius_count
                / orientation_count
            )

            for radius in radii:
                # ----------------------------------------------------
                # 全向源粒子
                # ----------------------------------------------------
                positive_compatible = all(
                    math.dist(
                        source,
                        observation["position"],
                    )
                    <= radius + 1e-7
                    for observation in positive
                )
                no_signal_compatible = not any(
                    math.dist(
                        source,
                        observation["position"],
                    )
                    <= radius + 1e-7
                    for observation in no_signals
                )

                if (
                    positive_compatible
                    and no_signal_compatible
                ):
                    hypotheses.append({
                        "source": source,
                        "radius": radius,
                        "direction": None,
                        "weight": omni_prior,
                        "prior_weight": omni_prior,
                        "source_group": source_index,
                        "source_class": "omnidirectional",
                        "channel": record.channel_id,
                    })

                # ----------------------------------------------------
                # 定向源粒子
                # ----------------------------------------------------
                for direction in orientations:
                    positive_compatible = all(
                        (
                            math.dist(
                                source,
                                observation["position"],
                            )
                            <= radius + 1e-7
                            and self._particle_front(
                                direction,
                                source,
                                observation["position"],
                            )
                        )
                        for observation in positive
                    )
                    if not positive_compatible:
                        continue

                    inconsistent_no_signal = any(
                        (
                            math.dist(
                                source,
                                observation["position"],
                            )
                            <= radius + 1e-7
                            and self._particle_front(
                                direction,
                                source,
                                observation["position"],
                            )
                        )
                        for observation in no_signals
                    )
                    if inconsistent_no_signal:
                        continue

                    hypotheses.append({
                        "source": source,
                        "radius": radius,
                        "direction": direction,
                        "weight": directional_prior,
                        "prior_weight": directional_prior,
                        "source_group": source_index,
                        "source_class": "directional",
                        "channel": record.channel_id,
                    })

        total_weight = sum(
            item["weight"]
            for item in hypotheses
        )
        if total_weight <= 1e-15:
            return []

        for item in hypotheses:
            item["weight"] /= total_weight

        self.counts[
            "recovery_likelihood_weighted_hypothesis_build"
        ] += 1
        return hypotheses

    @staticmethod
    def _weighted_quantile(
        values_and_weights: Sequence[
            tuple[float, float]
        ],
        quantile: float,
    ) -> float:
        if not values_and_weights:
            return 0.0

        ordered = sorted(
            (
                float(value),
                max(0.0, float(weight)),
            )
            for value, weight in values_and_weights
        )
        total_weight = sum(
            weight
            for _, weight in ordered
        )
        if total_weight <= 1e-15:
            return 0.0

        threshold = min(
            1.0,
            max(0.0, quantile),
        ) * total_weight

        accumulated = 0.0
        for value, weight in ordered:
            accumulated += weight
            if accumulated + 1e-15 >= threshold:
                return value

        return ordered[-1][0]

    def _model_reliability(
        self,
        record: ChannelRecord,
    ) -> float:
        relevant = [
            item
            for item in record.progress_history
            if item.get("measurement_kind") in {
                "active",
                "recovery",
                "opportunistic",
            }
        ]

        success_count = sum(
            item.get("result") in {
                "direction",
                "near",
            }
            for item in relevant
        )
        attempt_count = len(relevant)

        strength = (
            self.tuning.model_reliability_prior_strength
        )
        reliability = (
            strength + success_count
        ) / (
            strength + attempt_count
        )

        return min(
            1.0,
            max(
                self.tuning.model_reliability_floor,
                reliability,
            ),
        )

    def _joint_receive_probability(
        self,
        point: Point,
        hypotheses: Sequence[dict],
    ) -> float:
        if not hypotheses:
            return 0.0

        total_probability = 0.0
        groups: dict[int, dict[str, float]] = {}

        for item in hypotheses:
            weight = float(item["weight"])
            group = int(
                item.get("source_group", 0)
            )
            group_entry = groups.setdefault(
                group,
                {
                    "weight": 0.0,
                    "receive_weight": 0.0,
                },
            )
            group_entry["weight"] += weight

            source = item["source"]
            receives = (
                math.dist(point, source)
                <= item["radius"] + 1e-7
                and self._particle_front(
                    item["direction"],
                    source,
                    point,
                )
            )
            if not receives:
                continue

            total_probability += weight
            group_entry["receive_weight"] += weight

        group_probabilities: list[
            tuple[float, float]
        ] = []

        for entry in groups.values():
            group_weight = entry["weight"]
            if group_weight <= 1e-15:
                continue
            group_probabilities.append((
                entry["receive_weight"]
                / group_weight,
                group_weight,
            ))

        lower_probability = (
            self._weighted_quantile(
                group_probabilities,
                self.tuning.posterior_receive_quantile,
            )
        )

        blend = self.tuning.posterior_robust_blend
        robust_probability = (
            (1.0 - blend) * total_probability
            + blend * lower_probability
        )

        channel = hypotheses[0].get("channel")
        if channel in self.records:
            reliability = self._model_reliability(
                self.records[int(channel)]
            )
        else:
            reliability = 1.0

        calibrated_probability = (
            robust_probability * reliability
        )

        return min(
            1.0,
            max(0.0, calibrated_probability),
        )

    def _adaptive_receive_floor(
        self,
        record: ChannelRecord,
        base_floor: float,
    ) -> float:
        """失败后降低模型信任，而不是提高原始概率门槛。"""
        reliability = self._model_reliability(record)

        # 初始可靠度为1时保持原门槛；
        # 可靠度下降后最多小幅降低约25%。
        scale = 0.75 + 0.25 * reliability
        return max(
            self.tuning.calibrated_receive_floor_min,
            base_floor * scale,
        )

    # ================================================================
    # 优先级1：单示向安全重捕获
    # ================================================================

    @staticmethod
    def _post_positive_no_signal_count(
        record: ChannelRecord,
    ) -> int:
        if not record.bearing_observations:
            return 0

        first_positive_time = min(
            float(item["virtual_time_s"])
            for item in record.bearing_observations
        )

        return sum(
            (
                item.get("measurement_kind")
                in {"active", "recovery"}
                and float(
                    item.get("virtual_time_s", -math.inf)
                )
                >= first_positive_time
            )
            for item in record.no_signal_observations
        )

    def _single_bearing_safe_candidates(
        self,
        record: ChannelRecord,
    ) -> list[dict]:
        if len(record.bearing_observations) != 1:
            return []

        failure_count = (
            self._post_positive_no_signal_count(
                record
            )
        )
        if (
            failure_count
            >= self.tuning.single_bearing_probe_limit
        ):
            return []

        observation = record.bearing_observations[0]
        origin = tuple(observation["position"])
        bearing_deg = float(
            observation["bearing_deg"]
        )

        if failure_count == 0:
            patterns = (
                (
                    self.tuning.single_bearing_first_step_m,
                    0.0,
                ),
                (
                    0.80
                    * self.tuning.single_bearing_first_step_m,
                    5.0,
                ),
                (
                    0.80
                    * self.tuning.single_bearing_first_step_m,
                    -5.0,
                ),
            )
        else:
            patterns = (
                (
                    self.tuning.single_bearing_second_step_m,
                    0.0,
                ),
                (
                    1.50
                    * self.tuning.single_bearing_second_step_m,
                    8.0,
                ),
                (
                    1.50
                    * self.tuning.single_bearing_second_step_m,
                    -8.0,
                ),
            )

        result: list[dict] = []

        for step_m, offset_deg in patterns:
            angle = math.radians(
                bearing_deg + offset_deg
            )
            point = (
                origin[0]
                + step_m * math.cos(angle),
                origin[1]
                + step_m * math.sin(angle),
            )

            if self._point_was_tested(
                record,
                point,
            ):
                continue

            result.append({
                "point": point,
                "origin": "recovery_single_bearing_safe",
                "safe_step_m": step_m,
                "safe_offset_deg": offset_deg,
                "safe_failure_stage": failure_count,
            })

        return result

    def _build_single_bearing_proposal(
        self,
        record: ChannelRecord,
    ) -> dict | None:
        candidates = (
            self._single_bearing_safe_candidates(
                record
            )
        )
        if not candidates:
            return None

        try:
            sources = sample_source_positions(
                record,
                self.cfg,
            )
        except IncompleteRun:
            return None

        estimate = (
            sum(point[0] for point in sources)
            / len(sources),
            sum(point[1] for point in sources)
            / len(sources),
        )
        hypotheses = self._joint_hypotheses(
            record,
            sources,
        )

        # 中心线候选始终优先；只有中心线点不能评分或已测过时
        # 才尝试小角度备选点。
        for candidate in candidates:
            point = tuple(candidate["point"])

            conditional = score_candidate(
                record,
                candidate,
                sources,
                estimate,
                tuple(self.client.position),
                self.client.current_channel,
                self.cfg,
            )
            if conditional is None:
                continue

            model_probability = (
                self._joint_receive_probability(
                    point,
                    hypotheses,
                )
                if hypotheses
                else float(
                    conditional[
                        "robust_receive_score"
                    ]
                )
            )

            travel_distance_m = math.dist(
                self.client.position,
                point,
            )
            expected_value_m = (
                model_probability
                * (
                    float(
                        conditional[
                            "radius_reduction"
                        ]
                    )
                    + self.tuning.single_bearing_completion_value_m
                )
            )
            information_score = (
                expected_value_m
                / max(
                    float(
                        conditional["time_cost_s"]
                    ),
                    1e-9,
                )
            )

            return {
                **conditional,
                **candidate,
                "legacy_robust_receive_score": (
                    conditional[
                        "robust_receive_score"
                    ]
                ),
                "robust_receive_score": (
                    model_probability
                ),
                "joint_receive_probability": (
                    model_probability
                ),
                "information_expected_value_m": (
                    expected_value_m
                ),
                "information_score": (
                    information_score
                ),
                "weighted_score": information_score,
                "travel_distance_m": (
                    travel_distance_m
                ),
                "planned_from": tuple(
                    self.client.position
                ),
                "purpose": (
                    "recovery_single_bearing_safe_reacquire"
                ),
                "selection_stage": (
                    "recovery_single_bearing_safe_reacquire"
                ),
                "candidate_count": len(candidates),
                "scored_candidate_count": 1,
                "eligible_candidate_count": 1,
                "source_sample_count": len(sources),
                "hypothesis_count": len(
                    hypotheses
                ),
                "estimate": estimate,
                "receive_floor": None,
                "move_cap_m": None,
                "single_bearing_safe": True,
                "surprise_reference_probability": max(
                    model_probability,
                    self.tuning.single_bearing_reference_probability,
                ),
                "score_scope": (
                    "single_bearing_short_centerline_reacquire;"
                    "probability_calibrated_for_scheduler;"
                    "strict_geometry_unchanged"
                ),
            }

        return None

    def plan_active_measurement(
        self,
        record: ChannelRecord,
        *,
        record_plan: bool = True,
        allow_new_q2_seed: bool = True,
    ) -> dict:
        if (
            record.status == "FOUND"
            and len(record.bearing_observations) == 1
        ):
            proposal = (
                self._build_single_bearing_proposal(
                    record
                )
            )
            if proposal is not None:
                self.counts[
                    "recovery_single_bearing_plan"
                ] += 1
                if record_plan:
                    self._record_active_plan(
                        record,
                        proposal,
                    )
                return proposal

        return super().plan_active_measurement(
            record,
            record_plan=record_plan,
            allow_new_q2_seed=(
                allow_new_q2_seed
            ),
        )

    def _active_scheduler_task(
        self,
        record: ChannelRecord,
        proposal: dict,
    ) -> dict:
        task = super()._active_scheduler_task(
            record,
            proposal,
        )

        if not proposal.get(
            "single_bearing_safe",
            False,
        ):
            return task

        probability = max(
            0.55,
            float(
                proposal.get(
                    "joint_receive_probability",
                    proposal.get(
                        "robust_receive_score",
                        0.0,
                    ),
                )
            ),
        )
        reacquisition_bonus_m = (
            self.tuning.single_bearing_completion_value_m
            * probability
        )

        task["expected_value_m"] += (
            reacquisition_bonus_m
        )
        task["reacquisition_bonus_m"] = (
            reacquisition_bonus_m
        )
        task["base_score"] = (
            task["expected_value_m"]
            / max(task["time_cost_s"], 1e-9)
        )
        task["scheduler_score"] = (
            task["base_score"]
            * self._scheduler_age_factor(record)
        )
        task["recovery_single_bearing_priority"] = True
        return task

    def _record_active_plan(
        self,
        record: ChannelRecord,
        proposal: dict,
        **scheduler_data,
    ) -> None:
        probability = float(
            proposal.get(
                "surprise_reference_probability",
                proposal.get(
                    "joint_receive_probability",
                    proposal.get(
                        "robust_receive_score",
                        0.0,
                    ),
                ),
            )
        )

        self._active_execution_prediction[
            record.channel_id
        ] = {
            "probability": probability,
            "single_bearing_safe": bool(
                proposal.get(
                    "single_bearing_safe",
                    False,
                )
            ),
            "point": tuple(proposal["point"]),
            "origin": proposal.get("origin"),
            "selection_stage": proposal.get(
                "selection_stage"
            ),
        }

        super()._record_active_plan(
            record,
            proposal,
            **scheduler_data,
        )

    # ================================================================
    # 优先级2：恢复点元数据和过期重规划
    # ================================================================

    def _build_information_recovery_point(
        self,
        record: ChannelRecord,
    ) -> Point:
        receive_floor = self._adaptive_receive_floor(
            record,
            self.tuning.recovery_receive_floor,
        )
        move_cap_m = self._adaptive_move_cap(
            record,
            self.tuning.recovery_move_cap_m,
            self.tuning.adaptive_recovery_min_move_cap_m,
        )

        proposal = self._select_information_measurement(
            record,
            allow_new_q2_seed=False,
            receive_floor=receive_floor,
            move_cap_m=move_cap_m,
            strict_limits=True,
            purpose=(
                "recovery_calibrated_information_recovery"
            ),
        )

        planned_from = tuple(
            self.client.position
        )
        point = tuple(proposal["point"])

        self._recovery_plan_meta[
            record.channel_id
        ] = {
            "point": point,
            "planned_from": planned_from,
            "move_cap_m": move_cap_m,
            "probability": float(
                proposal[
                    "joint_receive_probability"
                ]
            ),
            "origin": proposal.get("origin"),
            "created_virtual_time_s": (
                self.client.virtual_time_s
            ),
        }

        self.counts[
            "adaptive_recovery_plan"
        ] += 1
        self.counts[
            "recovery_plan_recorded"
        ] += 1

        self.event(
            "information_recovery_planned",
            channel=record.channel_id,
            point=point,
            planned_from=planned_from,
            joint_receive_probability=(
                proposal[
                    "joint_receive_probability"
                ]
            ),
            information_score=(
                proposal["information_score"]
            ),
            predicted_radius_reduction=(
                proposal["radius_reduction"]
            ),
            travel_distance_m=(
                proposal["travel_distance_m"]
            ),
            adaptive_receive_floor=receive_floor,
            adaptive_move_cap_m=move_cap_m,
            probability_calibration=(
                "likelihood_weighted_position_groups;"
                "lower_quantile_blend;"
                "empirical_model_reliability"
            ),
        )
        return point

    def _discard_information_recovery(
        self,
        record: ChannelRecord,
        reason: str,
        **diagnostics,
    ) -> None:
        self._recovery_queues.pop(
            record.channel_id,
            None,
        )
        self._recovery_plan_meta.pop(
            record.channel_id,
            None,
        )
        record.recovery_active = False
        record.recovery_mode = None

        self.event(
            "recovery_information_recovery_discarded",
            channel=record.channel_id,
            reason=reason,
            **diagnostics,
        )

    def _ensure_recovery_plan_fresh(
        self,
        record: ChannelRecord,
    ) -> None:
        if (
            record.recovery_mode
            != "information_reacquire"
        ):
            return

        queue = self._recovery_queues.get(
            record.channel_id
        )
        if not queue:
            return

        meta = self._recovery_plan_meta.get(
            record.channel_id
        )
        if meta is None:
            self._discard_information_recovery(
                record,
                "missing_plan_metadata",
            )
            self.counts[
                "recovery_stale_recovery_replanned"
            ] += 1
            self.start_recovery(record)
            return

        current = tuple(self.client.position)
        point = tuple(queue[0])
        planned_from = tuple(
            meta["planned_from"]
        )

        moved_since_planning_m = math.dist(
            current,
            planned_from,
        )
        current_travel_m = math.dist(
            current,
            point,
        )

        current_move_cap_m = (
            self._adaptive_move_cap(
                record,
                self.tuning.recovery_move_cap_m,
                self.tuning.adaptive_recovery_min_move_cap_m,
            )
        )

        stale_by_origin = (
            moved_since_planning_m
            > self.tuning.active_cache_replan_distance_m
        )
        stale_by_move_cap = (
            current_travel_m
            > (
                current_move_cap_m
                * self.tuning.recovery_stale_cap_ratio
            )
        )

        if not (
            stale_by_origin
            or stale_by_move_cap
        ):
            return

        self._discard_information_recovery(
            record,
            "robot_moved_after_recovery_planning",
            moved_since_planning_m=(
                moved_since_planning_m
            ),
            current_travel_m=current_travel_m,
            current_move_cap_m=current_move_cap_m,
            stale_by_origin=stale_by_origin,
            stale_by_move_cap=(
                stale_by_move_cap
            ),
        )
        self.counts[
            "recovery_stale_recovery_replanned"
        ] += 1
        self.start_recovery(record)

    def start_recovery(
        self,
        record: ChannelRecord,
    ) -> None:
        existing = self._recovery_queues.get(
            record.channel_id
        )

        # 高置信失败达到阈值时，即使已经存在信息恢复点，
        # 也不再继续依赖失准的概率模型。
        surprise_count = (
            self._surprise_misses.get(
                record.channel_id,
                0,
            )
        )

        if (
            surprise_count
            >= self.tuning.surprise_no_signal_limit
            and record.recovery_mode
            != "coarse_grid_last_resort"
        ):
            if existing:
                self._discard_information_recovery(
                    record,
                    "surprise_no_signal_limit_reached",
                    surprise_count=surprise_count,
                )

            self._active_plan_cache.pop(
                record.channel_id,
                None,
            )
            self._start_grid_fallback(
                record,
                (
                    "recovery_high_confidence_model_failure:"
                    f"{surprise_count}"
                ),
            )
            self.counts[
                "recovery_surprise_fast_fallback"
            ] += 1

            self.event(
                "recovery_fast_fallback_started",
                channel=record.channel_id,
                surprise_count=surprise_count,
                fallback_mode=(
                    "coarse_grid_last_resort"
                ),
            )
            return

        if (
            record.recovery_active
            and existing
        ):
            return

        super().start_recovery(record)

    def _recovery_scheduler_task(
        self,
        record: ChannelRecord,
    ) -> dict:
        self._ensure_recovery_plan_fresh(
            record
        )

        task = super()._recovery_scheduler_task(
            record
        )

        if (
            record.recovery_mode
            != "information_reacquire"
        ):
            return task

        # 信息恢复是完成该频道的必需动作。
        # 对当前仍在局部移动上限内的恢复点提高价值，
        # 防止调度器先去地图另一侧清除多个READY源。
        expected_value_m = max(
            float(task["expected_value_m"]),
            self.tuning.recovery_locality_value_m,
        )

        if (
            expected_value_m
            > float(task["expected_value_m"])
            + 1e-9
        ):
            self.counts[
                "recovery_locality_boost"
            ] += 1

        task["expected_value_m"] = (
            expected_value_m
        )
        task["base_score"] = (
            expected_value_m
            / max(task["time_cost_s"], 1e-9)
        )
        task["scheduler_score"] = (
            task["base_score"]
            * self._scheduler_age_factor(record)
        )
        task["recovery_locality_boost"] = True
        return task

    def recovery_step(
        self,
        record: ChannelRecord,
    ) -> str:
        """选中一次信息恢复后，尽量连续完成局部恢复探针。"""
        last_result = "no_signal"
        executed_count = 0

        for burst_index in range(
            self.tuning.recovery_burst_limit
        ):
            self._ensure_recovery_plan_fresh(
                record
            )

            queue = self._recovery_queues.get(
                record.channel_id
            )
            if not record.recovery_active or not queue:
                self.start_recovery(record)
                queue = self._recovery_queues.get(
                    record.channel_id
                )

            if not queue:
                raise IncompleteRun(
                    f"频道{record.channel_id}没有恢复候选"
                )

            mode_before = record.recovery_mode

            last_result = super().recovery_step(
                record
            )
            executed_count += 1

            if burst_index > 0:
                self.counts[
                    "recovery_burst_extra_measure"
                ] += 1

            if last_result != "no_signal":
                break
            if record.status != "FOUND":
                break

            # 粗网格兜底保持逐步调度，避免一次调用执行过多节点。
            if (
                mode_before
                != "information_reacquire"
            ):
                break

            if (
                burst_index + 1
                >= self.tuning.recovery_burst_limit
            ):
                break

            # 生成下一局部信息恢复点。
            self.start_recovery(record)
            next_queue = self._recovery_queues.get(
                record.channel_id
            )

            # 连续失败触发粗网格时，将粗网格留给下一轮调度。
            if (
                record.recovery_mode
                != "information_reacquire"
                or not next_queue
            ):
                break

            next_point = tuple(next_queue[0])
            next_leg_m = math.dist(
                self.client.position,
                next_point,
            )
            if (
                next_leg_m
                > self.tuning.recovery_burst_leg_cap_m
            ):
                break

        if executed_count > 1:
            self.counts[
                "recovery_burst_round"
            ] += 1
            self.event(
                "recovery_burst_completed",
                channel=record.channel_id,
                executed_count=executed_count,
                last_result=last_result,
                status_after=record.status,
                recovery_mode_after=(
                    record.recovery_mode
                ),
            )

        return last_result

    # ================================================================
    # 优先级3：高置信失败统计和快速降级
    # ================================================================

    def measure(
        self,
        record: ChannelRecord,
        point: Point,
        *,
        measurement_kind: str,
        global_index: int | None = None,
    ) -> str:
        prediction: dict | None = None

        if measurement_kind == "active":
            prediction = (
                self._active_execution_prediction.get(
                    record.channel_id
                )
            )
        elif (
            measurement_kind == "recovery"
            and record.recovery_mode
            == "information_reacquire"
        ):
            meta = self._recovery_plan_meta.get(
                record.channel_id
            )
            if meta is not None:
                prediction = {
                    "probability": meta.get(
                        "probability",
                        0.0,
                    ),
                    "single_bearing_safe": False,
                    "point": meta.get("point"),
                    "origin": meta.get("origin"),
                }

        result = super().measure(
            record,
            point,
            measurement_kind=measurement_kind,
            global_index=global_index,
        )

        if measurement_kind == "active":
            self._active_execution_prediction.pop(
                record.channel_id,
                None,
            )
        elif measurement_kind == "recovery":
            self._recovery_plan_meta.pop(
                record.channel_id,
                None,
            )

        if prediction is not None and prediction.get(
            "single_bearing_safe",
            False,
        ):
            self.counts[
                "recovery_single_bearing_measure"
            ] += 1
            if result in {"direction", "near"}:
                self.counts[
                    "recovery_single_bearing_positive"
                ] += 1
            elif result == "no_signal":
                self.counts[
                    "recovery_single_bearing_no_signal"
                ] += 1

        if measurement_kind not in {
            "active",
            "recovery",
        }:
            return result

        predicted_probability = float(
            prediction.get("probability", 0.0)
            if prediction is not None
            else 0.0
        )

        if result in {"direction", "near"}:
            self._surprise_misses[
                record.channel_id
            ] = 0
            self._information_attempts[
                record.channel_id
            ] = 0
            return result

        if (
            result == "no_signal"
            and predicted_probability
            >= self.tuning.surprise_no_signal_probability
        ):
            surprise_count = (
                self._surprise_misses.get(
                    record.channel_id,
                    0,
                )
                + 1
            )
            self._surprise_misses[
                record.channel_id
            ] = surprise_count
            self.counts[
                "recovery_surprise_no_signal"
            ] += 1

            self.event(
                "recovery_surprise_no_signal",
                channel=record.channel_id,
                measurement_kind=measurement_kind,
                point=point,
                predicted_probability=(
                    predicted_probability
                ),
                surprise_count=surprise_count,
                fast_fallback_threshold=(
                    self.tuning.surprise_no_signal_limit
                ),
            )

            if (
                surprise_count
                >= self.tuning.surprise_no_signal_limit
            ):
                # 下一轮任务构造直接进入恢复，不再等待普通
                # no_progress_trigger慢慢累计。
                record.no_progress_count = max(
                    record.no_progress_count,
                    self.cfg.no_progress_trigger,
                )

                if measurement_kind == "recovery":
                    self._information_attempts[
                        record.channel_id
                    ] = max(
                        self._information_attempts.get(
                            record.channel_id,
                            0,
                        ),
                        self.tuning.recovery_information_probe_limit,
                    )

        return result

    # ================================================================
    # 汇总
    # ================================================================

    def summary(
        self,
        outcome: str,
        reason: str,
        exit_confirmed: bool,
    ) -> dict:
        result = super().summary(
            outcome,
            reason,
            exit_confirmed,
        )

        result["recovery_tuning"] = asdict(self.tuning)
        result["recovery_optimization"] = {
            "single_bearing_plan_count": (
                self.counts[
                    "recovery_single_bearing_plan"
                ]
            ),
            "single_bearing_measure_count": (
                self.counts[
                    "recovery_single_bearing_measure"
                ]
            ),
            "single_bearing_positive_count": (
                self.counts[
                    "recovery_single_bearing_positive"
                ]
            ),
            "single_bearing_no_signal_count": (
                self.counts[
                    "recovery_single_bearing_no_signal"
                ]
            ),
            "surprise_no_signal_count": (
                self.counts[
                    "recovery_surprise_no_signal"
                ]
            ),
            "surprise_fast_fallback_count": (
                self.counts[
                    "recovery_surprise_fast_fallback"
                ]
            ),
            "stale_recovery_replanned_count": (
                self.counts[
                    "recovery_stale_recovery_replanned"
                ]
            ),
            "recovery_burst_round_count": (
                self.counts[
                    "recovery_burst_round"
                ]
            ),
            "recovery_burst_extra_measure_count": (
                self.counts[
                    "recovery_burst_extra_measure"
                ]
            ),
            "likelihood_weighted_hypothesis_build_count": (
                self.counts[
                    "recovery_likelihood_weighted_hypothesis_build"
                ]
            ),
            "certificate_preserved": (
                self.search_complete
                or self.discovered_count
                >= self.cfg.source_count_max
            ),
        }

        result["model_note"] = (
            result.get("model_note", "")
            + "；安全恢复层对单示向频道优先执行短距离中心线重捕获；"
            "information_reacquire恢复点在机器人移动后重新验证；"
            "局部信息恢复支持有界成组执行；联合粒子按照原始先验"
            "和观测存活似然加权，并使用位置组低分位概率和经验"
            "可靠度校准；连续高置信失败后快速转入确定性粗网格恢复"
        )
        return result


def _run_recovery_self_test() -> dict:
    result = _run_priority_search_self_test()
    tuning = _RecoveryTuning()

    assert tuning.single_bearing_first_step_m > 0.0
    assert tuning.single_bearing_second_step_m > 0.0
    assert tuning.single_bearing_probe_limit == 2
    assert tuning.recovery_information_probe_limit <= 3
    assert tuning.recovery_burst_limit > 0
    assert tuning.orientation_particle_count >= 72
    assert (
        0.0
        <= tuning.posterior_receive_quantile
        <= 1.0
    )

    quantile = _RecoveryRunner._weighted_quantile(
        [
            (0.1, 0.2),
            (0.5, 0.5),
            (0.9, 0.3),
        ],
        0.20,
    )
    assert abs(quantile - 0.1) <= 1e-9

    return {
        **result,
        "recovery_status": "ok",
        "recovery_single_bearing_safe_reacquire": True,
        "recovery_stale_recovery_replanning": True,
        "recovery_local_recovery_burst": True,
        "recovery_likelihood_weighted_posterior": True,
        "recovery_surprise_fast_fallback": True,
        "recovery_certificate_nodes_removed": False,
    }

@dataclass(frozen=True)
class V8Tuning(_RecoveryTuning):
    # ================================================================
    # 优先级0：最新成功点局部闭环
    # ================================================================

    beam_track_steps_m: tuple[float, ...] = (
        70.0,
        45.0,
        25.0,
    )
    beam_track_offsets_deg: tuple[float, ...] = (
        10.0,
        -12.0,
        8.0,
    )
    beam_track_failure_limit: int = 2

    # 安全恢复层的两个高置信失败过于敏感，V8提高到3次。
    surprise_no_signal_limit: int = 3

    # ================================================================
    # 优先级1：恢复网格动态排序与局部提交
    # ================================================================

    coarse_anchor_weight: float = 0.65
    coarse_anchor_prefix_count: int = 4
    coarse_route_ratio_cap: float = 1.08

    coarse_commit_limit: int = 2
    coarse_commit_leg_cap_m: float = 550.0

    # ================================================================
    # 优先级2：安全重捕获无回退门槛
    # ================================================================

    safe_max_incremental_detour_m: float = 550.0
    safe_absolute_move_cap_m: float = 1800.0
    safe_bonus_decay_m: float = 500.0

    # ================================================================
    # 优先级3：搜索节点时间收益与路线保护
    # ================================================================

    search_unknown_value_cap_m: float = 5000.0
    search_route_free_regret_m: float = 120.0
    search_route_max_regret_m: float = 320.0
    search_route_value_cover_ratio: float = 1.15
    search_route_regret_penalty: float = 1.0

    # ================================================================
    # 优先级4：严格证书下的点位优化
    # ================================================================

    certificate_edge_margin_m: float = 0.50

    def __post_init__(self):
        super().__post_init__()

        if (
            len(self.beam_track_steps_m)
            != len(self.beam_track_offsets_deg)
        ):
            raise ValueError(
                "beam_track_steps_m与"
                "beam_track_offsets_deg长度必须相同"
            )

        if not self.beam_track_steps_m:
            raise ValueError("局部闭环至少需要一个探针")

        for value in self.beam_track_steps_m:
            if (
                not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(
                    "beam_track_steps_m必须为有限正数"
                )

        for value in self.beam_track_offsets_deg:
            if not math.isfinite(float(value)):
                raise ValueError(
                    "beam_track_offsets_deg必须为有限数"
                )

        for name in (
            "beam_track_failure_limit",
            "coarse_anchor_prefix_count",
            "coarse_commit_limit",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}必须为正整数")

        for name in (
            "coarse_anchor_weight",
            "coarse_route_ratio_cap",
            "coarse_commit_leg_cap_m",
            "safe_max_incremental_detour_m",
            "safe_absolute_move_cap_m",
            "safe_bonus_decay_m",
            "search_unknown_value_cap_m",
            "search_route_free_regret_m",
            "search_route_max_regret_m",
            "search_route_value_cover_ratio",
            "search_route_regret_penalty",
            "certificate_edge_margin_m",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{name}必须为有限正数")

        if self.coarse_route_ratio_cap < 1.0:
            raise ValueError(
                "coarse_route_ratio_cap不能小于1"
            )

        if (
            self.search_route_free_regret_m
            > self.search_route_max_regret_m
        ):
            raise ValueError(
                "免费路线后悔值不能超过最大路线后悔值"
            )


def build_optimized_certified_search(
    config: Q4Config,
    tuning: V8Tuning,
) -> tuple[GridPlan, dict]:
    """构造经过严格验证的25点双环认证网格。

    不减少认证点，只优化内环半径和开放路线。
    """

    baseline_plan, baseline_diagnostics = (
        build_dual_ring_search(
            config,
            tuning,
        )
    )

    count = tuning.ring_point_count
    if count != 12:
        raise ValueError(
            "V8严格优化目前只对12+12双环结构提供证明"
        )

    step = 2.0 * math.pi / count
    half_step = math.pi / count

    target_support_radius = (
        config.target_radius_m
        + tuning.outer_tangent_margin_m
    )
    outer_radius = (
        target_support_radius
        / math.cos(half_step)
    )

    reception_radius = (
        config.reception_radius_min_m
    )

    # 环间交叉边相对于内外环的夹角为 half_step。
    # 解：
    # r^2 - 2R*cos(a)*r + R^2 - D^2 <= 0
    discriminant = (
        reception_radius * reception_radius
        - (
            outer_radius
            * math.sin(half_step)
        ) ** 2
    )
    if discriminant <= 0.0:
        raise ValueError(
            "给定外环下不存在满足接收半径的内环"
        )

    minimum_inner_radius = (
        outer_radius * math.cos(half_step)
        - math.sqrt(discriminant)
    )

    inner_radius = (
        minimum_inner_radius
        + tuning.certificate_edge_margin_m
    )

    if (
        inner_radius
        >= reception_radius
        - 1e-7
    ):
        raise ValueError(
            "优化后的内环半径不能由原点直接认证"
        )

    origin: Point = (0.0, 0.0)

    inner = [
        polar_point(
            inner_radius,
            index * step,
        )
        for index in range(count)
    ]
    outer = [
        polar_point(
            outer_radius,
            (index + 0.5) * step,
        )
        for index in range(count)
    ]

    triangles: list[Triangle] = []

    for index in range(count):
        next_index = (index + 1) % count
        triangles.append((
            origin,
            inner[index],
            inner[next_index],
        ))

    for index in range(count):
        next_index = (index + 1) % count

        triangles.append((
            inner[index],
            outer[index],
            inner[next_index],
        ))
        triangles.append((
            inner[next_index],
            outer[index],
            outer[next_index],
        ))

    max_edge = max(
        triangle_max_edge(triangle)
        for triangle in triangles
    )

    if max_edge > reception_radius + 1e-7:
        raise ValueError(
            f"V8认证三角形最大边{max_edge:.6f}米，"
            f"超过{reception_radius:.6f}米"
        )

    outer_min_distance = min(
        origin_segment_distance(
            outer[index],
            outer[(index + 1) % count],
        )
        for index in range(count)
    )

    if (
        outer_min_distance
        < config.target_radius_m - 1e-7
    ):
        raise ValueError(
            "V8外环没有完整包含目标圆盘"
        )

    points = [
        origin,
        *inner,
        *outer,
    ]

    structured_route = [
        origin,
        *inner,
        *reversed(outer),
    ]

    nearest_route = nearest_neighbor_route(
        points,
        origin,
    )

    route_candidates = [
        structured_route,
        two_opt_open(
            structured_route,
            origin,
        ),
        two_opt_open(
            nearest_route,
            origin,
        ),
    ]

    route = min(
        route_candidates,
        key=lambda candidate: (
            open_route_distance(
                origin,
                candidate,
            ),
            candidate,
        ),
    )

    if (
        len(route) != len(points)
        or set(route) != set(points)
    ):
        raise ValueError(
            "V8优化路线没有完整访问全部认证点"
        )

    route_distance = open_route_distance(
        origin,
        route,
    )

    # 若数值环境下没有获得收益，直接使用安全恢复层原网格。
    if (
        route_distance
        >= baseline_plan.route_distance_m - 1e-7
    ):
        diagnostics = {
            **baseline_diagnostics,
            "type": (
                "dual_ring_certified_triangulation_"
                "final_baseline_fallback"
            ),
            "final_geometry_optimized": False,
            "final_fallback_reason": (
                "optimized_route_not_shorter"
            ),
            "baseline_route_distance_m": (
                baseline_plan.route_distance_m
            ),
            "route_improvement_m": 0.0,
            "certificate_node_reduction": False,
            "strict_runtime_validation": True,
        }
        return baseline_plan, diagnostics

    plan = GridPlan(
        side_m=max_edge,
        points=points,
        route=list(route),
        triangle_count=len(triangles),
        route_distance_m=route_distance,
        triangles=triangles,
    )

    diagnostics = {
        "type": (
            "final_optimized_dual_ring_"
            "certified_triangulation"
        ),
        "point_count": len(points),
        "triangle_count": len(triangles),
        "inner_radius_m": inner_radius,
        "minimum_feasible_inner_radius_m": (
            minimum_inner_radius
        ),
        "outer_radius_m": outer_radius,
        "outer_boundary_min_distance_m": (
            outer_min_distance
        ),
        "max_certificate_edge_m": max_edge,
        "route_distance_m": route_distance,
        "baseline_route_distance_m": (
            baseline_plan.route_distance_m
        ),
        "route_improvement_m": (
            baseline_plan.route_distance_m
            - route_distance
        ),
        "certificate_node_reduction": False,
        "strict_runtime_validation": True,
    }

    return plan, diagnostics


class Q4V8Runner(_RecoveryRunner):
    def __init__(
        self,
        client,
        *,
        config: Q4Config,
        tuning: V8Tuning | None = None,
        planner_config=None,
        geometry_config=None,
        output_dir: Path | None = None,
    ):
        self.tuning = tuning or V8Tuning()

        super().__init__(
            client,
            config=config,
            tuning=self.tuning,
            planner_config=planner_config,
            geometry_config=geometry_config,
            output_dir=output_dir,
        )

        self._beam_exhausted: set[int] = set()

        self.counts.update({
            "final_beam_track_round": 0,
            "final_beam_track_measure": 0,
            "final_beam_track_direction": 0,
            "final_beam_track_no_signal": 0,
            "final_beam_track_ready": 0,
            "final_beam_track_forced_fallback": 0,
            "final_coarse_grid_reordered": 0,
            "final_coarse_batch_round": 0,
            "final_coarse_batch_extra_measure": 0,
            "final_safe_plan_deferred": 0,
            "final_safe_bonus_discounted": 0,
            "final_search_gate_evaluation_rejected": 0,
            "final_search_expected_measurements_saved": 0.0,
            "final_certificate_fallback": 0,
        })

        try:
            (
                optimized_plan,
                optimized_diagnostics,
            ) = build_optimized_certified_search(
                self.cfg,
                self.tuning,
            )

            self.global_grid = optimized_plan
            self.dual_ring_diagnostics = (
                optimized_diagnostics
            )
            self.search_points = list(
                optimized_plan.route
            )

            # 新点集必须清除优先调度层按节点编号保存的概率缓存。
            self._unknown_probability_cache.clear()
            self._unknown_omni_masks = None
            self._unknown_directional_masks = None

        except Exception as error:
            # 保留super初始化得到的安全恢复层原始网格。
            self.counts[
                "final_certificate_fallback"
            ] += 1

            self.dual_ring_diagnostics = {
                **self.dual_ring_diagnostics,
                "final_geometry_optimized": False,
                "final_fallback_reason": repr(error),
                "certificate_node_reduction": False,
                "strict_runtime_validation": True,
            }

    # ================================================================
    # 优先级0：成功重捕获后的局部闭环
    # ================================================================

    @staticmethod
    def _bearing_step_point(
        observation: dict,
        step_m: float,
        offset_deg: float,
    ) -> Point:
        angle = math.radians(
            float(observation["bearing_deg"])
            + offset_deg
        )
        origin = tuple(observation["position"])

        return (
            origin[0] + step_m * math.cos(angle),
            origin[1] + step_m * math.sin(angle),
        )

    def _beam_track_transaction(
        self,
        record: ChannelRecord,
    ) -> None:
        """以最新成功方向为锚点，连续执行小范围闭环。"""

        if (
            record.status != "FOUND"
            or not record.bearing_observations
        ):
            return

        self.counts["final_beam_track_round"] += 1
        consecutive_failures = 0

        self.event(
            "final_beam_track_started",
            channel=record.channel_id,
            bearing_count=len(
                record.bearing_observations
            ),
            anchor=record.bearing_observations[
                -1
            ]["position"],
        )

        for probe_index, (
            step_m,
            offset_deg,
        ) in enumerate(
            zip(
                self.tuning.beam_track_steps_m,
                self.tuning.beam_track_offsets_deg,
            ),
            start=1,
        ):
            if record.status != "FOUND":
                break

            self.check_budget()

            anchor_observation = (
                record.bearing_observations[-1]
            )

            offsets = (
                offset_deg,
                -offset_deg,
                0.0,
            )

            point = None
            selected_offset = None

            for candidate_offset in offsets:
                candidate = self._bearing_step_point(
                    anchor_observation,
                    step_m,
                    candidate_offset,
                )
                if not self._point_was_tested(
                    record,
                    candidate,
                ):
                    point = candidate
                    selected_offset = (
                        candidate_offset
                    )
                    break

            if point is None:
                continue

            travel_distance_m = math.dist(
                self.client.position,
                point,
            )
            time_cost_s = (
                travel_distance_m
                / self.cfg.dog_speed_m_per_s
                + self.cfg.detection_time_s
                + (
                    self.cfg.channel_switch_time_s
                    if (
                        self.client.current_channel
                        != record.channel_id
                    )
                    else 0.0
                )
            )

            proposal = {
                "point": point,
                "origin": (
                    "final_latest_positive_anchor_track"
                ),
                "purpose": (
                    "final_local_beam_track_transaction"
                ),
                "selection_stage": (
                    "final_latest_positive_anchor_track"
                ),
                "beam_track_probe_index": (
                    probe_index
                ),
                "beam_track_step_m": step_m,
                "beam_track_offset_deg": (
                    selected_offset
                ),
                "planned_from": tuple(
                    self.client.position
                ),
                "travel_distance_m": (
                    travel_distance_m
                ),
                "time_cost_s": time_cost_s,
                "robust_receive_score": 0.60,
                "joint_receive_probability": 0.60,
                "radius_reduction": 0.0,
                "side_penalty": 0,
                "predicted_clearable": False,
                "single_bearing_safe": False,
                "score_scope": (
                    "latest_positive_anchor;"
                    "bounded_local_transaction;"
                    "strict_geometry_unchanged"
                ),
            }

            self._record_active_plan(
                record,
                proposal,
                scheduling_basis=(
                    "final_committed_local_beam_track"
                ),
                scheduler_score=None,
                scheduler_expected_value_m=None,
                scheduler_travel_distance_m=(
                    travel_distance_m
                ),
            )

            result = self.measure(
                record,
                point,
                measurement_kind="active",
            )

            self.counts[
                "final_beam_track_measure"
            ] += 1

            if result in {"direction", "near"}:
                consecutive_failures = 0
                self.counts[
                    "final_beam_track_direction"
                ] += int(result == "direction")
            else:
                consecutive_failures += 1
                self.counts[
                    "final_beam_track_no_signal"
                ] += 1

            self.event(
                "final_beam_track_result",
                channel=record.channel_id,
                probe_index=probe_index,
                point=point,
                result=result,
                consecutive_failures=(
                    consecutive_failures
                ),
                status=record.status,
                clearance_radius_m=(
                    record.clearance_radius
                ),
            )

            if record.status == "READY":
                self.counts[
                    "final_beam_track_ready"
                ] += 1

                if not self.clear(record):
                    record.no_progress_count = max(
                        record.no_progress_count,
                        self.cfg.no_progress_trigger,
                    )
                    self.start_recovery(record)
                return

            if (
                consecutive_failures
                >= self.tuning.beam_track_failure_limit
            ):
                self._beam_exhausted.add(
                    record.channel_id
                )

                # 局部安全走廊已经失败，不继续依赖概率模型，
                # 直接进入经过动态排序的确定性粗网格。
                self._information_attempts[
                    record.channel_id
                ] = (
                    self.tuning
                    .recovery_information_probe_limit
                )
                record.no_progress_count = max(
                    record.no_progress_count,
                    self.cfg.no_progress_trigger,
                )

                self.counts[
                    "final_beam_track_forced_fallback"
                ] += 1

                self.start_recovery(record)
                return

    def _execute_localization_task(
        self,
        task: dict,
    ) -> None:
        proposal = task.get("proposal") or {}
        is_safe_reacquisition = (
            task.get("action_type") == "active"
            and proposal.get(
                "single_bearing_safe",
                False,
            )
        )

        if not is_safe_reacquisition:
            super()._execute_localization_task(task)
            return

        record = task["record"]
        bearing_count_before = len(
            record.bearing_observations
        )

        # 直接调用基础执行层执行主体，暂不触发自适应搜索层的跨频道顺路清除；
        # 先完成当前频道的局部事务。
        _CoreRunner._execute_localization_task(
            self,
            task,
        )

        if record.status == "READY":
            self.clear(record)
        elif (
            record.status == "FOUND"
            and len(record.bearing_observations)
            > bearing_count_before
        ):
            self._beam_track_transaction(record)

        # 完成当前局部闭环后再恢复自适应搜索层/优先调度层顺路清除。
        self._clear_ready_nearby(
            self.tuning.localization_inline_clear_radius_m,
            "localization",
        )

    # ================================================================
    # 优先级1：粗网格动态排序
    # ================================================================

    @staticmethod
    def _last_positive_anchor(
        record: ChannelRecord,
    ) -> Point | None:
        if record.bearing_observations:
            return tuple(
                record.bearing_observations[
                    -1
                ]["position"]
            )
        if record.near_signal_observations:
            return tuple(
                record.near_signal_observations[
                    -1
                ]["position"]
            )
        return None

    def _coarse_prefix_route(
        self,
        points: Sequence[Point],
        current: Point,
        anchor: Point | None,
        prefix_count: int,
    ) -> list[Point]:
        remaining = [
            tuple(point)
            for point in points
        ]
        prefix: list[Point] = []
        position = tuple(current)

        prefix_count = min(
            prefix_count,
            len(remaining),
        )

        while (
            remaining
            and len(prefix) < prefix_count
        ):
            progress = (
                len(prefix)
                / max(1, prefix_count)
            )
            anchor_weight = (
                self.tuning.coarse_anchor_weight
                * (1.0 - 0.65 * progress)
            )

            selected = min(
                remaining,
                key=lambda point: (
                    math.dist(position, point)
                    + (
                        anchor_weight
                        * math.dist(anchor, point)
                        if anchor is not None
                        else 0.0
                    ),
                    math.dist(position, point),
                    point[1],
                    point[0],
                ),
            )

            prefix.append(selected)
            remaining.remove(selected)
            position = selected

        if remaining:
            tail = nearest_neighbor_route(
                remaining,
                position,
            )
            tail = two_opt_open(
                tail,
                position,
            )
        else:
            tail = []

        return [
            *prefix,
            *tail,
        ]

    def _start_grid_fallback(
        self,
        record: ChannelRecord,
        reason: str,
    ) -> None:
        super()._start_grid_fallback(
            record,
            reason,
        )

        queue = self._recovery_queues.get(
            record.channel_id
        )
        if not queue:
            return

        original_route = [
            tuple(point)
            for point in queue
        ]
        current = tuple(self.client.position)
        anchor = self._last_positive_anchor(
            record
        )

        original_distance_m = (
            open_route_distance(
                current,
                original_route,
            )
        )

        selected_route = original_route
        selected_prefix_count = 0

        if anchor is not None:
            for prefix_count in range(
                min(
                    self.tuning.coarse_anchor_prefix_count,
                    len(original_route),
                ),
                0,
                -1,
            ):
                candidate = (
                    self._coarse_prefix_route(
                        original_route,
                        current,
                        anchor,
                        prefix_count,
                    )
                )
                candidate_distance_m = (
                    open_route_distance(
                        current,
                        candidate,
                    )
                )

                if (
                    candidate_distance_m
                    <= (
                        original_distance_m
                        * self.tuning
                        .coarse_route_ratio_cap
                        + 1e-7
                    )
                ):
                    selected_route = candidate
                    selected_prefix_count = (
                        prefix_count
                    )
                    break
        else:
            candidate = (
                nearest_neighbor_route(
                    original_route,
                    current,
                )
            )
            candidate = two_opt_open(
                candidate,
                current,
            )

            if (
                open_route_distance(
                    current,
                    candidate,
                )
                <= original_distance_m + 1e-7
            ):
                selected_route = candidate

        if (
            len(selected_route)
            != len(original_route)
            or set(selected_route)
            != set(original_route)
        ):
            raise IncompleteRun(
                "V8粗网格重排丢失了认证节点"
            )

        self._recovery_queues[
            record.channel_id
        ] = deque(selected_route)

        reordered_distance_m = (
            open_route_distance(
                current,
                selected_route,
            )
        )

        self.counts[
            "final_coarse_grid_reordered"
        ] += 1

        self.event(
            "final_coarse_grid_reordered",
            channel=record.channel_id,
            anchor=anchor,
            point_count=len(selected_route),
            anchor_prefix_count=(
                selected_prefix_count
            ),
            original_route_distance_m=(
                original_distance_m
            ),
            reordered_route_distance_m=(
                reordered_distance_m
            ),
            first_points=selected_route[:5],
            same_certificate_point_set=True,
        )

    def recovery_step(
        self,
        record: ChannelRecord,
    ) -> str:
        if (
            record.recovery_mode
            != "coarse_grid_last_resort"
        ):
            return super().recovery_step(record)

        executed_count = 0
        last_result = "no_signal"

        for _ in range(
            self.tuning.coarse_commit_limit
        ):
            self.check_budget()

            last_result = (
                super().recovery_step(record)
            )
            executed_count += 1

            if last_result != "no_signal":
                break

            queue = self._recovery_queues.get(
                record.channel_id
            )
            if (
                not record.recovery_active
                or not queue
            ):
                break

            next_point = tuple(queue[0])
            if (
                math.dist(
                    self.client.position,
                    next_point,
                )
                > self.tuning.coarse_commit_leg_cap_m
            ):
                break

        if executed_count > 1:
            self.counts[
                "final_coarse_batch_round"
            ] += 1
            self.counts[
                "final_coarse_batch_extra_measure"
            ] += executed_count - 1

            self.event(
                "final_coarse_batch_completed",
                channel=record.channel_id,
                executed_count=executed_count,
                last_result=last_result,
                recovery_active=(
                    record.recovery_active
                ),
            )

        return last_result

    # ================================================================
    # 优先级2：安全重捕获无回退限制
    # ================================================================

    def _other_localization_anchors(
        self,
        record: ChannelRecord,
    ) -> list[Point]:
        anchors: list[Point] = []

        for other in self.records.values():
            if (
                other.channel_id
                == record.channel_id
                or other.status
                not in {"FOUND", "READY"}
            ):
                continue

            if (
                other.status == "READY"
                and other.clearance_center
                is not None
            ):
                anchors.append(
                    tuple(
                        other.clearance_center
                    )
                )
                continue

            queue = self._recovery_queues.get(
                other.channel_id
            )
            if other.recovery_active and queue:
                anchors.append(
                    tuple(queue[0])
                )
                continue

            cached = self._active_plan_cache.get(
                other.channel_id
            )
            if (
                cached is not None
                and cached.get("point")
                is not None
            ):
                anchors.append(
                    tuple(cached["point"])
                )
                continue

            if other.bearing_observations:
                anchors.append(
                    tuple(
                        other.bearing_observations[
                            -1
                        ]["position"]
                    )
                )

        return anchors

    @staticmethod
    def _insertion_detour_to_anchor(
        current: Point,
        inserted: Point,
        anchor: Point,
    ) -> float:
        return max(
            0.0,
            math.dist(current, inserted)
            + math.dist(inserted, anchor)
            - math.dist(current, anchor),
        )

    def _build_single_bearing_proposal(
        self,
        record: ChannelRecord,
    ) -> dict | None:
        proposal = (
            super()
            ._build_single_bearing_proposal(
                record
            )
        )
        if proposal is None:
            return None

        current = tuple(self.client.position)
        point = tuple(proposal["point"])
        travel_distance_m = math.dist(
            current,
            point,
        )

        anchors = (
            self._other_localization_anchors(
                record
            )
        )

        if anchors:
            incremental_detour_m = min(
                self._insertion_detour_to_anchor(
                    current,
                    point,
                    anchor,
                )
                for anchor in anchors
            )
        else:
            # 当前频道已是最后一个待处理频道时，
            # 不把前往该源的必要移动当成额外绕路。
            incremental_detour_m = 0.0

        if anchors and (
            incremental_detour_m
            > (
                self.tuning
                .safe_max_incremental_detour_m
            )
            or travel_distance_m
            > self.tuning.safe_absolute_move_cap_m
        ):
            self.counts[
                "final_safe_plan_deferred"
            ] += 1

            self.event(
                "final_safe_plan_deferred",
                channel=record.channel_id,
                point=point,
                travel_distance_m=(
                    travel_distance_m
                ),
                incremental_detour_m=(
                    incremental_detour_m
                ),
                other_anchor_count=len(anchors),
                detour_cap_m=(
                    self.tuning
                    .safe_max_incremental_detour_m
                ),
                absolute_cap_m=(
                    self.tuning
                    .safe_absolute_move_cap_m
                ),
            )
            return None

        proposal.update({
            "safe_route_detour_m": (
                incremental_detour_m
            ),
            "safe_other_anchor_count": (
                len(anchors)
            ),
            "safe_no_regret_gate_passed": True,
            "move_cap_m": (
                self.tuning.safe_absolute_move_cap_m
            ),
        })
        return proposal

    def _active_scheduler_task(
        self,
        record: ChannelRecord,
        proposal: dict,
    ) -> dict:
        task = super()._active_scheduler_task(
            record,
            proposal,
        )

        if not proposal.get(
            "single_bearing_safe",
            False,
        ):
            return task

        detour_m = float(
            proposal.get(
                "safe_route_detour_m",
                0.0,
            )
        )

        old_bonus_m = float(
            task.get(
                "reacquisition_bonus_m",
                0.0,
            )
        )

        adjusted_bonus_m = (
            old_bonus_m
            / (
                1.0
                + detour_m
                / self.tuning.safe_bonus_decay_m
            )
        )

        if adjusted_bonus_m < old_bonus_m - 1e-9:
            self.counts[
                "final_safe_bonus_discounted"
            ] += 1

        task["expected_value_m"] = max(
            0.0,
            float(task["expected_value_m"])
            - old_bonus_m
            + adjusted_bonus_m,
        )
        task["reacquisition_bonus_m"] = (
            adjusted_bonus_m
        )
        task["base_score"] = (
            task["expected_value_m"]
            / max(task["time_cost_s"], 1e-9)
        )
        task["scheduler_score"] = (
            task["base_score"]
            * self._scheduler_age_factor(record)
        )
        task["final_safe_route_detour_m"] = (
            detour_m
        )
        task["final_safe_bonus_discounted_m"] = (
            old_bonus_m - adjusted_bonus_m
        )

        return task

    # ================================================================
    # 优先级3：UNKNOWN时间收益与路线经济性门槛
    # ================================================================

    def _nearest_neighbor_tail_distance(
        self,
        start: Point,
        indices: Sequence[int],
    ) -> float:
        if not indices:
            return 0.0

        route = nearest_neighbor_route(
            [
                self.search_points[index]
                for index in indices
            ],
            start,
        )
        route = two_opt_open(
            route,
            start,
        )
        return open_route_distance(
            start,
            route,
        )

    def _unknown_node_value(
        self,
        candidate_index: int,
        *,
        future_node_count: int,
        early_stop_value_m: float,
    ) -> dict:
        metrics = super()._unknown_node_value(
            candidate_index,
            future_node_count=(
                future_node_count
            ),
            early_stop_value_m=(
                early_stop_value_m
            ),
        )

        expected_hits = float(
            metrics.get(
                "expected_hits",
                0.0,
            )
        )

        measurement_equivalent_m = (
            self.cfg.detection_time_s
            + self.cfg.channel_switch_time_s
        ) * self.cfg.dog_speed_m_per_s

        expected_saved_measurements = (
            expected_hits
            * future_node_count
        )

        # 不再使用优先调度层的0.4缩放，直接以预计节省时间计价。
        ordinary_value_m = (
            expected_saved_measurements
            * measurement_equivalent_m
        )

        early_stop_bonus_m = float(
            metrics.get(
                "early_stop_bonus_m",
                0.0,
            )
        )

        metrics["ordinary_value_m"] = (
            ordinary_value_m
        )
        metrics["expected_saved_measurements"] = (
            expected_saved_measurements
        )
        metrics["value_m"] = min(
            self.tuning.search_unknown_value_cap_m,
            ordinary_value_m
            + early_stop_bonus_m,
        )

        return metrics

    def _search_candidate_metrics(
        self,
        candidate_index: int,
        remaining: set[int],
        current: Point,
        baseline_tail_m: float,
    ) -> dict:
        metrics = super()._search_candidate_metrics(
            candidate_index,
            remaining,
            current,
            baseline_tail_m,
        )

        total_value_m = (
            float(metrics["localization_value"])
            + float(
                metrics[
                    "unknown_discovery_value"
                ]
            )
        )

        metrics[
            "expected_saved_measurements"
        ] = (
            float(
                metrics["unknown_expected_hits"]
            )
            * max(0, len(remaining) - 1)
        )

        metrics["combined_score"] = (
            total_value_m
            - self.tuning.search_route_regret_penalty
            * float(metrics["route_regret_m"])
        )

        return metrics

    def _normal_search_evaluations(
        self,
        remaining: set[int],
    ) -> list[dict]:
        current = tuple(self.client.position)

        ordered_by_distance = sorted(
            remaining,
            key=lambda index: (
                math.dist(
                    current,
                    self.search_points[index],
                ),
                index,
            ),
        )

        candidate_indices = ordered_by_distance[
            :self.tuning.dynamic_route_candidate_limit
        ]

        raw_evaluations = [
            self._search_candidate_metrics(
                candidate_index,
                remaining,
                current,
                math.inf,
            )
            for candidate_index
            in candidate_indices
        ]

        baseline_route_m = min(
            float(item["route_distance_m"])
            for item in raw_evaluations
        )

        baseline_item = min(
            raw_evaluations,
            key=lambda item: (
                item["route_distance_m"],
                item["index"],
            ),
        )

        accepted: list[dict] = []

        for item in raw_evaluations:
            regret_m = max(
                0.0,
                float(item["route_distance_m"])
                - baseline_route_m,
            )

            item["route_regret_m"] = regret_m

            total_value_m = (
                float(item["localization_value"])
                + float(
                    item[
                        "unknown_discovery_value"
                    ]
                )
            )

            item["combined_score"] = (
                total_value_m
                - self.tuning
                .search_route_regret_penalty
                * regret_m
            )

            economically_covered = (
                total_value_m
                >= (
                    self.tuning
                    .search_route_value_cover_ratio
                    * regret_m
                )
            )

            accepted_by_gate = (
                regret_m
                <= (
                    self.tuning
                    .search_route_free_regret_m
                )
                or (
                    regret_m
                    <= (
                        self.tuning
                        .search_route_max_regret_m
                    )
                    and economically_covered
                )
            )

            item[
                "final_route_economically_covered"
            ] = economically_covered
            item[
                "final_route_gate_passed"
            ] = accepted_by_gate

            if accepted_by_gate:
                accepted.append(item)
            else:
                self.counts[
                    "final_search_gate_evaluation_rejected"
                ] += 1

        if baseline_item not in accepted:
            baseline_item["route_regret_m"] = 0.0
            baseline_item["combined_score"] = (
                float(
                    baseline_item[
                        "localization_value"
                    ]
                )
                + float(
                    baseline_item[
                        "unknown_discovery_value"
                    ]
                )
            )
            baseline_item[
                "final_route_economically_covered"
            ] = True
            baseline_item[
                "final_route_gate_passed"
            ] = True
            accepted.append(baseline_item)

        return accepted

    def _select_next_search_node(
        self,
        remaining: set[int],
    ) -> tuple[int, dict]:
        index, selected = (
            super()._select_next_search_node(
                remaining
            )
        )

        saved_measurements = float(
            selected.get(
                "expected_saved_measurements",
                float(
                    selected.get(
                        "unknown_expected_hits",
                        0.0,
                    )
                )
                * max(0, len(remaining) - 1),
            )
        )

        self.counts[
            "final_search_expected_measurements_saved"
        ] += saved_measurements

        self.event(
            "final_search_economic_selection",
            global_index=index,
            expected_saved_measurements=(
                saved_measurements
            ),
            route_regret_m=selected.get(
                "route_regret_m"
            ),
            route_gate_passed=selected.get(
                "final_route_gate_passed",
                True,
            ),
            route_economically_covered=(
                selected.get(
                    "final_route_economically_covered",
                    True,
                )
            ),
        )

        return index, selected

    # ================================================================
    # 汇总
    # ================================================================

    def summary(
        self,
        outcome: str,
        reason: str,
        exit_confirmed: bool,
    ) -> dict:
        result = super().summary(
            outcome,
            reason,
            exit_confirmed,
        )

        result["final_tuning"] = asdict(self.tuning)
        result["final_optimization"] = {
            "beam_track_round_count": (
                self.counts[
                    "final_beam_track_round"
                ]
            ),
            "beam_track_measure_count": (
                self.counts[
                    "final_beam_track_measure"
                ]
            ),
            "beam_track_direction_count": (
                self.counts[
                    "final_beam_track_direction"
                ]
            ),
            "beam_track_no_signal_count": (
                self.counts[
                    "final_beam_track_no_signal"
                ]
            ),
            "beam_track_ready_count": (
                self.counts[
                    "final_beam_track_ready"
                ]
            ),
            "beam_track_forced_fallback_count": (
                self.counts[
                    "final_beam_track_forced_fallback"
                ]
            ),
            "coarse_grid_reordered_count": (
                self.counts[
                    "final_coarse_grid_reordered"
                ]
            ),
            "coarse_batch_round_count": (
                self.counts[
                    "final_coarse_batch_round"
                ]
            ),
            "coarse_batch_extra_measure_count": (
                self.counts[
                    "final_coarse_batch_extra_measure"
                ]
            ),
            "safe_plan_deferred_count": (
                self.counts[
                    "final_safe_plan_deferred"
                ]
            ),
            "safe_bonus_discounted_count": (
                self.counts[
                    "final_safe_bonus_discounted"
                ]
            ),
            "search_gate_rejected_count": (
                self.counts[
                    "final_search_gate_evaluation_rejected"
                ]
            ),
            "search_expected_measurements_saved": (
                self.counts[
                    "final_search_expected_measurements_saved"
                ]
            ),
            "certificate_fallback_count": (
                self.counts[
                    "final_certificate_fallback"
                ]
            ),
            "geometry_route_improvement_m": (
                self.dual_ring_diagnostics.get(
                    "route_improvement_m",
                    0.0,
                )
            ),
            "certificate_node_reduction": False,
            "certificate_preserved": (
                self.search_complete
                or self.discovered_count
                >= self.cfg.source_count_max
            ),
        }

        result["model_note"] = (
            result.get("model_note", "")
            + "；V8在安全重捕获成功后使用最新成功点进行有界"
            "局部闭环；局部失败后使用保留完整节点集合的锚点优先"
            "粗网格；安全重捕获受路线插入代价约束；UNKNOWN收益"
            "按预计节省的测量时间计算并受路线经济门槛保护；"
            "全局认证使用严格验证的25点优化双环，不删除证明节点"
        )

        return result


def run_self_test() -> dict:
    inherited = _run_recovery_self_test()

    tuning = V8Tuning()
    config = Q4Config(
        global_grid_side_m=1000.0,
    )

    plan, diagnostics = (
        build_optimized_certified_search(
            config,
            tuning,
        )
    )

    assert len(plan.points) == 25
    assert len(plan.route) == 25
    assert len(plan.triangles) == 36
    assert set(plan.route) == set(plan.points)

    maximum_edge = max(
        triangle_max_edge(triangle)
        for triangle in plan.triangles
    )
    assert (
        maximum_edge
        <= config.reception_radius_min_m
        + 1e-7
    )

    outer_distance = (
        diagnostics[
            "outer_boundary_min_distance_m"
        ]
    )
    assert (
        outer_distance
        >= config.target_radius_m - 1e-7
    )

    assert (
        diagnostics["route_distance_m"]
        <= diagnostics[
            "baseline_route_distance_m"
        ] + 1e-7
    )

    return {
        **inherited,
        "final_status": "ok",
        "final_latest_positive_closed_loop": True,
        "final_anchor_ordered_coarse_grid": True,
        "final_safe_no_regret_gate": True,
        "final_unknown_time_value": True,
        "final_search_route_economic_gate": True,
        "final_certificate_point_count": (
            len(plan.points)
        ),
        "final_certificate_triangle_count": (
            len(plan.triangles)
        ),
        "final_max_certificate_edge_m": (
            maximum_edge
        ),
        "final_outer_boundary_min_distance_m": (
            outer_distance
        ),
        "final_geometry_route_improvement_m": (
            diagnostics.get(
                "route_improvement_m",
                0.0,
            )
        ),
        "final_certificate_nodes_removed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
    )
    parser.add_argument("--robot-id")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:2026",
    )
    parser.add_argument(
        "--geometry-config",
        type=Path,
    )
    parser.add_argument(
        "--planner-config",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument(
        "--no-q2-seed",
        action="store_true",
    )
    parser.add_argument(
        "--scheduler-active-shortlist",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--q2-timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
    )
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(
            run_self_test(),
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    if not args.robot_id:
        parser.error(
            "在线运行必须提供 --robot-id；"
            "离线检查请使用 --self-test"
        )

    tuning = V8Tuning()

    config = Q4Config(
        global_grid_side_m=1000.0,
        enable_q2_seed=not args.no_q2_seed,
        enable_parallel_search_localization=True,
        enable_value_time_scheduler=True,
        scheduler_active_shortlist=(
            args.scheduler_active_shortlist
        ),
        opportunistic_max_per_node=(
            tuning.opportunistic_max_per_node
        ),
        opportunistic_max_per_channel=(
            tuning.opportunistic_max_per_channel
        ),
        q2_timeout_s=args.q2_timeout,
        min_request_interval_s=(
            args.request_interval
        ),
        no_progress_trigger=3,
    )

    geometry_config = (
        load_config(args.geometry_config)
        if args.geometry_config is not None
        else GeometryConfig()
    )

    planner_config = (
        PlannerConfig(**json.loads(
            args.planner_config.read_text(
                encoding="utf-8-sig"
            )
        ))
        if args.planner_config is not None
        else PlannerConfig()
    )

    output_dir = (
        args.output_dir
        or Path("runs") / datetime.now().strftime(
            "q4_final_%Y%m%d_%H%M%S_%f"
        )
    )

    client = SimulatorClient(
        args.robot_id,
        args.base_url,
    )

    runner = Q4V8Runner(
        client,
        config=config,
        tuning=tuning,
        planner_config=planner_config,
        geometry_config=geometry_config,
        output_dir=output_dir,
    )

    logging.basicConfig(
        filename=output_dir / "client.log",
        level=logging.INFO,
        encoding="utf-8",
        format=(
            "%(asctime)s %(levelname)s %(message)s"
        ),
    )

    print(
        f"问题四V8策略启动，输出目录：{output_dir}",
        flush=True,
    )

    summary = runner.run()

    average_clear_time_s = (
        summary["virtual_time_s"]
        / summary["cleared_count"]
        if summary["cleared_count"] > 0
        else None
    )

    print(json.dumps({
        "outcome": summary["outcome"],
        "reason": summary["reason"],
        "discovered_count": (
            summary["discovered_count"]
        ),
        "cleared_count": (
            summary["cleared_count"]
        ),
        "virtual_time_s": (
            summary["virtual_time_s"]
        ),
        "average_clear_time_s": (
            average_clear_time_s
        ),
        "total_distance_m": (
            summary["total_distance_m"]
        ),
        "dynamic_search": (
            summary["dynamic_search"]
        ),
        "certified_grid": (
            summary["global_grid"]
        ),
        "priority_optimization": (
            summary["priority_optimization"]
        ),
        "recovery_optimization": (
            summary["recovery_optimization"]
        ),
        "final_optimization": (
            summary["final_optimization"]
        ),
        "counts": summary["counts"],
    }, ensure_ascii=False, indent=2))

    return (
        0
        if summary["outcome"] == "success"
        else 1
    )


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
