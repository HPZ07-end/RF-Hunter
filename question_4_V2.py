"""问题四螺旋增强版：有覆盖证书的阿基米德螺旋搜索 + READY 驱动补测。

本文件不读取隐藏真值，只依赖同目录中的公开模块：
    client.py       - SimulatorClient 通信接口
    geometry_V7.py  - 示向角域半平面与严格几何求交
    planner.py      - 问题二的第二检测点规划器（仅作首个候选种子）

在线运行：
    python question_4_spiral.py --robot-id 你的参赛队号

离线自检：
    python question_4_spiral.py --self-test

本版重点：
* 全局检测点严格按 r=bθ 的阿基米德螺旋顺序生成，不再使用固定三角格点路线。
* 对螺旋相邻圈建立三角剖分证书；证书三角形最大边严格小于1000米，
  从而保留“任意朝向的源至少能在一个检测点收到”的发现保证。
* 每到达一个螺旋点，先检测未知频道，再利用当前位置为已发现频道补充测向。
* 补测以“尽快变为 READY”为首要目标：可达 READY 的方案抢占普通限额，
  同时降低普通补测门槛、扩大每点和每频道额度并减少过度前瞻等待。
* 即使未知频道已经全部发现，只要仍有 FOUND 源，也继续利用剩余螺旋点补测。
* 补测达到清除条件后只登记 READY；除 near 外不在搜索途中离开螺旋清除。
* 正常示向度才会裁剪连续位置外包范围；no_signal 只记录，不裁剪位置。
* 只有 near 反馈或最小包围圆半径不超过清除阈值时才调用 clear。
* 候选点显式使用源位置样本，并评价剩余定向朝向下的接收覆盖率。
* 恢复分为 500 米粗重捕获与 4.9 米小范围近距离网格，禁止大范围细扫。
* 严格几何数值退化时使用扩张半平面外包，但该外包不能签发 MEC 清除证书。
* 搜索结束后按“预计定位价值 / 实际增量耗时”统一调度清除、主动测向与恢复任务。
* localization_turn_count 仅保留为运行统计；不再作为频道选择的第一优先级。
* 未选中的主动测向方案会缓存，并按机器人当前位置重新计算移动耗时，避免重复规划。
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
from types import SimpleNamespace
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

    # 问题四保证参数：r=bθ，pitch 是相邻两圈的径向间距。
    # 620米/圈、每圈17点时，默认认证三角形最大边约976.9米。
    spiral_pitch_m: float = 620.0
    spiral_samples_per_turn: int = 17
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
    opportunistic_max_per_node: int = 6
    opportunistic_ready_bonus_per_node: int = 3
    opportunistic_max_per_channel: int = 12
    opportunistic_lookahead_nodes: int = 4
    opportunistic_lookahead_ratio: float = 0.70
    opportunistic_min_receive_score: float = 0.20
    opportunistic_min_possible_fraction: float = 0.25
    opportunistic_min_relative_reduction: float = 0.005
    opportunistic_min_absolute_reduction_m: float = 0.5
    opportunistic_force_ready_receive_score: float = 0.12
    opportunistic_continue_until_ready: bool = True

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
            "spiral_pitch_m",
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
            "opportunistic_ready_bonus_per_node",
            "opportunistic_max_per_channel",
            "opportunistic_lookahead_nodes",
            "spiral_samples_per_turn",
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
        if self.spiral_pitch_m >= self.reception_radius_min_m:
            raise ValueError("螺旋圈距必须严格小于保证接收半径1000米")
        if self.spiral_samples_per_turn < 8:
            raise ValueError("spiral_samples_per_turn 至少为8")
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
        if type(self.opportunistic_continue_until_ready) is not bool:
            raise ValueError("opportunistic_continue_until_ready 必须是布尔值")
        if type(self.enable_value_time_scheduler) is not bool:
            raise ValueError("enable_value_time_scheduler 必须是布尔值")
        if not 0.0 < self.no_signal_side_width_deg <= 180.0:
            raise ValueError("no_signal_side_width_deg 必须位于 (0,180]")
        unit_interval_parameters = (
            "opportunistic_lookahead_ratio",
            "opportunistic_min_receive_score",
            "opportunistic_min_possible_fraction",
            "opportunistic_min_relative_reduction",
            "opportunistic_force_ready_receive_score",
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


@dataclass
class SpiralPlan:
    pitch_m: float
    b_m_per_rad: float
    samples_per_turn: int
    outer_turn_index: int
    points: list[Point]
    route: list[Point]
    triangles: list[Triangle]
    triangle_count: int
    route_distance_m: float
    max_triangle_edge_m: float
    outer_boundary_min_distance_m: float


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


def _triangle_max_edge(triangle: Triangle) -> float:
    a, b, c = triangle
    return max(math.dist(a, b), math.dist(b, c), math.dist(c, a))


def build_spiral_search(config: Q4Config) -> SpiralPlan:
    """建立按阿基米德螺旋排序、同时带有严格覆盖证书的检测点序列。

    仅把点排在螺旋线上并按固定弧长取样，并不足以保证定向源一定被发现。
    这里把相邻螺旋圈之间的四边形拆成三角形，并验证：

    1. 最外圈折线到原点的最小距离覆盖整个目标圆盘；
    2. 每个证书三角形的最大边不超过1000米。

    对圆盘内任意源 G，G 必位于某个证书三角形中。三角形三个顶点
    都在 G 的1000米接收半径内；又因为 G 是三个顶点的凸组合，任意
    过 G 的朝向半平面至少含一个顶点，故定向源也至少会被一次检测到。
    """
    pitch = float(config.spiral_pitch_m)
    samples = config.spiral_samples_per_turn
    angle_step = 2.0 * math.pi / samples
    b_m_per_rad = pitch / (2.0 * math.pi)

    def spiral_point(index: int) -> Point:
        theta = index * angle_step
        radius = b_m_per_rad * theta
        return (radius * math.cos(theta), radius * math.sin(theta))

    # 第 k 圈外边界由 P[km]..P[(k+1)m] 及同角度径向封口组成。
    # 找到完全包住1800米目标圆盘的最早一圈。
    outer_turn = 1
    outer_boundary_min = 0.0
    while outer_turn <= 64:
        boundary = [
            spiral_point(outer_turn * samples + j)
            for j in range(samples + 1)
        ]
        outer_boundary_min = min(
            [_point_segment_distance((0.0, 0.0), a, b)
             for a, b in zip(boundary, boundary[1:])]
            + [_point_segment_distance(
                (0.0, 0.0), boundary[-1], boundary[0]
            )]
        )
        if outer_boundary_min >= config.target_radius_m - 1e-8:
            break
        outer_turn += 1
    else:
        raise ValueError("无法在64圈内建立螺旋覆盖边界")

    last_index = (outer_turn + 1) * samples
    points = [spiral_point(index) for index in range(last_index + 1)]
    triangles: list[Triangle] = []

    # 第一圈曲线与径向封口围成的内区。
    origin = points[0]
    for j in range(1, samples):
        triangles.append((origin, points[j], points[j + 1]))

    # 相邻螺旋圈间的条带；两条径向边的长度均等于 pitch。
    for turn in range(outer_turn):
        for j in range(samples):
            a = points[turn * samples + j]
            b = points[turn * samples + j + 1]
            c = points[(turn + 1) * samples + j]
            d = points[(turn + 1) * samples + j + 1]
            triangles.extend(((a, b, c), (b, d, c)))

    max_edge = max(_triangle_max_edge(triangle) for triangle in triangles)
    if max_edge > config.reception_radius_min_m + 1e-8:
        raise ValueError(
            "当前螺旋离散过稀，覆盖证书最大边"
            f"{max_edge:.3f}米超过{config.reception_radius_min_m:.3f}米；"
            "请增加 spiral_samples_per_turn 或减小 spiral_pitch_m"
        )

    # 客户端两次 measure 间做直线运动，因此 route 是螺旋线的折线近似；
    # 所有采样点仍严格满足 r=bθ，且访问次序严格按 θ 增大。
    route = list(points)
    return SpiralPlan(
        pitch_m=pitch,
        b_m_per_rad=b_m_per_rad,
        samples_per_turn=samples,
        outer_turn_index=outer_turn,
        points=points,
        route=route,
        triangles=triangles,
        triangle_count=len(triangles),
        route_distance_m=open_route_distance((0.0, 0.0), route),
        max_triangle_edge_m=max_edge,
        outer_boundary_min_distance_m=outer_boundary_min,
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


class Q4Runner:
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
        self.global_spiral = build_spiral_search(self.cfg)
        self.search_points = self.global_spiral.route
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
            "opportunistic_ready_priority_selected": 0,
            "opportunistic_search_extended_for_ready": 0,
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
                reason = "all_certified_spiral_nodes_no_signal"
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
        """评价一个本来就会经过的螺旋点，不把螺旋段移动重复计入成本。"""
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
            {"point": tuple(point), "origin": "spiral_route_waypoint"},
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
        normal_eligible = (
            scored["robust_receive_score"]
            >= self.cfg.opportunistic_min_receive_score
            and scored["possible_receive_fraction"]
            >= self.cfg.opportunistic_min_possible_fraction
            and (
                scored["predicted_clearable"]
                or scored["radius_reduction"] >= minimum_reduction
            )
        )
        # 如果模型预测本次补测可直接使定位范围取得MEC清除证书，则采用更低但
        # 仍为正的鲁棒接收门槛。失败的 no_signal 只耗时，不会错误裁剪位置范围。
        ready_eligible = (
            scored["predicted_clearable"]
            and scored["robust_receive_score"]
            >= self.cfg.opportunistic_force_ready_receive_score
            and scored["possible_receive_fraction"] > 0.0
        )
        # 未知频道已经处理完后，剩余螺旋本身就是免费的既定路线资源。
        # 此时允许有正缩减、且仍有鲁棒接收可能的中间补测逐步推动到 READY，
        # 不要求单次就跨过20米阈值。
        tail_ready_drive = (
            not resume_unknown_search
            and scored["robust_receive_score"]
            >= self.cfg.opportunistic_force_ready_receive_score
            and scored["possible_receive_fraction"] > 0.0
            and scored["radius_reduction"] > self.cfg.progress_abs_m
        )
        return {
            **scored,
            "eligible": normal_eligible or ready_eligible or tail_ready_drive,
            "normal_eligible": normal_eligible,
            "ready_priority": ready_eligible or tail_ready_drive,
            "predicted_ready_now": ready_eligible,
            "tail_ready_drive": tail_ready_drive,
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
        """选择当前螺旋点的补测频道，并让可直接 READY 的方案抢占执行。"""
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

        ranked = sorted(
            proposals,
            key=lambda item: (
                item["ready_priority"],
                item["predicted_clearable"],
                item["opportunity_score"],
                item["robust_receive_score"],
                item["radius_reduction"],
                -item["effective_incremental_time_s"],
                -item["channel"],
            ),
            reverse=True,
        )
        selected = ranked[:self.cfg.opportunistic_max_per_node]
        # 普通批次已满时，仍允许少量“本次即可 READY”的频道越过普通上限。
        # 这只增加当前位置的测量/切频，不会造成路线偏离。
        selected_channels = {item["channel"] for item in selected}
        ready_extras = [
            item for item in ranked
            if item["ready_priority"] and item["channel"] not in selected_channels
        ][:self.cfg.opportunistic_ready_bonus_per_node]
        selected.extend(ready_extras)
        self.counts["opportunistic_ready_priority_selected"] += sum(
            bool(item["ready_priority"]) for item in selected
        )
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
                        "ready_priority": item["ready_priority"],
                        "tail_ready_drive": item["tail_ready_drive"],
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
        """在不离开当前螺旋点的前提下，对已发现频道补充测向。"""
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
                # near 的清除点就是当前螺旋点，不产生路线偏离，因此立即清除。
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
            found = [
                record for record in self.records.values()
                if record.status == "FOUND"
            ]
            should_extend_for_ready = (
                self.cfg.enable_parallel_search_localization
                and self.cfg.opportunistic_continue_until_ready
                and bool(found)
            )
            if not unknown and not should_extend_for_ready:
                break
            if not unknown and should_extend_for_ready:
                self.counts["opportunistic_search_extended_for_ready"] += 1
                self.event(
                    "spiral_search_extended_for_ready",
                    global_index=index,
                    point=point,
                    found_channels=[record.channel_id for record in found],
                )
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
            # 此步骤不改变螺旋覆盖证书及“未知频道完整检测”的发现保证。
            self.opportunistic_localization_at_waypoint(index, point)
            self.next_search_index = index + 1
            if self.discovered_count >= self.cfg.source_count_max:
                # 16个源达到题目上界后，未知频道可立即判不存在；但 FOUND 源
                # 仍继续利用后续螺旋点争取变为 READY。
                self.mark_absent()

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
            "global_spiral": {
                "equation": "r = b * theta",
                "pitch_m": self.global_spiral.pitch_m,
                "b_m_per_rad": self.global_spiral.b_m_per_rad,
                "samples_per_turn": self.global_spiral.samples_per_turn,
                "outer_turn_index": self.global_spiral.outer_turn_index,
                "point_count": len(self.global_spiral.points),
                "triangle_count": self.global_spiral.triangle_count,
                "route_distance_m": self.global_spiral.route_distance_m,
                "max_certificate_triangle_edge_m": (
                    self.global_spiral.max_triangle_edge_m
                ),
                "outer_boundary_min_distance_m": (
                    self.global_spiral.outer_boundary_min_distance_m
                ),
                "proof": (
                    "目标圆盘由相邻螺旋圈的三角剖分覆盖；每个证书三角形"
                    "最大边不超过1000米，且任意过源的前向半平面至少含一个顶点"
                ),
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
                "带三角覆盖证书的阿基米德螺旋搜索途中对已发现频道执行READY驱动补测；"
                "no_signal 不裁剪连续位置；主动选择联合采样位置与方向；"
                "严格几何失败时的浮点外包不签发MEC清除证书"
            ),
            "formal_log_note": "正式加密日志仍需从模拟器界面导出",
        }


@dataclass(frozen=True)
class _FakeSource:
    position: Point
    reception_radius_m: float
    directional: bool = False
    direction_deg: float = 0.0


class _FakeClient:
    """仅供 --self-test；公开字段和动作返回形式与 SimulatorClient 对齐。"""

    def __init__(self, sources: dict[int, _FakeSource], config: Q4Config):
        self.sources = dict(sources)
        self.cfg = config
        self.position: Point | None = None
        self.current_channel: int | None = None
        self.virtual_time_s: float | None = None
        self.max_virtual_duration_s: float | None = None
        self.max_real_duration_s: float | None = None
        self.pending_request_id = None
        self.cleared_channels: set[int] = set()
        self._state = "new"

    @property
    def remaining_real_time_s(self):
        return 1200.0 if self._state == "active" else None

    def _advance(self, point: Point, operation_s: float, switch: bool = False) -> None:
        self.virtual_time_s += math.dist(self.position, point) / self.cfg.dog_speed_m_per_s
        self.virtual_time_s += operation_s
        if switch:
            self.virtual_time_s += self.cfg.channel_switch_time_s
        self.position = tuple(point)

    def _base_response(self) -> dict:
        return {
            "accepted": True,
            "real_timestamp_ms": int(self.virtual_time_s * 1000),
            "virtual_time_s": self.virtual_time_s,
        }

    def enter(self):
        self._state = "active"
        self.position = (0.0, 0.0)
        self.current_channel = 1
        self.virtual_time_s = 0.0
        self.max_virtual_duration_s = 360000.0
        self.max_real_duration_s = 1200.0
        return {
            **self._base_response(),
            "max_virtual_duration_s": 360000,
            "max_real_duration_s": 1200,
            "remaining_real_duration_s": 1200,
        }

    def measure(self, x: float, y: float, channel: int):
        point = (float(x), float(y))
        switched = channel != self.current_channel
        self._advance(point, self.cfg.detection_time_s, switched)
        self.current_channel = channel
        source = None if channel in self.cleared_channels else self.sources.get(channel)
        if source is None:
            return {**self._base_response(), "measure_result": "no_signal"}

        distance = math.dist(point, source.position)
        front = True
        if source.directional:
            angle = math.radians(source.direction_deg)
            direction = (math.cos(angle), math.sin(angle))
            front = (
                direction[0] * (point[0] - source.position[0])
                + direction[1] * (point[1] - source.position[1])
                >= -1e-9
            )
        if distance > source.reception_radius_m + 1e-9 or not front:
            return {**self._base_response(), "measure_result": "no_signal"}
        if distance <= self.cfg.strong_signal_radius_m + 1e-9:
            return {**self._base_response(), "measure_result": "near"}
        bearing = math.degrees(
            math.atan2(source.position[1] - point[1], source.position[0] - point[0])
        ) % 360.0
        return {**self._base_response(), "measure_result": "direction", "svd_deg": bearing}

    def clear(self, x: float, y: float, channel: int):
        point = (float(x), float(y))
        source = None if channel in self.cleared_channels else self.sources.get(channel)
        success = source is not None and math.dist(point, source.position) <= self.cfg.clearance_radius_m + 1e-9
        operation = self.cfg.optical_time_s + (self.cfg.clearance_time_s if success else 0.0)
        self._advance(point, operation)
        if success:
            self.cleared_channels.add(channel)
        return {
            **self._base_response(),
            "clear_result": "success" if success else "no_target_in_range",
        }

    def exit(self):
        self._state = "exited"
        return {**self._base_response(), "exit_reason": "user_exit"}


def run_self_test() -> dict:
    config = Q4Config(
        enable_q2_seed=False,
        source_samples=20,
        error_samples=3,
        min_request_interval_s=0.0,
    )
    spiral = build_spiral_search(config)
    assert spiral.triangle_count == 118, spiral.triangle_count
    assert len(spiral.points) == 69, len(spiral.points)
    assert spiral.outer_turn_index == 3, spiral.outer_turn_index
    assert spiral.max_triangle_edge_m < config.reception_radius_min_m
    assert spiral.outer_boundary_min_distance_m >= config.target_radius_m
    assert 31_000.0 < spiral.route_distance_m < 31_500.0
    for index, point in enumerate(spiral.points):
        theta = index * 2.0 * math.pi / spiral.samples_per_turn
        expected_radius = spiral.b_m_per_rad * theta
        assert abs(math.dist((0.0, 0.0), point) - expected_radius) <= 1e-7

    rng = random.Random(20260912)
    for _ in range(1000):
        radius = config.target_radius_m * math.sqrt(rng.random())
        angle = 2.0 * math.pi * rng.random()
        source = (radius * math.cos(angle), radius * math.sin(angle))
        direction_angle = 2.0 * math.pi * rng.random()
        direction = (math.cos(direction_angle), math.sin(direction_angle))
        containing = [
            triangle for triangle in spiral.triangles
            if point_in_triangle(source, triangle)
        ]
        assert containing, source
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

    center, radius = minimum_enclosing_circle([(0.0, 0.0), (40.0, 0.0), (20.0, 10.0)])
    assert math.dist(center, (20.0, 0.0)) <= 1e-7
    assert abs(radius - 20.0) <= 1e-7

    # 回归检查：第四问中的 no_signal 不能像问题三那样裁剪位置范围。
    no_signal_fake = _FakeClient(
        {2: _FakeSource(position=(500.0, 0.0), reception_radius_m=1200.0)},
        config,
    )
    no_signal_runner = Q4Runner(
        no_signal_fake,
        config=config,
        planner_config=PlannerConfig(),
        geometry_config=GeometryConfig(),
    )
    no_signal_runner.action("enter")
    no_signal_record = no_signal_runner.records[2]
    assert no_signal_runner.measure(
        no_signal_record,
        (-500.0, 0.0),
        measurement_kind="active",
    ) == "direction"
    polygon_before = list(no_signal_record.outer_polygon)
    diameter_before = no_signal_record.diameter
    radius_before = no_signal_record.clearance_radius
    assert no_signal_runner.measure(
        no_signal_record,
        (3000.0, 0.0),
        measurement_kind="active",
    ) == "no_signal"
    assert polygon_before == no_signal_record.outer_polygon
    assert diameter_before == no_signal_record.diameter
    assert radius_before == no_signal_record.clearance_radius
    no_signal_fake.exit()

    # 报告中的 ch19 形状：274.24° 的样本和关键候选必须位于正确角域。
    direction_record = ChannelRecord(19)
    direction_record.bearing_observations.append({
        "position": (0.0, 0.0),
        "bearing_deg": 274.24,
        "measurement_kind": "unknown",
        "virtual_time_s": 0.0,
        "real_timestamp_ms": 0,
    })
    update_region(direction_record, config, GeometryConfig())
    direction_sources = sample_source_positions(direction_record, config)
    _, direction_candidates = candidate_points(
        direction_record,
        direction_sources,
        (0.0, 0.0),
        config,
        None,
    )
    source_candidates = [
        item["point"] for item in direction_candidates if item["origin"] == "source_sample"
    ]
    assert source_candidates
    assert all(point[1] < 0.0 for point in source_candidates)
    assert any(point[0] > 0.0 for point in source_candidates)
    direction_fake = _FakeClient({}, config)
    direction_fake.enter()
    direction_runner = Q4Runner(
        direction_fake,
        config=config,
        planner_config=PlannerConfig(),
        geometry_config=GeometryConfig(),
    )
    direction_runner.records[19] = direction_record
    direction_proposal = direction_runner.plan_active_measurement(direction_record)
    # 旧版会直接跳到 y≈-1000 的失败侧；修复版先选高接收相容候选。
    assert direction_proposal["robust_receive_score"] > 0.5
    assert math.dist(direction_proposal["point"], (0.0, 0.0)) < 800.0
    direction_record.no_progress_count = config.no_progress_trigger
    direction_runner.start_recovery(direction_record)
    assert direction_record.recovery_mode == "coarse_reacquire"
    assert len(direction_runner._recovery_queues[19]) < config.fine_recovery_max_nodes
    direction_fake.exit()

    # 强制模拟 geometry_V7 的 NUMERICAL_ISSUE，验证不会终止且不会签发MEC证书。
    original_solver = globals()["solve_halfplanes"]
    globals()["solve_halfplanes"] = lambda *args, **kwargs: SimpleNamespace(
        status="NUMERICAL_ISSUE",
        vertices=[],
        diameter=None,
        diameter_pair=None,
        diagnostics={"warnings": ["forced numerical issue"]},
    )
    try:
        fallback_record = ChannelRecord(4)
        fallback_record.bearing_observations.append({
            "position": (0.0, 0.0),
            "bearing_deg": 45.0,
            "measurement_kind": "unknown",
            "virtual_time_s": 0.0,
            "real_timestamp_ms": 0,
        })
        fallback_info = update_region(fallback_record, config, GeometryConfig())
    finally:
        globals()["solve_halfplanes"] = original_solver
    assert fallback_info["fallback_used"] is True
    assert fallback_record.outer_polygon
    assert fallback_record.status == "FOUND"
    assert fallback_record.certificate_source is None
    assert fallback_record.geometry_certifiable is False

    # 调度回归：即便近频道历史轮转次数更高，也应优先执行更省时的实际任务。
    scheduler_fake = _FakeClient({}, config)
    scheduler_fake.enter()
    scheduler_runner = Q4Runner(
        scheduler_fake,
        config=config,
        planner_config=PlannerConfig(),
        geometry_config=GeometryConfig(),
    )
    far_record = scheduler_runner.records[1]
    far_record.status = "READY"
    far_record.clearance_center = (1000.0, 0.0)
    far_record.clearance_radius = config.strong_signal_radius_m
    far_record.certificate_source = "near_feedback"
    far_record.localization_turn_count = 0
    near_record = scheduler_runner.records[2]
    near_record.status = "READY"
    near_record.clearance_center = (100.0, 0.0)
    near_record.clearance_radius = config.strong_signal_radius_m
    near_record.certificate_source = "near_feedback"
    near_record.localization_turn_count = 99
    scheduler_tasks = scheduler_runner._build_localization_tasks(
        [far_record, near_record]
    )
    scheduler_choice = max(
        scheduler_tasks,
        key=scheduler_runner._scheduler_selection_key,
    )
    assert scheduler_choice["record"].channel_id == 2
    scheduler_fake.exit()

    # 端到端离线案例：定向源先由粗网格发现，再由主动定位/MEC或恢复网格清除。
    fake = _FakeClient(
        {
            7: _FakeSource(
                position=(310.0, 230.0),
                reception_radius_m=1350.0,
                directional=True,
                direction_deg=215.0,
            )
        },
        config,
    )
    runner = Q4Runner(fake, config=config, planner_config=PlannerConfig(), geometry_config=GeometryConfig())
    summary = runner.run()
    assert summary["outcome"] == "success", summary["reason"]
    assert summary["cleared_count"] == 1, summary["cleared_count"]
    assert all(record.status in TERMINAL_STATES for record in runner.records.values())
    assert summary["counts"]["opportunistic_measure"] >= 1
    assert any(
        event.get("event") == "measurement_update"
        and event.get("measurement_kind") == "opportunistic"
        for event in runner.events
    )
    assert summary["counts"]["opportunistic_ready_queued"] >= 1

    # 搜索控制回归：16个源已发现但仍是 FOUND 时，不能像旧版一样立刻离开
    # 全局路线；应继续利用后续螺旋点，直到路线结束或不再有 FOUND 源。
    extension_fake = _FakeClient({}, config)
    extension_fake.enter()
    extension_runner = Q4Runner(
        extension_fake,
        config=config,
        planner_config=PlannerConfig(),
        geometry_config=GeometryConfig(),
    )
    extension_runner.search_points = [(0.0, 0.0), (100.0, 0.0)]
    for channel in range(1, 17):
        extension_record = extension_runner.records[channel]
        extension_record.status = "FOUND"
        extension_record.bearing_observations.append({
            "position": (0.0, 0.0),
            "bearing_deg": 0.0,
            "measurement_kind": "test_fixture",
            "virtual_time_s": 0.0,
            "real_timestamp_ms": 0,
        })
    extension_calls: list[int] = []

    def record_extension_call(global_index: int, point: Point) -> None:
        extension_calls.append(global_index)

    extension_runner.opportunistic_localization_at_waypoint = record_extension_call
    extension_runner.search()
    assert extension_calls == [0, 1], extension_calls
    assert extension_runner.counts["opportunistic_search_extended_for_ready"] == 1
    extension_fake.exit()

    # 所有 no_signal 事件同时写明没有用于位置裁剪，便于审计运行日志。
    assert all(
        event.get("no_signal_position_pruned") is False
        for event in runner.events
        if event.get("event") == "measurement_update"
        and event.get("result") == "no_signal"
    )

    return {
        "status": "ok",
        "global_spiral_points": len(spiral.points),
        "global_spiral_triangles": spiral.triangle_count,
        "global_spiral_route_distance_m": spiral.route_distance_m,
        "spiral_max_certificate_edge_m": spiral.max_triangle_edge_m,
        "spiral_outer_boundary_min_distance_m": (
            spiral.outer_boundary_min_distance_m
        ),
        "random_directional_coverage_cases": 1000,
        "no_signal_region_invariance": True,
        "directional_candidate_regression": True,
        "large_wedge_recovery_mode": "coarse_reacquire",
        "numerical_geometry_fallback": True,
        "parallel_search_localization": True,
        "global_value_time_scheduler": True,
        "round_count_not_primary": True,
        "opportunistic_measure_count": summary["counts"]["opportunistic_measure"],
        "scheduler_round_count": summary["counts"]["scheduler_round"],
        "integration_outcome": summary["outcome"],
        "integration_cleared_count": summary["cleared_count"],
        "integration_virtual_time_s": summary["virtual_time_s"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-id")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--geometry-config", type=Path)
    parser.add_argument("--planner-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-q2-seed", action="store_true")
    parser.add_argument(
        "--no-parallel-search-localization",
        action="store_true",
        help="关闭螺旋搜索途中的已发现频道补测，用于基线对照",
    )
    parser.add_argument("--spiral-pitch", type=float, default=620.0)
    parser.add_argument("--spiral-samples-per-turn", type=int, default=17)
    parser.add_argument("--opportunistic-max-per-node", type=int, default=6)
    parser.add_argument("--opportunistic-ready-bonus-per-node", type=int, default=3)
    parser.add_argument("--opportunistic-max-per-channel", type=int, default=12)
    parser.add_argument("--opportunistic-lookahead-nodes", type=int, default=4)
    parser.add_argument(
        "--stop-spiral-after-discovery",
        action="store_true",
        help="未知频道消失后不再沿剩余螺旋点把 FOUND 源补测到 READY",
    )
    parser.add_argument(
        "--legacy-localization-scheduler",
        action="store_true",
        help="恢复旧版轮转次数优先调度，仅用于 A/B 对照",
    )
    parser.add_argument("--scheduler-active-shortlist", type=int, default=4)
    parser.add_argument("--q2-timeout", type=float, default=30.0)
    parser.add_argument("--no-progress-trigger", type=int, default=3)
    parser.add_argument("--request-interval", type=float, default=0.03)
    parser.add_argument("--recovery-batch-size", type=int, default=20)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))
        return 0
    if not args.robot_id:
        parser.error("在线运行必须提供 --robot-id；离线检查请使用 --self-test")

    config = Q4Config(
        spiral_pitch_m=args.spiral_pitch,
        spiral_samples_per_turn=args.spiral_samples_per_turn,
        enable_q2_seed=not args.no_q2_seed,
        enable_parallel_search_localization=(
            not args.no_parallel_search_localization
        ),
        enable_value_time_scheduler=(
            not args.legacy_localization_scheduler
        ),
        scheduler_active_shortlist=args.scheduler_active_shortlist,
        opportunistic_max_per_node=args.opportunistic_max_per_node,
        opportunistic_ready_bonus_per_node=(
            args.opportunistic_ready_bonus_per_node
        ),
        opportunistic_max_per_channel=args.opportunistic_max_per_channel,
        opportunistic_lookahead_nodes=args.opportunistic_lookahead_nodes,
        opportunistic_continue_until_ready=(
            not args.stop_spiral_after_discovery
        ),
        q2_timeout_s=args.q2_timeout,
        no_progress_trigger=args.no_progress_trigger,
        min_request_interval_s=args.request_interval,
        recovery_batch_size=args.recovery_batch_size,
    )
    geometry_config = (
        load_config(args.geometry_config)
        if args.geometry_config is not None
        else GeometryConfig()
    )
    planner_config = (
        PlannerConfig(**json.loads(args.planner_config.read_text(encoding="utf-8-sig")))
        if args.planner_config is not None
        else PlannerConfig()
    )
    output_dir = args.output_dir or Path("runs") / datetime.now().strftime("q4_%Y%m%d_%H%M%S_%f")
    client = SimulatorClient(args.robot_id, args.base_url)
    runner = Q4Runner(
        client,
        config=config,
        planner_config=planner_config,
        geometry_config=geometry_config,
        output_dir=output_dir,
    )
    logging.basicConfig(
        filename=output_dir / "client.log",
        level=logging.INFO,
        encoding="utf-8",
        format="%(asctime)s %(levelname)s %(message)s",
    )
    print(f"问题四阿基米德螺旋/READY增强策略启动，输出目录：{output_dir}", flush=True)
    summary = runner.run()
    average_time_per_cleared_s = (
        summary["virtual_time_s"] / summary["cleared_count"]
        if summary["cleared_count"] > 0
        else None
    )
    print(json.dumps(
        {
            "outcome": summary["outcome"],
            "reason": summary["reason"],
            "discovered_count": summary["discovered_count"],
            "cleared_count": summary["cleared_count"],
            "virtual_time_s": summary["virtual_time_s"],
            "program_elapsed_s": summary["program_elapsed_s"],
            "average_time_per_cleared_s": average_time_per_cleared_s,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if summary["outcome"] == "success" else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
