"""两检测点、四个测向半平面的专用浮点几何评价器。

本模块只负责快速路径：

* 固定输入为两次检测，每次测向产生两个半平面；
* 枚举至多 6 个边界交点，并直接计算凸多边形直径；
* 只有在有界、非退化且残差健康时才返回 certified=True；
* 近平行、近三线共点、近退化或其他不确定情况返回
  FALLBACK_REQUIRED，由调用方交给 geometry_V7.solve_halfplanes。

它不替代 geometry_V7，也不负责最终精确校核。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence

from geometry_V7 import GeometryConfig


Point = tuple[float, float]
Halfplane = tuple[float, float, float]


@dataclass(frozen=True)
class FastGeometryResult:
    """快速评价结果；只有 certified=True 时 diameter 才可直接采用。"""

    status: str
    certified: bool
    diameter: float | None = None
    vertices: tuple[Point, ...] = ()
    diameter_pair: tuple[Point, Point] | None = None
    max_constraint_residual_m: float | None = None
    fallback_reason: str | None = None
    diagnostics: dict = field(default_factory=dict)


def bearing_halfplanes(
    point: Sequence[float],
    bearing_deg: float,
    error_deg: float,
) -> tuple[Halfplane, Halfplane]:
    """生成与 geometry_V7.build_halfplanes 相同的两个单位法向半平面。"""
    if len(point) != 2:
        raise ValueError("检测点必须包含两个坐标")
    x, y = float(point[0]), float(point[1])
    bearing = float(bearing_deg)
    error = float(error_deg)
    if not all(math.isfinite(value) for value in (x, y, bearing, error)):
        raise ValueError("检测点、示向度和误差角必须有限")
    if not 0.0 < error < 90.0:
        raise ValueError("误差角必须位于 (0,90) 度")

    theta = bearing % 360.0
    lower = math.radians(theta - error)
    upper = math.radians(theta + error)
    rows = []
    for a, b in (
        (math.sin(lower), -math.cos(lower)),
        (-math.sin(upper), math.cos(upper)),
    ):
        c = math.fsum((a * x, b * y))
        rows.append((a, b, c))
    return rows[0], rows[1]


def _fallback(reason: str, **diagnostics) -> FastGeometryResult:
    return FastGeometryResult(
        status="FALLBACK_REQUIRED",
        certified=False,
        fallback_reason=reason,
        diagnostics=diagnostics,
    )


def _normal_gap(rows: Sequence[Halfplane]) -> float:
    angles = sorted(math.atan2(b, a) % (2.0 * math.pi) for a, b, _ in rows)
    gaps = [angles[index + 1] - angles[index] for index in range(len(angles) - 1)]
    gaps.append(angles[0] + 2.0 * math.pi - angles[-1])
    return max(gaps)


def _intersection(first: Halfplane, second: Halfplane) -> tuple[Point, float]:
    a, b, c = first
    d, e, f = second
    determinant = math.fsum((a * e, -d * b))
    x = math.fsum((c * e, -b * f)) / determinant
    y = math.fsum((a * f, -c * d)) / determinant
    return (x, y), determinant


def _residual(row: Halfplane, point: Point) -> float:
    a, b, c = row
    return math.fsum((a * point[0], b * point[1], -c))


def solve_two_bearings_fast(
    first_halfplanes: Iterable[Halfplane],
    second_halfplanes: Iterable[Halfplane],
    *,
    config: GeometryConfig,
) -> FastGeometryResult:
    """快速求四半平面交集的直径；病态输入不猜测，要求调用方回退。"""
    rows = tuple(
        tuple(float(value) for value in row)
        for row in (*tuple(first_halfplanes), *tuple(second_halfplanes))
    )
    if len(rows) != 4 or any(len(row) != 3 for row in rows):
        raise ValueError("专用评价器必须恰好接收两组、共四个半平面")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("半平面系数必须有限")

    # bearing_halfplanes 生成单位法向量。仍检查一次，避免其他调用者传入
    # 缩放严重或退化的约束，使固定阈值失去意义。
    normal_errors = [abs(math.hypot(a, b) - 1.0) for a, b, _ in rows]
    if max(normal_errors) > 1e-10:
        return _fallback(
            "NON_UNIT_NORMAL",
            max_normalization_error=max(normal_errors),
        )

    determinant_guard = max(1e-9, 32.0 * config.near_parallel_det_tol)
    bounded_gap_guard = max(1e-9, 32.0 * config.near_parallel_det_tol)
    max_gap = _normal_gap(rows)
    if max_gap >= math.pi - bounded_gap_guard:
        return _fallback(
            "UNBOUNDED_OR_NEAR_PARALLEL_NORMAL_CONE",
            max_normal_gap_rad=max_gap,
        )

    coordinate_scale = max(1.0, *(abs(row[2]) for row in rows))
    feasibility_tolerance = (
        config.feasibility_abs_tol_m
        + config.feasibility_rel_tol * coordinate_scale
    )
    topology_guard = max(1e-9, 8.0 * feasibility_tolerance)

    vertices: list[Point] = []
    pair_determinants: list[float] = []
    max_residual = -math.inf

    for first_index, second_index in combinations(range(4), 2):
        first = rows[first_index]
        second = rows[second_index]
        determinant = math.fsum((first[0] * second[1], -second[0] * first[1]))
        pair_determinants.append(abs(determinant))
        if abs(determinant) <= determinant_guard:
            return _fallback(
                "NEAR_PARALLEL_BOUNDARIES",
                minimum_abs_determinant=min(pair_determinants),
            )

        point, _ = _intersection(first, second)
        if not all(math.isfinite(value) for value in point):
            return _fallback("NONFINITE_INTERSECTION")

        residuals = [_residual(row, point) for row in rows]
        max_residual = max(max_residual, *residuals)
        other_residuals = [
            residual
            for index, residual in enumerate(residuals)
            if index not in (first_index, second_index)
        ]

        if any(residual > topology_guard for residual in other_residuals):
            continue
        if any(abs(residual) <= topology_guard for residual in other_residuals):
            return _fallback(
                "NEAR_THREE_BOUNDARY_INTERSECTION",
                maximum_constraint_residual_m=max_residual,
            )
        vertices.append(point)

    if len(vertices) < 3:
        return _fallback(
            "EMPTY_OR_DEGENERATE_INTERSECTION",
            feasible_vertex_count=len(vertices),
        )

    vertex_scale = max(
        coordinate_scale,
        *(abs(value) for point in vertices for value in point),
    )
    merge_tolerance = (
        config.vertex_merge_abs_tol_m
        + config.vertex_merge_rel_tol * vertex_scale
    )
    for first, second in combinations(vertices, 2):
        if math.dist(first, second) <= 8.0 * merge_tolerance:
            return _fallback("NEAR_DUPLICATE_VERTICES")

    center = (
        math.fsum(point[0] for point in vertices) / len(vertices),
        math.fsum(point[1] for point in vertices) / len(vertices),
    )
    vertices.sort(key=lambda point: math.atan2(point[1] - center[1], point[0] - center[0]))

    area_twice = abs(
        math.fsum(
            vertices[index][0] * vertices[(index + 1) % len(vertices)][1]
            - vertices[(index + 1) % len(vertices)][0] * vertices[index][1]
            for index in range(len(vertices))
        )
    )
    pairs = list(combinations(range(len(vertices)), 2))
    farthest = max(
        pairs,
        key=lambda pair: (
            (vertices[pair[0]][0] - vertices[pair[1]][0]) ** 2
            + (vertices[pair[0]][1] - vertices[pair[1]][1]) ** 2
        ),
    )
    first_vertex = vertices[farthest[0]]
    second_vertex = vertices[farthest[1]]
    diameter = math.dist(first_vertex, second_vertex)
    if not math.isfinite(diameter):
        return _fallback("NONFINITE_DIAMETER")

    degeneracy_tolerance = (
        config.degeneracy_abs_tol_m
        + config.degeneracy_rel_tol * max(vertex_scale, diameter)
    )
    minimum_height = area_twice / max(diameter, 1.0)
    if minimum_height <= 8.0 * degeneracy_tolerance:
        return _fallback(
            "NEAR_DEGENERATE_POLYGON",
            area_twice=area_twice,
            minimum_height_m=minimum_height,
        )

    final_residual = max(
        _residual(row, point)
        for point in vertices
        for row in rows
    )
    if final_residual > topology_guard:
        return _fallback(
            "RESIDUAL_EXCEEDS_TOLERANCE",
            maximum_constraint_residual_m=final_residual,
        )

    return FastGeometryResult(
        status="POLYGON",
        certified=True,
        diameter=diameter,
        vertices=tuple(vertices),
        diameter_pair=(first_vertex, second_vertex),
        max_constraint_residual_m=final_residual,
        diagnostics={
            "backend": "two_bearing_four_halfplane_float",
            "vertex_count": len(vertices),
            "minimum_abs_determinant": min(pair_determinants),
            "max_normal_gap_rad": max_gap,
            "topology_guard_m": topology_guard,
        },
    )


class TwoBearingFastEvaluator:
    """缓存第一次观测，使 planner 每次只构造第二次观测的两个半平面。"""

    def __init__(
        self,
        first_point: Sequence[float],
        first_bearing_deg: float,
        bearing_error_deg: float,
        *,
        config: GeometryConfig,
    ):
        self.error_deg = float(bearing_error_deg)
        self.config = config
        self.first_halfplanes = bearing_halfplanes(
            first_point,
            first_bearing_deg,
            self.error_deg,
        )

    def evaluate(
        self,
        second_point: Sequence[float],
        second_bearing_deg: float,
    ) -> FastGeometryResult:
        second_halfplanes = bearing_halfplanes(
            second_point,
            second_bearing_deg,
            self.error_deg,
        )
        return solve_two_bearings_fast(
            self.first_halfplanes,
            second_halfplanes,
            config=self.config,
        )

