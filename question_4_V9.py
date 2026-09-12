"""问题四 V9：21点凸包认证搜索与25点V8对照版本。

21点方案：
    原点1点；
    内环半径995米，角度15°+45°k，k=0,...,7；
    外环半径1863.6米，角度30°k，k=0,...,11。

正确性判据：
    对任意目标圆内的可能源位置x，取所有与x距离不超过998米
    的检测点。如果x位于这些检测点的凸包内，则任意经过x的
    定向发射前半平面至少包含一个距离不超过998米的检测点。

连续认证采用自适应区间细分，不是随机抽样。

运行21点版本：
    python question_4_V9.py --robot-id 你的队号 --grid-mode 21

运行25点对照版本：
    python question_4_V9.py --robot-id 你的队号 --grid-mode 25

离线认证：
    python question_4_V9.py --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
from typing import Sequence

import question_4_V8 as v8


v7 = v8.v7
v6 = v8.v6
v5 = v8.v5
v4 = v8.v4
v3 = v8.v3

Point = tuple[float, float]


@dataclass(frozen=True)
class V9Tuning(v8.V8Tuning):
    grid21_inner_radius_m: float = 995.0
    grid21_inner_count: int = 8
    grid21_inner_phase_deg: float = 15.0

    grid21_outer_radius_m: float = 1863.6
    grid21_outer_count: int = 12
    grid21_outer_phase_deg: float = 0.0

    # 使用998米完成证明，保留相对于1000米最低接收距离的2米余量。
    grid21_certificate_distance_m: float = 998.0

    # 连续区间认证参数。
    grid21_certificate_max_depth: int = 18
    grid21_certificate_max_boxes: int = 500_000
    grid21_hull_tolerance_m2: float = 1e-7
    grid21_symmetry_tolerance_m: float = 1e-6

    def __post_init__(self):
        super().__post_init__()

        for name in (
            "grid21_inner_count",
            "grid21_outer_count",
            "grid21_certificate_max_depth",
            "grid21_certificate_max_boxes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}必须为正整数")

        for name in (
            "grid21_inner_radius_m",
            "grid21_outer_radius_m",
            "grid21_certificate_distance_m",
            "grid21_hull_tolerance_m2",
            "grid21_symmetry_tolerance_m",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{name}必须为有限正数")

        if self.grid21_inner_count != 8:
            raise ValueError("21点认证要求内环恰好8点")

        if self.grid21_outer_count != 12:
            raise ValueError("21点认证要求外环恰好12点")

def polar_point(
    radius_m: float,
    angle_deg: float,
) -> Point:
    angle_rad = math.radians(angle_deg)
    return (
        radius_m * math.cos(angle_rad),
        radius_m * math.sin(angle_rad),
    )


def build_21_point_coordinates(
    tuning: V9Tuning,
) -> list[Point]:
    points: list[Point] = [(0.0, 0.0)]

    points.extend(
        polar_point(
            tuning.grid21_inner_radius_m,
            tuning.grid21_inner_phase_deg
            + 45.0 * index,
        )
        for index in range(
            tuning.grid21_inner_count
        )
    )

    points.extend(
        polar_point(
            tuning.grid21_outer_radius_m,
            tuning.grid21_outer_phase_deg
            + 30.0 * index,
        )
        for index in range(
            tuning.grid21_outer_count
        )
    )

    if len(points) != 21:
        raise ValueError(
            f"21点坐标生成错误：得到{len(points)}点"
        )

    if len(set(points)) != len(points):
        raise ValueError("21点方案存在重复坐标")

    return points


def cross(
    origin: Point,
    point_a: Point,
    point_b: Point,
) -> float:
    return (
        (point_a[0] - origin[0])
        * (point_b[1] - origin[1])
        - (point_a[1] - origin[1])
        * (point_b[0] - origin[0])
    )


def convex_hull(
    points: Sequence[Point],
) -> list[Point]:
    ordered = sorted(
        set(tuple(map(float, point)) for point in points)
    )

    if len(ordered) <= 1:
        return ordered

    lower: list[Point] = []
    for point in ordered:
        while (
            len(lower) >= 2
            and cross(
                lower[-2],
                lower[-1],
                point,
            ) <= 0.0
        ):
            lower.pop()
        lower.append(point)

    upper: list[Point] = []
    for point in reversed(ordered):
        while (
            len(upper) >= 2
            and cross(
                upper[-2],
                upper[-1],
                point,
            ) <= 0.0
        ):
            upper.pop()
        upper.append(point)

    lower.pop()
    upper.pop()
    return lower + upper


def point_in_convex_polygon(
    point: Point,
    polygon: Sequence[Point],
    tolerance_m2: float,
) -> bool:
    if len(polygon) < 3:
        return False

    return all(
        cross(
            polygon[index],
            polygon[(index + 1) % len(polygon)],
            point,
        )
        >= -tolerance_m2
        for index in range(len(polygon))
    )


def verify_quarter_turn_symmetry(
    points: Sequence[Point],
    tolerance_m: float,
) -> None:
    """验证点集绕原点旋转90度后保持不变。"""

    for point in points:
        rotated = (-point[1], point[0])

        if not any(
            math.dist(rotated, candidate)
            <= tolerance_m
            for candidate in points
        ):
            raise ValueError(
                "21点集合不满足90度旋转对称性："
                f"{point}旋转后找不到对应点"
            )


def box_corners(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[Point, Point, Point, Point]:
    return (
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_min),
        (x_max, y_max),
    )


def box_intersects_quarter_disk(
    x_min: float,
    y_min: float,
    target_radius_m: float,
) -> bool:
    # 所有待认证盒均位于第一象限，因此盒内距原点最近点为(x_min,y_min)。
    return (
        x_min * x_min + y_min * y_min
        <= target_radius_m * target_radius_m
    )


def guaranteed_near_points_for_box(
    points: Sequence[Point],
    corners: Sequence[Point],
    certificate_distance_m: float,
) -> list[Point]:
    """返回对盒内所有位置均不超过认证距离的检测点。

    到固定点的距离是凸函数，其在矩形上的最大值必定在角点取得。
    """

    result: list[Point] = []

    for detector in points:
        maximum_distance = max(
            math.dist(detector, corner)
            for corner in corners
        )

        if maximum_distance <= certificate_distance_m:
            result.append(detector)

    return result


def certify_21_point_continuous_coverage(
    points: Sequence[Point],
    *,
    target_radius_m: float,
    certificate_distance_m: float,
    actual_reception_distance_m: float,
    max_depth: int,
    max_boxes: int,
    hull_tolerance_m2: float,
    symmetry_tolerance_m: float,
) -> dict:
    """连续认证目标圆内任意源位置均满足局部凸包条件。

    每个通过认证的矩形盒满足：

    1. 选出的检测点到盒内任意位置均不超过998米；
    2. 矩形四个角均位于这些检测点的凸包内；
    3. 因矩形和凸包均为凸集，整个矩形均位于该凸包内。

    点集具有90度旋转对称性，因此只需认证第一象限四分之一圆。
    """

    if certificate_distance_m >= actual_reception_distance_m:
        raise ValueError(
            "连续认证距离必须严格小于实际最低接收距离"
        )

    verify_quarter_turn_symmetry(
        points,
        symmetry_tolerance_m,
    )

    # 元素为(x_min,x_max,y_min,y_max,depth)。
    pending = [(
        0.0,
        target_radius_m,
        0.0,
        target_radius_m,
        0,
    )]

    total_box_count = 0
    certified_box_count = 0
    outside_box_count = 0
    split_box_count = 0
    maximum_certified_depth = 0
    minimum_guaranteed_point_count = len(points)

    while pending:
        (
            x_min,
            x_max,
            y_min,
            y_max,
            depth,
        ) = pending.pop()

        total_box_count += 1

        if total_box_count > max_boxes:
            raise ValueError(
                "21点连续认证超过最大区域数量："
                f"{max_boxes}"
            )

        if not box_intersects_quarter_disk(
            x_min,
            y_min,
            target_radius_m,
        ):
            outside_box_count += 1
            continue

        corners = box_corners(
            x_min,
            x_max,
            y_min,
            y_max,
        )

        guaranteed_points = (
            guaranteed_near_points_for_box(
                points,
                corners,
                certificate_distance_m,
            )
        )

        hull = convex_hull(guaranteed_points)

        certified = (
            len(hull) >= 3
            and all(
                point_in_convex_polygon(
                    corner,
                    hull,
                    hull_tolerance_m2,
                )
                for corner in corners
            )
        )

        if certified:
            certified_box_count += 1
            maximum_certified_depth = max(
                maximum_certified_depth,
                depth,
            )
            minimum_guaranteed_point_count = min(
                minimum_guaranteed_point_count,
                len(guaranteed_points),
            )
            continue

        if depth >= max_depth:
            raise ValueError(
                "21点连续凸包认证失败："
                f"区域=({x_min:.9f},{x_max:.9f})×"
                f"({y_min:.9f},{y_max:.9f})，"
                f"深度={depth}，"
                f"保证近点数={len(guaranteed_points)}"
            )

        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        next_depth = depth + 1

        pending.extend((
            (
                x_min,
                x_mid,
                y_min,
                y_mid,
                next_depth,
            ),
            (
                x_mid,
                x_max,
                y_min,
                y_mid,
                next_depth,
            ),
            (
                x_min,
                x_mid,
                y_mid,
                y_max,
                next_depth,
            ),
            (
                x_mid,
                x_max,
                y_mid,
                y_max,
                next_depth,
            ),
        ))
        split_box_count += 1

    return {
        "passed": True,
        "method": (
            "continuous_adaptive_box_"
            "guaranteed_near_convex_hull"
        ),
        "symmetry_reduction": (
            "90_degree_rotational_symmetry;"
            "first_quadrant_certifies_full_disk"
        ),
        "target_radius_m": target_radius_m,
        "certificate_distance_m": (
            certificate_distance_m
        ),
        "actual_min_reception_distance_m": (
            actual_reception_distance_m
        ),
        "distance_reserve_m": (
            actual_reception_distance_m
            - certificate_distance_m
        ),
        "total_box_count": total_box_count,
        "certified_box_count": (
            certified_box_count
        ),
        "outside_box_count": outside_box_count,
        "split_box_count": split_box_count,
        "maximum_certified_depth": (
            maximum_certified_depth
        ),
        "minimum_guaranteed_point_count": (
            minimum_guaranteed_point_count
        ),
        "random_sampling_used": False,
    }


def build_21_point_certified_search(
    config: v3.Q4Config,
    tuning: V9Tuning,
) -> tuple[v3.GridPlan, dict]:
    points = build_21_point_coordinates(tuning)

    certificate = (
        certify_21_point_continuous_coverage(
            points,
            target_radius_m=(
                config.target_radius_m
            ),
            certificate_distance_m=(
                tuning.grid21_certificate_distance_m
            ),
            actual_reception_distance_m=(
                config.reception_radius_min_m
            ),
            max_depth=(
                tuning.grid21_certificate_max_depth
            ),
            max_boxes=(
                tuning.grid21_certificate_max_boxes
            ),
            hull_tolerance_m2=(
                tuning.grid21_hull_tolerance_m2
            ),
            symmetry_tolerance_m=(
                tuning.grid21_symmetry_tolerance_m
            ),
        )
    )

    if not certificate["passed"]:
        raise ValueError(
            "21点连续几何认证没有通过"
        )

    origin: Point = (0.0, 0.0)

    route = v3.nearest_neighbor_route(
        points,
        origin,
    )
    route = v3.two_opt_open(
        route,
        origin,
    )

    if (
        len(route) != 21
        or set(route) != set(points)
    ):
        raise ValueError(
            "21点开放路线没有完整访问全部检测点"
        )

    route_distance_m = (
        v3.open_route_distance(
            origin,
            route,
        )
    )

    plan = v3.GridPlan(
        side_m=(
            tuning.grid21_certificate_distance_m
        ),
        points=list(points),
        route=list(route),
        triangle_count=0,
        route_distance_m=route_distance_m,
        triangles=[],
    )

    outer_boundary_apothem_m = (
        tuning.grid21_outer_radius_m
        * math.cos(
            math.pi
            / tuning.grid21_outer_count
        )
    )

    diagnostics = {
        "type": (
            "21_point_dual_ring_"
            "continuous_convex_hull_certificate"
        ),
        "point_count": len(points),
        "inner_point_count": (
            tuning.grid21_inner_count
        ),
        "inner_radius_m": (
            tuning.grid21_inner_radius_m
        ),
        "inner_angles_deg": [
            tuning.grid21_inner_phase_deg
            + 45.0 * index
            for index in range(
                tuning.grid21_inner_count
            )
        ],
        "outer_point_count": (
            tuning.grid21_outer_count
        ),
        "outer_radius_m": (
            tuning.grid21_outer_radius_m
        ),
        "outer_angles_deg": [
            tuning.grid21_outer_phase_deg
            + 30.0 * index
            for index in range(
                tuning.grid21_outer_count
            )
        ],
        "outer_boundary_apothem_m": (
            outer_boundary_apothem_m
        ),
        "route_distance_m": route_distance_m,
        "triangle_count": 0,
        "certificate_criterion": (
            "every_source_is_inside_the_convex_hull_"
            "of_detection_points_within_998m"
        ),
        "continuous_certificate": certificate,
        "proof": (
            "目标圆内任意源位置均位于其998米范围内检测点的"
            "凸包中；若某个定向发射前半平面不包含任何这些点，"
            "则存在经过源位置的直线把全部局部检测点严格分离到"
            "反方向，这与源位置属于其凸包矛盾。因此任意方向源"
            "至少能被一个距离不超过998米的检测点发现。"
        ),
    }

    return plan, diagnostics


class Q4V9Runner(v8.Q4V8Runner):
    def __init__(
        self,
        client,
        *,
        config: v3.Q4Config,
        tuning: V9Tuning | None = None,
        grid_mode: str = "21",
        planner_config=None,
        geometry_config=None,
        output_dir: Path | None = None,
    ):
        self.v9 = tuning or V9Tuning()

        if grid_mode not in {"21", "25"}:
            raise ValueError(
                "grid_mode必须为'21'或'25'"
            )

        self.grid_mode = grid_mode

        # 完整初始化V8。定位、恢复、清除和调度均保持V8实现。
        super().__init__(
            client,
            config=config,
            tuning=self.v9,
            planner_config=planner_config,
            geometry_config=geometry_config,
            output_dir=output_dir,
        )

        self._v9_baseline_25_diagnostics = dict(
            self.dual_ring_diagnostics
        )
        self._v9_21_certificate: dict | None = None

        self.counts.update({
            "v9_grid21_selected": 0,
            "v9_grid25_selected": 0,
            "v9_continuous_certificate_passed": 0,
        })

        if self.grid_mode == "21":
            plan, diagnostics = (
                build_21_point_certified_search(
                    self.cfg,
                    self.v9,
                )
            )

            baseline_distance_m = float(
                self._v9_baseline_25_diagnostics[
                    "route_distance_m"
                ]
            )

            diagnostics[
                "baseline_25_route_distance_m"
            ] = baseline_distance_m
            diagnostics[
                "route_difference_vs_25_m"
            ] = (
                plan.route_distance_m
                - baseline_distance_m
            )
            diagnostics[
                "nodes_removed_vs_25"
            ] = 4

            self.global_grid = plan
            self.search_points = list(plan.route)
            self.dual_ring_diagnostics = (
                diagnostics
            )

            self._v9_21_certificate = (
                diagnostics[
                    "continuous_certificate"
                ]
            )

            # V6缓存与认证节点编号绑定，替换点集后必须清空。
            self._v6_unknown_probability_cache.clear()
            self._v6_unknown_omni_masks = None
            self._v6_unknown_directional_masks = None

            self.counts[
                "v9_grid21_selected"
            ] += 1
            self.counts[
                "v9_continuous_certificate_passed"
            ] += 1
        else:
            self.counts[
                "v9_grid25_selected"
            ] += 1

            self.dual_ring_diagnostics = {
                **self.dual_ring_diagnostics,
                "comparison_variant": (
                    "25_point_v8_control"
                ),
                "nodes_removed_vs_25": 0,
            }

    def mark_absent(self) -> None:
        if self.grid_mode == "25":
            super().mark_absent()
            return

        for record in self.records.values():
            if record.status != "UNKNOWN":
                continue

            if (
                self.discovered_count
                >= self.cfg.source_count_max
            ):
                reason = (
                    "source_count_upper_bound_reached"
                )
            elif (
                self.search_complete
                and len(record.global_nodes_tested)
                == len(self.search_points)
            ):
                reason = (
                    "all_21_point_continuous_"
                    "convex_hull_certificate_nodes_"
                    "no_signal"
                )
            else:
                continue

            record.status = "ABSENT"
            record.absent_reason = reason

            self.event(
                "absent",
                channel=record.channel_id,
                reason=reason,
                certificate_type=(
                    "continuous_local_convex_hull"
                ),
                certificate_distance_m=(
                    self.v9
                    .grid21_certificate_distance_m
                ),
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

        if self.grid_mode == "21":
            # 覆盖V4遗留的三角网格证明说明。
            result["global_grid"] = dict(
                self.dual_ring_diagnostics
            )

            result[
                "v8_optimization"
            ][
                "certificate_node_reduction"
            ] = True

        discovered_count = int(
            result["discovered_count"]
        )
        cleared_count = int(
            result["cleared_count"]
        )

        clear_to_discovered_ratio = (
            cleared_count / discovered_count
            if discovered_count > 0
            else None
        )
        clear_to_maximum_ratio = (
            cleared_count
            / self.cfg.source_count_max
        )

        baseline_route_m = float(
            self._v9_baseline_25_diagnostics[
                "route_distance_m"
            ]
        )
        selected_route_m = float(
            self.dual_ring_diagnostics[
                "route_distance_m"
            ]
        )

        result["v9_tuning"] = asdict(self.v9)
        result["v9_optimization"] = {
            "grid_mode": self.grid_mode,
            "certification_node_count": len(
                self.search_points
            ),
            "baseline_25_node_count": 25,
            "nodes_removed_vs_25": (
                4
                if self.grid_mode == "21"
                else 0
            ),
            "selected_route_distance_m": (
                selected_route_m
            ),
            "baseline_25_route_distance_m": (
                baseline_route_m
            ),
            "route_difference_vs_25_m": (
                selected_route_m
                - baseline_route_m
            ),
            "continuous_certificate": (
                self._v9_21_certificate
            ),
            "localization_module_changed": False,
            "clear_module_changed": False,
            "recovery_module_changed": False,
            "scheduler_module_changed": False,
            "certificate_preserved": (
                (
                    self.grid_mode == "25"
                    or (
                        self._v9_21_certificate
                        is not None
                        and self._v9_21_certificate[
                            "passed"
                        ]
                    )
                )
                and (
                    self.search_complete
                    or self.discovered_count
                    >= self.cfg.source_count_max
                )
            ),
        }

        result["grid_comparison_metrics"] = {
            "grid_mode": self.grid_mode,
            "cleared_count": cleared_count,
            "discovered_count": discovered_count,
            "clear_to_discovered_ratio": (
                clear_to_discovered_ratio
            ),
            "clear_to_maximum_ratio": (
                clear_to_maximum_ratio
            ),
            "total_virtual_time_s": (
                result["virtual_time_s"]
            ),
            "average_clear_time_s": (
                result["virtual_time_s"]
                / cleared_count
                if cleared_count > 0
                else None
            ),
            "total_distance_m": (
                result["total_distance_m"]
            ),
            "measure_count": (
                result["counts"]["measure"]
            ),
            "unknown_measure_count": (
                result["counts"][
                    "unknown_measure"
                ]
            ),
            "visited_search_node_count": len(
                self.dynamic_search_visit_order
            ),
        }

        result["model_note"] = (
            result.get("model_note", "")
            + (
                "；V9使用21点局部凸包连续认证搜索；"
                "定位、恢复、清除及价值调度继承V8且未修改"
                if self.grid_mode == "21"
                else
                "；V9本次使用V8的25点认证网格作为对照组"
            )
        )

        return result


def run_self_test() -> dict:
    inherited = v8.run_self_test()

    tuning = V9Tuning()
    config = v3.Q4Config(
        global_grid_side_m=1000.0,
    )

    plan21, diagnostics21 = (
        build_21_point_certified_search(
            config,
            tuning,
        )
    )

    plan25, diagnostics25 = (
        v8.build_optimized_certified_search(
            config,
            tuning,
        )
    )

    certificate = diagnostics21[
        "continuous_certificate"
    ]

    assert certificate["passed"]
    assert certificate["random_sampling_used"] is False
    assert len(plan21.points) == 21
    assert len(plan21.route) == 21
    assert set(plan21.points) == set(plan21.route)
    assert len(plan25.points) == 25

    assert (
        tuning.grid21_certificate_distance_m
        < config.reception_radius_min_m
    )

    assert (
        abs(
            certificate["distance_reserve_m"]
            - 2.0
        )
        <= 1e-9
    )

    assert (
        certificate["maximum_certified_depth"]
        <= tuning.grid21_certificate_max_depth
    )

    return {
        **inherited,
        "v9_status": "ok",
        "v9_21_point_certificate_passed": True,
        "v9_certificate_method": (
            certificate["method"]
        ),
        "v9_random_sampling_used": False,
        "v9_target_radius_m": (
            config.target_radius_m
        ),
        "v9_certificate_distance_m": (
            tuning.grid21_certificate_distance_m
        ),
        "v9_actual_min_reception_distance_m": (
            config.reception_radius_min_m
        ),
        "v9_distance_reserve_m": (
            certificate["distance_reserve_m"]
        ),
        "v9_total_certificate_boxes": (
            certificate["total_box_count"]
        ),
        "v9_certified_boxes": (
            certificate["certified_box_count"]
        ),
        "v9_outside_boxes": (
            certificate["outside_box_count"]
        ),
        "v9_maximum_certified_depth": (
            certificate[
                "maximum_certified_depth"
            ]
        ),
        "v9_21_point_route_distance_m": (
            plan21.route_distance_m
        ),
        "v9_25_point_route_distance_m": (
            plan25.route_distance_m
        ),
        "v9_route_difference_m": (
            plan21.route_distance_m
            - plan25.route_distance_m
        ),
        "v9_nodes_removed": 4,
        "v9_localization_module_changed": False,
        "v9_clear_module_changed": False,
        "v9_recovery_module_changed": False,
        "v9_scheduler_module_changed": False,
        "v9_21_diagnostics": diagnostics21,
        "v9_25_diagnostics": diagnostics25,
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
        "--grid-mode",
        choices=("21", "25"),
        default="21",
        help=(
            "21使用凸包连续认证方案；"
            "25使用V8认证网格作为对照"
        ),
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
            "离线认证请使用 --self-test"
        )

    tuning = V9Tuning()

    config = v3.Q4Config(
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
        v3.load_config(args.geometry_config)
        if args.geometry_config is not None
        else v3.GeometryConfig()
    )

    planner_config = (
        v3.PlannerConfig(**json.loads(
            args.planner_config.read_text(
                encoding="utf-8-sig"
            )
        ))
        if args.planner_config is not None
        else v3.PlannerConfig()
    )

    output_dir = (
        args.output_dir
        or Path("runs") / datetime.now().strftime(
            f"q4_v9_{args.grid_mode}pt_"
            "%Y%m%d_%H%M%S_%f"
        )
    )

    client = v3.SimulatorClient(
        args.robot_id,
        args.base_url,
    )

    runner = Q4V9Runner(
        client,
        config=config,
        tuning=tuning,
        grid_mode=args.grid_mode,
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
        f"问题四V9策略启动："
        f"{args.grid_mode}点认证方案，"
        f"输出目录：{output_dir}",
        flush=True,
    )

    summary = runner.run()

    comparison = summary[
        "grid_comparison_metrics"
    ]

    print(json.dumps({
        "outcome": summary["outcome"],
        "reason": summary["reason"],
        "grid_mode": args.grid_mode,
        "certification_node_count": (
            len(runner.search_points)
        ),
        "discovered_count": (
            summary["discovered_count"]
        ),
        "cleared_count": (
            summary["cleared_count"]
        ),
        "clear_to_discovered_ratio": (
            comparison[
                "clear_to_discovered_ratio"
            ]
        ),
        "clear_to_maximum_ratio": (
            comparison[
                "clear_to_maximum_ratio"
            ]
        ),
        "virtual_time_s": (
            summary["virtual_time_s"]
        ),
        "average_clear_time_s": (
            comparison["average_clear_time_s"]
        ),
        "total_distance_m": (
            summary["total_distance_m"]
        ),
        "unknown_measure_count": (
            comparison["unknown_measure_count"]
        ),
        "visited_search_node_count": (
            comparison[
                "visited_search_node_count"
            ]
        ),
        "global_grid": (
            summary["global_grid"]
        ),
        "v8_optimization": (
            summary["v8_optimization"]
        ),
        "v9_optimization": (
            summary["v9_optimization"]
        ),
        "grid_comparison_metrics": (
            comparison
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