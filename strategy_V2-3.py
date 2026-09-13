"""
问题三策略 V2-3

基于 strategy_V2-2-1.py，实现：

优先级0：
    联合评价 FOUND 源的多个定位候选与 READY 源清除任务。

优先级1：
    机会补测由固定次数扩展为“基础次数 + 边际收益补测”。

优先级2：
    将目标区域外包由正方形收紧为目标圆的24边外接多边形。

优先级3：
    保留原有7点严格认证，但动态排列剩余认证点访问顺序。

正确性约束：
    1. UNKNOWN 频道仍需完成全部7点认证才能判定 ABSENT；
    2. 发现16个不同源后才允许提前结束搜索；
    3. READY 仍由 near 反馈或全历史角域 + MEC 严格判定；
    4. 概率和采样只影响调度顺序，不参与最终清除证明；
    5. 清除失败恢复逻辑继续复用基础版本。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.util
import itertools
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any, Iterable


# ============================================================================
# 加载基础版本
# ============================================================================

BASE_FILE = Path(__file__).with_name("strategy_V2-2-1.py")
BASE_MODULE_NAME = "q3_strategy_v221_base"


def load_base_module():
    if not BASE_FILE.exists():
        raise FileNotFoundError(
            f"找不到基础策略文件：{BASE_FILE}\n"
            "请将本文件与 strategy_V2-2-1.py 放在同一目录。"
        )

    spec = importlib.util.spec_from_file_location(
        BASE_MODULE_NAME,
        BASE_FILE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础策略：{BASE_FILE}")

    module = importlib.util.module_from_spec(spec)

    # Windows multiprocessing spawn 需要能够通过模块名找到基础模块。
    sys.modules[BASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()

Point = tuple[float, float]

SPEED_MPS = 5.0
MEASURE_OPERATION_S = 5.0
SWITCH_OPERATION_S = 1.0

TARGET_RADIUS_M = 1800.0
TARGET_OUTER_POLYGON_SIDES = 24


# ============================================================================
# 优先级2：目标圆24边外接多边形
# ============================================================================

def make_target_outer_halfplanes(
    radius_m: float = TARGET_RADIUS_M,
    sides: int = TARGET_OUTER_POLYGON_SIDES,
) -> list[tuple[float, float, float]]:
    """
    返回目标圆的正多边形外接半平面。

    每个半平面：
        cos(theta) * x + sin(theta) * y <= radius_m

    所有半径不超过 radius_m 的点都满足约束，因此不会裁掉合法源。
    """
    if sides < 8:
        raise ValueError("外接多边形边数不能小于8。")

    return [
        (
            math.cos(2.0 * math.pi * index / sides),
            math.sin(2.0 * math.pi * index / sides),
            float(radius_m),
        )
        for index in range(sides)
    ]


# 必须在模块加载阶段设置。
# Windows规划子进程重新导入本文件时，也会得到相同的外边界。
base.BOX = make_target_outer_halfplanes()


# ============================================================================
# 新增参数
# ============================================================================

@dataclass(frozen=True)
class V23Config(base.Q3Config):
    # 优先级0：联合定位与清除
    joint_candidates_per_channel: int = 4
    joint_free_route_regret_m: float = 100.0
    joint_max_route_regret_m: float = 420.0
    joint_value_cover_ratio: float = 1.10
    joint_ready_value_m: float = 900.0
    joint_reduction_weight: float = 0.70
    joint_wait_bonus_m: float = 18.0
    joint_measure_direct_cap_m: float = 1800.0

    # 优先级1：机会补测
    opportunity_extra_limit: int = 2
    opportunity_extra_min_possible_receive: float = 0.82
    opportunity_extra_min_guaranteed_receive: float = 0.30
    opportunity_extra_min_ready_probability: float = 0.52
    opportunity_extra_min_expected_net_s: float = 5.0
    opportunity_extra_value_cover_ratio: float = 1.12
    opportunity_future_leg_floor_m: float = 250.0
    opportunity_future_leg_cap_m: float = 800.0

    # 优先级3：动态认证点顺序
    search_unknown_state_count: int = 384
    search_unknown_value_scale: float = 0.60
    search_found_reduction_weight: float = 0.30
    search_found_value_cap_m: float = 1000.0
    search_found_ready_bonus_m: float = 500.0
    search_free_route_regret_m: float = 40.0
    search_max_route_regret_m: float = 230.0
    search_value_cover_ratio: float = 1.12
    search_inline_clear_batch_limit: int = 2

    def __post_init__(self):
        super().__post_init__()

        integer_fields = (
            "joint_candidates_per_channel",
            "opportunity_extra_limit",
            "search_unknown_state_count",
            "search_inline_clear_batch_limit",
        )

        for field_name in integer_fields:
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"{field_name} 必须是非负整数"
                )

        if self.joint_candidates_per_channel < 1:
            raise ValueError(
                "joint_candidates_per_channel 必须至少为1"
            )

        if self.search_unknown_state_count < 48:
            raise ValueError(
                "search_unknown_state_count 不能小于48"
            )

        probability_fields = (
            "opportunity_extra_min_possible_receive",
            "opportunity_extra_min_guaranteed_receive",
            "opportunity_extra_min_ready_probability",
        )

        for field_name in probability_fields:
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"{field_name} 必须位于 [0,1]"
                )

        nonnegative_fields = (
            "joint_free_route_regret_m",
            "joint_max_route_regret_m",
            "joint_value_cover_ratio",
            "joint_ready_value_m",
            "joint_reduction_weight",
            "joint_wait_bonus_m",
            "joint_measure_direct_cap_m",
            "opportunity_extra_min_expected_net_s",
            "opportunity_extra_value_cover_ratio",
            "opportunity_future_leg_floor_m",
            "opportunity_future_leg_cap_m",
            "search_unknown_value_scale",
            "search_found_reduction_weight",
            "search_found_value_cap_m",
            "search_found_ready_bonus_m",
            "search_free_route_regret_m",
            "search_max_route_regret_m",
            "search_value_cover_ratio",
        )

        for field_name in nonnegative_fields:
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"{field_name} 必须是有限非负数"
                )

        if (
            self.joint_max_route_regret_m
            < self.joint_free_route_regret_m
        ):
            raise ValueError(
                "joint_max_route_regret_m 不能小于 "
                "joint_free_route_regret_m"
            )

        if (
            self.search_max_route_regret_m
            < self.search_free_route_regret_m
        ):
            raise ValueError(
                "search_max_route_regret_m 不能小于 "
                "search_free_route_regret_m"
            )

        if (
            self.opportunity_future_leg_cap_m
            < self.opportunity_future_leg_floor_m
        ):
            raise ValueError(
                "opportunity_future_leg_cap_m 不能小于 "
                "opportunity_future_leg_floor_m"
            )


# ============================================================================
# 通用数学函数
# ============================================================================

def distance(first: Point, second: Point) -> float:
    return math.dist(first, second)


def clamp(
    value: float,
    lower: float,
    upper: float,
) -> float:
    return max(lower, min(upper, value))


def bearing_from_to(
    point: Point,
    source: Point,
) -> float:
    return (
        math.degrees(
            math.atan2(
                source[1] - point[1],
                source[0] - point[0],
            )
        )
        % 360.0
    )


def poisson_binomial_tail(
    probabilities: Iterable[float],
    required_hits: int,
) -> float:
    probabilities = [
        clamp(float(probability), 0.0, 1.0)
        for probability in probabilities
    ]

    if required_hits <= 0:
        return 1.0

    if required_hits > len(probabilities):
        return 0.0

    distribution = [0.0] * (len(probabilities) + 1)
    distribution[0] = 1.0

    used = 0
    for probability in probabilities:
        for hits in range(used, -1, -1):
            previous = distribution[hits]
            distribution[hits] = (
                previous * (1.0 - probability)
            )
            distribution[hits + 1] += (
                previous * probability
            )
        used += 1

    return sum(distribution[required_hits:])


# ============================================================================
# V2-3运行器
# ============================================================================

class Q3V23Runner(base.Q3Runner):
    def __init__(
        self,
        client,
        *,
        config=None,
        planner_config=None,
        geometry_config=None,
        output_dir=None,
        plan_function=base.bounded_plan,
    ):
        # 防止基础模块被其他代码重新设置。
        base.BOX = make_target_outer_halfplanes()

        super().__init__(
            client,
            config=config or V23Config(),
            planner_config=planner_config,
            geometry_config=geometry_config,
            output_dir=output_dir,
            plan_function=plan_function,
        )

        if not isinstance(self.cfg, V23Config):
            raise TypeError(
                "Q3V23Runner 必须使用 V23Config"
            )

        self.dynamic_coverage_visit_order: list[int] = []

        self._unknown_states = self._build_unknown_states(
            self.cfg.search_unknown_state_count
        )

        self._opportunity_metric_cache: dict[
            tuple,
            dict[str, float],
        ] = {}

        self.v23_metrics: dict[str, float | int] = {
            "search_route_decisions": 0,
            "search_nonshortest_selected": 0,
            "search_route_gate_rejected": 0,
            "search_expected_discovery_value_m": 0.0,
            "search_expected_early_finish_value_m": 0.0,
            "search_found_opportunity_value_m": 0.0,
            "search_inline_batch_clears": 0,
            "joint_scheduler_rounds": 0,
            "joint_candidate_actions": 0,
            "joint_route_gate_rejected": 0,
            "joint_measure_selected": 0,
            "joint_clear_selected": 0,
            "joint_predicted_ready_selected": 0,
            "opportunity_extra_considered": 0,
            "opportunity_extra_accepted": 0,
            "opportunity_extra_rejected": 0,
            "opportunity_expected_saved_s": 0.0,
        }

    # ========================================================================
    # 优先级3：动态七点认证顺序
    # ========================================================================

    def _build_unknown_states(
        self,
        requested_count: int,
    ) -> list[tuple[float, float, float]]:
        possible_reception_radii = (
            1000.0,
            1250.0,
            1500.0,
        )

        position_count = max(
            16,
            math.ceil(
                requested_count
                / len(possible_reception_radii)
            ),
        )

        golden_angle = (
            math.pi * (3.0 - math.sqrt(5.0))
        )

        states: list[tuple[float, float, float]] = []

        for index in range(position_count):
            radial_fraction = (
                index + 0.5
            ) / position_count

            radial_distance = (
                TARGET_RADIUS_M
                * math.sqrt(radial_fraction)
            )

            angle = index * golden_angle

            x = radial_distance * math.cos(angle)
            y = radial_distance * math.sin(angle)

            for reception_radius in (
                possible_reception_radii
            ):
                states.append(
                    (x, y, reception_radius)
                )

        return states[:requested_count]

    def _feasible_unknown_states(
        self,
        channel: int,
    ) -> list[tuple[float, float, float]]:
        tested_indices = [
            index
            for index in range(len(self.points))
            if self.scan_done[index][channel - 1]
        ]

        if not tested_indices:
            return self._unknown_states

        feasible = []

        for x, y, reception_radius in (
            self._unknown_states
        ):
            source = (x, y)

            if all(
                distance(
                    source,
                    self.points[index],
                )
                > reception_radius
                for index in tested_indices
            ):
                feasible.append(
                    (x, y, reception_radius)
                )

        return feasible

    def _unknown_hit_probability(
        self,
        channel: int,
        coverage_index: int,
    ) -> float:
        record = self.records[channel]

        if record.status != "UNKNOWN":
            return 0.0

        states = self._feasible_unknown_states(
            channel
        )
        if not states:
            return 0.0

        point = self.points[coverage_index]

        hit_count = sum(
            distance((x, y), point)
            <= reception_radius
            for x, y, reception_radius in states
        )

        return hit_count / len(states)

    def _coverage_route_distance(
        self,
        order: Iterable[int],
    ) -> float:
        position = self.client.position
        total = 0.0

        for index in order:
            point = self.points[index]
            total += distance(position, point)
            position = point

        return total

    def _best_coverage_orders_by_first(
        self,
        remaining: list[int],
    ) -> dict[int, tuple[float, tuple[int, ...]]]:
        if not remaining:
            return {}

        if len(remaining) == 1:
            index = remaining[0]
            return {
                index: (
                    distance(
                        self.client.position,
                        self.points[index],
                    ),
                    (index,),
                )
            }

        best: dict[
            int,
            tuple[float, tuple[int, ...]],
        ] = {}

        # 最多7!种，可以直接枚举。
        for order in itertools.permutations(remaining):
            route_distance = (
                self._coverage_route_distance(order)
            )
            first = order[0]

            old = best.get(first)

            if (
                old is None
                or route_distance < old[0]
            ):
                best[first] = (
                    route_distance,
                    order,
                )

        return best

    def _found_search_value(
        self,
        coverage_index: int,
    ) -> float:
        point = self.points[coverage_index]
        total_value = 0.0

        for record in self.records.values():
            if record.status != "FOUND":
                continue

            if record.clearance_radius is None:
                continue

            if base.point_already_tested(
                record,
                point,
                self.cfg.repeated_point_m,
            ):
                continue

            try:
                estimate = base.source_estimate(record)
            except base.IncompleteRun:
                continue

            estimated_distance = distance(
                estimate,
                point,
            )

            if (
                estimated_distance
                > self.cfg.opportunistic_receive_m
            ):
                continue

            cross_angle = base.candidate_cross_angle(
                record,
                point,
                estimate,
            )

            angular_quality = math.sin(
                math.radians(
                    clamp(
                        cross_angle,
                        0.0,
                        90.0,
                    )
                )
            )

            reduction_proxy = min(
                self.cfg.search_found_value_cap_m,
                record.clearance_radius
                * angular_quality
                * self.cfg.search_found_reduction_weight,
            )

            ready_bonus = 0.0
            if (
                record.clearance_radius <= 90.0
                and cross_angle
                >= self.cfg.min_cross_angle_deg
            ):
                ready_bonus = (
                    self.cfg.search_found_ready_bonus_m
                )

            total_value += (
                reduction_proxy + ready_bonus
            )

        return total_value

    def _select_next_coverage_point(
        self,
        remaining: list[int],
    ) -> tuple[int, dict[str, Any]]:
        route_options = (
            self._best_coverage_orders_by_first(
                remaining
            )
        )

        # 机器人进入后位于原点，先测原点没有额外移动。
        if (
            not self.dynamic_coverage_visit_order
            and 0 in remaining
        ):
            route_distance, order = (
                route_options[0]
            )

            return 0, {
                "reason": "origin_first",
                "selected_index": 0,
                "selected_order": list(order),
                "route_distance_m": route_distance,
                "route_regret_m": 0.0,
                "unknown_value_m": 0.0,
                "early_finish_value_m": 0.0,
                "found_value_m": 0.0,
            }

        baseline_distance = min(
            route_distance
            for route_distance, _ in (
                route_options.values()
            )
        )

        unknown_channels = [
            channel
            for channel, record
            in self.records.items()
            if record.status == "UNKNOWN"
        ]

        discovered_count = self.discovered_count

        candidates = []

        for coverage_index in remaining:
            route_distance, order = (
                route_options[coverage_index]
            )

            route_regret = max(
                0.0,
                route_distance - baseline_distance,
            )

            probabilities = [
                self._unknown_hit_probability(
                    channel,
                    coverage_index,
                )
                for channel in unknown_channels
            ]

            future_node_count = max(
                0,
                len(remaining) - 1,
            )

            # 5秒测量 + 预估1秒切频，按5m/s折算。
            avoided_measure_equivalent_m = 30.0

            unknown_value = (
                sum(probabilities)
                * future_node_count
                * avoided_measure_equivalent_m
                * self.cfg.search_unknown_value_scale
            )

            needed_to_finish = max(
                0,
                16 - discovered_count,
            )

            finish_probability = (
                poisson_binomial_tail(
                    probabilities,
                    needed_to_finish,
                )
            )

            distance_to_first = distance(
                self.client.position,
                self.points[coverage_index],
            )

            avoidable_tail_distance = max(
                0.0,
                route_distance - distance_to_first,
            )

            early_finish_value = (
                finish_probability
                * avoidable_tail_distance
            )

            found_value = (
                self._found_search_value(
                    coverage_index
                )
            )

            total_value = (
                unknown_value
                + early_finish_value
                + found_value
            )

            allowed = (
                route_regret
                <= self.cfg.search_free_route_regret_m
                or (
                    route_regret
                    <= self.cfg.search_max_route_regret_m
                    and total_value
                    >= (
                        self.cfg.search_value_cover_ratio
                        * route_regret
                    )
                )
            )

            if not allowed:
                self.v23_metrics[
                    "search_route_gate_rejected"
                ] += 1

            candidates.append(
                {
                    "index": coverage_index,
                    "order": order,
                    "route_distance_m": route_distance,
                    "route_regret_m": route_regret,
                    "unknown_value_m": unknown_value,
                    "early_finish_value_m": (
                        early_finish_value
                    ),
                    "finish_probability": (
                        finish_probability
                    ),
                    "found_value_m": found_value,
                    "total_value_m": total_value,
                    "allowed": allowed,
                    "score": (
                        total_value - route_regret
                    ),
                }
            )

        allowed_candidates = [
            candidate
            for candidate in candidates
            if candidate["allowed"]
        ]

        if not allowed_candidates:
            allowed_candidates = candidates

        selected = max(
            allowed_candidates,
            key=lambda candidate: (
                candidate["score"],
                -candidate["route_distance_m"],
                -candidate["index"],
            ),
        )

        shortest_candidate = min(
            candidates,
            key=lambda candidate: (
                candidate["route_distance_m"],
                candidate["index"],
            ),
        )

        self.v23_metrics[
            "search_route_decisions"
        ] += 1

        if (
            selected["index"]
            != shortest_candidate["index"]
        ):
            self.v23_metrics[
                "search_nonshortest_selected"
            ] += 1

        self.v23_metrics[
            "search_expected_discovery_value_m"
        ] += selected["unknown_value_m"]

        self.v23_metrics[
            "search_expected_early_finish_value_m"
        ] += selected["early_finish_value_m"]

        self.v23_metrics[
            "search_found_opportunity_value_m"
        ] += selected["found_value_m"]

        return selected["index"], {
            "reason": "dynamic_search_value",
            "selected_index": selected["index"],
            "selected_order": list(
                selected["order"]
            ),
            "route_distance_m": selected[
                "route_distance_m"
            ],
            "route_regret_m": selected[
                "route_regret_m"
            ],
            "unknown_value_m": selected[
                "unknown_value_m"
            ],
            "early_finish_value_m": selected[
                "early_finish_value_m"
            ],
            "finish_probability": selected[
                "finish_probability"
            ],
            "found_value_m": selected[
                "found_value_m"
            ],
        }

    def _search_channels_at_point(
        self,
        coverage_index: int,
    ) -> list[int]:
        unknown_channels = [
            channel
            for channel, record
            in self.records.items()
            if (
                record.status == "UNKNOWN"
                and not self.scan_done[
                    coverage_index
                ][channel - 1]
            )
        ]

        current_channel = (
            self.client.current_channel
        )

        return sorted(
            unknown_channels,
            key=lambda channel: (
                channel != current_channel,
                -self._unknown_hit_probability(
                    channel,
                    coverage_index,
                ),
                channel,
            ),
        )

    def _shortest_remaining_coverage_anchor(
        self,
        remaining: list[int],
    ) -> Point | None:
        if not remaining:
            return None

        options = (
            self._best_coverage_orders_by_first(
                remaining
            )
        )

        best_first = min(
            options,
            key=lambda index: (
                options[index][0],
                index,
            ),
        )

        return self.points[best_first]

    def _perform_search_inline_clears(
        self,
        remaining: list[int],
    ) -> None:
        if not self.cfg.enable_insert:
            return

        if not remaining:
            return

        completed = 0

        while (
            completed
            < self.cfg.search_inline_clear_batch_limit
        ):
            anchor = (
                self._shortest_remaining_coverage_anchor(
                    remaining
                )
            )

            if anchor is None:
                return

            choices = []

            for channel, record in (
                self.records.items()
            ):
                if (
                    record.status != "READY"
                    or record.clearance_center is None
                ):
                    continue

                insertion_cost_s = (
                    base.insertion_cost(
                        self.client.position,
                        record.clearance_center,
                        anchor,
                    )
                )

                choices.append(
                    (
                        insertion_cost_s,
                        channel,
                        record,
                    )
                )

            if not choices:
                return

            insertion_cost_s, _, record = min(
                choices,
                key=lambda item: (
                    item[0],
                    item[1],
                ),
            )

            accepted = (
                insertion_cost_s
                <= self.cfg.insert_threshold_s
            )

            self.event(
                "dynamic_insertion_decision",
                channel=record.channel_id,
                cost_s=insertion_cost_s,
                accepted=accepted,
                remaining_coverage_points=remaining,
            )

            if not accepted:
                return

            success = self.clear(
                record,
                insertion=True,
            )

            if not success:
                return

            completed += 1
            self.v23_metrics[
                "search_inline_batch_clears"
            ] += 1

    def search(self):
        self.phase = "search"
        remaining = list(range(len(self.points)))

        while remaining:
            self.check_budget()

            if self.discovered_count >= 16:
                self.mark_absent()
                break

            if not any(
                record.status == "UNKNOWN"
                for record in self.records.values()
            ):
                break

            (
                coverage_index,
                diagnostics,
            ) = self._select_next_coverage_point(
                remaining
            )

            point = self.points[coverage_index]

            self.event(
                "dynamic_coverage_target",
                coverage_index=coverage_index,
                point=point,
                remaining_indices=list(remaining),
                diagnostics=diagnostics,
            )

            # SimulatorClient 没有 move_to。
            # 第一次 measure 会自动移动至该认证点。
            for channel in (
                self._search_channels_at_point(
                    coverage_index
                )
            ):
                record = self.records[channel]

                if record.status != "UNKNOWN":
                    continue

                self.measure(
                    record,
                    point,
                    coverage_index=coverage_index,
                    measurement_kind="unknown",
                )

                if self.discovered_count >= 16:
                    break

            self.dynamic_coverage_visit_order.append(
                coverage_index
            )

            remaining.remove(coverage_index)

            self.next_coverage_index = len(
                self.dynamic_coverage_visit_order
            )

            if self.discovered_count >= 16:
                self.mark_absent()
                break

            self.perform_opportunistic_measurements(
                point,
                arrival_kind="coverage_scan",
                coverage_index=coverage_index,
            )

            self._perform_search_inline_clears(
                remaining
            )

        self.mark_absent()

    # ========================================================================
    # 优先级1：自适应机会补测
    # ========================================================================

    def _empty_opportunity_metrics(
        self,
    ) -> dict[str, float]:
        return {
            "expected_receive_fraction": 0.0,
            "guaranteed_receive_fraction_robust": 0.0,
            "expected_ready_probability": 0.0,
            "guaranteed_ready_fraction": 0.0,
            "expected_reduction_m": 0.0,
        }

    def _robust_opportunity_metrics(
        self,
        record,
        point: Point,
    ) -> dict[str, float]:
        cache_key = (
            record.channel_id,
            record.revision,
            round(point[0], 5),
            round(point[1], 5),
        )

        cached = (
            self._opportunity_metric_cache.get(
                cache_key
            )
        )

        if cached is not None:
            return cached

        if (
            record.status != "FOUND"
            or record.clearance_radius is None
            or not record.outer_polygon
        ):
            result = (
                self._empty_opportunity_metrics()
            )
            self._opportunity_metric_cache[
                cache_key
            ] = result
            return result

        try:
            samples = base.sample_sources(
                record,
                self.cfg,
                self.pcfg,
            )
        except base.IncompleteRun:
            samples = []

        if not samples:
            result = (
                self._empty_opportunity_metrics()
            )
            self._opportunity_metric_cache[
                cache_key
            ] = result
            return result

        current_radius = (
            record.clearance_radius
        )

        ready_threshold = (
            20.0 - self.cfg.clear_margin_m
        )

        receive_sum = 0.0
        guaranteed_receive_count = 0
        ready_probability_sum = 0.0
        guaranteed_ready_count = 0
        reduction_sum = 0.0

        bearing_error = (
            self.gcfg.bearing_error_deg
        )

        for source in samples:
            (
                lower_radius,
                upper_radius,
            ) = base.reception_radius_interval(
                source,
                record,
                self.pcfg,
            )

            source_distance = distance(
                source,
                point,
            )

            if source_distance <= lower_radius:
                receive_probability = 1.0

            elif source_distance > upper_radius:
                receive_probability = 0.0

            elif upper_radius > lower_radius:
                receive_probability = (
                    upper_radius - source_distance
                ) / (
                    upper_radius - lower_radius
                )

            else:
                receive_probability = 0.0

            receive_probability = clamp(
                receive_probability,
                0.0,
                1.0,
            )

            receive_sum += receive_probability

            guaranteed_receive = (
                source_distance <= lower_radius
            )

            if guaranteed_receive:
                guaranteed_receive_count += 1

            if receive_probability <= 0.0:
                continue

            true_bearing = bearing_from_to(
                point,
                source,
            )

            future_radii = []

            for error in (
                -bearing_error,
                0.0,
                bearing_error,
            ):
                try:
                    future_geometry = (
                        base.lightweight_future_geometry(
                            record.outer_polygon,
                            point,
                            true_bearing + error,
                            bearing_error,
                        )
                    )
                except base.IncompleteRun:
                    future_geometry = None

                if future_geometry is None:
                    future_radius = current_radius
                else:
                    future_radius = min(
                        current_radius,
                        float(future_geometry[1]),
                    )

                future_radii.append(
                    future_radius
                )

            worst_future_radius = max(
                future_radii
            )

            reduction = max(
                0.0,
                current_radius
                - worst_future_radius,
            )

            reduction_sum += (
                receive_probability * reduction
            )

            will_be_ready = (
                worst_future_radius
                <= ready_threshold + 1e-9
            )

            if will_be_ready:
                ready_probability_sum += (
                    receive_probability
                )

                if guaranteed_receive:
                    guaranteed_ready_count += 1

        sample_count = len(samples)

        result = {
            "expected_receive_fraction": (
                receive_sum / sample_count
            ),
            "guaranteed_receive_fraction_robust": (
                guaranteed_receive_count
                / sample_count
            ),
            "expected_ready_probability": (
                ready_probability_sum
                / sample_count
            ),
            "guaranteed_ready_fraction": (
                guaranteed_ready_count
                / sample_count
            ),
            "expected_reduction_m": (
                reduction_sum / sample_count
            ),
        }

        self._opportunity_metric_cache[
            cache_key
        ] = result

        return result

    def opportunistic_candidates(
        self,
        point,
        *,
        arrival_kind,
        coverage_index=None,
        primary_channel=None,
    ):
        candidates = (
            super().opportunistic_candidates(
                point,
                arrival_kind=arrival_kind,
                coverage_index=coverage_index,
                primary_channel=primary_channel,
            )
        )

        augmented = []

        for candidate in candidates:
            record = candidate["record"]

            metrics = (
                self._robust_opportunity_metrics(
                    record,
                    point,
                )
            )

            item = dict(candidate)
            item.update(metrics)
            augmented.append(item)

        return augmented

    def _opportunity_future_leg_m(
        self,
        record,
    ) -> float:
        if record.clearance_radius is None:
            return (
                self.cfg.opportunity_future_leg_floor_m
            )

        estimate = (
            0.70 * record.clearance_radius
        )

        return clamp(
            estimate,
            self.cfg.opportunity_future_leg_floor_m,
            self.cfg.opportunity_future_leg_cap_m,
        )

    def _evaluate_extra_opportunity(
        self,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        record = candidate["record"]

        operation_s = MEASURE_OPERATION_S

        if (
            self.client.current_channel
            != record.channel_id
        ):
            operation_s += SWITCH_OPERATION_S

        future_leg_m = (
            self._opportunity_future_leg_m(
                record
            )
        )

        avoided_future_s = (
            future_leg_m / SPEED_MPS
            + MEASURE_OPERATION_S
        )

        ready_probability = candidate[
            "expected_ready_probability"
        ]

        if (
            record.clearance_radius is None
            or record.clearance_radius <= 1e-9
        ):
            reduction_fraction = 0.0
        else:
            reduction_fraction = clamp(
                candidate["expected_reduction_m"]
                / record.clearance_radius,
                0.0,
                1.0,
            )

        expected_saved_s = (
            ready_probability
            * avoided_future_s
            + 0.25
            * reduction_fraction
            * avoided_future_s
        )

        expected_net_s = (
            expected_saved_s - operation_s
        )

        reception_robust = (
            candidate[
                "guaranteed_receive_fraction_robust"
            ]
            >= (
                self.cfg
                .opportunity_extra_min_guaranteed_receive
            )
            or candidate[
                "expected_receive_fraction"
            ] >= 0.95
        )

        allowed = (
            candidate["possible_receive_fraction"]
            >= (
                self.cfg
                .opportunity_extra_min_possible_receive
            )
            and ready_probability
            >= (
                self.cfg
                .opportunity_extra_min_ready_probability
            )
            and reception_robust
            and expected_net_s
            >= (
                self.cfg
                .opportunity_extra_min_expected_net_s
            )
            and expected_saved_s
            >= (
                self.cfg
                .opportunity_extra_value_cover_ratio
                * operation_s
            )
        )

        result = dict(candidate)

        result.update(
            {
                "extra_allowed": allowed,
                "expected_saved_s": expected_saved_s,
                "expected_net_s": expected_net_s,
                "operation_s_v23": operation_s,
                "future_leg_m": future_leg_m,
            }
        )

        return result

    def perform_opportunistic_measurements(
        self,
        point,
        *,
        arrival_kind,
        coverage_index=None,
        primary_channel=None,
    ):
        """
        保持基础版本的方法签名，因为继承的 clear() 会调用本方法。
        """
        self.counts[
            "opportunistic_arrivals_considered"
        ] += 1

        completed = 0

        base_limit = (
            self.cfg.max_opportunistic_measurements
        )

        total_limit = (
            base_limit
            + self.cfg.opportunity_extra_limit
        )

        while completed < total_limit:
            candidates = (
                self.opportunistic_candidates(
                    point,
                    arrival_kind=arrival_kind,
                    coverage_index=coverage_index,
                    primary_channel=primary_channel,
                )
            )

            if not candidates:
                break

            if completed < base_limit:
                selected = max(
                    candidates,
                    key=lambda item: (
                        item["predicted_clearable"],
                        item.get(
                            "expected_ready_probability",
                            0.0,
                        ),
                        item[
                            "guaranteed_receive_fraction"
                        ],
                        item[
                            "possible_receive_fraction"
                        ],
                        item["benefit"],
                        item["radius_reduction"],
                        -item["channel"],
                    ),
                )

                selection_kind = "base"

            else:
                self.v23_metrics[
                    "opportunity_extra_considered"
                ] += 1

                evaluated = [
                    self._evaluate_extra_opportunity(
                        candidate
                    )
                    for candidate in candidates
                ]

                accepted = [
                    candidate
                    for candidate in evaluated
                    if candidate["extra_allowed"]
                ]

                if not accepted:
                    self.v23_metrics[
                        "opportunity_extra_rejected"
                    ] += 1

                    self.event(
                        "opportunistic_extra_stopped",
                        point=tuple(point),
                        arrival_kind=arrival_kind,
                        coverage_index=coverage_index,
                        primary_channel=primary_channel,
                        completed=completed,
                    )
                    break

                selected = max(
                    accepted,
                    key=lambda item: (
                        item["expected_net_s"],
                        item[
                            "expected_ready_probability"
                        ],
                        item[
                            "guaranteed_ready_fraction"
                        ],
                        item[
                            "expected_reduction_m"
                        ],
                        -item["channel"],
                    ),
                )

                selection_kind = "extra"

                self.v23_metrics[
                    "opportunity_extra_accepted"
                ] += 1

                self.v23_metrics[
                    "opportunity_expected_saved_s"
                ] += selected["expected_saved_s"]

            self.event(
                "opportunistic_selected_v23",
                point=tuple(point),
                arrival_kind=arrival_kind,
                coverage_index=coverage_index,
                primary_channel=primary_channel,
                selection_kind=selection_kind,
                channel=selected["channel"],
                predicted_radius=selected[
                    "predicted_radius"
                ],
                predicted_clearable=selected[
                    "predicted_clearable"
                ],
                radius_reduction=selected[
                    "radius_reduction"
                ],
                expected_ready_probability=(
                    selected.get(
                        "expected_ready_probability"
                    )
                ),
                expected_receive_fraction=(
                    selected.get(
                        "expected_receive_fraction"
                    )
                ),
                expected_saved_s=selected.get(
                    "expected_saved_s"
                ),
                expected_net_s=selected.get(
                    "expected_net_s"
                ),
            )

            self.measure(
                selected["record"],
                point,
                measurement_kind="opportunistic",
            )

            completed += 1

        # 与基础版语义一致：统计发生过机会补测的到达次数。
        if completed:
            self.counts[
                "opportunistic_arrivals_used"
            ] += 1

        self.event(
            "opportunistic_arrival_completed",
            point=tuple(point),
            arrival_kind=arrival_kind,
            coverage_index=coverage_index,
            primary_channel=primary_channel,
            measurement_count=completed,
        )

        return completed

    # ========================================================================
    # 优先级0：联合定位候选与清除调度
    # ========================================================================

    def _normalise_proposal(
        self,
        record,
        proposal: dict[str, Any],
    ) -> dict[str, Any]:
        if record.clearance_radius is not None:
            current_radius = (
                record.clearance_radius
            )
        elif record.diameter is not None:
            current_radius = (
                0.5 * record.diameter
            )
        else:
            current_radius = TARGET_RADIUS_M

        predicted_radius = float(
            proposal.get(
                "predicted_radius",
                current_radius,
            )
        )

        guaranteed_receive_fraction = float(
            proposal.get(
                "guaranteed_receive_fraction",
                proposal.get(
                    "selected_guaranteed_receive_fraction",
                    0.0,
                ),
            )
        )

        possible_receive_fraction = float(
            proposal.get(
                "possible_receive_fraction",
                proposal.get(
                    "selected_possible_receive_fraction",
                    0.0,
                ),
            )
        )

        return {
            "point": tuple(
                map(float, proposal["point"])
            ),
            "predicted_radius": predicted_radius,
            "predicted_clearable": bool(
                proposal.get(
                    "predicted_clearable",
                    False,
                )
            ),
            "predicted_radius_reduction_m": float(
                proposal.get(
                    "predicted_radius_reduction_m",
                    max(
                        0.0,
                        current_radius
                        - predicted_radius,
                    ),
                )
            ),
            "guaranteed_receive_fraction": (
                guaranteed_receive_fraction
            ),
            "possible_receive_fraction": (
                possible_receive_fraction
            ),
            "cross_angle_deg": float(
                proposal.get(
                    "cross_angle_deg",
                    proposal.get(
                        "selected_cross_angle_deg",
                        0.0,
                    ),
                )
            ),
            "source": proposal.get(
                "origin",
                proposal.get(
                    "method",
                    "planner",
                ),
            ),
        }

    def _candidate_proposals_for_record(
        self,
        record,
        primary_proposal: dict[str, Any],
    ) -> list[dict[str, Any]]:
        primary = self._normalise_proposal(
            record,
            primary_proposal,
        )

        proposals = [primary]

        cache_entry = (
            self._static_plan_cache.get(
                record.channel_id
            )
        )

        if cache_entry is None:
            return proposals

        static_plan = (
            cache_entry.get("static_plan") or {}
        )

        scored_candidates = (
            static_plan.get(
                "scored_candidates"
            )
            or []
        )

        if not scored_candidates:
            return proposals

        normalised = [
            self._normalise_proposal(
                record,
                candidate,
            )
            for candidate in scored_candidates
        ]

        clearable = [
            proposal
            for proposal in normalised
            if proposal["predicted_clearable"]
        ]

        if clearable:
            pool = clearable
        else:
            best_radius = min(
                proposal["predicted_radius"]
                for proposal in normalised
            )

            radius_allowance = max(
                5.0,
                0.08 * max(20.0, best_radius),
            )

            pool = [
                proposal
                for proposal in normalised
                if proposal["predicted_radius"]
                <= best_radius + radius_allowance
            ]

        pool.sort(
            key=lambda proposal: (
                not proposal[
                    "predicted_clearable"
                ],
                proposal["predicted_radius"],
                -proposal[
                    "guaranteed_receive_fraction"
                ],
                distance(
                    self.client.position,
                    proposal["point"],
                ),
            )
        )

        proposals.extend(
            pool[
                :self.cfg
                .joint_candidates_per_channel
            ]
        )

        unique = []
        seen = set()

        for proposal in proposals:
            point_key = (
                round(
                    proposal["point"][0],
                    5,
                ),
                round(
                    proposal["point"][1],
                    5,
                ),
            )

            if point_key in seen:
                continue

            seen.add(point_key)
            unique.append(proposal)

            if (
                len(unique)
                >= self.cfg
                .joint_candidates_per_channel
            ):
                break

        return unique

    def _action_value_m(
        self,
        action: dict[str, Any],
    ) -> float:
        record = self.records[
            action["channel"]
        ]

        wait_bonus = (
            min(10, record.replan_level)
            * self.cfg.joint_wait_bonus_m
        )

        if action["action_type"] == "CLEAR":
            return (
                self.cfg.joint_ready_value_m
                + wait_bonus
            )

        proposal = action["proposal"]

        reduction = max(
            0.0,
            proposal[
                "predicted_radius_reduction_m"
            ],
        )

        receive_quality = clamp(
            0.65
            * proposal[
                "guaranteed_receive_fraction"
            ]
            + 0.35
            * proposal[
                "possible_receive_fraction"
            ],
            0.15,
            1.0,
        )

        value = (
            reduction
            * self.cfg.joint_reduction_weight
            * receive_quality
        )

        if proposal["predicted_clearable"]:
            value += (
                self.cfg.joint_ready_value_m
            )

        return value + wait_bonus

    def _greedy_tail_route_distance(
        self,
        start: Point,
        actions: list[dict[str, Any]],
        current_channel,
    ) -> float:
        position = start
        remaining = list(actions)
        total = 0.0

        while remaining:
            selected = min(
                remaining,
                key=lambda action: (
                    distance(
                        position,
                        action["point"],
                    )
                    + (
                        SPEED_MPS
                        * SWITCH_OPERATION_S
                        if (
                            action["action_type"]
                            == "MEASURE"
                            and current_channel
                            != action["channel"]
                        )
                        else 0.0
                    ),
                    action["channel"],
                ),
            )

            total += distance(
                position,
                selected["point"],
            )

            position = selected["point"]

            if (
                selected["action_type"]
                == "MEASURE"
            ):
                current_channel = (
                    selected["channel"]
                )

            remaining.remove(selected)

        return total

    def _forced_first_route_cost(
        self,
        action: dict[str, Any],
        representatives: list[dict[str, Any]],
    ) -> float:
        current = self.client.position
        first_point = action["point"]

        tail = [
            representative
            for representative in representatives
            if representative["channel"]
            != action["channel"]
        ]

        if action["action_type"] == "MEASURE":
            next_channel = action["channel"]
        else:
            next_channel = (
                self.client.current_channel
            )

        return (
            distance(current, first_point)
            + self._greedy_tail_route_distance(
                first_point,
                tail,
                next_channel,
            )
        )

    def _build_joint_actions(
        self,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        joint_actions = []
        representatives = []

        for channel, record in (
            self.records.items()
        ):
            if record.status == "READY":
                if record.clearance_center is None:
                    continue

                action = {
                    "channel": channel,
                    "action_type": "CLEAR",
                    "point": tuple(
                        record.clearance_center
                    ),
                    "proposal": None,
                }

                joint_actions.append(action)
                representatives.append(action)
                continue

            if record.status != "FOUND":
                continue

            replan_budget = (
                self.dynamic_replan_budget(record)
            )

            if (
                record.replan_level
                > replan_budget["limit"]
            ):
                continue

            try:
                primary = self.proposal(record)

            except base.BudgetStop:
                raise

            except base.IncompleteRun as error:
                self.replan(
                    record,
                    str(error),
                )
                continue

            proposals = (
                self._candidate_proposals_for_record(
                    record,
                    primary,
                )
            )

            if not proposals:
                self.replan(
                    record,
                    "联合调度没有合法定位候选",
                )
                continue

            channel_actions = [
                {
                    "channel": channel,
                    "action_type": "MEASURE",
                    "point": proposal["point"],
                    "proposal": proposal,
                }
                for proposal in proposals
            ]

            joint_actions.extend(
                channel_actions
            )

            representatives.append(
                channel_actions[0]
            )

        return joint_actions, representatives

    def _select_joint_action(
        self,
        joint_actions: list[dict[str, Any]],
        representatives: list[dict[str, Any]],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
    ]:
        evaluated = []

        for action in joint_actions:
            route_cost = (
                self._forced_first_route_cost(
                    action,
                    representatives,
                )
            )

            direct_distance = distance(
                self.client.position,
                action["point"],
            )

            value = self._action_value_m(
                action
            )

            evaluated.append(
                {
                    "action": action,
                    "route_cost_m": route_cost,
                    "direct_distance_m": (
                        direct_distance
                    ),
                    "value_m": value,
                }
            )

        baseline_route_cost = min(
            item["route_cost_m"]
            for item in evaluated
        )

        allowed = []

        for item in evaluated:
            action = item["action"]

            route_regret = max(
                0.0,
                item["route_cost_m"]
                - baseline_route_cost,
            )

            direct_allowed = (
                action["action_type"] == "CLEAR"
                or action["proposal"][
                    "predicted_clearable"
                ]
                or item["direct_distance_m"]
                <= (
                    self.cfg
                    .joint_measure_direct_cap_m
                )
            )

            route_allowed = (
                route_regret
                <= (
                    self.cfg
                    .joint_free_route_regret_m
                )
                or (
                    route_regret
                    <= (
                        self.cfg
                        .joint_max_route_regret_m
                    )
                    and item["value_m"]
                    >= (
                        self.cfg
                        .joint_value_cover_ratio
                        * route_regret
                    )
                )
            )

            item["route_regret_m"] = (
                route_regret
            )

            item["allowed"] = (
                direct_allowed and route_allowed
            )

            item["score"] = (
                item["value_m"]
                - route_regret
                - 0.12
                * item["direct_distance_m"]
            )

            if item["allowed"]:
                allowed.append(item)
            else:
                self.v23_metrics[
                    "joint_route_gate_rejected"
                ] += 1

        if allowed:
            selected_item = max(
                allowed,
                key=lambda item: (
                    item["score"],
                    item["action"][
                        "action_type"
                    ] == "CLEAR",
                    -item["route_cost_m"],
                ),
            )

            reason = "joint_value_route"

        else:
            fallback_route = (
                base.dynamic_open_tsp(
                    representatives,
                    self.client.position,
                    self.client.current_channel,
                )
            )

            fallback_action = fallback_route[0]

            selected_item = min(
                evaluated,
                key=lambda item: (
                    item["action"]["channel"]
                    != fallback_action["channel"],
                    item["action"]["action_type"]
                    != fallback_action[
                        "action_type"
                    ],
                    distance(
                        item["action"]["point"],
                        fallback_action["point"],
                    ),
                ),
            )

            reason = "base_tsp_fallback"

        selected = selected_item["action"]

        diagnostics = {
            "reason": reason,
            "candidate_count": len(evaluated),
            "baseline_route_cost_m": (
                baseline_route_cost
            ),
            "selected_route_cost_m": (
                selected_item["route_cost_m"]
            ),
            "selected_route_regret_m": (
                selected_item["route_regret_m"]
            ),
            "selected_direct_distance_m": (
                selected_item["direct_distance_m"]
            ),
            "selected_value_m": (
                selected_item["value_m"]
            ),
            "selected_score": (
                selected_item["score"]
            ),
        }

        return selected, diagnostics

    def localize(self):
        # 必须使用 localization。
        # 基础版 phase_virtual_s 没有 localize 这个键。
        self.phase = "localization"

        while any(
            record.status in ("FOUND", "READY")
            for record in self.records.values()
        ):
            self.check_budget()

            (
                joint_actions,
                representatives,
            ) = self._build_joint_actions()

            if not joint_actions:
                pending = [
                    channel
                    for channel, record
                    in self.records.items()
                    if record.status
                    in ("FOUND", "READY")
                ]

                can_continue = any(
                    self.records[channel].status
                    == "FOUND"
                    and self.records[
                        channel
                    ].replan_level
                    <= self.dynamic_replan_budget(
                        self.records[channel]
                    )["limit"]
                    for channel in pending
                )

                if can_continue:
                    continue

                raise base.IncompleteRun(
                    f"重规划预算耗尽，频道"
                    f"{pending}仍未完成；"
                    "没有将它们标记为成功"
                )

            self.v23_metrics[
                "joint_scheduler_rounds"
            ] += 1

            self.v23_metrics[
                "joint_candidate_actions"
            ] += len(joint_actions)

            self.counts["tsp_replans"] += 1

            (
                selected,
                diagnostics,
            ) = self._select_joint_action(
                joint_actions,
                representatives,
            )

            channel = selected["channel"]
            record = self.records[channel]
            proposal = selected["proposal"]

            self.event(
                "joint_target_selected",
                channel=channel,
                action_type=selected[
                    "action_type"
                ],
                point=selected["point"],
                predicted_radius=(
                    proposal["predicted_radius"]
                    if proposal is not None
                    else record.clearance_radius
                ),
                predicted_clearable=(
                    proposal[
                        "predicted_clearable"
                    ]
                    if proposal is not None
                    else True
                ),
                diagnostics=diagnostics,
            )

            if selected["action_type"] == "CLEAR":
                self.v23_metrics[
                    "joint_clear_selected"
                ] += 1

                self.clear(
                    record,
                    insertion=False,
                )
                continue

            self.v23_metrics[
                "joint_measure_selected"
            ] += 1

            if proposal["predicted_clearable"]:
                self.v23_metrics[
                    "joint_predicted_ready_selected"
                ] += 1

            record.q2_attempted = True

            self.measure(
                record,
                selected["point"],
                measurement_kind="active",
            )

            self.perform_opportunistic_measurements(
                selected["point"],
                arrival_kind="active_measurement",
                primary_channel=record.channel_id,
            )

            if (
                record.status == "FOUND"
                and record.no_progress_count
                >= self.cfg.no_progress_trigger
            ):
                self.replan(
                    record,
                    "联合定位连续无足够进展",
                )

    # ========================================================================
    # 摘要
    # ========================================================================

    def summary(
        self,
        outcome,
        reason,
        exit_confirmed,
    ):
        result = super().summary(
            outcome,
            reason,
            exit_confirmed,
        )

        result["strategy_version"] = "V2-3"

        result["coverage_visit_order"] = list(
            self.dynamic_coverage_visit_order
        )

        result["target_outer_bound"] = {
            "type": (
                "circumscribed_regular_polygon"
            ),
            "target_radius_m": TARGET_RADIUS_M,
            "sides": TARGET_OUTER_POLYGON_SIDES,
            "polygon_vertex_radius_m": (
                TARGET_RADIUS_M
                / math.cos(
                    math.pi
                    / TARGET_OUTER_POLYGON_SIDES
                )
            ),
        }

        result["v23_metrics"] = dict(
            self.v23_metrics
        )

        result["v23_config"] = asdict(
            self.cfg
        )

        return result


# ============================================================================
# 命令行
# ============================================================================

def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
    )

    # client.py 明确要求 robot_id 是字符串，因此不使用 type=int。
    parser.add_argument(
        "--robot-id",
        required=True,
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:2026",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=1130.0,
    )

    parser.add_argument(
        "--no-insert",
        "--disable-insert",
        dest="no_insert",
        action="store_true",
    )

    parser.add_argument(
        "--insert-threshold",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--clear-margin",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--planning-timeout",
        type=float,
        default=45.0,
    )

    parser.add_argument(
        "--max-clear-failures",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-dynamic-replans",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--replan-soft-budget",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--no-signal-repeat",
        type=float,
        default=225.0,
    )

    parser.add_argument(
        "--opportunity-base-limit",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--opportunity-extra-limit",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--joint-candidates",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--search-state-count",
        type=int,
        default=384,
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
        "--output-root",
        type=Path,
        default=Path("runs"),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
    )

    return parser


# ============================================================================
# 主程序
# ============================================================================

def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    config = V23Config(
        rho=args.rho,
        enable_insert=not args.no_insert,
        insert_threshold_s=(
            args.insert_threshold
        ),
        clear_margin_m=args.clear_margin,
        planning_timeout_s=(
            args.planning_timeout
        ),
        max_clear_failures_per_channel=(
            args.max_clear_failures
        ),
        max_dynamic_replans=(
            args.max_dynamic_replans
        ),
        dynamic_replan_soft_budget_s=(
            args.replan_soft_budget
        ),
        no_signal_repeat_m=(
            args.no_signal_repeat
        ),
        max_opportunistic_measurements=(
            args.opportunity_base_limit
        ),
        opportunity_extra_limit=(
            args.opportunity_extra_limit
        ),
        joint_candidates_per_channel=(
            args.joint_candidates
        ),
        search_unknown_state_count=(
            args.search_state_count
        ),
    )

    geometry_config = (
        base.load_config(
            args.geometry_config
        )
        if args.geometry_config
        else base.GeometryConfig()
    )

    planner_config = (
        base.PlannerConfig(
            **json.loads(
                args.planner_config.read_text(
                    encoding="utf-8-sig"
                )
            )
        )
        if args.planner_config
        else base.PlannerConfig()
    )

    timestamp = datetime.now().strftime(
        "q3_v23_%Y%m%d_%H%M%S_%f"
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.output_root / timestamp
    )

    # 与 client.py 的真实接口一致。
    client_instance = base.SimulatorClient(
        args.robot_id,
        args.base_url,
    )

    # Q3Runner 会在这里创建 output_dir。
    # 因此主程序不能提前 mkdir。
    runner = Q3V23Runner(
        client_instance,
        config=config,
        planner_config=planner_config,
        geometry_config=geometry_config,
        output_dir=output_dir,
    )

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(message)s"
        ),
        handlers=[
            logging.FileHandler(
                output_dir / "client.log",
                encoding="utf-8",
            ),
            logging.StreamHandler(
                sys.stdout
            ),
        ],
        force=True,
    )

    print(
        "问题三 V2-3 策略启动，"
        f"输出目录：{output_dir}",
        flush=True,
    )

    # run() 已经负责：
    # enter、search、localize、exit、异常处理、
    # summary.json写入和JSONL日志流关闭。
    summary = runner.run()

    displayed = {
        "outcome": summary.get("outcome"),
        "reason": summary.get("reason"),
        "discovered_count": summary.get(
            "discovered_count"
        ),
        "cleared_count": summary.get(
            "cleared_count"
        ),
        "virtual_time_s": summary.get(
            "virtual_time_s"
        ),
        "average_clear_time_s": summary.get(
            "average_clear_time_s"
        ),
        "program_elapsed_s": summary.get(
            "program_elapsed_s"
        ),
        "total_distance_m": summary.get(
            "total_distance_m"
        ),
        "coverage_visit_order": summary.get(
            "coverage_visit_order"
        ),
        "opportunity_extra_accepted": (
            summary.get(
                "v23_metrics",
                {},
            ).get(
                "opportunity_extra_accepted"
            )
        ),
        "joint_measure_selected": (
            summary.get(
                "v23_metrics",
                {},
            ).get(
                "joint_measure_selected"
            )
        ),
        "joint_clear_selected": (
            summary.get(
                "v23_metrics",
                {},
            ).get(
                "joint_clear_selected"
            )
        ),
        "output_dir": str(output_dir),
    }

    print(
        json.dumps(
            displayed,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    return (
        0
        if summary.get("outcome") == "success"
        else 1
    )


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())