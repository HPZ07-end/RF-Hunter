"""问题四 V6：UNKNOWN优先发现、READY全局路线及插入代价清除。

在 V5 的基础上增加：

1. UNKNOWN 频道预期发现收益参与认证节点排序；
2. READY 清除中心采用多起点最近邻 + 2-opt 全局路线；
3. 顺路清除由固定半径改为路线插入代价判定。

正确性不变量：

- 不删除、不减少 V4 的双环认证节点；
- 未达到 16 个干扰源时，UNKNOWN 频道仍须完成全部认证节点；
- 只清除具有可靠 READY 证书的频道；
- UNKNOWN 的概率模型只用于改变节点访问顺序，不用于判定不存在。

依赖：
    question_4_V5.py
    question_4_V4.py
    question_4_V3.py
    client.py
    geometry_V7.py
    planner.py

运行：
    python question_4_V6.py --robot-id 你的队号

离线检查：
    python question_4_V6.py --self-test
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

import question_4_V5 as v5


v3 = v5.v3
Point = tuple[float, float]


@dataclass(frozen=True)
class V6Tuning(v5.V5Tuning):
    # ================================================================
    # 优先级1：UNKNOWN预期发现收益
    # ================================================================

    # V5 默认为8；增至12，但仍受路线后悔值约束。
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


class Q4V6Runner(v5.Q4V5Runner):
    def __init__(
        self,
        client,
        *,
        config: v3.Q4Config,
        tuning: V6Tuning | None = None,
        planner_config=None,
        geometry_config=None,
        output_dir: Path | None = None,
    ):
        self.v6 = tuning or V6Tuning()

        super().__init__(
            client,
            config=config,
            tuning=self.v6,
            planner_config=planner_config,
            geometry_config=geometry_config,
            output_dir=output_dir,
        )

        # UNKNOWN概率评价缓存。
        self._v6_unknown_probability_cache: dict[
            tuple[int, int],
            float,
        ] = {}
        self._v6_unknown_omni_masks: list[int] | None = None
        self._v6_unknown_directional_masks: list[int] | None = None

        # 顺路清除可以指定清除后必须访问的认证节点。
        self._v6_forced_search_index: int | None = None

        self.counts.update({
            "v6_unknown_aware_selection": 0,
            "v6_unknown_value_selected_m": 0.0,
            "v6_expected_unknown_hits_selected": 0.0,
            "v6_early_stop_probability_sum": 0.0,
            "v6_forced_search_anchor": 0,
            "v6_ready_route_plan": 0,
            "v6_ready_route_first_override": 0,
            "v6_ready_route_improvement_m": 0.0,
            "v6_inline_clear_search": 0,
            "v6_inline_clear_localization": 0,
            "v6_inline_rejected_by_detour": 0,
        })

    # ================================================================
    # 优先级1：UNKNOWN源的后验接收概率
    # ================================================================

    def _build_unknown_signal_masks(self) -> None:
        """预计算每个未知源粒子能在哪些认证节点被接收。"""
        if self._v6_unknown_omni_masks is not None:
            return

        position_count = self.v6.unknown_position_samples
        orientation_count = self.v6.unknown_orientation_samples

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

        self._v6_unknown_omni_masks = omni_masks
        self._v6_unknown_directional_masks = (
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
        record: v3.ChannelRecord,
        candidate_index: int,
    ) -> float:
        """计算在历史UNKNOWN无信号条件下，候选节点的接收概率。"""
        self._build_unknown_signal_masks()

        tested_mask = self._tested_node_mask(
            record.global_nodes_tested
        )
        cache_key = (tested_mask, candidate_index)

        cached = self._v6_unknown_probability_cache.get(
            cache_key
        )
        if cached is not None:
            return cached

        candidate_bit = 1 << candidate_index

        omni_survival, omni_joint_hit = (
            self._posterior_class_statistics(
                self._v6_unknown_omni_masks or [],
                tested_mask,
                candidate_bit,
            )
        )
        directional_survival, directional_joint_hit = (
            self._posterior_class_statistics(
                self._v6_unknown_directional_masks or [],
                tested_mask,
                candidate_bit,
            )
        )

        directional_prior = (
            self.v6.unknown_directional_prior
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
        self._v6_unknown_probability_cache[
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
            * self.v6.unknown_discovery_value_scale
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
            * self.v6.unknown_early_stop_weight
        )

        total_value_m = min(
            self.v6.unknown_value_cap_m,
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
            - self.v6.dynamic_route_regret_penalty
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
            :self.v6.dynamic_route_candidate_limit
        ]

        baseline_index = candidate_indices[0]
        baseline_tail_m = (
            self._nearest_neighbor_tail_distance(
                current,
                list(remaining),
            )
        )

        allowed_regret_m = max(
            self.v6.dynamic_route_regret_floor_m,
            self.v6.dynamic_route_regret_ratio
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
            self._v6_forced_search_index is not None
            and self._v6_forced_search_index in remaining
        ):
            return self._v6_forced_search_index

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

        forced_index = self._v6_forced_search_index
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

            self._v6_forced_search_index = None
            self.counts["v6_forced_search_anchor"] += 1
        else:
            self._v6_forced_search_index = None
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
                "v6_unknown_aware_selection"
            ] += 1

        self.counts[
            "dynamic_search_node_selection"
        ] += 1
        self.counts[
            "dynamic_search_total_regret_m"
        ] += selected["route_regret_m"]
        self.counts[
            "v6_unknown_value_selected_m"
        ] += selected["unknown_discovery_value"]
        self.counts[
            "v6_expected_unknown_hits_selected"
        ] += selected["unknown_expected_hits"]
        self.counts[
            "v6_early_stop_probability_sum"
        ] += selected[
            "unknown_early_stop_probability"
        ]

        self.event(
            "v6_search_node_value",
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
        record: v3.ChannelRecord,
    ) -> Point:
        if record.clearance_center is None:
            raise v3.IncompleteRun(
                f"频道{record.channel_id}缺少清除中心"
            )
        return tuple(record.clearance_center)

    def _ready_path_distance(
        self,
        start: Point,
        route: Sequence[v3.ChannelRecord],
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
        records: Sequence[v3.ChannelRecord],
        start: Point,
        *,
        forced_first: v3.ChannelRecord | None = None,
    ) -> list[v3.ChannelRecord]:
        remaining = {
            record.channel_id: record
            for record in records
        }
        route: list[v3.ChannelRecord] = []
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
        route: Sequence[v3.ChannelRecord],
        start: Point,
    ) -> list[v3.ChannelRecord]:
        result = list(route)
        if len(result) < 3:
            return result

        for _ in range(
            self.v6.ready_route_two_opt_passes
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
        records: Sequence[v3.ChannelRecord],
    ) -> tuple[
        list[v3.ChannelRecord],
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
                    + self.v6.ready_route_first_leg_slack_m
                )
            )
        ][:self.v6.ready_route_start_limit]

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

        self.counts["v6_ready_route_plan"] += 1
        self.counts[
            "v6_ready_route_improvement_m"
        ] += improvement_m

        if (
            optimized_first["record"].channel_id
            != selected["record"].channel_id
        ):
            self.counts[
                "v6_ready_route_first_override"
            ] += 1

        horizon = max(
            1,
            self.v6.scheduler_horizon,
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
            "v6_ready_route_selected",
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
            :self.v6.dynamic_route_candidate_limit
        ]

    def _clear_ready_nearby(
        self,
        maximum_distance_m: float,
        phase_name: str,
    ) -> int:
        """用插入代价替代V5的固定欧氏距离阈值。"""
        cleared = 0

        while (
            cleared
            < self.v6.inline_clear_batch_limit
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
            selected: v3.ChannelRecord | None = None
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
                        > self.v6.inline_max_direct_distance_m
                    ):
                        continue

                    detour_m = self._insertion_detour(
                        current,
                        center,
                        anchor_point,
                    )
                    if (
                        detour_m
                        > self.v6.search_inline_detour_cap_m
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
                        "v6_inline_rejected_by_detour"
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
                            > self.v6.inline_max_direct_distance_m
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
                            > self.v6.localization_inline_detour_cap_m
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
                            "v6_inline_rejected_by_detour"
                        ] += 1
                        return cleared

            if selected is None:
                return cleared

            if phase_name == "search":
                self._v6_forced_search_index = (
                    selected_anchor_index
                )

            if not self.clear(selected):
                return cleared

            cleared += 1

            legacy_counter = (
                "v5_inline_clear_search"
                if phase_name == "search"
                else "v5_inline_clear_localization"
            )
            new_counter = (
                "v6_inline_clear_search"
                if phase_name == "search"
                else "v6_inline_clear_localization"
            )
            self.counts[legacy_counter] += 1
            self.counts[new_counter] += 1

            self.event(
                "v6_inline_clear_completed",
                channel=selected.channel_id,
                source_phase=phase_name,
                decision_basis="route_insertion_cost",
                insertion_detour_m=(
                    selected_detour_m
                ),
                detour_cap_m=(
                    self.v6.search_inline_detour_cap_m
                    if phase_name == "search"
                    else self.v6.localization_inline_detour_cap_m
                ),
                direct_distance_limit_m=(
                    self.v6.inline_max_direct_distance_m
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

        result["v6_tuning"] = asdict(self.v6)
        result["v6_optimization"] = {
            "unknown_probability_cache_entries": len(
                self._v6_unknown_probability_cache
            ),
            "unknown_aware_node_selection_count": (
                self.counts[
                    "v6_unknown_aware_selection"
                ]
            ),
            "unknown_value_selected_m": (
                self.counts[
                    "v6_unknown_value_selected_m"
                ]
            ),
            "expected_unknown_hits_selected": (
                self.counts[
                    "v6_expected_unknown_hits_selected"
                ]
            ),
            "ready_route_plan_count": (
                self.counts[
                    "v6_ready_route_plan"
                ]
            ),
            "ready_route_estimated_improvement_m": (
                self.counts[
                    "v6_ready_route_improvement_m"
                ]
            ),
            "inline_clear_search_count": (
                self.counts[
                    "v6_inline_clear_search"
                ]
            ),
            "inline_clear_localization_count": (
                self.counts[
                    "v6_inline_clear_localization"
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
            + "；V6将UNKNOWN后验发现收益加入认证节点排序，"
            "但不减少双环认证节点；READY清除采用多起点最近邻"
            "和2-opt开放路线；顺路清除使用路线插入代价，"
            "不再以固定欧氏半径作为主要判据"
        )
        return result


def run_self_test() -> dict:
    result = v5.run_self_test()
    tuning = V6Tuning()

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
    straight = Q4V6Runner._insertion_detour(
        (0.0, 0.0),
        (5.0, 0.0),
        (10.0, 0.0),
    )
    off_route = Q4V6Runner._insertion_detour(
        (0.0, 0.0),
        (5.0, 5.0),
        (10.0, 0.0),
    )
    assert abs(straight) <= 1e-9
    assert off_route > 0.0

    probability = (
        Q4V6Runner._poisson_binomial_tail(
            [0.5, 0.5],
            1,
        )
    )
    assert abs(probability - 0.75) <= 1e-9

    return {
        **result,
        "v6_status": "ok",
        "v6_unknown_discovery_ordering": True,
        "v6_global_ready_route": True,
        "v6_insertion_cost_inline_clear": True,
        "v6_certificate_nodes_removed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    tuning = V6Tuning()

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
            "q4_v6_%Y%m%d_%H%M%S_%f"
        )
    )

    client = v3.SimulatorClient(
        args.robot_id,
        args.base_url,
    )

    runner = Q4V6Runner(
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
        f"问题四V6策略启动，输出目录：{output_dir}",
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
        "cleared_count": summary["cleared_count"],
        "virtual_time_s": summary["virtual_time_s"],
        "average_clear_time_s": (
            average_clear_time_s
        ),
        "total_distance_m": (
            summary["total_distance_m"]
        ),
        "dynamic_search": (
            summary["dynamic_search"]
        ),
        "dual_ring": summary["global_grid"],
        "v6_optimization": (
            summary["v6_optimization"]
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