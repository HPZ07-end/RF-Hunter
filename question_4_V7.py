"""问题四 V7：安全重捕获、局部恢复事务和概率校准。

在V6基础上实现：

1. 单示向安全重捕获：
   只有一个方向观测时，先沿成功示向中心线做短距离补测，
   避免直接选择大角度、大距离候选而退出定向波束。

2. 恢复点失效与局部恢复事务：
   恢复点在机器人移动后重新验证；
   信息恢复尽量成组执行，避免生成恢复点后横跨地图再返回。

3. 联合概率校准和快速降级：
   联合位置、半径、朝向粒子按照原始先验及观测存活似然加权；
   使用位置组低分位接收概率；
   连续高置信预测失败后快速进入确定性粗网格恢复。

依赖：
    question_4_V6.py
    question_4_V5.py
    question_4_V4.py
    question_4_V3.py
    client.py
    geometry_V7.py
    planner.py

运行：
    python question_4_V7.py --robot-id 你的队号

离线检查：
    python question_4_V7.py --self-test
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

import question_4_V6 as v6


v5 = v6.v5
v3 = v6.v3
Point = tuple[float, float]


@dataclass(frozen=True)
class V7Tuning(v6.V6Tuning):
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

    # V4原来最多尝试5次信息恢复；V7缩短为3次。
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

    # 将V4的36个朝向粒子提高到72个，减小半平面边界误差。
    orientation_particle_count: int = 72

    # 位置组接收概率的保守分位数。
    posterior_receive_quantile: float = 0.20

    # 最终概率 = 均值和低分位概率的加权结果。
    posterior_robust_blend: float = 0.45

    # 经验可靠度的Beta型先验强度；初始可靠度为1。
    model_reliability_prior_strength: float = 3.0

    # 多次失败后仍保留的最低模型可靠度。
    model_reliability_floor: float = 0.35

    # V5原来在失败后提高门槛；V7不再提高，
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


class Q4V7Runner(v6.Q4V6Runner):
    def __init__(
        self,
        client,
        *,
        config: v3.Q4Config,
        tuning: V7Tuning | None = None,
        planner_config=None,
        geometry_config=None,
        output_dir: Path | None = None,
    ):
        self.v7 = tuning or V7Tuning()

        super().__init__(
            client,
            config=config,
            tuning=self.v7,
            planner_config=planner_config,
            geometry_config=geometry_config,
            output_dir=output_dir,
        )

        # 当前真正准备执行的主动测量预测。
        self._v7_active_execution_prediction: dict[
            int,
            dict,
        ] = {}

        # information_reacquire恢复点的生成位置、预测概率和移动上限。
        self._v7_recovery_plan_meta: dict[
            int,
            dict,
        ] = {}

        # 连续高置信预测失败次数。
        self._v7_surprise_misses: dict[int, int] = {}

        self.counts.update({
            "v7_single_bearing_plan": 0,
            "v7_single_bearing_measure": 0,
            "v7_single_bearing_positive": 0,
            "v7_single_bearing_no_signal": 0,
            "v7_likelihood_weighted_hypothesis_build": 0,
            "v7_surprise_no_signal": 0,
            "v7_surprise_fast_fallback": 0,
            "v7_recovery_plan_recorded": 0,
            "v7_stale_recovery_replanned": 0,
            "v7_recovery_burst_round": 0,
            "v7_recovery_burst_extra_measure": 0,
            "v7_recovery_locality_boost": 0,
        })

    # ================================================================
    # 优先级3：联合粒子按存活似然加权
    # ================================================================

    def _joint_hypotheses(
        self,
        record: v3.ChannelRecord,
        sources: Sequence[Point],
    ) -> list[dict]:
        """构建位置、半径和朝向联合后验。

        与V4的关键区别：

        V4会先对每个位置样本内部重新归一化，导致只有极少
        朝向存活的位置仍可能获得与大量朝向存活位置相同的总权重。

        V7先给所有原始粒子分配先验权重，再删除与观测冲突的
        粒子，最后只做一次全局归一化。因此位置样本的后验质量
        与其真实存活似然一致。

        no_signal仍然只影响规划粒子，不参与ABSENT证明和MEC证书。
        """
        if not sources:
            return []

        positive = list(record.bearing_observations)
        no_signals = list(record.no_signal_observations)

        orientation_count = (
            self.v7.orientation_particle_count
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
            "v7_likelihood_weighted_hypothesis_build"
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
        record: v3.ChannelRecord,
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
            self.v7.model_reliability_prior_strength
        )
        reliability = (
            strength + success_count
        ) / (
            strength + attempt_count
        )

        return min(
            1.0,
            max(
                self.v7.model_reliability_floor,
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
                self.v7.posterior_receive_quantile,
            )
        )

        blend = self.v7.posterior_robust_blend
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
        record: v3.ChannelRecord,
        base_floor: float,
    ) -> float:
        """失败后降低模型信任，而不是提高原始概率门槛。"""
        reliability = self._model_reliability(record)

        # 初始可靠度为1时保持原门槛；
        # 可靠度下降后最多小幅降低约25%。
        scale = 0.75 + 0.25 * reliability
        return max(
            self.v7.calibrated_receive_floor_min,
            base_floor * scale,
        )

    # ================================================================
    # 优先级1：单示向安全重捕获
    # ================================================================

    @staticmethod
    def _post_positive_no_signal_count(
        record: v3.ChannelRecord,
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
        record: v3.ChannelRecord,
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
            >= self.v7.single_bearing_probe_limit
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
                    self.v7.single_bearing_first_step_m,
                    0.0,
                ),
                (
                    0.80
                    * self.v7.single_bearing_first_step_m,
                    5.0,
                ),
                (
                    0.80
                    * self.v7.single_bearing_first_step_m,
                    -5.0,
                ),
            )
        else:
            patterns = (
                (
                    self.v7.single_bearing_second_step_m,
                    0.0,
                ),
                (
                    1.50
                    * self.v7.single_bearing_second_step_m,
                    8.0,
                ),
                (
                    1.50
                    * self.v7.single_bearing_second_step_m,
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
                "origin": "v7_single_bearing_safe",
                "safe_step_m": step_m,
                "safe_offset_deg": offset_deg,
                "safe_failure_stage": failure_count,
            })

        return result

    def _build_single_bearing_proposal(
        self,
        record: v3.ChannelRecord,
    ) -> dict | None:
        candidates = (
            self._single_bearing_safe_candidates(
                record
            )
        )
        if not candidates:
            return None

        try:
            sources = v3.sample_source_positions(
                record,
                self.cfg,
            )
        except v3.IncompleteRun:
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

            conditional = v3.score_candidate(
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
                    + self.v7.single_bearing_completion_value_m
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
                    "v7_single_bearing_safe_reacquire"
                ),
                "selection_stage": (
                    "v7_single_bearing_safe_reacquire"
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
                    self.v7.single_bearing_reference_probability,
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
        record: v3.ChannelRecord,
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
                    "v7_single_bearing_plan"
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
        record: v3.ChannelRecord,
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
            self.v7.single_bearing_completion_value_m
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
        task["v7_single_bearing_priority"] = True
        return task

    def _record_active_plan(
        self,
        record: v3.ChannelRecord,
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

        self._v7_active_execution_prediction[
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
        record: v3.ChannelRecord,
    ) -> Point:
        receive_floor = self._adaptive_receive_floor(
            record,
            self.v7.recovery_receive_floor,
        )
        move_cap_m = self._adaptive_move_cap(
            record,
            self.v7.recovery_move_cap_m,
            self.v7.adaptive_recovery_min_move_cap_m,
        )

        proposal = self._select_information_measurement(
            record,
            allow_new_q2_seed=False,
            receive_floor=receive_floor,
            move_cap_m=move_cap_m,
            strict_limits=True,
            purpose=(
                "v7_calibrated_information_recovery"
            ),
        )

        planned_from = tuple(
            self.client.position
        )
        point = tuple(proposal["point"])

        self._v7_recovery_plan_meta[
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
            "v5_adaptive_recovery_plan"
        ] += 1
        self.counts[
            "v7_recovery_plan_recorded"
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
        record: v3.ChannelRecord,
        reason: str,
        **diagnostics,
    ) -> None:
        self._recovery_queues.pop(
            record.channel_id,
            None,
        )
        self._v7_recovery_plan_meta.pop(
            record.channel_id,
            None,
        )
        record.recovery_active = False
        record.recovery_mode = None

        self.event(
            "v7_information_recovery_discarded",
            channel=record.channel_id,
            reason=reason,
            **diagnostics,
        )

    def _ensure_recovery_plan_fresh(
        self,
        record: v3.ChannelRecord,
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

        meta = self._v7_recovery_plan_meta.get(
            record.channel_id
        )
        if meta is None:
            self._discard_information_recovery(
                record,
                "missing_plan_metadata",
            )
            self.counts[
                "v7_stale_recovery_replanned"
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
                self.v7.recovery_move_cap_m,
                self.v7.adaptive_recovery_min_move_cap_m,
            )
        )

        stale_by_origin = (
            moved_since_planning_m
            > self.v7.active_cache_replan_distance_m
        )
        stale_by_move_cap = (
            current_travel_m
            > (
                current_move_cap_m
                * self.v7.recovery_stale_cap_ratio
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
            "v7_stale_recovery_replanned"
        ] += 1
        self.start_recovery(record)

    def start_recovery(
        self,
        record: v3.ChannelRecord,
    ) -> None:
        existing = self._recovery_queues.get(
            record.channel_id
        )

        # 高置信失败达到阈值时，即使已经存在信息恢复点，
        # 也不再继续依赖失准的概率模型。
        surprise_count = (
            self._v7_surprise_misses.get(
                record.channel_id,
                0,
            )
        )

        if (
            surprise_count
            >= self.v7.surprise_no_signal_limit
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
                    "v7_high_confidence_model_failure:"
                    f"{surprise_count}"
                ),
            )
            self.counts[
                "v7_surprise_fast_fallback"
            ] += 1

            self.event(
                "v7_fast_fallback_started",
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
        record: v3.ChannelRecord,
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
            self.v7.recovery_locality_value_m,
        )

        if (
            expected_value_m
            > float(task["expected_value_m"])
            + 1e-9
        ):
            self.counts[
                "v7_recovery_locality_boost"
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
        task["v7_recovery_locality_boost"] = True
        return task

    def recovery_step(
        self,
        record: v3.ChannelRecord,
    ) -> str:
        """选中一次信息恢复后，尽量连续完成局部恢复探针。"""
        last_result = "no_signal"
        executed_count = 0

        for burst_index in range(
            self.v7.recovery_burst_limit
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
                raise v3.IncompleteRun(
                    f"频道{record.channel_id}没有恢复候选"
                )

            mode_before = record.recovery_mode

            last_result = super().recovery_step(
                record
            )
            executed_count += 1

            if burst_index > 0:
                self.counts[
                    "v7_recovery_burst_extra_measure"
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
                >= self.v7.recovery_burst_limit
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
                > self.v7.recovery_burst_leg_cap_m
            ):
                break

        if executed_count > 1:
            self.counts[
                "v7_recovery_burst_round"
            ] += 1
            self.event(
                "v7_recovery_burst_completed",
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
        record: v3.ChannelRecord,
        point: Point,
        *,
        measurement_kind: str,
        global_index: int | None = None,
    ) -> str:
        prediction: dict | None = None

        if measurement_kind == "active":
            prediction = (
                self._v7_active_execution_prediction.get(
                    record.channel_id
                )
            )
        elif (
            measurement_kind == "recovery"
            and record.recovery_mode
            == "information_reacquire"
        ):
            meta = self._v7_recovery_plan_meta.get(
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
            self._v7_active_execution_prediction.pop(
                record.channel_id,
                None,
            )
        elif measurement_kind == "recovery":
            self._v7_recovery_plan_meta.pop(
                record.channel_id,
                None,
            )

        if prediction is not None and prediction.get(
            "single_bearing_safe",
            False,
        ):
            self.counts[
                "v7_single_bearing_measure"
            ] += 1
            if result in {"direction", "near"}:
                self.counts[
                    "v7_single_bearing_positive"
                ] += 1
            elif result == "no_signal":
                self.counts[
                    "v7_single_bearing_no_signal"
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
            self._v7_surprise_misses[
                record.channel_id
            ] = 0
            self._v4_information_attempts[
                record.channel_id
            ] = 0
            return result

        if (
            result == "no_signal"
            and predicted_probability
            >= self.v7.surprise_no_signal_probability
        ):
            surprise_count = (
                self._v7_surprise_misses.get(
                    record.channel_id,
                    0,
                )
                + 1
            )
            self._v7_surprise_misses[
                record.channel_id
            ] = surprise_count
            self.counts[
                "v7_surprise_no_signal"
            ] += 1

            self.event(
                "v7_surprise_no_signal",
                channel=record.channel_id,
                measurement_kind=measurement_kind,
                point=point,
                predicted_probability=(
                    predicted_probability
                ),
                surprise_count=surprise_count,
                fast_fallback_threshold=(
                    self.v7.surprise_no_signal_limit
                ),
            )

            if (
                surprise_count
                >= self.v7.surprise_no_signal_limit
            ):
                # 下一轮任务构造直接进入恢复，不再等待普通
                # no_progress_trigger慢慢累计。
                record.no_progress_count = max(
                    record.no_progress_count,
                    self.cfg.no_progress_trigger,
                )

                if measurement_kind == "recovery":
                    self._v4_information_attempts[
                        record.channel_id
                    ] = max(
                        self._v4_information_attempts.get(
                            record.channel_id,
                            0,
                        ),
                        self.v7.recovery_information_probe_limit,
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

        result["v7_tuning"] = asdict(self.v7)
        result["v7_optimization"] = {
            "single_bearing_plan_count": (
                self.counts[
                    "v7_single_bearing_plan"
                ]
            ),
            "single_bearing_measure_count": (
                self.counts[
                    "v7_single_bearing_measure"
                ]
            ),
            "single_bearing_positive_count": (
                self.counts[
                    "v7_single_bearing_positive"
                ]
            ),
            "single_bearing_no_signal_count": (
                self.counts[
                    "v7_single_bearing_no_signal"
                ]
            ),
            "surprise_no_signal_count": (
                self.counts[
                    "v7_surprise_no_signal"
                ]
            ),
            "surprise_fast_fallback_count": (
                self.counts[
                    "v7_surprise_fast_fallback"
                ]
            ),
            "stale_recovery_replanned_count": (
                self.counts[
                    "v7_stale_recovery_replanned"
                ]
            ),
            "recovery_burst_round_count": (
                self.counts[
                    "v7_recovery_burst_round"
                ]
            ),
            "recovery_burst_extra_measure_count": (
                self.counts[
                    "v7_recovery_burst_extra_measure"
                ]
            ),
            "likelihood_weighted_hypothesis_build_count": (
                self.counts[
                    "v7_likelihood_weighted_hypothesis_build"
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
            + "；V7对单示向频道优先执行短距离中心线重捕获；"
            "information_reacquire恢复点在机器人移动后重新验证；"
            "局部信息恢复支持有界成组执行；联合粒子按照原始先验"
            "和观测存活似然加权，并使用位置组低分位概率和经验"
            "可靠度校准；连续高置信失败后快速转入确定性粗网格恢复"
        )
        return result


def run_self_test() -> dict:
    result = v6.run_self_test()
    tuning = V7Tuning()

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

    quantile = Q4V7Runner._weighted_quantile(
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
        "v7_status": "ok",
        "v7_single_bearing_safe_reacquire": True,
        "v7_stale_recovery_replanning": True,
        "v7_local_recovery_burst": True,
        "v7_likelihood_weighted_posterior": True,
        "v7_surprise_fast_fallback": True,
        "v7_certificate_nodes_removed": False,
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

    tuning = V7Tuning()

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
            "q4_v7_%Y%m%d_%H%M%S_%f"
        )
    )

    client = v3.SimulatorClient(
        args.robot_id,
        args.base_url,
    )

    runner = Q4V7Runner(
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
        f"问题四V7策略启动，输出目录：{output_dir}",
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
        "dual_ring": summary["global_grid"],
        "v6_optimization": (
            summary["v6_optimization"]
        ),
        "v7_optimization": (
            summary["v7_optimization"]
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