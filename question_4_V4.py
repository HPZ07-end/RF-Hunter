"""问题四 V4：双环认证搜索 + 信息驱动恢复 + 短视野联合调度。

依赖：
    question_4_V3.py
    client.py
    geometry_V7.py
    planner.py

运行：
    python question_4_V4.py --robot-id 你的队号

离线检查双环覆盖证书：
    python question_4_V4.py --self-test
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
import random
from typing import Sequence

import question_4_V3 as v3


Point = tuple[float, float]
Triangle = tuple[Point, Point, Point]


@dataclass(frozen=True)
class V4Tuning:
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
    config: v3.Q4Config,
    tuning: V4Tuning,
) -> tuple[v3.GridPlan, dict]:
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
    route_distance = v3.open_route_distance(origin, route)

    plan = v3.GridPlan(
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


class Q4V4Runner(v3.Q4Runner):
    def __init__(
        self,
        client,
        *,
        config: v3.Q4Config,
        tuning: V4Tuning | None = None,
        planner_config=None,
        geometry_config=None,
        output_dir: Path | None = None,
    ):
        self.v4 = tuning or V4Tuning()

        super().__init__(
            client,
            config=config,
            planner_config=planner_config,
            geometry_config=geometry_config,
            output_dir=output_dir,
        )

        # super 只进行了内存初始化，尚未向客户端发出任何动作。
        # 在 enter/search 前用双环计划替换V3网格。
        self.global_grid, self.dual_ring_diagnostics = (
            build_dual_ring_search(self.cfg, self.v4)
        )
        self.search_points = list(self.global_grid.route)

        self._v4_information_attempts: dict[int, int] = {}
        self._v4_last_selected_channel: int | None = None

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
        record: v3.ChannelRecord,
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
                             self.v4.orientation_particle_count),
                    math.sin(2.0 * math.pi * index /
                             self.v4.orientation_particle_count),
                )
                for index in range(self.v4.orientation_particle_count)
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
        record: v3.ChannelRecord,
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
                    "origin": "v4_success_anchor",
                })
        return result

    def _select_information_measurement(
        self,
        record: v3.ChannelRecord,
        *,
        allow_new_q2_seed: bool,
        receive_floor: float,
        move_cap_m: float,
        strict_limits: bool,
        purpose: str,
    ) -> dict:
        sources = v3.sample_source_positions(record, self.cfg)
        hypotheses = self._joint_hypotheses(record, sources)

        q2_seed = (
            self.q2_candidate(record)
            if allow_new_q2_seed
            else record.q2_seed
        )
        estimate, candidates = v3.candidate_points(
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

            conditional = v3.score_candidate(
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
                # 让V3后续调度器使用联合模型概率。
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
            raise v3.IncompleteRun(
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
            raise v3.IncompleteRun(
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
        record: v3.ChannelRecord,
        *,
        record_plan: bool = True,
        allow_new_q2_seed: bool = True,
    ) -> dict:
        proposal = self._select_information_measurement(
            record,
            allow_new_q2_seed=allow_new_q2_seed,
            receive_floor=self.v4.active_receive_floor,
            move_cap_m=self.v4.active_move_cap_m,
            strict_limits=False,
            purpose="v4_receive_constrained_active",
        )
        if record_plan:
            self._record_active_plan(record, proposal)
        return proposal

    def _build_information_recovery_point(
        self,
        record: v3.ChannelRecord,
    ) -> Point:
        proposal = self._select_information_measurement(
            record,
            allow_new_q2_seed=False,
            receive_floor=self.v4.recovery_receive_floor,
            move_cap_m=self.v4.recovery_move_cap_m,
            strict_limits=True,
            purpose="v4_information_recovery",
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
        record: v3.ChannelRecord,
        reason: str,
    ) -> None:
        plan = v3.build_recovery_grid(
            record.outer_polygon,
            self.v4.recovery_fallback_side_m,
            self.client.position,
        )
        route = [
            point for point in plan.route
            if not self._point_was_tested(record, point)
        ]
        if not route:
            raise v3.IncompleteRun(
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
            route_distance_m=v3.open_route_distance(
                self.client.position,
                route,
            ),
            side_m=plan.side_m,
            recovery_mode=record.recovery_mode,
            fallback_reason=reason,
        )

    def start_recovery(self, record: v3.ChannelRecord) -> None:
        self._active_plan_cache.pop(record.channel_id, None)

        existing = self._recovery_queues.get(record.channel_id)
        if record.recovery_active and existing:
            return

        attempts = self._v4_information_attempts.get(
            record.channel_id,
            0,
        )

        if attempts >= self.v4.recovery_information_probe_limit:
            self._start_grid_fallback(
                record,
                "information_probe_limit_reached",
            )
            return

        try:
            point = self._build_information_recovery_point(record)
        except v3.IncompleteRun as error:
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

    def recovery_step(self, record: v3.ChannelRecord) -> str:
        queue = self._recovery_queues.get(record.channel_id)
        if not record.recovery_active or not queue:
            self.start_recovery(record)
            queue = self._recovery_queues[record.channel_id]

        if not queue:
            raise v3.IncompleteRun(
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
                self._v4_information_attempts[record.channel_id] = (
                    self._v4_information_attempts.get(
                        record.channel_id,
                        0,
                    ) + 1
                )
            else:
                self._v4_information_attempts[record.channel_id] = 0
        elif result != "no_signal":
            self._v4_information_attempts[record.channel_id] = 0

        return result

    def _recovery_scheduler_task(
        self,
        record: v3.ChannelRecord,
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
        pending: Sequence[v3.ChannelRecord],
    ) -> list[dict]:
        # V3只更新缓存方案的移动时间，却不更新候选点本身。
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
                > self.v4.active_cache_replan_distance_m
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
        )[:self.v4.scheduler_task_limit]

        horizon = min(
            self.v4.scheduler_horizon,
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
                        if distance <= self.v4.scheduler_cluster_radius_m:
                            value *= (
                                1.0
                                + self.v4.scheduler_cluster_bonus
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
            states = expanded[:self.v4.scheduler_beam_width]

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
        if self._v4_last_selected_channel is not None:
            stay_tasks = [
                task for task in tasks
                if (
                    task["record"].channel_id
                    == self._v4_last_selected_channel
                )
            ]
            if stay_tasks:
                stay = max(
                    stay_tasks,
                    key=self._scheduler_selection_key,
                )
                ratio = (
                    self.v4.scheduler_recovery_commit_ratio
                    if stay["action_type"] == "recovery"
                    else self.v4.scheduler_commit_ratio
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
                raise v3.IncompleteRun(
                    "V4联合调度器没有生成可执行任务"
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
                    "v4_three_step_beam_search_with_channel_commit"
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
            self._v4_last_selected_channel = selected.channel_id
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
        result["v4_tuning"] = asdict(self.v4)
        result["model_note"] = (
            "双环认证搜索；no_signal仅更新规划粒子，"
            "不裁剪连续位置证书；信息重捕获失败后才使用"
            "500米网格；定位阶段采用短视野联合路线调度"
        )
        return result


def run_self_test() -> dict:
    config = v3.Q4Config(
        global_grid_side_m=1000.0,
        enable_q2_seed=False,
        min_request_interval_s=0.0,
    )
    tuning = V4Tuning()
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
            if v3.point_in_triangle(source, triangle)
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
    parser.add_argument("--geometry-config", type=Path)
    parser.add_argument("--planner-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-q2-seed", action="store_true")
    parser.add_argument(
        "--legacy-localization-scheduler",
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
    parser.add_argument("--self-test", action="store_true")
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

    # global_grid_side_m 仅满足V3构造阶段的安全配置；
    # Runner初始化后会在任何客户端动作前替换为双环认证计划。
    config = v3.Q4Config(
        global_grid_side_m=1000.0,
        enable_q2_seed=not args.no_q2_seed,
        enable_parallel_search_localization=True,
        enable_value_time_scheduler=(
            not args.legacy_localization_scheduler
        ),
        scheduler_active_shortlist=(
            args.scheduler_active_shortlist
        ),
        q2_timeout_s=args.q2_timeout,
        min_request_interval_s=args.request_interval,
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
            "q4_v4_%Y%m%d_%H%M%S_%f"
        )
    )

    client = v3.SimulatorClient(
        args.robot_id,
        args.base_url,
    )
    runner = Q4V4Runner(
        client,
        config=config,
        tuning=V4Tuning(),
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

    print(
        f"问题四V4策略启动，输出目录：{output_dir}",
        flush=True,
    )
    summary = runner.run()

    average_clear_time_s = (
        summary["virtual_time_s"] / summary["cleared_count"]
        if summary["cleared_count"] > 0
        else None
    )

    print(json.dumps({
        "outcome": summary["outcome"],
        "reason": summary["reason"],
        "discovered_count": summary["discovered_count"],
        "cleared_count": summary["cleared_count"],
        "virtual_time_s": summary["virtual_time_s"],
        "average_clear_time_s": average_clear_time_s,
        "total_distance_m": summary["total_distance_m"],
        "dual_ring": summary["global_grid"],
        "counts": summary["counts"],
    }, ensure_ascii=False, indent=2))

    return 0 if summary["outcome"] == "success" else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())