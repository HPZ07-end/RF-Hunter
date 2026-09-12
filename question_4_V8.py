"""问题四 V8：局部闭环、恢复路径重排、无回退调度和认证搜索优化。

在 V7 基础上实现：

优先级0：
    安全重捕获成功后，以最新成功点为锚点，连续进行短距离
    方向跟踪，避免重新进入跨地图普通调度。

优先级1：
    保留完整粗恢复网格，但根据最后成功点和当前位置动态排序；
    粗网格被选中后允许连续执行两个相邻节点。

优先级2：
    安全重捕获增加路线插入代价限制，并根据绕路代价衰减
    调度奖励，避免为了一个单示向源横跨地图。

优先级3：
    UNKNOWN 节点价值直接按“预计节省的后续测量时间”计算；
    动态节点必须通过路线经济性约束。

优先级4：
    不删除未经证明的认证节点。保留原点、12点内环和12点外环，
    将内环半径压缩到严格满足1000米边长证书的最小安全值，
    并优化25点开放路线。运行时验证失败则退回V7网格。

运行：
    python question_4_V8.py --robot-id 你的队号

离线检查：
    python question_4_V8.py --self-test
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
from typing import Sequence

import question_4_V7 as v7


v6 = v7.v6
v5 = v6.v5
v4 = v5.v4
v3 = v7.v3

Point = tuple[float, float]
Triangle = tuple[Point, Point, Point]


@dataclass(frozen=True)
class V8Tuning(v7.V7Tuning):
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

    # V7的两个高置信失败过于敏感，V8提高到3次。
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
    config: v3.Q4Config,
    tuning: V8Tuning,
) -> tuple[v3.GridPlan, dict]:
    """构造经过严格验证的25点双环认证网格。

    不减少认证点，只优化内环半径和开放路线。
    """

    baseline_plan, baseline_diagnostics = (
        v4.build_dual_ring_search(
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
        v4.polar_point(
            inner_radius,
            index * step,
        )
        for index in range(count)
    ]
    outer = [
        v4.polar_point(
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
        v4.triangle_max_edge(triangle)
        for triangle in triangles
    )

    if max_edge > reception_radius + 1e-7:
        raise ValueError(
            f"V8认证三角形最大边{max_edge:.6f}米，"
            f"超过{reception_radius:.6f}米"
        )

    outer_min_distance = min(
        v4.origin_segment_distance(
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

    nearest_route = v3.nearest_neighbor_route(
        points,
        origin,
    )

    route_candidates = [
        structured_route,
        v3.two_opt_open(
            structured_route,
            origin,
        ),
        v3.two_opt_open(
            nearest_route,
            origin,
        ),
    ]

    route = min(
        route_candidates,
        key=lambda candidate: (
            v3.open_route_distance(
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

    route_distance = v3.open_route_distance(
        origin,
        route,
    )

    # 若数值环境下没有获得收益，直接使用V7原网格。
    if (
        route_distance
        >= baseline_plan.route_distance_m - 1e-7
    ):
        diagnostics = {
            **baseline_diagnostics,
            "type": (
                "dual_ring_certified_triangulation_"
                "v8_baseline_fallback"
            ),
            "v8_geometry_optimized": False,
            "v8_fallback_reason": (
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

    plan = v3.GridPlan(
        side_m=max_edge,
        points=points,
        route=list(route),
        triangle_count=len(triangles),
        route_distance_m=route_distance,
        triangles=triangles,
    )

    diagnostics = {
        "type": (
            "v8_optimized_dual_ring_"
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


class Q4V8Runner(v7.Q4V7Runner):
    def __init__(
        self,
        client,
        *,
        config: v3.Q4Config,
        tuning: V8Tuning | None = None,
        planner_config=None,
        geometry_config=None,
        output_dir: Path | None = None,
    ):
        self.v8 = tuning or V8Tuning()

        super().__init__(
            client,
            config=config,
            tuning=self.v8,
            planner_config=planner_config,
            geometry_config=geometry_config,
            output_dir=output_dir,
        )

        self._v8_beam_exhausted: set[int] = set()

        self.counts.update({
            "v8_beam_track_round": 0,
            "v8_beam_track_measure": 0,
            "v8_beam_track_direction": 0,
            "v8_beam_track_no_signal": 0,
            "v8_beam_track_ready": 0,
            "v8_beam_track_forced_fallback": 0,
            "v8_coarse_grid_reordered": 0,
            "v8_coarse_batch_round": 0,
            "v8_coarse_batch_extra_measure": 0,
            "v8_safe_plan_deferred": 0,
            "v8_safe_bonus_discounted": 0,
            "v8_search_gate_evaluation_rejected": 0,
            "v8_search_expected_measurements_saved": 0.0,
            "v8_certificate_fallback": 0,
        })

        try:
            (
                optimized_plan,
                optimized_diagnostics,
            ) = build_optimized_certified_search(
                self.cfg,
                self.v8,
            )

            self.global_grid = optimized_plan
            self.dual_ring_diagnostics = (
                optimized_diagnostics
            )
            self.search_points = list(
                optimized_plan.route
            )

            # 新点集必须清除V6按节点编号保存的概率缓存。
            self._v6_unknown_probability_cache.clear()
            self._v6_unknown_omni_masks = None
            self._v6_unknown_directional_masks = None

        except Exception as error:
            # 保留super初始化得到的V7原始网格。
            self.counts[
                "v8_certificate_fallback"
            ] += 1

            self.dual_ring_diagnostics = {
                **self.dual_ring_diagnostics,
                "v8_geometry_optimized": False,
                "v8_fallback_reason": repr(error),
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
        record: v3.ChannelRecord,
    ) -> None:
        """以最新成功方向为锚点，连续执行小范围闭环。"""

        if (
            record.status != "FOUND"
            or not record.bearing_observations
        ):
            return

        self.counts["v8_beam_track_round"] += 1
        consecutive_failures = 0

        self.event(
            "v8_beam_track_started",
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
                self.v8.beam_track_steps_m,
                self.v8.beam_track_offsets_deg,
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
                    "v8_latest_positive_anchor_track"
                ),
                "purpose": (
                    "v8_local_beam_track_transaction"
                ),
                "selection_stage": (
                    "v8_latest_positive_anchor_track"
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
                    "v8_committed_local_beam_track"
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
                "v8_beam_track_measure"
            ] += 1

            if result in {"direction", "near"}:
                consecutive_failures = 0
                self.counts[
                    "v8_beam_track_direction"
                ] += int(result == "direction")
            else:
                consecutive_failures += 1
                self.counts[
                    "v8_beam_track_no_signal"
                ] += 1

            self.event(
                "v8_beam_track_result",
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
                    "v8_beam_track_ready"
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
                >= self.v8.beam_track_failure_limit
            ):
                self._v8_beam_exhausted.add(
                    record.channel_id
                )

                # 局部安全走廊已经失败，不继续依赖概率模型，
                # 直接进入经过动态排序的确定性粗网格。
                self._v4_information_attempts[
                    record.channel_id
                ] = (
                    self.v8
                    .recovery_information_probe_limit
                )
                record.no_progress_count = max(
                    record.no_progress_count,
                    self.cfg.no_progress_trigger,
                )

                self.counts[
                    "v8_beam_track_forced_fallback"
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

        # 直接调用V3执行主体，暂不触发V5的跨频道顺路清除；
        # 先完成当前频道的局部事务。
        v3.Q4Runner._execute_localization_task(
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

        # 完成当前局部闭环后再恢复V5/V6顺路清除。
        self._clear_ready_nearby(
            self.v8.localization_inline_clear_radius_m,
            "localization",
        )

    # ================================================================
    # 优先级1：粗网格动态排序
    # ================================================================

    @staticmethod
    def _last_positive_anchor(
        record: v3.ChannelRecord,
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
                self.v8.coarse_anchor_weight
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
            tail = v3.nearest_neighbor_route(
                remaining,
                position,
            )
            tail = v3.two_opt_open(
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
        record: v3.ChannelRecord,
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
            v3.open_route_distance(
                current,
                original_route,
            )
        )

        selected_route = original_route
        selected_prefix_count = 0

        if anchor is not None:
            for prefix_count in range(
                min(
                    self.v8.coarse_anchor_prefix_count,
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
                    v3.open_route_distance(
                        current,
                        candidate,
                    )
                )

                if (
                    candidate_distance_m
                    <= (
                        original_distance_m
                        * self.v8
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
                v3.nearest_neighbor_route(
                    original_route,
                    current,
                )
            )
            candidate = v3.two_opt_open(
                candidate,
                current,
            )

            if (
                v3.open_route_distance(
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
            raise v3.IncompleteRun(
                "V8粗网格重排丢失了认证节点"
            )

        self._recovery_queues[
            record.channel_id
        ] = deque(selected_route)

        reordered_distance_m = (
            v3.open_route_distance(
                current,
                selected_route,
            )
        )

        self.counts[
            "v8_coarse_grid_reordered"
        ] += 1

        self.event(
            "v8_coarse_grid_reordered",
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
        record: v3.ChannelRecord,
    ) -> str:
        if (
            record.recovery_mode
            != "coarse_grid_last_resort"
        ):
            return super().recovery_step(record)

        executed_count = 0
        last_result = "no_signal"

        for _ in range(
            self.v8.coarse_commit_limit
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
                > self.v8.coarse_commit_leg_cap_m
            ):
                break

        if executed_count > 1:
            self.counts[
                "v8_coarse_batch_round"
            ] += 1
            self.counts[
                "v8_coarse_batch_extra_measure"
            ] += executed_count - 1

            self.event(
                "v8_coarse_batch_completed",
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
        record: v3.ChannelRecord,
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
        record: v3.ChannelRecord,
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
                self.v8
                .safe_max_incremental_detour_m
            )
            or travel_distance_m
            > self.v8.safe_absolute_move_cap_m
        ):
            self.counts[
                "v8_safe_plan_deferred"
            ] += 1

            self.event(
                "v8_safe_plan_deferred",
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
                    self.v8
                    .safe_max_incremental_detour_m
                ),
                absolute_cap_m=(
                    self.v8
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
                self.v8.safe_absolute_move_cap_m
            ),
        })
        return proposal

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
                / self.v8.safe_bonus_decay_m
            )
        )

        if adjusted_bonus_m < old_bonus_m - 1e-9:
            self.counts[
                "v8_safe_bonus_discounted"
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
        task["v8_safe_route_detour_m"] = (
            detour_m
        )
        task["v8_safe_bonus_discounted_m"] = (
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

        route = v3.nearest_neighbor_route(
            [
                self.search_points[index]
                for index in indices
            ],
            start,
        )
        route = v3.two_opt_open(
            route,
            start,
        )
        return v3.open_route_distance(
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

        # 不再使用V6的0.4缩放，直接以预计节省时间计价。
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
            self.v8.search_unknown_value_cap_m,
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
            - self.v8.search_route_regret_penalty
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
            :self.v8.dynamic_route_candidate_limit
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
                - self.v8
                .search_route_regret_penalty
                * regret_m
            )

            economically_covered = (
                total_value_m
                >= (
                    self.v8
                    .search_route_value_cover_ratio
                    * regret_m
                )
            )

            accepted_by_gate = (
                regret_m
                <= (
                    self.v8
                    .search_route_free_regret_m
                )
                or (
                    regret_m
                    <= (
                        self.v8
                        .search_route_max_regret_m
                    )
                    and economically_covered
                )
            )

            item[
                "v8_route_economically_covered"
            ] = economically_covered
            item[
                "v8_route_gate_passed"
            ] = accepted_by_gate

            if accepted_by_gate:
                accepted.append(item)
            else:
                self.counts[
                    "v8_search_gate_evaluation_rejected"
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
                "v8_route_economically_covered"
            ] = True
            baseline_item[
                "v8_route_gate_passed"
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
            "v8_search_expected_measurements_saved"
        ] += saved_measurements

        self.event(
            "v8_search_economic_selection",
            global_index=index,
            expected_saved_measurements=(
                saved_measurements
            ),
            route_regret_m=selected.get(
                "route_regret_m"
            ),
            route_gate_passed=selected.get(
                "v8_route_gate_passed",
                True,
            ),
            route_economically_covered=(
                selected.get(
                    "v8_route_economically_covered",
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

        result["v8_tuning"] = asdict(self.v8)
        result["v8_optimization"] = {
            "beam_track_round_count": (
                self.counts[
                    "v8_beam_track_round"
                ]
            ),
            "beam_track_measure_count": (
                self.counts[
                    "v8_beam_track_measure"
                ]
            ),
            "beam_track_direction_count": (
                self.counts[
                    "v8_beam_track_direction"
                ]
            ),
            "beam_track_no_signal_count": (
                self.counts[
                    "v8_beam_track_no_signal"
                ]
            ),
            "beam_track_ready_count": (
                self.counts[
                    "v8_beam_track_ready"
                ]
            ),
            "beam_track_forced_fallback_count": (
                self.counts[
                    "v8_beam_track_forced_fallback"
                ]
            ),
            "coarse_grid_reordered_count": (
                self.counts[
                    "v8_coarse_grid_reordered"
                ]
            ),
            "coarse_batch_round_count": (
                self.counts[
                    "v8_coarse_batch_round"
                ]
            ),
            "coarse_batch_extra_measure_count": (
                self.counts[
                    "v8_coarse_batch_extra_measure"
                ]
            ),
            "safe_plan_deferred_count": (
                self.counts[
                    "v8_safe_plan_deferred"
                ]
            ),
            "safe_bonus_discounted_count": (
                self.counts[
                    "v8_safe_bonus_discounted"
                ]
            ),
            "search_gate_rejected_count": (
                self.counts[
                    "v8_search_gate_evaluation_rejected"
                ]
            ),
            "search_expected_measurements_saved": (
                self.counts[
                    "v8_search_expected_measurements_saved"
                ]
            ),
            "certificate_fallback_count": (
                self.counts[
                    "v8_certificate_fallback"
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
    inherited = v7.run_self_test()

    tuning = V8Tuning()
    config = v3.Q4Config(
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
        v4.triangle_max_edge(triangle)
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
        "v8_status": "ok",
        "v8_latest_positive_closed_loop": True,
        "v8_anchor_ordered_coarse_grid": True,
        "v8_safe_no_regret_gate": True,
        "v8_unknown_time_value": True,
        "v8_search_route_economic_gate": True,
        "v8_certificate_point_count": (
            len(plan.points)
        ),
        "v8_certificate_triangle_count": (
            len(plan.triangles)
        ),
        "v8_max_certificate_edge_m": (
            maximum_edge
        ),
        "v8_outer_boundary_min_distance_m": (
            outer_distance
        ),
        "v8_geometry_route_improvement_m": (
            diagnostics.get(
                "route_improvement_m",
                0.0,
            )
        ),
        "v8_certificate_nodes_removed": False,
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
            "q4_v8_%Y%m%d_%H%M%S_%f"
        )
    )

    client = v3.SimulatorClient(
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
        "v6_optimization": (
            summary["v6_optimization"]
        ),
        "v7_optimization": (
            summary["v7_optimization"]
        ),
        "v8_optimization": (
            summary["v8_optimization"]
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