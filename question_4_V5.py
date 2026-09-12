"""问题四 V5：在V4基础上增加动态搜索、顺路清除和自适应测量。

依赖：
    question_4_V4.py
    question_4_V3.py
    client.py
    geometry_V7.py
    planner.py

运行：
    python question_4_V5.py --robot-id 你的队号

离线检查：
    python question_4_V5.py --self-test
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

import question_4_V4 as v4


v3 = v4.v3
Point = tuple[float, float]


@dataclass(frozen=True)
class V5Tuning(v4.V4Tuning):
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


class Q4V5Runner(v4.Q4V4Runner):
    def __init__(
        self,
        client,
        *,
        config: v3.Q4Config,
        tuning: V5Tuning | None = None,
        planner_config=None,
        geometry_config=None,
        output_dir: Path | None = None,
    ):
        self.v5 = tuning or V5Tuning()

        super().__init__(
            client,
            config=config,
            tuning=self.v5,
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
            "v5_opportunistic_planned": 0,
            "v5_inline_clear_search": 0,
            "v5_inline_clear_localization": 0,
            "v5_adaptive_active_plan": 0,
            "v5_adaptive_recovery_plan": 0,
        })

    # ================================================================
    # 优先级5：根据历史成功率动态调整接收概率下限
    # ================================================================

    def _adaptive_receive_floor(
        self,
        record: v3.ChannelRecord,
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
            self.v5.adaptive_receive_ceiling,
            base_floor
            + self.v5.adaptive_receive_penalty * failure_rate,
        )

    def _adaptive_move_cap(
        self,
        record: v3.ChannelRecord,
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
        record: v3.ChannelRecord,
        *,
        record_plan: bool = True,
        allow_new_q2_seed: bool = True,
    ) -> dict:
        receive_floor = self._adaptive_receive_floor(
            record,
            self.v5.active_receive_floor,
        )
        move_cap = self._adaptive_move_cap(
            record,
            self.v5.active_move_cap_m,
            self.v5.adaptive_active_min_move_cap_m,
        )

        proposal = self._select_information_measurement(
            record,
            allow_new_q2_seed=allow_new_q2_seed,
            receive_floor=receive_floor,
            move_cap_m=move_cap,
            strict_limits=False,
            purpose="v5_adaptive_receive_constrained_active",
        )
        proposal["adaptive_receive_floor"] = receive_floor
        proposal["adaptive_move_cap_m"] = move_cap

        self.counts["v5_adaptive_active_plan"] += 1
        if record_plan:
            self._record_active_plan(record, proposal)
        return proposal

    def _build_information_recovery_point(
        self,
        record: v3.ChannelRecord,
    ) -> Point:
        receive_floor = self._adaptive_receive_floor(
            record,
            self.v5.recovery_receive_floor,
        )
        move_cap = self._adaptive_move_cap(
            record,
            self.v5.recovery_move_cap_m,
            self.v5.adaptive_recovery_min_move_cap_m,
        )

        proposal = self._select_information_measurement(
            record,
            allow_new_q2_seed=False,
            receive_floor=receive_floor,
            move_cap_m=move_cap,
            strict_limits=True,
            purpose="v5_adaptive_information_recovery",
        )

        self.counts["v5_adaptive_recovery_plan"] += 1
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
        record: v3.ChannelRecord,
        point: Point,
    ) -> float:
        if record.status != "FOUND":
            return 0.0
        if self._point_was_tested(record, point):
            return 0.0
        if not record.bearing_observations:
            return 0.0

        try:
            sources = v3.sample_source_positions(
                record,
                self.cfg,
            )
        except v3.IncompleteRun:
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
            self.v5.opportunistic_receive_floor,
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
                self.v5.dynamic_ready_bonus_m
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
                :self.v5.dynamic_route_found_limit
            ]
        )

    def _nearest_neighbor_tail_distance(
        self,
        start: Point,
        indices: Sequence[int],
    ) -> float:
        if not indices:
            return 0.0

        route = v3.nearest_neighbor_route(
            [self.search_points[index] for index in indices],
            start,
        )
        return v3.open_route_distance(start, route)

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
            :self.v5.dynamic_route_candidate_limit
        ]

        baseline_index = candidates[0]
        baseline_tail = self._nearest_neighbor_tail_distance(
            current,
            list(remaining),
        )

        allowed_regret = max(
            self.v5.dynamic_route_regret_floor_m,
            self.v5.dynamic_route_regret_ratio
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
                - self.v5.dynamic_route_regret_penalty
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

    def _score_v5_opportunity(
        self,
        record: v3.ChannelRecord,
        point: Point,
    ) -> dict | None:
        if record.status != "FOUND":
            return None
        if self._point_was_tested(record, point):
            return None
        if (
            record.opportunistic_measure_count
            >= self.v5.opportunistic_max_per_channel
        ):
            return None

        try:
            sources = v3.sample_source_positions(
                record,
                self.cfg,
            )
        except v3.IncompleteRun:
            return None

        estimate = (
            sum(item[0] for item in sources) / len(sources),
            sum(item[1] for item in sources) / len(sources),
        )
        candidate = {
            "point": tuple(point),
            "origin": "v5_dynamic_certified_node",
        }

        conditional = v3.score_candidate(
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
            self.v5.opportunistic_receive_floor,
        )
        minimum_reduction = max(
            self.v5.opportunistic_min_reduction_m,
            self.v5.opportunistic_min_relative_reduction
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
                self.v5.opportunistic_ready_bonus_m
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
            proposal = self._score_v5_opportunity(
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
        )[:self.v5.opportunistic_max_per_node]

        if selected:
            self.counts["v5_opportunistic_planned"] += len(
                selected
            )
            self.event(
                "opportunistic_batch_planned",
                global_index=global_index,
                point=point,
                strategy="v5_joint_posterior",
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
                "v5_inline_clear_search"
                if phase_name == "search"
                else "v5_inline_clear_localization"
            )
            self.counts[counter] += 1
            self.event(
                "v5_inline_clear_completed",
                channel=selected.channel_id,
                source_phase=phase_name,
                maximum_distance_m=maximum_distance_m,
            )

    def _execute_localization_task(self, task: dict) -> None:
        super()._execute_localization_task(task)

        self._clear_ready_nearby(
            self.v5.localization_inline_clear_radius_m,
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
                self.v5.search_inline_clear_radius_m,
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
            raise v3.IncompleteRun(
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

        result["v5_tuning"] = asdict(self.v5)
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
            "V4双环证书、信息恢复和联合调度保持不变；"
            "V5只动态重排尚未访问的认证节点，不删除节点；"
            "机会补测使用联合位置-朝向-半径概率；"
            "附近READY源顺路清除；主动测量门槛根据历史"
            "no_signal比例自适应"
        )
        return result


def run_self_test() -> dict:
    result = v4.run_self_test()
    tuning = V5Tuning()

    assert tuning.opportunistic_max_per_node > 0
    assert tuning.opportunistic_max_per_channel > 0
    assert (
        tuning.search_inline_clear_radius_m
        < tuning.localization_inline_clear_radius_m
    )

    return {
        **result,
        "v5_status": "ok",
        "v5_dynamic_route": True,
        "v5_inline_clear": True,
        "v5_adaptive_measurement": True,
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

    tuning = V5Tuning()
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
            "q4_v5_%Y%m%d_%H%M%S_%f"
        )
    )

    client = v3.SimulatorClient(
        args.robot_id,
        args.base_url,
    )
    runner = Q4V5Runner(
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
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print(
        f"问题四V5策略启动，输出目录：{output_dir}",
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
        "discovered_count": summary["discovered_count"],
        "cleared_count": summary["cleared_count"],
        "virtual_time_s": summary["virtual_time_s"],
        "average_clear_time_s": average_clear_time_s,
        "total_distance_m": summary["total_distance_m"],
        "dynamic_search": summary["dynamic_search"],
        "dual_ring": summary["global_grid"],
        "counts": summary["counts"],
    }, ensure_ascii=False, indent=2))

    return 0 if summary["outcome"] == "success" else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())