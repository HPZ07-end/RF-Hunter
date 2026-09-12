"""问题三基线主程序：严格对应《问题3-v1》默认 B3 流程。

同目录依赖（原文件内容无需修改，仅统一文件名）：
    client.py, geometry_V7.py, geometry_two_point_fast.py, planner.py
运行（模拟器已登录、已启动问题三测试且接口就绪）：
    python main_q3.py --robot-id 你的参赛队号
关闭插入对照：
    python main_q3.py --robot-id 你的参赛队号 --no-insert
参数对照：
    python main_q3.py --robot-id 你的参赛队号 --rho 1123 --insert-threshold 120
可选既有问题一配置：--geometry-config config.h
可选问题二配置：--planner-config planner_config.json（PlannerConfig 字段的JSON对象）

输出 runs/q3_时间戳/{summary.json, operations.jsonl, events.jsonl, client.log}。
正式测试加密日志必须从模拟器界面另行导出，本程序不能获取案例编码或隐藏总数。
所有采样仅用于下一测向点评分；清除证书使用全部历史角域+先验方框的完整外包顶点。
首次选点调用原 solve_q2；后续适配采用有限源/误差/候选采样，不冒称连续全局最优。
默认配置是可运行工程参数；有限次重规划失败会报告未完成，绝不会当成清除成功。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import combinations
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
import time
import traceback

from client import SimulatorClient, ClientError, TransportError
from geometry_V7 import GeometryConfig, build_halfplanes, solve_halfplanes, load_config
from planner import PlannerConfig, SourceRegion, solve_q2

Point = tuple[float, float]
GOOD_REGIONS = {"POLYGON", "SEGMENT", "POINT"}
TERMINAL = {"CLEARED", "ABSENT"}
BOX = [(1., 0., 1800.), (-1., 0., 1800.), (0., 1., 1800.), (0., -1., 1800.)]


class IncompleteRun(RuntimeError):
    """算法/证书异常；保留未完成状态，并报告具体原因。"""


class BudgetStop(IncompleteRun):
    """即将耗尽现实或虚拟时间；不是任务完成。"""


@dataclass(frozen=True)
class Q3Config:
    rho: float = 1200.0
    enable_insert: bool = True
    insert_threshold_s: float = 60.0
    clear_margin_m: float = 0.01
    no_progress_trigger: int = 3
    max_replans: int = 3
    planning_timeout_s: float = 45.0
    real_reserve_s: float = 10.0
    source_samples: int = 24
    candidate_count: int = 32
    error_samples: int = 3
    near_optimal_abs_m: float = 0.5
    near_optimal_rel: float = 0.05
    repeated_point_m: float = 0.5
    progress_abs_m: float = 0.01
    progress_rel: float = 0.001

    def __post_init__(self):
        lo = 900 * math.sqrt(3) - math.sqrt(190000)
        hi = 1000 * math.sqrt(3)
        if not math.isfinite(self.rho) or not lo <= self.rho <= hi:
            raise ValueError(f"rho 必须在保证覆盖区间 [{lo}, {hi}] 内")
        if not 0 <= self.clear_margin_m < 20:
            raise ValueError("clear_margin_m 必须在 [0,20) 内")
        for key in ("no_progress_trigger", "max_replans", "source_samples", "candidate_count", "error_samples"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} 必须是正整数")
        if self.source_samples < 3 or self.error_samples < 2:
            raise ValueError("source_samples 至少3，error_samples 至少2（包含误差端点）")
        for key in ("planning_timeout_s", "real_reserve_s", "repeated_point_m"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} 必须是有限正数")
        for key in ("insert_threshold_s", "near_optimal_abs_m", "near_optimal_rel", "progress_abs_m", "progress_rel"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} 必须是有限非负数")


@dataclass
class ChannelRecord:
    channel_id: int
    status: str = "UNKNOWN"
    bearing_observations: list = field(default_factory=list)
    no_signal_observations: list = field(default_factory=list)
    near_signal_observations: list = field(default_factory=list)
    outer_polygon: list = field(default_factory=list)
    region_status: str | None = None
    diameter: float | None = None
    clearance_center: Point | None = None
    clearance_radius: float | None = None
    certificate_source: str | None = None
    absent_reason: str | None = None
    last_action: str | None = None
    failure_count: int = 0
    no_progress_count: int = 0
    replan_level: int = 0
    progress_history: list = field(default_factory=list)
    revision: int = 0
    q2_attempted: bool = False
    # 只缓存耗时的Q2首次建议；多历史建议随机器人位置重评近优候选。
    q2_proposal: dict | None = None


def coverage_points(rho=1200.0):
    return [(0., 0.)] + [(rho * math.cos(k * math.pi / 3),
                         rho * math.sin(k * math.pi / 3)) for k in range(6)]


def halfplanes(record):
    rows = list(BOX)
    if record.bearing_observations:
        obs = [(*o["position"], o["bearing_deg"]) for o in record.bearing_observations]
        rows.extend(map(tuple, build_halfplanes(obs, error_deg=1.0).tolist()))
    return rows


def minimum_enclosing_circle(vertices):
    """枚举1/2/3点支持圆；所有候选重新计算最大顶点距离，返回实测外包半径。

    近共线三点圆可能病态，跳过后仍有逐顶点检查的保守候选；绝不低估半径。
    最坏 O(h^4)，仅用于小规模凸多边形。不是直径中点圆的替代命名。
    """
    pts = list(dict.fromkeys(tuple(map(float, p)) for p in vertices))
    if not pts:
        raise IncompleteRun("空顶点集不能出具清除证书")
    candidates = [(p, 0.) for p in pts]
    for a, b in combinations(pts, 2):
        q = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        candidates.append((q, math.dist(a, b) / 2))
    for a, b, c in combinations(pts, 3):
        ux, uy = b[0]-a[0], b[1]-a[1]
        vx, vy = c[0]-a[0], c[1]-a[1]
        cross = ux*vy-uy*vx
        if abs(cross) <= 1e-12 * max(1., math.hypot(ux, uy)*math.hypot(vx, vy)):
            continue
        u2, v2 = ux*ux+uy*uy, vx*vx+vy*vy
        q = (a[0]+(u2*vy-v2*uy)/(2*cross), a[1]+(ux*v2-vx*u2)/(2*cross))
        if all(math.isfinite(v) for v in q):
            candidates.append((q, math.dist(q, a)))
    best = None
    for q, r in candidates:
        actual = max(math.dist(q, p) for p in pts)
        if actual <= r + 1e-7 * max(1., r):
            candidate = (actual, q)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        # 理论上端点圆或三点圆应存在；若数值失效，报告而不是造证书。
        raise IncompleteRun("最小包围圆支持圆枚举失败")
    r, q = best
    return q, max(math.dist(q, p) for p in pts)


def update_region(record, cfg, geometry_cfg):
    result = solve_halfplanes(halfplanes(record), config=geometry_cfg)
    record.region_status = result.status
    if result.status not in GOOD_REGIONS or not result.vertices:
        raise IncompleteRun(f"频道{record.channel_id}几何状态{result.status}: {result.diagnostics}")
    if record.diameter is not None and result.diameter > record.diameter + 1e-5:
        raise IncompleteRun(f"频道{record.channel_id}增加历史约束后直径异常增大")
    record.outer_polygon = result.vertices
    record.diameter = result.diameter
    q, radius = minimum_enclosing_circle(result.vertices)
    record.clearance_center, record.clearance_radius = q, radius
    record.certificate_source = "all_history_box_mec" if radius <= 20-cfg.clear_margin_m else None
    record.status = "READY" if record.certificate_source else "FOUND"


def radical_inverse(index, base):
    result, fraction = 0., 1./base
    while index:
        index, digit = divmod(index, base)
        result += digit*fraction
        fraction /= base
    return result


def sample_sources(record, cfg, planner_cfg):
    """从外包多边形及Q2的SourceRegion生成样本，并检查全部物理约束。

    无信号只留账，不裁剪几何；正常测向点构成成功接收点集合 H。
    """
    rows = halfplanes(record)
    target = cfg.source_samples * (2 ** record.replan_level)
    samples = []

    def add(p):
        p = tuple(map(float, p))
        if math.hypot(*p) > 1800 + 1e-7:
            return
        if any(a*p[0]+b*p[1]-c > 1e-7 for a, b, c in rows):
            return
        if any(not 5.0 < math.dist(p, o["position"]) <= 1500+1e-7
               for o in record.bearing_observations):
            return
        if not any(math.dist(p, q) < 1e-6 for q in samples):
            samples.append(p)

    vertices = record.outer_polygon
    center = tuple(sum(p[k] for p in vertices)/len(vertices) for k in (0, 1))
    add(center)
    for p in vertices:
        add(p)
    # 凸多边形三角扇采样，避免很窄的多观测区域被包围盒拒绝采样漏掉。
    for i in range(1, target*12+1):
        a = vertices[(i-1) % len(vertices)]
        b = vertices[i % len(vertices)]
        u, v = math.sqrt(radical_inverse(i, 2)), radical_inverse(i, 3)
        add(((1-u)*center[0]+u*((1-v)*a[0]+v*b[0]),
             (1-u)*center[1]+u*((1-v)*a[1]+v*b[1])))
        if len(samples) >= target:
            break
    # 复用问题二第一次角域+圆盘的参数化；对所有历史再筛选。
    first = record.bearing_observations[0]
    region = SourceRegion(tuple(first["position"]), first["bearing_deg"], planner_cfg)
    for i in range(1, target*24+1):
        if len(samples) >= target:
            break
        p = region.point((radical_inverse(i, 2), radical_inverse(i, 3)))
        if p is not None:
            add(p)
    if not samples:
        raise IncompleteRun(f"频道{record.channel_id}无满足全部历史/目标圆/接收距离的源样本")
    return samples


def plan_history(record, robot_position, cfg, planner_cfg, geometry_cfg):
    """Q3适配：全历史+方框的有限情景最坏直径，近优集合按距离择优。

    不接收情景按原直径计（没有新约束）；near按至多10米直径计。
    正常测向情景加入模拟角域后调用问题一；该评分不用于可靠清除证书。
    """
    sources = sample_sources(record, cfg, planner_cfg)
    center = tuple(sum(p[k] for p in sources)/len(sources) for k in (0, 1))
    candidates = [center, tuple(robot_position)]
    if record.q2_proposal:
        candidates.append(tuple(record.q2_proposal["point"]))
    n = cfg.candidate_count * (2 ** record.replan_level)
    spread = max(math.dist(center, p) for p in sources)
    # 工程候选多样性；无进展时旋转并加密候选和源样本。
    for i in range(n):
        angle = 2*math.pi*(i/n + record.replan_level*0.137)
        radius = (30., 80., 200., 500., min(900., max(80., spread)))[i % 5]
        candidates.append((center[0]+radius*math.cos(angle), center[1]+radius*math.sin(angle)))
    previous = [o["position"] for o in record.bearing_observations+record.no_signal_observations]
    unique = []
    for p in candidates:
        if any(math.dist(p, s) < cfg.repeated_point_m for s in previous+unique):
            continue
        if all(math.isfinite(v) and abs(v) <= 2_000_000 for v in p):
            unique.append(p)
    if not unique:
        raise IncompleteRun("无非重复测向候选")
    rows = halfplanes(record)
    errors = [-1+2*i/(cfg.error_samples-1) for i in range(cfg.error_samples)]
    scored = []
    prediction_fallbacks = {}
    for p in unique:
        worst = 0.
        for g in sources:
            d = math.dist(p, g)
            radius_lower = max(1000., *(math.dist(g, o["position"]) for o in record.bearing_observations))
            if d > radius_lower + 1e-7:
                worst = max(worst, record.diameter)
                continue
            if d <= 5:
                worst = max(worst, min(10., record.diameter))
                continue
            angle = math.degrees(math.atan2(g[1]-p[1], g[0]-p[0])) % 360
            for error in errors:
                future = rows + list(map(tuple, build_halfplanes([(*p, angle+error)], 1.).tolist()))
                result = solve_halfplanes(future, config=geometry_cfg)
                if result.status not in GOOD_REGIONS:
                    # 样本恰落旧边界且新误差取端点时，浮点生成系数可导致空/极薄域。
                    # 预测评分保守退回“本次无缩减”的旧直径，并记录；不删真实历史，
                    # 不扩大用于清除证书的区域，也不把异常当成直径0。
                    prediction_fallbacks[result.status] = prediction_fallbacks.get(result.status, 0)+1
                    worst = max(worst, record.diameter)
                    continue
                worst = max(worst, result.diameter)
        scored.append((worst, math.dist(p, robot_position), p))
    best_score = min(v[0] for v in scored)
    tolerance = max(cfg.near_optimal_abs_m, cfg.near_optimal_rel*best_score)
    near_best = [v for v in scored if v[0] <= best_score+tolerance]
    score, _, point = min(near_best, key=lambda v: (v[1], v[0], v[2]))
    return {"point": point, "score": score, "method": "q3_all_history_sampled",
            "source_count": len(sources), "candidate_count": len(unique),
            "error_samples_deg": errors, "score_scope": "all_bearings_and_prior_box",
            "best_sampled_score": best_score, "near_optimal_tolerance_m": tolerance,
            "prediction_fallback_counts": prediction_fallbacks}


def _planning_worker(connection, record, position, cfg, planner_cfg, geometry_cfg):
    """独立进程只计算；不接触模拟器。主进程可在预算耗尽时终止计算。"""
    try:
        if len(record.bearing_observations) == 1 and not record.q2_attempted:
            obs = record.bearing_observations[0]
            result = solve_q2((*obs["position"], obs["bearing_deg"]),
                              config=planner_cfg, geometry_config=geometry_cfg)
            if result.second_point is None or result.worst_diameter is None or not math.isfinite(result.worst_diameter):
                raise IncompleteRun(f"问题二未返回有限有效选点: {result.status}")
            point = tuple(result.second_point)
            if not all(math.isfinite(v) and abs(v) <= 2_000_000 for v in point):
                raise IncompleteRun("问题二返回非法坐标")
            if math.dist(point, obs["position"]) < cfg.repeated_point_m:
                raise IncompleteRun("问题二返回重复测向点")
            output = {"point": point, "score": result.worst_diameter,
                      "method": "original_q2", "planner_status": result.status,
                      "receive_margin_m": result.receive_margin_m,
                      "score_scope": "q2_two_bearing_model"}
        else:
            output = plan_history(record, position, cfg, planner_cfg, geometry_cfg)
        connection.send((True, output))
    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def bounded_plan(record, position, cfg, planner_cfg, geometry_cfg, budget_s):
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_planning_worker,
                              args=(child, record, position, cfg, planner_cfg, geometry_cfg))
    process.start()
    child.close()
    try:
        if not parent.poll(max(0., min(cfg.planning_timeout_s, budget_s))):
            raise IncompleteRun(f"频道{record.channel_id}单次选点超时；未获得可用建议")
        try:
            ok, result = parent.recv()
        except EOFError as error:
            raise IncompleteRun("选点子进程异常结束") from error
        if not ok:
            raise IncompleteRun(result)
        return result
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()


def insertion_cost(position, clearance, next_point):
    extra = math.dist(position, clearance)+math.dist(clearance, next_point)-math.dist(position, next_point)
    if extra < -1e-6:
        raise IncompleteRun("插入增量显著为负")
    return max(0., extra)/5 + 5.  # clear不切频道


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class Q3Runner:
    """主状态机。client可注入本地测试替身；在线决策只读取公开接口返回值。"""
    def __init__(self, client, *, config=None, planner_config=None, geometry_config=None,
                 output_dir=None, plan_function=bounded_plan):
        self.client = client
        self.cfg = config or Q3Config()
        self.pcfg = planner_config or PlannerConfig()
        self.gcfg = geometry_config or GeometryConfig()
        # 模型常数必须与问题三固定物理约束一致，允许调整搜索数值参数。
        fixed = {"target_radius_m":1800., "reception_radius_min_m":1000.,
                 "reception_radius_max_m":1500., "no_bearing_radius_m":5., "bearing_error_deg":1.}
        if any(getattr(self.pcfg, k) != v for k, v in fixed.items()) or self.gcfg.bearing_error_deg != 1.:
            raise ValueError("问题一/二配置的题目物理常数必须与问题三一致")
        self.plan_function = plan_function
        self.records = {c: ChannelRecord(c) for c in range(1, 21)}
        self.points = coverage_points(self.cfg.rho)
        self.scan_done = [[False]*20 for _ in self.points]
        self.next_coverage_index = 0
        self.operations, self.events, self.trajectory = [], [], []
        self.phase = "initialization"
        self.total_distance_m = 0.
        self.counts = {"measure":0, "switch":0, "optical":0, "clear_success":0, "insertion":0}
        self.phase_virtual_s = {"search":0., "localization":0.}
        self.entered_at = None
        self.end_at = None
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=False)
        self._streams = {}
        try:
            if self.output_dir:
                for name in ("operations", "events"):
                    self._streams[name] = (self.output_dir / f"{name}.jsonl").open("w", encoding="utf-8")
        except Exception:
            self.close()
            raise

    @property
    def discovered_count(self):
        return sum(bool(r.bearing_observations or r.near_signal_observations) for r in self.records.values())

    @property
    def cleared_count(self):
        return sum(r.status == "CLEARED" for r in self.records.values())

    def close(self):
        for stream in self._streams.values():
            stream.close()

    def _append(self, kind, entry):
        entry = json_safe(entry)
        (self.operations if kind == "operations" else self.events).append(entry)
        if kind in self._streams:
            self._streams[kind].write(json.dumps(entry, ensure_ascii=False, allow_nan=False)+"\n")
            self._streams[kind].flush()

    def event(self, kind, **data):
        self._append("events", {"event":kind, "phase":self.phase,
                     "virtual_time_s":self.client.virtual_time_s, **data})

    def check_budget(self, move_to=None, action="measure", channel=None):
        remaining = self.client.remaining_real_time_s
        if remaining is not None and remaining <= self.cfg.real_reserve_s:
            raise BudgetStop("现实预算不足，保留退出余量；任务未完成")
        if self.client.virtual_time_s is not None:
            cost = 0.
            if move_to is not None:
                cost = math.dist(self.client.position, move_to)/5 + 5
                if action == "measure" and channel != self.client.current_channel:
                    cost += 1
            if self.client.virtual_time_s+cost >= self.client.max_virtual_duration_s:
                raise BudgetStop("虚拟时间不足；任务未完成")

    def action(self, method, position=None, channel=None):
        self.check_budget(position, method, channel)
        before_position = self.client.position
        before_channel = self.client.current_channel
        before_time = self.client.virtual_time_s or 0.
        started = time.monotonic()
        args = () if position is None else (*position, channel)
        try:
            result = getattr(self.client, method)(*args)
        except Exception as error:
            self._append("operations", {"method":method, "position":position, "channel":channel,
                 "phase":self.phase, "accepted":getattr(error, "response", {}).get("accepted"), "execution_confirmed":False,
                 "error":repr(error), "pending_request_id":self.client.pending_request_id,
                 "http_status":getattr(error, "status", None), "response":getattr(error, "response", None)})
            raise
        delta = result["virtual_time_s"]-before_time
        distance = math.dist(before_position, position) if position is not None else 0.
        switched = int(method == "measure" and channel != before_channel)
        expected = 0.
        if method in ("measure", "clear"):
            self.total_distance_m += distance
            if method == "measure":
                self.counts["measure"] += 1
                self.counts["switch"] += switched
                expected = distance/5 + 5 + switched
            else:
                self.counts["optical"] += 1
                success = result["clear_result"] == "success"
                self.counts["clear_success"] += int(success)
                expected = distance/5 + (5 if success else 3)
            self.phase_virtual_s[self.phase] += delta
            self.trajectory.append({"from":before_position, "to":position, "phase":self.phase,
                                    "method":method, "channel":channel, "virtual_time_s":result["virtual_time_s"]})
        self._append("operations", {"method":method, "position":position, "channel":channel,
            "phase":self.phase, "accepted":True, "execution_confirmed":True, "response":result,
            "distance_m":distance, "switch_count":switched, "virtual_delta_s":delta,
            "expected_delta_s":expected, "request_elapsed_s":time.monotonic()-started})
        if abs(delta-expected) > 2e-5:
            self.event("time_accounting_mismatch", actual=delta, expected=expected)
        return result

    def measure(self, record, point, coverage_index=None):
        old_diameter = record.diameter
        result = self.action("measure", point, record.channel_id)
        if coverage_index is not None:
            self.scan_done[coverage_index][record.channel_id-1] = True
        datum = {"position":tuple(point), "virtual_time_s":result["virtual_time_s"],
                 "real_timestamp_ms":result["real_timestamp_ms"]}
        record.last_action = "measure"
        record.revision += 1
        code = result["measure_result"]
        if code == "direction":
            record.bearing_observations.append({**datum, "bearing_deg":result["svd_deg"]})
            record.status = "FOUND"
            update_region(record, self.cfg, self.gcfg)
        elif code == "near":
            record.near_signal_observations.append(datum)
            record.status = "READY"
            record.clearance_center, record.clearance_radius = tuple(point), 5.
            record.certificate_source = "near_feedback"
        elif code == "no_signal":
            record.no_signal_observations.append(datum)
        else:
            raise IncompleteRun(f"未知检测结果 {code}")
        improved = (record.status == "READY" or (code == "direction" and
            (old_diameter is None or old_diameter-record.diameter >
             max(self.cfg.progress_abs_m, self.cfg.progress_rel*old_diameter))))
        record.no_progress_count = 0 if improved else record.no_progress_count+1
        record.progress_history.append({"result":code, "diameter":record.diameter,
            "status":record.status, "improved":improved, "virtual_time_s":result["virtual_time_s"],
            "outer_polygon":record.outer_polygon, "clearance_center":record.clearance_center,
            "clearance_radius":record.clearance_radius})
        self.event("measurement_update", channel=record.channel_id, result=code, status=record.status)

    def clear(self, record, insertion=False):
        if record.status != "READY" or not record.certificate_source:
            raise IncompleteRun("禁止无可靠证书清除")
        if record.certificate_source == "all_history_box_mec":
            radius = max(math.dist(record.clearance_center, p) for p in record.outer_polygon)
            if radius > 20-self.cfg.clear_margin_m:
                raise IncompleteRun("执行前逐顶点证书复核失败")
        result = self.action("clear", record.clearance_center, record.channel_id)
        record.last_action = "clear"
        if insertion:
            self.counts["insertion"] += 1
        if result["clear_result"] == "success":
            record.status = "CLEARED"
            self.event("cleared", channel=record.channel_id, insertion=insertion,
                       certificate_source=record.certificate_source)
        else:
            record.failure_count += 1
            # 静止源+可靠证书下失败意味着假设/数值/协议不一致，停止并保留证据。
            raise IncompleteRun(f"频道{record.channel_id}持可靠证书却清除失败，保留READY待诊断")

    def mark_absent(self):
        for record in self.records.values():
            if record.status != "UNKNOWN":
                continue
            if self.discovered_count >= 16:
                reason = "16_distinct_sources_discovered"
            elif all(row[record.channel_id-1] for row in self.scan_done):
                reason = "seven_coverage_points_all_no_signal"
            else:
                continue
            record.status, record.absent_reason = "ABSENT", reason
            self.event("absent", channel=record.channel_id, reason=reason)

    def search(self):
        self.phase = "search"
        for index, point in enumerate(self.points):
            self.next_coverage_index = index
            unknown = [c for c, r in self.records.items() if r.status == "UNKNOWN"]
            if not unknown:
                break  # 所有频道已发现/排除，不再需要未知扫描；不伪造访问轨迹。
            current = self.client.current_channel
            order = ([current] if current in unknown else []) + [c for c in unknown if c != current]
            for channel in order:
                self.measure(self.records[channel], point, index)
                if self.discovered_count == 16:
                    self.mark_absent()
                    break
            self.next_coverage_index = index+1
            if self.discovered_count == 16:
                break
            if index < 6 and self.cfg.enable_insert:
                choices = [(insertion_cost(self.client.position, r.clearance_center, self.points[index+1]), c, r)
                           for c, r in self.records.items() if r.status == "READY"]
                if choices:
                    cost, _, record = min(choices, key=lambda v: (v[0], v[1]))
                    self.event("insertion_decision", channel=record.channel_id, cost_s=cost,
                               accepted=cost <= self.cfg.insert_threshold_s)
                    if cost <= self.cfg.insert_threshold_s:
                        self.clear(record, insertion=True)
        self.mark_absent()

    def proposal(self, record):
        self.check_budget()
        if len(record.bearing_observations) == 1 and not record.q2_attempted and record.q2_proposal:
            return record.q2_proposal
        remaining = self.client.remaining_real_time_s
        budget = self.cfg.planning_timeout_s if remaining is None else remaining-self.cfg.real_reserve_s
        result = self.plan_function(record, self.client.position, self.cfg, self.pcfg, self.gcfg, budget)
        if result["method"] == "original_q2":
            record.q2_proposal = result
        self.event("planned", channel=record.channel_id, **result)
        self.check_budget()
        return result

    def replan(self, record, reason):
        record.failure_count += 1
        record.replan_level += 1
        record.no_progress_count = 0
        # 原Q2已尝试但失败/无信号时，启用全历史候选适配，避免原地重复同一建议。
        record.q2_attempted = True
        self.event("replan", channel=record.channel_id, level=record.replan_level, reason=reason)

    def localize(self):
        self.phase = "localization"
        while any(r.status in ("FOUND", "READY") for r in self.records.values()):
            self.check_budget()
            choices = []
            for channel, record in self.records.items():
                if record.status not in ("FOUND", "READY") or record.replan_level > self.cfg.max_replans:
                    continue
                try:
                    proposal = None if record.status == "READY" else self.proposal(record)
                except BudgetStop:
                    raise
                except IncompleteRun as error:
                    self.replan(record, str(error))
                    continue
                point = record.clearance_center if proposal is None else proposal["point"]
                cost = math.dist(self.client.position, point)/5 + 5
                if proposal is not None:
                    cost += int(channel != self.client.current_channel)
                choices.append((cost, channel, proposal))
            if not choices:
                pending = [c for c, r in self.records.items() if r.status in ("FOUND", "READY")]
                if any(self.records[c].replan_level <= self.cfg.max_replans for c in pending):
                    continue
                raise IncompleteRun(f"重规划预算耗尽，频道{pending}仍未完成；没有将它们标为成功")
            _, channel, proposal = min(choices, key=lambda v:(v[0], v[1]))
            record = self.records[channel]
            self.event("target_selected", channel=channel)
            # 按基线连续处理选中源，直到清除或无进展触发暂存。
            while record.status in ("FOUND", "READY"):
                if record.status == "READY":
                    self.clear(record)
                    break
                if proposal is None:
                    try:
                        proposal = self.proposal(record)
                    except BudgetStop:
                        raise
                    except IncompleteRun as error:
                        self.replan(record, str(error))
                        break
                record.q2_attempted = True
                self.measure(record, tuple(proposal["point"]))
                proposal = None
                if record.no_progress_count >= self.cfg.no_progress_trigger:
                    self.replan(record, "连续测向无足够直径缩减/无信号")
                    break

    def complete(self):
        return self.cleared_count == 16 or all(r.status in TERMINAL for r in self.records.values())

    def run(self):
        outcome, reason = "incomplete", "尚未完成"
        exit_confirmed = False
        try:
            self.action("enter")
            self.entered_at = time.monotonic()
            self.trajectory.append({"to":(0.,0.), "phase":"enter", "virtual_time_s":0.})
            self.search()
            self.localize()
            if not self.complete():
                raise IncompleteRun("存在非终态频道，不能正常宣告任务完成")
            # 完成后退出，不再计入定位动作；exit前保留通信预算。
            self.phase = "finish"
            self.action("exit")
            exit_confirmed = True
            outcome, reason = "success", "全部频道终态，或确认清除16源；主动退出已确认"
        except (IncompleteRun, ClientError, KeyboardInterrupt) as error:
            reason = f"{type(error).__name__}: {error}"
            self.event("run_incomplete", reason=reason)
            # 通信异常/结果未知时不能发不同动作，也不能用exit查询结束原因。
            # 本地算法异常且会话预算尚存时，主动结束本次不完整测试，节省等待时间。
            remaining = self.client.remaining_real_time_s
            if (not isinstance(error, ClientError) and self.entered_at is not None
                    and self.client.pending_request_id is None
                    and remaining is not None and remaining > 0
                    and self.client.virtual_time_s < self.client.max_virtual_duration_s):
                try:
                    self.phase = "incomplete_exit"
                    # 不执行普通动作的预留预算检查；预留时间本来就是给exit的。
                    response = self.client.exit()
                    self._append("operations", {"method":"exit", "phase":self.phase,
                                 "accepted":True, "response":response})
                    exit_confirmed = True
                except ClientError as exit_error:
                    self.event("incomplete_exit_failed", reason=repr(exit_error))
        except Exception as error:
            reason = f"unexpected_error: {error}"
            self.event("unexpected_error", traceback=traceback.format_exc())
            # 未预期错误不继续动作，保留诊断，界面可手工中止。
        finally:
            self.end_at = time.monotonic()
            summary = self.summary(outcome, reason, exit_confirmed)
            if self.output_dir:
                (self.output_dir / "summary.json").write_text(
                    json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            self.close()
        return summary

    def summary(self, outcome, reason, exit_confirmed):
        virtual = self.client.virtual_time_s
        account = self.total_distance_m/5 + self.counts["switch"] + 5*self.counts["measure"] + 3*self.counts["optical"] + 2*self.counts["clear_success"]
        return {"outcome":outcome, "reason":reason, "all_targets_resolved":self.complete(),
            "exit_confirmed":exit_confirmed, "discovered_count":self.discovered_count,
            "cleared_count":self.cleared_count, "total_source_count":None,
            "clearance_ratio":None, "truth_note":"总数/清除比例须在演练结束后由界面真值补算，在线不读取",
            "virtual_time_s":virtual, "accounted_virtual_time_s":account,
            "average_clear_time_s":virtual/self.cleared_count if self.cleared_count and virtual is not None else None,
            "program_elapsed_s":self.end_at-self.entered_at if self.entered_at is not None else None,
            "program_time_note":"本地enter响应后至流程结束的近似耗时；正式计时以模拟器为准",
            "total_distance_m":self.total_distance_m, "counts":self.counts,
            "phase_virtual_s":self.phase_virtual_s, "next_coverage_index":self.next_coverage_index,
            "coverage_points":self.points, "scan_done":self.scan_done,
            "coverage_proof_endpoint_distances_m":[math.sqrt(r*r+self.cfg.rho**2-math.sqrt(3)*r*self.cfg.rho) for r in (1000,1800)],
            "trajectory":self.trajectory, "channels":{c:asdict(r) for c,r in self.records.items()},
            "q3_config":asdict(self.cfg), "planner_config":asdict(self.pcfg), "geometry_config":asdict(self.gcfg),
            "formal_log_note":"正式加密日志请在模拟器界面导出，并保留原文件名"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--rho", type=float, default=1200.)
    parser.add_argument("--no-insert", action="store_true")
    parser.add_argument("--insert-threshold", type=float, default=60.)
    parser.add_argument("--clear-margin", type=float, default=0.01)
    parser.add_argument("--planning-timeout", type=float, default=45.)
    parser.add_argument("--geometry-config", type=Path)
    parser.add_argument("--planner-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = Q3Config(rho=args.rho, enable_insert=not args.no_insert,
                      insert_threshold_s=args.insert_threshold, clear_margin_m=args.clear_margin,
                      planning_timeout_s=args.planning_timeout)
    gcfg = load_config(args.geometry_config) if args.geometry_config else GeometryConfig()
    pcfg = PlannerConfig(**json.loads(args.planner_config.read_text(encoding="utf-8-sig"))) if args.planner_config else PlannerConfig()
    out = args.output_dir or Path("runs") / datetime.now().strftime("q3_%Y%m%d_%H%M%S_%f")
    client = SimulatorClient(args.robot_id, args.base_url)
    runner = Q3Runner(client, config=config, planner_config=pcfg, geometry_config=gcfg, output_dir=out)
    logging.basicConfig(filename=out/"client.log", level=logging.INFO,
                        encoding="utf-8", format="%(asctime)s %(levelname)s %(message)s")
    print(f"问题三基线启动，输出目录：{out}", flush=True)
    summary = runner.run()
    print(json.dumps({k:summary[k] for k in ("outcome","reason","discovered_count","cleared_count",
                     "virtual_time_s","average_clear_time_s","program_elapsed_s")}, ensure_ascii=False, indent=2))
    return 0 if summary["outcome"] == "success" else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
