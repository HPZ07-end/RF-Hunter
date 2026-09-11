"""独立极端案例与规模测试，不修改geometry.py或原test_geometry.py。

运行：python test_geometry_extreme.py
自定规模：python test_geometry_extreme.py --sizes 10 50 100 300 1000 --timeout 60
重复计时：python test_geometry_extreme.py --sizes 10 100 --repeat 3

板块一：构造可复现案例与解析期望。
板块二：独立子进程计时、超时控制和结果校核。
板块三：逐案例保存cases.csv/report.json/unittest.log/README.txt。

PASS=达到明确期望；WARN=病态输入被报告数值异常；FAIL=答案或校核不符；
TIMEOUT=超出单案例墙钟预算，不能视为通过或算法错误；ERROR=非预期异常。
默认所有案例使用当前Python环境的真实后端，不模拟或关闭SciPy。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

import numpy as np
import geometry_V7 as g
from geometry_test_timing import StageTimer, COLUMNS as TIMING_COLUMNS, NOTE as TIMING_NOTE

SEED = 20260911


# ==================== 一、极端案例构造 ====================
def make_cases(sizes):
    cases = []

    def add(name, group, data, *, kind="halfplanes", note="", **expected):
        cases.append(dict(case_id=name, group=group, input_kind=kind, input=data,
                          note=note, expected=expected))

    # 顶点(-1,0),(1,0),(0,h)：当h接近1时直径恒为2，圆心(0,0)。
    for label, height, covers in (("inside", 1 - 5e-7, True), ("on", 1.0, True),
                                  ("outside_within_tolerance", 1 + 5e-7, True),
                                  ("outside_tolerance", 1 + 5e-5, False)):
        add(f"circle_boundary_{label}", "coverage_boundary",
            [(0, -1, 0), (height, 1, height), (-height, 1, height)],
            status="POLYGON", diameter=2.0, covers=covers,
            exact_covers=height <= 1, note="解析三角形，覆盖容差内外对照；非测向输入")

    for width in (1e-3, 1e-7, 1e-10, 1e-12):
        add(f"thin_rectangle_{width:g}", "degeneracy",
            [(1, 0, 1), (-1, 0, 0), (0, 1, width), (0, -1, 0)],
            status="POLYGON", diameter=math.hypot(1, width), allow_numerical_issue=True,
            note="极薄但有面积，不能静默压成线段")
    add("exact_segment", "degeneracy", [(1, 0, 1), (-1, 0, 1), (0, 1, 0), (0, -1, 0)],
        status="SEGMENT", diameter=2.0)
    add("exact_point", "degeneracy", [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
        status="POINT", diameter=0.0)
    for gap in (1e-3, 1e-8, 1e-12):
        add(f"almost_infeasible_{gap:g}", "feasibility_boundary", [(1, 0, 0), (-1, 0, -gap)],
            status="EMPTY", allow_numerical_issue=True,
            note="x<=0且x>=gap，解析空集；LP容差冲突应报告数值异常")
    add("parallel_strip", "parallel", [(0, 1, 1), (0, -1, 0)], status="UNBOUNDED")
    for slope in (1e-6, 1e-10, 1e-12):
        add(f"near_parallel_{slope:g}", "parallel", [(-1, 0, 0), (0, -1, 0), (slope, 1, 1)],
            status="POLYGON", diameter=math.hypot(1 / slope, 1), allow_numerical_issue=True,
            note="解析顶点含(1/slope,0)，微小行列式不能直接跳过")

    base = [(-800, 0, 0), (0, -800, 90)]
    for shift in (0, 1e6, 1e9):
        add(f"translation_{shift:g}", "coordinate_scale", [(x + shift, y - shift, t) for x, y, t in base],
            kind="observations", status="POLYGON", diameter=39.50834065898743,
            diameter_atol=1e-4 if shift else 1e-8, allow_numerical_issue=bool(shift),
            note="整体平移不改变直径；大坐标是数值压力数据，不是题目实际位置")
    add("angle_wrap", "bearing_boundary", [(-800, 0, 360), (0, -800, -270)],
        kind="observations", status="POLYGON", diameter=39.50834065898743)
    for epsilon in (1e-8, 89.999999):
        add(f"configured_error_{epsilon:.9g}", "bearing_boundary", base,
            kind="observations", status="UNBOUNDED" if epsilon > 45 else "POLYGON",
            error_deg=epsilon, allow_numerical_issue=True,
            note="仅测试可配置误差边界；非默认±1度。近90度时两角域存在共同东北无界方向")
    # 真源为(0,0)，误差恰为±1°，检验真实点落在角域边界时的浮点行为。
    add("bearing_error_exact_limit", "bearing_boundary", [(-800, 0, 1), (0, -800, 89)],
        kind="observations", status="POLYGON", source=[0, 0], allow_numerical_issue=True)

    rng = random.Random(SEED)
    observations = []
    for i in range(max(sizes)):
        # 前三点分散在三个方向，此后使用不同角度/半径，形成真正不同的检测点。
        alpha = math.radians((i % 3) * 120 + rng.uniform(-15, 15))
        radius = rng.uniform(300, 1400)
        x, y = radius * math.cos(alpha), radius * math.sin(alpha)
        theta = math.degrees(math.atan2(-y, -x)) + rng.uniform(-0.8, 0.8)
        observations.append((x, y, theta))
    for n in sizes:
        add(f"distinct_points_{n}", "distinct_scale", observations[:n], kind="observations",
            status="POLYGON", source=[0, 0], note="不同检测点，递增数据集为同一固定种子序列的前缀")
    for n in (1000, 10000):
        add(f"repeated_observations_{n}", "duplicate_scale", (base * (n // 2)), kind="observations",
            status="POLYGON", diameter=39.50834065898743,
            note="仅两个不同检测点，测量重复；用于检验去重，不能等同于n个不同点的压力")
    return cases


# ==================== 二、求解计时与独立校核 ====================
def check_result(case, result, cfg):
    expected = case["expected"]
    if result.status == "NUMERICAL_ISSUE" and expected.get("allow_numerical_issue"):
        return "WARN", ["未给出几何答案：病态数据触发数值异常，请查看diagnostics"], None
    problems = []
    if result.status != expected["status"]:
        problems.append(f"期望状态{expected['status']}，实际{result.status}")
    if "diameter" in expected and (result.diameter is None or not math.isclose(
            result.diameter, expected["diameter"], rel_tol=1e-9, abs_tol=expected.get("diameter_atol", 1e-8))):
        problems.append("直径与解析值不符")
    if "covers" in expected and result.same_diameter_covers != expected["covers"]:
        problems.append("覆盖判断与期望不符")
    if "exact_covers" in expected and result.diagnostics.get("coefficient_exact_covers") != expected["exact_covers"]:
        problems.append("精确系数覆盖判断与解析期望不符")
    if result.status == "EMPTY" and result.diameter is not None:
        problems.append("空集直径不应赋值")
    if result.status == "UNBOUNDED" and result.diameter != math.inf:
        problems.append("无界区域应返回无限直径")
    if result.status in ("EMPTY", "UNBOUNDED") and result.same_diameter_covers is not None:
        problems.append("空集/无界的有限圆覆盖判断应为None")
    residual = None
    if result.vertices:
        rows = (g.build_halfplanes(case["input"], cfg.bearing_error_deg)
                if case["input_kind"] == "observations" else np.asarray(case["input"], dtype=float))
        rows = rows / np.hypot(rows[:, 0], rows[:, 1])[:, None]
        vertices = np.asarray(result.vertices)
        scale = max(1.0, float(np.max(np.abs(vertices))), float(np.max(np.abs(rows[:, 2]))))
        tol = cfg.feasibility_abs_tol_m + cfg.feasibility_rel_tol * scale
        residual = float(np.max(rows[:, :2] @ vertices.T - rows[:, 2, None]))
        if residual > tol:
            problems.append("顶点不满足全部原始约束")
        if "source" in expected and np.max(rows[:, :2] @ expected["source"] - rows[:, 2]) > tol:
            problems.append("构造的真实源未通过约束检查")
        # 与求解器的Fraction距离平方比较相独立，用浮点欧氏距离重算顶点直径。
        measured = max((math.dist(a, b) for a in result.vertices for b in result.vertices), default=0)
        if not math.isclose(measured, result.diameter, rel_tol=1e-9, abs_tol=tol):
            problems.append("直径不等于输出顶点的最大距离")
    return ("FAIL" if problems else "PASS"), problems, residual


def worker(case, connection):
    """顶层函数供Windows spawn调用；只计时求解，不计启动/校核/写文件。"""
    try:
        cfg = g.load_config()
        if "error_deg" in case["expected"]:
            cfg = replace(cfg, bearing_error_deg=case["expected"]["error_deg"])
        with StageTimer() as timer:
            start = perf_counter()
            result = (g.solve_q1(case["input"], config=cfg) if case["input_kind"] == "observations"
                      else g.solve_halfplanes(case["input"], config=cfg))
            elapsed_ms = (perf_counter() - start) * 1000
        outcome, messages, residual = check_result(case, result, cfg)
        connection.send(dict(result=result.to_dict(), elapsed_ms=elapsed_ms,
                             test_outcome=outcome, checks=messages, audit_max_residual_m=residual,
                             effective_config=asdict(cfg), **timer.report(elapsed_ms)))
    except Exception:
        connection.send(dict(test_outcome="ERROR", traceback=traceback.format_exc()))
    finally:
        connection.close()


def run_case(case, timeout):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(case, sender))
    start = perf_counter()
    process.start()
    sender.close()
    try:
        if receiver.poll(timeout):
            try:
                answer = receiver.recv()
            except EOFError:
                answer = dict(test_outcome="ERROR", checks=["子进程退出，未返回完整结果"])
        else:
            answer = dict(test_outcome="TIMEOUT", status="TIMEOUT", elapsed_ms=None,
                          checks=[f"含进程启动/校核的单案例墙钟时间超过{timeout}秒；求解耗时未知"])
    finally:
        receiver.close()
        process.join(timeout=0.2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
    answer["case_wall_ms"] = (perf_counter() - start) * 1000
    result = answer.pop("result", {})
    answer.update(result)
    return answer


# ==================== 三、报告生成与命令行 ====================
def json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def save_report(folder, metadata, records):
    counts = {name: sum(r["test_outcome"] == name for r in records)
              for name in ("PASS", "WARN", "FAIL", "TIMEOUT", "ERROR")}
    report = json_value(dict(metadata=metadata, summary=dict(case_count=len(records), **counts), cases=records))
    # 每个案例完成后更新；临时文件替换避免中断留下半截JSON。
    temp = folder / "report.json.tmp"
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(folder / "report.json")
    columns = ["case_id", "group", "repeat", "test_outcome", "input_kind", "observation_count",
               "unique_detection_point_count", "constraint_count_input", "constraint_count_solver",
               "elapsed_ms", "case_wall_ms", "status", "vertex_count", "diameter", "circle_radius",
               "max_vertex_distance", "coverage_excess", "same_diameter_covers", "backend",
               "audit_max_residual_m", "checks"]
    columns += TIMING_COLUMNS
    with (folder / "cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in report["cases"]:
            writer.writerow({**record, "checks": " | ".join(record.get("checks", []))})
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 30, 100, 200, 500])
    parser.add_argument("--timeout", type=float, default=30, help="每个案例墙钟预算，秒（包括启动/校核）")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if any(n < 3 for n in args.sizes) or not math.isfinite(args.timeout) or args.timeout <= 0 or args.repeat < 1:
        parser.error("sizes必须>=3，timeout必须为正有限数，repeat必须>=1")
    sizes = sorted(set(args.sizes))
    folder = args.output_dir or Path(__file__).parent / "results" / datetime.now().strftime("geometry_extreme_%Y%m%d_%H%M%S_%f")
    if folder.exists():
        parser.error("输出目录已存在，不能覆盖历史实验")
    folder.mkdir(parents=True)
    metadata = dict(started_at=datetime.now().astimezone().isoformat(), python_executable=sys.executable,
                    python_version=sys.version, seed=SEED, sizes=sizes, repeats=args.repeat,
                    timeout_s=args.timeout, config=asdict(g.load_config()), packages={},
                    completed=False, planned_case_count=0,
                    timing_note=TIMING_NOTE,
                    source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ("geometry_V7.py", "config.h", "test_geometry.py", "test_geometry_extreme.py")})
    for package in ("numpy", "scipy"):
        try:
            metadata["packages"][package] = version(package)
        except PackageNotFoundError:
            metadata["packages"][package] = None
    (folder / "README.txt").write_text(
        "问题一独立极端测试报告\n" + TIMING_NOTE + "\n\n"
        "cases.csv逐案例汇总；report.json保留输入、期望、配置、结果、诊断、代码哈希；\n"
        "unittest.log为本独立运行器的逐案例日志（不是原unittest测试套件）。\n"
        "PASS达到期望；WARN为允许的病态数值异常，未得答案；FAIL为不符；ERROR为异常；TIMEOUT未完成。\n"
        "所有案例使用当前环境真实求解后端，不模拟SciPy；不修改原测试/求解脚本。\n"
        "observation_count为观测条数；unique_detection_point_count为不同坐标数；\n"
        "constraint_count_input为输入约束数（测向为2n）；constraint_count_solver为归一化去重后数。\n"
        "直接半平面案例没有检测点，检测点数为空；无返回结果时顶点数等未知，不填0。\n"
        "elapsed_ms单位毫秒，只计求解；case_wall_ms含进程启动、校核和通信；每个进程无预热。\n"
        "TIMEOUT表示墙钟预算不足，elapsed_ms为空，不能当作测得的求解时间或通过。\n"
        "直径/半径/最大距离/超出量单位米；Infinity为无界；null/空白为不适用或未知。\n"
        "coverage_excess保留原始越界量，JSON中同时查看coefficient_exact_covers和容差。\n"
        "重复观测压力与不同检测点压力分组，10000条重复观测并不等于10000个不同点。\n"
        "随机源(0,0)，固定种子20260911，不同点规模采用同一序列前缀，检查直径不增加。\n"
        "几何临界/超大坐标/非默认误差为数值压力构造，并非题目提供的正常观测数据。\n"
        "--sizes自定义不同检测点规模；--timeout控制每例预算；--repeat增加重复计时。\n"
        "每例结束即保存，不覆盖历史结果；metadata.completed=false表示运行未完成。\n",
        encoding="utf-8")
    cases = make_cases(sizes)
    metadata["planned_case_count"] = len(cases) * args.repeat
    records = []
    previous_diameter = {}
    save_report(folder, metadata, records)
    with (folder / "unittest.log").open("w", encoding="utf-8", buffering=1) as log:
        for case in cases:
            for repetition in range(1, args.repeat + 1):
                print(f"Running {case['case_id']} [{repetition}/{args.repeat}]", flush=True)
                answer = run_case(case, args.timeout)
                observations = case["input_kind"] == "observations"
                record = dict(case, repeat=repetition,
                              observation_count=len(case["input"]) if observations else None,
                              unique_detection_point_count=len({tuple(v[:2]) for v in case["input"]}) if observations else None,
                              constraint_count_input=len(case["input"]) * (2 if observations else 1), **answer)
                diag = record.get("diagnostics", {})
                record["constraint_count_solver"] = diag.get("constraint_count")
                record["backend"] = diag.get("classification_backend")
                record["vertex_count"] = len(record["vertices"]) if "vertices" in record else None
                if case["group"] == "distinct_scale" and record["test_outcome"] == "PASS":
                    prior = previous_diameter.get(repetition)
                    if prior is not None and record["diameter"] > prior + 1e-7:
                        record["test_outcome"] = "FAIL"
                        record.setdefault("checks", []).append("增加观测后直径增大")
                    previous_diameter[repetition] = record["diameter"]
                records.append(record)
                counts = save_report(folder, metadata, records)
                line = f"{case['case_id']} repeat={repetition} {record['test_outcome']} status={record.get('status')} elapsed_ms={record.get('elapsed_ms')} diameter={record.get('diameter')}"
                print(line, flush=True)
                log.write(line + "\n" + "\n".join(record.get("checks", [])) + "\n")
                if "traceback" in record:
                    log.write(record["traceback"] + "\n")
        metadata["completed"] = True
        counts = save_report(folder, metadata, records)
        log.write(json.dumps(counts) + "\n")
    print(f"Reports: {folder.resolve()}\n{counts}")
    return 1 if counts["FAIL"] or counts["ERROR"] or counts["TIMEOUT"] else 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
