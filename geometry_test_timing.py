"""仅供测试的计时包装；不写入或改变geometry.py的算法语句。

函数级计时为包含子调用的墙钟时间。_measure内部通过AST在原语句之间
插入时间标记，避免逐行追踪放大Fraction循环开销。找不到边界时明确报错，
修改直径实现后应同步检查本文件的分段边界，禁止静默沿用错误范围。
"""
import ast
from contextlib import ExitStack
import inspect
import textwrap
from time import perf_counter
from unittest.mock import patch

import geometry_V7 as g

STAGES = ["config", "build_halfplanes", "classify", "projection_bounds", "linprog",
          "vertices", "measure", "measure_setup", "diameter", "coverage",
          "degeneracy", "sort_vertices", "residual_and_output"]
COLUMNS = ["stage_" + name + "_ms" for name in STAGES] + ["stage_other_ms"]
NOTE = ("stage_*_ms单位毫秒；null表示未执行。classify包含projection_bounds和linprog；"
        "measure包含setup、diameter、coverage、degeneracy、sort_vertices、residual_and_output，"
        "父子时间不能重复相加。vertices为边界两两求交+全部约束筛选+去重。"
        "diameter为最远点对搜索、距离平方和最终距离计算；coverage为圆心/半径/覆盖检查；"
        "other为总耗时减config/build_halfplanes/classify/vertices/measure，含归一化、"
        "有理数转换、调度及计时开销。所有时间含少量插桩开销，不含写报告和测试断言。"
        "直径比较应同时参考vertex_count，检测点数量不是直径模块的直接输入规模。")


def _instrument_measure(mark):
    tree = ast.parse(textwrap.dedent(inspect.getsource(g._measure)))
    function = tree.body[0]
    starts = {"squared": "diameter", "midpoint": "coverage", "status": "degeneracy",
            "residual": "residual_and_output"}
    found = set()
    body = [ast.parse("__timing_mark('measure_setup')").body[0]]
    for statement in function.body:
        key = None
        if isinstance(statement, ast.FunctionDef):
            key = statement.name
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if len(targets) == 1 and isinstance(targets[0], ast.Name):
                key = targets[0].id
        stage = starts.get(key)
        if isinstance(statement, ast.If) and "cfg.sort_vertices_ccw" in ast.unparse(statement.test):
            stage = "sort_vertices"
        if stage:
            found.add(stage)
            body.append(ast.parse(f"__timing_mark('{stage}')").body[0])
        body.append(statement)
    expected = set(starts.values()) | {"sort_vertices"}
    if found != expected:
        raise RuntimeError("_measure结构已变化，请更新geometry_test_timing.py的分段边界")
    function.body = body
    ast.fix_missing_locations(tree)
    namespace = dict(g.__dict__, __timing_mark=mark)
    exec(compile(tree, "<geometry_test_timing:_measure>", "exec"), namespace)
    return namespace["_measure"]


class StageTimer:
    def __init__(self):
        self.times = {}
        self.calls = {}
        self.current = None
        self.tick = None

    def add(self, name, seconds):
        self.times[name] = self.times.get(name, 0.0) + seconds * 1000
        self.calls[name] = self.calls.get(name, 0) + 1

    def mark(self, name):
        now = perf_counter()
        if self.current is not None:
            self.add(self.current, now - self.tick)
        self.current, self.tick = name, now

    def wrap(self, name, function):
        def timed(*args, **kwargs):
            start = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                if name == "measure":
                    self.mark(None)
                self.add(name, perf_counter() - start)
        return timed

    def __enter__(self):
        self.stack = ExitStack()
        measure = _instrument_measure(self.mark)
        for attr, stage in [("load_config", "config"), ("build_halfplanes", "build_halfplanes"),
                            ("_classify", "classify"), ("_projection_bounds", "projection_bounds"),
                            ("linprog", "linprog"), ("_vertices", "vertices"), ("_measure", "measure")]:
            function = measure if attr == "_measure" else getattr(g, attr)
            if function is not None:
                self.stack.enter_context(patch.object(g, attr, self.wrap(stage, function)))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def report(self, total_ms):
        result = {"stage_" + name + "_ms": self.times.get(name) for name in STAGES}
        result["stage_other_ms"] = total_ms - sum(self.times.get(n, 0) for n in
                                                 ("config", "build_halfplanes", "classify", "vertices", "measure"))
        result["stage_call_counts"] = self.calls.copy()
        return result
