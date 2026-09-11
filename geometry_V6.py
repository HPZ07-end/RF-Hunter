"""问题一：纯测向角域交集、区域直径及同直径圆盘覆盖判定。

执行顺序与代码板块（文件中的板块编号与此处一致）
================================================
一、配置与数据结构：load_config -> GeometryConfig / GeometryResult。
二、输入转半平面：solve_q1 -> build_halfplanes（每个观测生成两行）。
三、状态判断：solve_halfplanes -> _classify（可行性 + 四个坐标极值）。
四、顶点枚举：_vertices（两两求交 -> 检查全部约束 -> 去重）。
五、直径及覆盖：_measure（退化判定 -> 最远点对 -> 中点 -> 覆盖）。
六、公共入口与命令行：solve_q1 / solve_halfplanes / _main。

用法：
    result = solve_q1([(-800, 0, 0), (0, -800, 90)])
    print(result.status, result.diameter, result.same_diameter_covers)
    # 或 solve_q1([(-800, 0), (0, -800)], [0, 90])
    # 文档两个算例：python geometry.py --demo

重要约定：
* 输入为同一源的(x, y, 示向度)，米/度；误差默认±1°，不是随机标准差。
* 只求角域交集，不加目标圆、接收圆、近距离排除圆或人为包围矩形。
* 错误输入/配置抛ValueError；几何空集不是异常；数值失败单独返回。
* Fraction校核针对浮点三角函数生成的系数，不是对真实角度的精确证明。
* 临界结果在diagnostics标注；本模块不执行移动、检测或清除动作。
"""

#改动函数_verticals

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass, field, fields, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
try:
    from scipy.optimize import linprog
except ImportError:
    # 未安装SciPy时，已有的二维有理数消元可独立完成状态判断。
    # 不伪造LP结果；diagnostics中明确记录所用后端。
    linprog = None


# ==================== 一、配置与数据结构 ====================
@dataclass(frozen=True)
class GeometryConfig:
    """字段名对应config.h中的RF_Q1_宏（转成小写）。"""

    min_observations: int = 1
    bearing_error_deg: float = 1.0
    region_model: str = "bearing_halfplanes_only"
    lp_method: str = "highs"
    lp_presolve: int = 1
    lp_primal_feasibility_tol: float = 1e-9
    lp_dual_feasibility_tol: float = 1e-9
    feasibility_abs_tol_m: float = 1e-6
    feasibility_rel_tol: float = 1e-12
    vertex_merge_abs_tol_m: float = 1e-7
    vertex_merge_rel_tol: float = 1e-12
    degeneracy_abs_tol_m: float = 1e-7
    degeneracy_rel_tol: float = 1e-12
    coverage_abs_tol_m: float = 1e-6
    coverage_rel_tol: float = 1e-12
    near_parallel_det_tol: float = 1e-10
    decimal_precision: int = 60
    sort_vertices_ccw: int = 1
    return_diagnostics: int = 1
    display_decimal_places: int = 8

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in {"region_model", "lp_method"}:
                if not isinstance(value, str):
                    raise ValueError(f"{item.name}必须为字符串")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{item.name}必须为数值字面量")
            if not math.isfinite(value):
                raise ValueError(f"{item.name}必须有限")
            if "tol" in item.name and value <= 0:
                raise ValueError(f"{item.name}必须大于0")
        for name in ("min_observations", "decimal_precision", "display_decimal_places",
                     "lp_presolve", "sort_vertices_ccw", "return_diagnostics"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name}必须为整数")
        if self.min_observations != 1:
            raise ValueError("问题一允许任意n>=1，min_observations必须为1")
        if not 0 < self.bearing_error_deg < 90:
            raise ValueError("必须满足0 < bearing_error_deg < 90")
        if self.region_model != "bearing_halfplanes_only":
            raise ValueError("仅实现纯测向半平面交集模型")
        if self.lp_method not in {"highs", "highs-ds", "highs-ipm"}:
            raise ValueError("lp_method必须使用HiGHS系列")
        for name in ("lp_presolve", "sort_vertices_ccw", "return_diagnostics"):
            if getattr(self, name) not in (0, 1):
                raise ValueError(f"{name}必须为0或1")
        if min(self.lp_primal_feasibility_tol, self.lp_dual_feasibility_tol) < 1e-10:
            raise ValueError("HiGHS可行性容差不得小于1e-10")
        if not 30 <= self.decimal_precision <= 500:
            raise ValueError("decimal_precision应在30到500之间")
        if not 0 <= self.display_decimal_places <= 16:
            raise ValueError("display_decimal_places应在0到16之间")
        if self.near_parallel_det_tol >= 1:
            raise ValueError("near_parallel_det_tol必须小于1")


def load_config(path: str | Path | None = None) -> GeometryConfig:
    """严格读取同目录config.h；只接受字面量，不执行宏或Python代码。"""
    path = Path(path) if path is not None else Path(__file__).with_name("config.h")
    text = re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8-sig"), flags=re.S)
    values = {}
    allowed = {f.name for f in fields(GeometryConfig)}
    for line in text.splitlines():
        line = line.split("//", 1)[0].strip()
        if not re.match(r"#\s*define\s+RF_Q1_", line):
            continue
        match = re.fullmatch(r"#\s*define\s+RF_Q1_(\w+)\s+(.+)", line)
        if not match:
            raise ValueError(f"无法解析配置行：{line}")
        name, literal = match.groups()
        name = name.lower()
        if name not in allowed or name in values:
            raise ValueError(f"未知或重复配置：{name}")
        try:
            values[name] = ast.literal_eval(literal)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"配置{name}必须是单个字面量") from exc
    missing = allowed - values.keys()
    if missing:
        raise ValueError(f"缺少配置：{sorted(missing)}")
    return GeometryConfig(**values)


@dataclass
class GeometryResult:
    status: str
    vertices: list[tuple[float, float]] = field(default_factory=list)
    diameter: float | None = None
    diameter_pair: tuple[tuple[float, float], tuple[float, float]] | None = None
    circle_center: tuple[float, float] | None = None
    circle_radius: float | None = None
    max_vertex_distance: float | None = None
    coverage_excess: float | None = None
    same_diameter_covers: bool | None = None
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """保留完整精度；无界直径为float('inf')，严格JSON需自行编码。"""
        return asdict(self)


def _tol(cfg, kind, scale):
    return (getattr(cfg, kind + "_abs_tol_m")
            + getattr(cfg, kind + "_rel_tol") * max(1.0, scale))


# ==================== 二、输入转半平面 ====================
def _matrix(data: Iterable, columns: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(list(data), dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label}必须是{columns}列的有限数值") from exc
    if array.size == 0:
        raise ValueError(f"{label}不能为空")
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f"{label}必须是{columns}列的有限数值")
    return array


def build_halfplanes(observations: Iterable, error_deg: float = 1.0) -> np.ndarray:
    """输入n行(a,b,theta)，输出2n行(A,B,C)，表示Ax+By<=C。"""
    if not isinstance(error_deg, (int, float)) or not math.isfinite(error_deg) or not 0 < error_deg < 90:
        raise ValueError("误差角必须有限且位于(0,90)度")
    obs = _matrix(observations, 3, "observations")
    rows = []
    for a, b, theta in obs:
        # 先归一化中心角；不是用角度区间比较，因此可安全跨越0度。
        theta = float(theta) % 360.0
        lower = math.radians(theta - error_deg)
        upper = math.radians(theta + error_deg)
        # 下边界左侧，上边界右侧。第二行符号不能与第一行写成相同。
        for A, B in ((math.sin(lower), -math.cos(lower)),
                     (-math.sin(upper), math.cos(upper))):
            C = math.fsum((A * float(a), B * float(b)))
            if not math.isfinite(C):
                raise ValueError("坐标尺度过大，无法构造有限约束")
            rows.append((A, B, C))
    return np.asarray(rows, dtype=float)


# ==================== 三、可行性与有界性 ====================
def _projection_bounds(rows, swap=False):
    """精确有理数Fourier-Motzkin消元，用于校核LP状态，不引入包围盒。

    消去y得到x的一维约束；swap=True则相反。O(m²)。
    精确性限于输入浮点系数的有理数表示，不涵盖三角函数建模误差。
    返回(可行, 下界或None, 上界或None)。
    """
    lower_y, upper_y, one_dim = [], [], []
    for a, b, c in rows:
        if swap:
            a, b = b, a
        if b < 0:
            lower_y.append((a, b, c))
        elif b > 0:
            upper_y.append((a, b, c))
        else:
            one_dim.append((a, c))
    for a1, b1, c1 in lower_y:
        for a2, b2, c2 in upper_y:
            one_dim.append((b2 * a1 - b1 * a2, b2 * c1 - b1 * c2))
    lo = hi = None
    for a, c in one_dim:
        if a == 0:
            if c < 0:
                return False, None, None
        elif a > 0:
            bound = c / a
            hi = bound if hi is None else min(hi, bound)
        else:
            bound = c / a
            lo = bound if lo is None else max(lo, bound)
    return not (lo is not None and hi is not None and lo > hi), lo, hi


def _classify(rows, rational, cfg, diag):
    # 独立消元校核防止LP容差把薄区域/近乎平行误判为空集或无界。
    xbounds = _projection_bounds(rational)
    ybounds = _projection_bounds(rational, swap=True)
    if not xbounds[0] or not ybounds[0]:
        exact_state = "EMPTY"
    elif any(v is None for v in (*xbounds[1:], *ybounds[1:])):
        exact_state = "UNBOUNDED"
    else:
        exact_state = "BOUNDED"
    diag["coefficient_exact_classification"] = exact_state
    diag["lp_statuses"] = []
    if linprog is None:
        diag["classification_backend"] = "fraction_fourier_motzkin"
        return exact_state
    diag["classification_backend"] = "scipy_highs_with_fraction_check"
    options = {"presolve": bool(cfg.lp_presolve),
               "primal_feasibility_tolerance": cfg.lp_primal_feasibility_tol,
               "dual_feasibility_tolerance": cfg.lp_dual_feasibility_tol}
    lp_state = "BOUNDED"
    for objective in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        result = linprog(objective, A_ub=rows[:, :2], b_ub=rows[:, 2],
                         bounds=[(None, None), (None, None)],
                         method=cfg.lp_method, options=options)
        diag["lp_statuses"].append({"objective": objective, "status": result.status,
                                    "message": result.message})
        if result.status not in (0, 2, 3):
            diag["warnings"].append("线性规划求解失败，不能视为空集或无界")
            return "NUMERICAL_ISSUE"
        if result.status == 2:
            lp_state = "EMPTY"
            break
        if result.status == 3:
            lp_state = "UNBOUNDED"
            break
        scale = max(1.0, float(np.max(np.abs(rows[:, 2]))), float(np.max(np.abs(result.x))))
        residual = max(math.fsum((a * result.x[0], b * result.x[1], -c)) for a, b, c in rows)
        if residual > _tol(cfg, "feasibility", scale):
            diag["warnings"].append("线性规划返回点未通过约束残差检查")
            return "NUMERICAL_ISSUE"
    if lp_state != exact_state:
        diag["warnings"].append("LP与有理数消元状态不一致：输入可能近退化或病态")
        return "NUMERICAL_ISSUE"
    return lp_state


# ==================== 四、边界求交、全约束筛选、去重 ====================
def _vertices(rows, rational, cfg, diag):
    """增量半平面裁剪求多边形顶点。

    与原版两两枚举 O(m³) 相比：
    * 每次裁剪只处理当前多边形与新约束的边，单次 O(h)，h 是当前顶点数；
    * 总复杂度典型 O(m·h)，最坏 O(m²)，显著低于 O(m³)。

    为了从有界区域开始裁剪，先用输入约束的量级构造一个保守包围盒作为初始容器。
    包围盒的边在后续裁剪中会被真实约束完全切除，因此不影响最终结果。
    """
    m = len(rows)
    if m == 0:
        raise ArithmeticError("空约束列表不应进入 _vertices")

    # 归一化后 |a|,|b| <= 1；用 2*max|C|+1 作为半边长，对一般测向输入足够安全。
    xb = _projection_bounds(rational)
    yb = _projection_bounds(rational, swap=True)
    if not xb[0] or not yb[0]:
        raise ArithmeticError("增量裁剪前置 BOUNDED 判定失效")
    xlo, xhi = xb[1], xb[2]
    ylo, yhi = yb[1], yb[2]
    if xlo is None or xhi is None or ylo is None or yhi is None:
        raise ArithmeticError("增量裁剪前置 BOUNDED 判定失效")
    M_x = max(abs(xlo), abs(xhi)) + 1
    M_y = max(abs(ylo), abs(yhi)) + 1
    polygon = [(-M_x, -M_y), (M_x, -M_y), (M_x, M_y), (-M_x, M_y)]

    # 逐个半平面裁剪
    for k in range(m):
        a, b, c = rational[k]
        if a == 0 and b == 0:
            raise ArithmeticError("常量约束不应进入增量裁剪")
        new_polygon = []
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            f1 = a * x1 + b * y1 - c
            f2 = a * x2 + b * y2 - c
            in1 = f1 <= 0
            in2 = f2 <= 0
            if in1:
                new_polygon.append((x1, y1))
            if in1 != in2:
                # 边的两端在约束两侧，必有唯一切点；denom = f2 - f1 不为零。
                denom = f2 - f1
                t = -f1 / denom
                xi = x1 + t * (x2 - x1)
                yi = y1 + t * (y2 - y1)
                new_polygon.append((xi, yi))
        polygon = new_polygon
        if not polygon:
            raise ArithmeticError("增量裁剪得到空集，与前置 BOUNDED 判定不一致")

    # 顶点在边界上可能被多次加入，需要去重
    seen = set()
    unique_polygon = []
    for p in polygon:
        if p not in seen:
            seen.add(p)
            unique_polygon.append(p)

    # 与原版等价的 sanity check：所有输出顶点必须满足全部约束
    for x, y in unique_polygon:
        for a, b, c in rational:
            if a * x + b * y > c:
                raise ArithmeticError("增量裁剪输出顶点未通过约束检查")

    # 与原版一致的排序、类型转换、输出格式
    points = sorted(unique_polygon)
    if not points:
        raise ArithmeticError("已判定非空有界，却未找到顶点")
    floats = [(float(x), float(y)) for x, y in points]
    if not all(math.isfinite(v) for point in floats for v in point):
        raise ArithmeticError("顶点超出浮点表示范围")
    if len(set(floats)) != len(points):
        raise ArithmeticError("不同顶点转为浮点后重合，无法可靠输出")

    # 与原版一致的诊断：近邻顶点对
    scale = max(1.0, *(abs(v) for point in floats for v in point))
    threshold = _tol(cfg, "vertex_merge", scale)
    close = [math.dist(a, b) for a, b in combinations(floats, 2)
             if math.dist(a, b) <= threshold]
    diag["close_distinct_vertex_pairs"] = len(close)
    diag["max_merged_vertex_distance"] = 0.0
    if close:
        diag["warnings"].append("存在容差内但精确不同的顶点，已保留以避免改变拓扑")

    # 与原版一致的诊断：近乎平行边界对数（仅统计，不依赖枚举求交）
    near_count = 0
    for p, q in combinations(range(m), 2):
        a, b, _ = rational[p]
        d, e, _ = rational[q]
        determinant = a * e - d * b
        if determinant != 0 and abs(determinant) <= cfg.near_parallel_det_tol:
            near_count += 1
    diag["near_parallel_pairs"] = near_count
    if near_count:
        diag["warnings"].append("近乎平行边界已高精度重算并用有理数校核；未直接丢弃")

    return points, floats


# ==================== 五、退化、直径、同直径圆覆盖 ====================
def _measure(exact, vertices, rows, cfg, diag):
    if not vertices:
        raise ArithmeticError("已判定非空有界，却未找到顶点")
    if len(vertices) == 1:
        v = vertices[0]
        return GeometryResult("POINT", vertices, 0.0, (v, v), v, 0.0, 0.0, 0.0, True, diag)

    # 用距离平方比较；有理数避免平方溢出和并列最远点的错误选择。
    def squared(pair):
        i, j = pair
        return sum((a - b) ** 2 for a, b in zip(exact[i], exact[j]))
    i, j = max(combinations(range(len(exact)), 2), key=squared)
    A, B = exact[i], exact[j]
    distance_squared = squared((i, j))
    diameter = math.dist(vertices[i], vertices[j])
    midpoint = tuple((a + b) / 2 for a, b in zip(A, B))
    # 明确固定为二维坐标，避免类型检查器将生成器推断为不定长 tuple。
    center: tuple[float, float] = (float(midpoint[0]), float(midpoint[1]))
    if not math.isfinite(diameter):
        raise ArithmeticError("直径超出浮点表示范围")
    radius = diameter / 2
    rmax = max(math.dist(center, v) for v in vertices)
    # 同时保留精确系数模型下的覆盖结论，区分数学越界与容差内的近似接受。
    exact_covers = all(sum((a - b) ** 2 for a, b in zip(v, midpoint)) <= distance_squared / 4
                       for v in exact)
    scale = max(1.0, diameter, rmax, *(abs(v) for point in vertices for v in point),
                float(np.max(np.abs(rows[:, 2]))))
    cover_tol = _tol(cfg, "coverage", scale)
    excess = rmax - radius
    diag["coverage_tolerance_m"] = cover_tol
    diag["coefficient_exact_covers"] = exact_covers
    diag["coverage_numerically_borderline"] = abs(excess) <= cover_tol
    if not exact_covers and excess <= cover_tol:
        diag["warnings"].append("精确系数模型略超出圆盘，但在覆盖容差内；布尔结果为近似接受")
    cross = [(B[0] - A[0]) * (v[1] - A[1]) - (B[1] - A[1]) * (v[0] - A[0]) for v in exact]
    status = "SEGMENT" if all(value == 0 for value in cross) else "POLYGON"
    max_height = max(float(abs(value) / distance_squared) * diameter for value in cross)
    diag["near_degenerate"] = (status == "POLYGON" and
                              max_height <= _tol(cfg, "degeneracy", scale))
    if diag["near_degenerate"]:
        diag["warnings"].append("区域近乎退化但仍有面积，保留POLYGON与所有顶点")
    if cfg.sort_vertices_ccw and status == "POLYGON":
        centroid = tuple(float(sum(v[k] for v in exact) / len(exact)) for k in (0, 1))
        vertices = sorted(vertices, key=lambda v: math.atan2(v[1] - centroid[1], v[0] - centroid[0]))
    residual = max(math.fsum((a * x, b * y, -c)) for x, y in vertices for a, b, c in rows)
    diag["max_constraint_residual_m"] = residual
    if residual > _tol(cfg, "feasibility", scale):
        raise ArithmeticError("输出浮点顶点未通过约束残差检查")
    if cover_tol > 1e-3:
        diag["warnings"].append("坐标尺度导致覆盖容差超过1毫米，请按任务精度复核")
    return GeometryResult(status, vertices, diameter,
                          ((float(A[0]), float(A[1])), (float(B[0]), float(B[1]))),
                          center, radius, rmax, excess, excess <= cover_tol, diag)


# ==================== 六、公共入口与命令行 ====================
def solve_halfplanes(halfplanes: Iterable, *, config: GeometryConfig | None = None) -> GeometryResult:
    """求解任意非空约束列表[(A,B,C),...]，便于直接验收点/线段退化。

    零法向量约束作为常量真/假处理；不是正常测向观测的特殊假设。
    """
    cfg = config if config is not None else load_config()
    raw = _matrix(halfplanes, 3, "halfplanes")
    rows = []
    for a, b, c in raw:
        norm = math.hypot(a, b)
        if norm == 0:
            if c < 0:
                return GeometryResult("EMPTY", diagnostics={"reason": "矛盾常量约束"})
            continue
        row = tuple(float(v) / norm for v in (a, b, c))
        if not math.isfinite(norm) or not all(math.isfinite(v) for v in row):
            raise ValueError("约束尺度过大或过小，归一化失败")
        rows.append(row)
    if not rows:
        return GeometryResult("UNBOUNDED", diameter=math.inf,
                              diagnostics={"reason": "全部约束恒真"})
    # 消除完全重复约束；重复测量不增加信息。
    rows = np.asarray(sorted(set(rows)), dtype=float)
    rational = [tuple(Fraction.from_float(float(v)) for v in row) for row in rows]
    diag = {"warnings": [], "constraint_count": len(rows),
            "precision_scope": "有理数校核只针对已生成的浮点系数，不包含三角函数误差"}
    try:
        state = _classify(rows, rational, cfg, diag)
        if state != "BOUNDED":
            result = GeometryResult(state, diameter=math.inf if state == "UNBOUNDED" else None,
                                    diagnostics=diag)
        else:
            exact, vertices = _vertices(rows, rational, cfg, diag)
            result = _measure(exact, vertices, rows, cfg, diag)
    except (ArithmeticError, ValueError) as exc:
        diag["warnings"].append(str(exc))
        result = GeometryResult("NUMERICAL_ISSUE", diagnostics=diag)
    # 异常/风险诊断不能因关闭普通诊断而被静默吞掉。
    if not cfg.return_diagnostics and not diag["warnings"] and result.status != "NUMERICAL_ISSUE":
        result.diagnostics = {}
    return result


def solve_q1(observations: Iterable, bearings: Iterable | None = None, *,
             config: GeometryConfig | None = None, error_deg: float | None = None) -> GeometryResult:
    """完整问题一入口。支持三元组列表或检测点列表+示向度列表。

    返回GeometryResult。EMPTY直径None；UNBOUNDED直径inf；
    NUMERICAL_ISSUE不提供未经证实的几何答案；有限结果不提前舍入。
    """
    cfg = config if config is not None else load_config()
    if error_deg is not None:
        cfg = replace(cfg, bearing_error_deg=error_deg)
    if bearings is not None:
        points = _matrix(observations, 2, "检测点")
        try:
            angles = np.asarray(list(bearings), dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("示向度必须是一维有限数值列表") from exc
        if angles.shape != (len(points),) or not np.isfinite(angles).all():
            raise ValueError("示向度必须与检测点一一对应且有限")
        observations = np.column_stack((points, angles))
    return solve_halfplanes(build_halfplanes(observations, cfg.bearing_error_deg), config=cfg)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="运行问题1.pdf的两个算例")
    parser.add_argument("--input", type=Path, help="JSON文件：[[x,y,theta_deg],...]")
    parser.add_argument("--config", type=Path, help="config.h路径，默认与脚本同目录")
    args = parser.parse_args()
    if args.demo == (args.input is not None):
        parser.error("请选择--demo或--input之一")
    cfg = load_config(args.config)
    datasets = ([[( -800, 0, 0), (0, -800, 90)],
                 [(-800, 0, 0), (400, -400 * math.sqrt(3), 120),
                  (400, 400 * math.sqrt(3), 240)]] if args.demo
                else [json.loads(args.input.read_text(encoding="utf-8-sig"))])
    def display(value):
        if isinstance(value, float):
            return round(value, cfg.display_decimal_places) if math.isfinite(value) else "Infinity"
        if isinstance(value, dict):
            return {k: display(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [display(v) for v in value]
        return value
    for data in datasets:
        print(json.dumps(display(solve_q1(data, config=cfg).to_dict()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
