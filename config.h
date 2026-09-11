/*
 * B题问题一配置（坐标/距离单位：米；输入角度单位：度）
 *
 * 后续 geometry.py 的执行顺序与配置对应：
 * 1. 校验输入、角度转弧度、构造两个半平面 -> 板块一
 * 2. 判断可行性，再求四个坐标极值判断有界性 -> 板块二
 * 3. 枚举边界交点、检查全部约束、去重 -> 板块三
 * 4. 处理点/线段退化、求最远顶点对和直径 -> 板块三
 * 5. 求最远点对中点、判定同直径圆覆盖 -> 板块三、四
 * 6. 返回状态、几何结果及数值诊断 -> 板块四
 *
 * 本文件不是Python模块，不能直接import。
 * 后续读取器仅解析 RF_ 开头的#define，值限定为单个数字或双引号字符串。
 * 不使用eval，不需要C编译器；读取时校验必需键、类型、有限性和范围。
 * geometry.py的load_config负责读取，solve_q1为完整计算入口。
 */
#ifndef RF_HUNTER_CONFIG_H
#define RF_HUNTER_CONFIG_H

/* ==================== 一、输入与模型范围 ==================== */
#define RF_Q1_MIN_OBSERVATIONS 1

/* 要求0 < epsilon < 90，即角域宽度小于180度。
 * sin/cos前转弧度；用方向向量处理359/0跨越，不用角度区间比较。
 * 同一地点误差固定，重复观测不能通过平均值缩小此误差界。
 */
#define RF_Q1_BEARING_ERROR_DEG 1.0

/* 模型声明，不是任意可切换的选项；其他值必须拒绝或另行实现。
 * 不纳入1800米目标圆、1500米接收上界及5米近距离排除圆。
 * 禁止以人为大矩形截断无界区域。
 */
#define RF_Q1_REGION_MODEL "bearing_halfplanes_only"

/* ==================== 二、可行性与有界性 ==================== */
/* 安装SciPy时使用scipy.optimize.linprog，并用有理数消元校核。
 * 缺少SciPy时使用二维有理数消元独立判断，并在diagnostics中记录后端；
 * 此时以下LP选项不参与计算。无需用人为包围盒替代无界判定。
 * 必须显式bounds=[(None, None), (None, None)]，允许负坐标。
 * 先零目标求可行性，再求min x、max x、min y、max y。
 * 检查求解状态；数值失败不能当作EMPTY或UNBOUNDED。
 * 近退化问题还需结合残差校核求解器结论。
 */
#define RF_Q1_LP_METHOD "highs"
#define RF_Q1_LP_PRESOLVE 1
#define RF_Q1_LP_PRIMAL_FEASIBILITY_TOL 1e-9
#define RF_Q1_LP_DUAL_FEASIBILITY_TOL 1e-9

/* ==================== 三、几何数值计算 ==================== */
/* 每条Ax+By<=C先归一化法向量，使约束残差具有米的量纲。
 * 长度容差=ABS_TOL_M+REL_TOL*L。
 * L=max(1米, 输入及待检查点坐标的绝对值, 相关有限几何长度)。
 * 默认值适用于常见坐标尺度，不保证所有病态输入都可靠。
 * 若尺度过大导致容差不满足任务精度，应报告风险。
 */
#define RF_Q1_FEASIBILITY_ABS_TOL_M 1e-6
#define RF_Q1_FEASIBILITY_REL_TOL 1e-12

/* 去重、退化、覆盖使用各自的长度容差，不能与无量纲行列式阈值混用。
 * 退化用点到直线距离检查，不能直接将面积与米单位容差比较。
 * 合并近邻顶点应记录合并距离，避免掩盖极小边或真实面积。
 */
#define RF_Q1_VERTEX_MERGE_ABS_TOL_M 1e-7
#define RF_Q1_VERTEX_MERGE_REL_TOL 1e-12
#define RF_Q1_DEGENERACY_ABS_TOL_M 1e-7
#define RF_Q1_DEGENERACY_REL_TOL 1e-12
#define RF_Q1_COVERAGE_ABS_TOL_M 1e-6
#define RF_Q1_COVERAGE_REL_TOL 1e-12

/* 归一化后行列式无量纲。此阈值只触发高精度重算/警告，不能直接跳过。
 * 仅确认平行或重合才跳过；近乎平行可能产生有效的远处顶点。
 * decimal重算只能改善已生成系数的运算，不能恢复浮点三角函数精度。
 * 重算后仍无法可靠判定，应返回NUMERICAL_ISSUE。
 */
#define RF_Q1_NEAR_PARALLEL_DET_TOL 1e-10
#define RF_Q1_DECIMAL_PRECISION 60

/* ==================== 四、输出约定 ==================== */
/* 状态：EMPTY/UNBOUNDED/POINT/SEGMENT/POLYGON/NUMERICAL_ISSUE。
 * 空集直径None，无界直径正无穷，单点直径0。
 * 空集、无界及数值异常的覆盖判断为None，不伪造布尔结果。
 * 返回vertices、diameter、diameter_pair、circle_center、circle_radius、
 * max_vertex_distance、coverage_excess、same_diameter_covers、diagnostics。
 * 圆心=最远顶点对中点，不是顶点平均值；固定圆心的覆盖半径不是一般最小覆盖圆。
 * coverage_excess保留r_max-D/2的原始带符号值。
 * 容差内的覆盖结果标记数值临界，不能当成严格证明。
 * 仅显示时舍入，内部和模块间传递的坐标不提前舍入。
 */
#define RF_Q1_SORT_VERTICES_CCW 1
#define RF_Q1_RETURN_DIAGNOSTICS 1
#define RF_Q1_DISPLAY_DECIMAL_PLACES 8

/* ==================== 五、后续问题的共享题目常量 ==================== */
/* 不参与第一问纯角域交集；只供后续选点、搜索、清除模块使用。 */
#define RF_TARGET_REGION_RADIUS_M 1800.0
#define RF_RECEPTION_RADIUS_MIN_M 1000.0
#define RF_RECEPTION_RADIUS_MAX_M 1500.0
#define RF_CHANNEL_MIN 1
#define RF_CHANNEL_MAX 20
#define RF_INITIAL_CHANNEL 1
#define RF_INITIAL_X_M 0.0
#define RF_INITIAL_Y_M 0.0
#define RF_DOG_SPEED_M_PER_S 5.0
#define RF_CHANNEL_SWITCH_TIME_S 1.0
#define RF_DETECTION_TIME_S 5.0
#define RF_OPTICAL_LOCALIZATION_TIME_S 3.0
#define RF_CLEARANCE_TIME_S 2.0
#define RF_CLEARANCE_RADIUS_M 20.0
#define RF_STRONG_SIGNAL_RADIUS_M 5.0
#define RF_SOURCE_COUNT_MIN 10
#define RF_SOURCE_COUNT_MAX 16

/* 队号、接口地址、通信参数尚未提供，待读取附件和获得连接信息后再配置。 */
#endif /* RF_HUNTER_CONFIG_H */
