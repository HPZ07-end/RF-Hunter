问题一逐案例测试报告
stage_*_ms单位毫秒；null表示未执行。classify包含projection_bounds和linprog；measure包含setup、diameter、coverage、degeneracy、sort_vertices、residual_and_output，父子时间不能重复相加。vertices为边界两两求交+全部约束筛选+去重。diameter为最远点对搜索、距离平方和最终距离计算；coverage为圆心/半径/覆盖检查；other为总耗时减config/build_halfplanes/classify/vertices/measure，含归一化、有理数转换、调度及计时开销。所有时间含少量插桩开销，不含写报告和测试断言。直径比较应同时参考vertex_count，检测点数量不是直径模块的直接输入规模。

cases.csv：每次最外层求解一行，可用Excel打开。
report.json：完整输入、配置、顶点、最远点对、圆心、诊断和测试结果。
unittest.log：全部测试方法的通过/失败记录及失败堆栈。

字段说明：
case_id：测试方法名+方法内求解序号；不是模拟器正式测试编码。
observation_count：输入观测条数（重复观测也计数）。
unique_detection_point_count：输入中的不同检测点坐标数。
constraint_count_input：实际传入半平面求解器的约束数；合法测向输入通常为2n。
constraint_count_solver：归一化并删除完全重复项后的约束数，不是非冗余约束数。
elapsed_ms：一次完整求解的墙钟耗时，单位毫秒；不含写报告和断言，无预热。
status：几何状态或EXCEPTION（如预期的非法输入）。
vertex_count：输出顶点数。
diameter：定位区域直径，单位米；Infinity为无界，空白/null表示无有效值。
circle_radius/max_vertex_distance/coverage_excess：圆半径/最大顶点距离/超出量，单位米。
same_diameter_covers：是否在配置容差下可被同直径圆覆盖。
test_outcome：所属测试方法整体断言结果；PASS不等于几何status必须为POLYGON。
backend_mode/backend：真实SciPy、纯有理数或模拟求解器；模拟耗时不能用于性能比较。

随机测试使用固定种子20260911；每组先3点后4点，共100组200行。
直接半平面案例没有检测点，因此点数为空；未到达阶段的参数也为空。
仅构造约束或检查配置的测试不调用求解器，没有cases行，可在JSON的tests中查看。
各行test_outcome以整个测试方法为单位；若中途断言失败，后续案例不会运行。
完整精度保存在JSON/CSV中；文件写入使用UTF-8（CSV带BOM）。
