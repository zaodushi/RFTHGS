#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 PyVRP 求解一个 16 客户的 Capacitated VRP。
代码严格依据官方 quick_tutorial 文档编写。
"""

# -------------------- 1. 依赖 --------------------
from pyvrp import Model
from pyvrp.stop import MaxRuntime
from pyvrp.plotting import plot_coordinates, plot_solution
import matplotlib.pyplot as plt

# -------------------- 2. 数据 --------------------
# 参考 OR-Tools 官方示例的坐标与需求
COORDS = [
    (456, 320),  # 0 — depot
    (228,   0),  # 1
    (912,   0),  # 2
    (  0,  80),  # 3
    (114,  80),  # 4
    (570, 160),  # 5
    (798, 160),  # 6
    (342, 240),  # 7
    (684, 240),  # 8
    (570, 400),  # 9
    (912, 400),  # 10
    (114, 480),  # 11
    (228, 480),  # 12
    (342, 560),  # 13
    (684, 560),  # 14
    (  0, 640),  # 15
    (798, 640),  # 16
]

DEMANDS = [
     0,  # depot
     1, 1, 2, 4, 2, 4, 8, 8,
     1, 2, 1, 2, 4, 4, 8, 8
]

# -------------------- 3. 构建模型 --------------------
m = Model()

# 车辆类型：4 辆车，每辆容量 15
m.add_vehicle_type(4, capacity=15)

# 仓库
depot = m.add_depot(x=COORDS[0][0], y=COORDS[0][1])

# 客户
for idx in range(1, len(COORDS)):
    m.add_client(
        x=COORDS[idx][0],
        y=COORDS[idx][1],
        delivery=DEMANDS[idx],
    )

# 添加两两之间的边，距离使用曼哈顿距离
for frm in m.locations:
    for to in m.locations:
        dist = abs(frm.x - to.x) + abs(frm.y - to.y)
        m.add_edge(frm, to, distance=dist)

# -------------------- 4. 可视化坐标 --------------------
fig, ax = plt.subplots(figsize=(8, 8))
plot_coordinates(m.data(), ax=ax)
ax.set_title("CVRP instance – node coordinates")
plt.tight_layout()
plt.show()

# -------------------- 5. 求解 --------------------
# 运行 1 秒
res = m.solve(stop=MaxRuntime(5), display=True)

# -------------------- 6. 结果输出 --------------------
print(res)

# -------------------- 7. 绘制最优解 --------------------
fig, ax = plt.subplots(figsize=(8, 8))
plot_solution(res.best, m.data(), ax=ax)
ax.set_title("Best solution (distance = {})".format(res.best.distance()))
plt.tight_layout()
plt.show()