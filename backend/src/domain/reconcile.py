"""跨系统对账与可行车次分配。

信号台按「日期 × 路段」报总车次，值班台按「日期 × 时段」报全线分时总车次。
台账中已确认窗口的车次不可挪动；漏记窗口只知车次上下限。每个留出日构成一个
有界运输问题：路段行和等于信号台总数、时段列和等于值班台总数，每个路口-时段
单元格在确认值或漏记区间内取整数。约束矩阵全幺模，最小费用流给出整数极值解。
"""

from collections import deque
from json import dumps as json_dumps, loads as json_loads
from math import fsum

# 预计阻断次数为浮点线性目标，比较领先关系时的容差。
LEAD_TOL = 1e-9


def encode_key(key):
    """把 (路段, 路口, 时段) 元组键编码成可进 JSON 列的字符串。"""
    return json_dumps(list(key), ensure_ascii=False)


def decode_key(text):
    return tuple(json_loads(text))


class InfeasibleFlow(Exception):
    """无满足两边总数与区间约束的整数分配。"""


class _Edge:
    __slots__ = ("to", "rev", "cap", "cost")

    def __init__(self, to, rev, cap, cost):
        self.to, self.rev, self.cap, self.cost = to, rev, cap, cost


class MinCostFlow:
    """单源单汇最小费用流，SPFA 求最短路，支持负费用与整数容量。"""

    def __init__(self, size):
        self.graph = [[] for _ in range(size)]

    def add(self, source, target, cap, cost):
        forward = _Edge(target, len(self.graph[target]), cap, cost)
        backward = _Edge(source, len(self.graph[source]), 0, -cost)
        self.graph[source].append(forward)
        self.graph[target].append(backward)
        return forward

    def flow(self, source, sink, required):
        total_flow = 0
        total_cost = 0.0
        n = len(self.graph)
        while total_flow < required:
            dist = [None] * n
            dist[source] = 0.0
            prev_v = [-1] * n
            prev_e = [-1] * n
            queue = deque([source])
            in_queue = [False] * n
            in_queue[source] = True
            while queue:
                v = queue.popleft()
                in_queue[v] = False
                base = dist[v]
                for i, edge in enumerate(self.graph[v]):
                    if edge.cap <= 0:
                        continue
                    candidate = base + edge.cost
                    if dist[edge.to] is None or candidate < dist[edge.to]:
                        dist[edge.to] = candidate
                        prev_v[edge.to], prev_e[edge.to] = v, i
                        if not in_queue[edge.to]:
                            queue.append(edge.to)
                            in_queue[edge.to] = True
            if dist[sink] is None:
                raise InfeasibleFlow("可行流在剩余需求下中断")
            add = required - total_flow
            v = sink
            while v != source:
                add = min(add, self.graph[prev_v[v]][prev_e[v]].cap)
                v = prev_v[v]
            v = sink
            while v != source:
                edge = self.graph[prev_v[v]][prev_e[v]]
                edge.cap -= add
                self.graph[v][edge.rev].cap += add
                v = prev_v[v]
            total_flow += add
            total_cost += add * dist[sink]
        return total_flow, total_cost


def solve_transport(cells, row_totals, col_totals, costs=None):
    """求解有界运输问题，返回 {cell_key: 分配车次}。

    cells: list[(row, col, lower, upper, key)]，key 由调用方用于回填。
    row_totals / col_totals: dict，行（路段）与列（时段）的总车次。
    costs: 可选 dict[key, float]，缺省为 0，只求一个可行整数分配。
    """
    rows = sorted(row_totals)
    cols = sorted(col_totals)
    row_index = {row: i for i, row in enumerate(rows)}
    col_index = {col: i for i, col in enumerate(cols)}
    source = 0
    row_base = 1
    col_base = row_base + len(rows)
    sink = col_base + len(cols)
    network = MinCostFlow(sink + 1)
    row_supply = dict(row_totals)
    col_demand = dict(col_totals)
    handles = []
    for row, col, lower, upper, key in cells:
        if lower > upper:
            raise InfeasibleFlow(f"窗口 {key} 的车次下限超过上限")
        span = upper - lower
        edge = network.add(row_base + row_index[row], col_base + col_index[col], span,
                           (costs or {}).get(key, 0.0)) if span else None
        # 下界部分是必然通过的固定流量，先从供需中扣除。
        row_supply[row] -= lower
        col_demand[col] -= lower
        handles.append((key, edge, lower, span))
    if any(value < 0 for value in row_supply.values()) or any(value < 0 for value in col_demand.values()):
        raise InfeasibleFlow("已确认车次超过信号台或值班台总数")
    grand_row = fsum(row_totals.values())
    grand_col = fsum(col_totals.values())
    if grand_row != grand_col:
        raise InfeasibleFlow(f"信号台总车次 {grand_row:g} 与值班台总车次 {grand_col:g} 不一致")
    for row in rows:
        network.add(source, row_base + row_index[row], row_supply[row], 0.0)
    for col in cols:
        network.add(col_base + col_index[col], sink, col_demand[col], 0.0)
    fixed = fsum(lower for _, _, lower, _, _ in cells)
    network.flow(source, sink, int(grand_row - fixed))
    allocation = {}
    for key, edge, lower, span in handles:
        allocation[key] = lower if edge is None else lower + span - edge.cap
    return allocation


def _cell_key(row):
    return row["road"], row["intersection"], row["shift"]


def reconcile_day(day_rows, signal_totals, duty_totals, rate_of):
    """对单个留出日做对账并枚举各路口预计阻断次数的可行范围。

    rate_of: callable(row) -> 每车次预计阻断率（GLM，按窗口条件）。
    返回 dict：feasible、两套原数、可行分配、各路口 min/max、领先证人。
    """
    roads = sorted({row["road"] for row in day_rows})
    shifts = sorted({row["shift"] for row in day_rows})
    missing_total = {road for road in roads if road not in signal_totals}
    if missing_total or {shift for shift in shifts if shift not in duty_totals}:
        raise InfeasibleFlow("信号台或值班台总数缺失，无法对账")

    cells = []
    unit_rate = {}
    for row in day_rows:
        key = _cell_key(row)
        if row.get("exposure") is not None:
            lower = upper = row["exposure"]
        else:
            lower, upper = row["exposure_min"], row["exposure_max"]
        cells.append((row["road"], row["shift"], lower, upper, key))
        unit_rate[key] = rate_of(row)

    def predicted(allocation):
        return {key: allocation[key] * unit_rate[key] for key in allocation}

    def public(allocation):
        """把 tuple 键的解转成字符串键，供 JSON 存储与接口返回。"""
        return {encode_key(key): value for key, value in allocation.items()}

    # 零费用的基准可行分配。
    try:
        reference = solve_transport(cells, signal_totals, duty_totals)
    except InfeasibleFlow as error:
        return {
            "feasible": False,
            "reason": str(error),
            "signal_totals": signal_totals,
            "duty_totals": duty_totals,
        }

    intersections = sorted({key[1] for key in unit_rate})

    ranges = {}
    for intersection in intersections:
        keys = [key for key in unit_rate if key[1] == intersection]
        # 对「该路口总预计」分别求最小、最大。
        min_costs = {key: (unit_rate[key] if key[1] == intersection else 0.0) for key in unit_rate}
        max_costs = {key: (-unit_rate[key] if key[1] == intersection else 0.0) for key in unit_rate}
        min_alloc = solve_transport(cells, signal_totals, duty_totals, min_costs)
        max_alloc = solve_transport(cells, signal_totals, duty_totals, max_costs)
        min_value = fsum(min_alloc[key] * unit_rate[key] for key in keys)
        max_value = fsum(max_alloc[key] * unit_rate[key] for key in keys)
        ranges[intersection] = {"min": min_value, "max": max_value,
                                "min_allocation": public(min_alloc), "max_allocation": public(max_alloc)}

    # 每种可行分配都领先：对任意其他路口 j，min(本路口-j) >= 0。
    robust_leaders = set()
    witnesses = {}
    for leader in intersections:
        dominates = True
        for other in intersections:
            if other == leader:
                continue
            diff_costs = {key: ((unit_rate[key] if key[1] == leader else 0.0)
                                - (unit_rate[key] if key[1] == other else 0.0))
                          for key in unit_rate}
            alloc = solve_transport(cells, signal_totals, duty_totals, diff_costs)
            gap = fsum(alloc[key] * diff_costs[key] for key in unit_rate)
            if gap < -LEAD_TOL * (1 + abs(gap)):
                dominates = False
                witnesses[f"{leader}>{other}"] = {
                    "leader": leader, "challenger": other, "gap": gap,
                    "allocation": public(alloc), "predicted": public(predicted(alloc)),
                }
        if dominates:
            robust_leaders.add(leader)

    return {
        "feasible": True,
        "signal_totals": signal_totals,
        "duty_totals": duty_totals,
        "reference_allocation": public(reference),
        "reference_predicted": public(predicted(reference)),
        "ranges": ranges,
        "robust_leaders": sorted(robust_leaders),
        "witnesses": witnesses,
        "unit_rate": {encode_key(key): rate for key, rate in unit_rate.items()},
        "cells": [{"road": row["road"], "intersection": row["intersection"], "shift": row["shift"],
                   "kind": "confirmed" if row.get("exposure") is not None else "missing",
                   "exposure": row.get("exposure"),
                   "exposure_min": None if row.get("exposure") is not None else row["exposure_min"],
                   "exposure_max": None if row.get("exposure") is not None else row["exposure_max"]}
                  for row in day_rows],
    }
