"""对账与稳健领先逻辑的独立冒烟测试，运行：python3 -m tests.test_reconcile"""

from src.domain.analysis import analyze, validate_rows
from src.domain.demo import demo_rows
from src.domain.reconcile import InfeasibleFlow, MinCostFlow, decode_key, encode_key, solve_transport

ROADS = {
    "港湾路": ["货栈街口", "渡口街口"],
    "科创大道": ["智造街口", "东环街口"],
}
TRACK = {"货栈街口": "混行", "渡口街口": "混行", "智造街口": "独立通行", "东环街口": "独立通行"}


def window(day, road, intersection, shift, exposure=None, count=None,
           lo=None, hi=None, signal=None, duty=None):
    row = {"date": day, "road": road, "intersection": intersection, "shift": shift,
           "track": TRACK[intersection], "exposure": exposure, "count": count,
           "exposure_min": lo, "exposure_max": hi, "signal_total": signal, "duty_total": duty}
    return row


def test_min_cost_flow_basic():
    network = MinCostFlow(4)
    network.add(0, 1, 3, 0)
    network.add(0, 2, 2, 0)
    network.add(1, 3, 3, -1)
    network.add(2, 3, 2, 5)
    flow, cost = network.flow(0, 3, 4)
    assert flow == 4 and cost == -3 + 5, (flow, cost)
    print("ok min cost flow")


def test_transport_feasible_and_bounds():
    cells = [
        ("R1", "平峰", 10, 10, ("R1", "A", "平峰")),
        ("R1", "高峰", 0, 20, ("R1", "A", "高峰")),
        ("R2", "平峰", 5, 5, ("R2", "B", "平峰")),
        ("R2", "高峰", 0, 20, ("R2", "B", "高峰")),
    ]
    rows_total = {"R1": 20, "R2": 15}
    cols_total = {"平峰": 15, "高峰": 20}
    allocation = solve_transport(cells, rows_total, cols_total)
    assert sum(allocation.values()) == 35
    assert allocation[("R1", "A", "平峰")] == 10 and allocation[("R2", "B", "平峰")] == 5
    # 高峰 20 辆车在两个单元格之间分，列和必须为 20。
    assert allocation[("R1", "A", "高峰")] + allocation[("R2", "B", "高峰")] == 20
    print("ok transport feasible")


def test_transport_infeasible_grand_total():
    cells = [("R1", "平峰", 0, 10, ("R1", "A", "平峰"))]
    try:
        solve_transport(cells, {"R1": 9}, {"平峰": 10})
    except InfeasibleFlow:
        print("ok grand total mismatch rejected")
    else:
        raise AssertionError("应当判定对不上")


def test_legacy_ledger_unchanged():
    rows = demo_rows()
    # 旧台账只有 7 个字段。
    assert set(rows[0]) == {"date", "road", "intersection", "shift", "track", "exposure", "count"}
    result = analyze(rows, {"holdout_days": 3, "severity_rate": 9})
    assert result["mode"] == "legacy"
    assert result["tables"][3]["columns"][0]["key"] == "date"
    assert len(result["tables"][3]["columns"]) == 12  # 7 原始列 + 5 模型列
    assert "reconciliation" not in result
    print("ok legacy ledger output preserved")


def _build_reconciliation_rows(infeasible_day=False):
    rows = demo_rows()  # 14 天训练数据，字段齐全
    base = [dict(row, exposure_min=None, exposure_max=None, signal_total=None, duty_total=None) for row in rows]
    holdout_dates = sorted({row["date"] for row in base})[-3:]
    # 留出日重写为带外部总数的对账场景。
    for index, day in enumerate(holdout_dates):
        day_rows = [row for row in base if row["date"] == day]
        signal_sum, duty_sum = {}, {}
        for row in day_rows:
            signal_sum[row["road"]] = signal_sum.get(row["road"], 0) + row["exposure"]
            duty_sum[row["shift"]] = duty_sum.get(row["shift"], 0) + row["exposure"]
        for row in day_rows:
            row["signal_total"] = signal_sum[row["road"]]
            row["duty_total"] = duty_sum[row["shift"]]
        if index == 1:
            # 第二个留出日制造一个漏记窗口：清空车次与阻断，给出区间。
            target = next(row for row in day_rows if row["intersection"] == "货栈街口" and row["shift"] == "高峰")
            known = target["exposure"]
            target["exposure_min"], target["exposure_max"] = max(0, known - 6), known + 6
            target["exposure"] = target["count"] = None
            # 区间允许原数，保持两套总数仍可对上。
        if infeasible_day and index == 2:
            for row in day_rows:
                row["signal_total"] = row["signal_total"] + 100
    return base, holdout_dates


def test_reconciliation_feasible_with_missing_window():
    rows, holdout_dates = _build_reconciliation_rows()
    result = analyze(rows, {"holdout_days": 3, "severity_rate": 9})
    assert result["mode"] == "reconciliation"
    assert result["all_feasible"] is True
    assert not result["infeasible_dates"]
    reconciliation = result["reconciliation"]
    missing_day = reconciliation[1]
    assert missing_day["feasible"] is True
    alloc = missing_day["reference_allocation"]
    # 字符串键可 JSON 化且能解码回三元组。
    for text in alloc:
        assert len(decode_key(text)) == 3 and encode_key(decode_key(text)) == text
    # 行和等于信号台总数、列和等于值班台总数。
    row_sum, col_sum = {}, {}
    for text, vehicles in alloc.items():
        road, _, shift = decode_key(text)
        row_sum[road] = row_sum.get(road, 0) + vehicles
        col_sum[shift] = col_sum.get(shift, 0) + vehicles
    assert row_sum == missing_day["signal_totals"]
    assert col_sum == missing_day["duty_totals"]
    # 范围表每个路口都有非负下限且下限不超过上限。
    for span in missing_day["ranges"].values():
        assert 0 <= span["min"] <= span["max"] + 1e-12
    print("ok feasible reconciliation with missing window and totals")


def test_reconciliation_infeasible_day_blocks_advice():
    rows, holdout_dates = _build_reconciliation_rows(infeasible_day=True)
    result = analyze(rows, {"holdout_days": 3, "severity_rate": 9})
    assert result["mode"] == "reconciliation"
    assert result["all_feasible"] is False
    assert result["infeasible_dates"] == [holdout_dates[2]]
    assert "对不上" in result["headline"]
    bad = next(item for item in result["reconciliation"] if item["date"] == holdout_dates[2])
    assert bad["feasible"] is False and bad["reason"]
    # 对不上的日期没有稳健领先集合。
    assert result["robust_leaders"] == []
    print("ok infeasible day produces no advice and keeps raw totals")


def test_robust_lead_and_witness():
    rows, _ = _build_reconciliation_rows()
    result = analyze(rows, {"holdout_days": 3, "severity_rate": 9})
    for day in result["reconciliation"]:
        if not day.get("feasible"):
            continue
        leaders = set(day["robust_leaders"])
        # 证人只可能出现在未稳健领先者身上；领先者不应有被推翻记录。
        for witness in day["witnesses"].values():
            assert witness["leader"] not in leaders
            assert witness["gap"] < 0
            assert set(witness["allocation"]) == set(day["reference_allocation"])
    print("ok robust leaders and refuting witnesses")


def test_validation_rules():
    # 漏记窗口缺下限。
    bad = window("2026-11-18", "港湾路", "货栈街口", "高峰", lo=5)
    bad.update({"exposure": None, "count": None, "exposure_max": None})
    try:
        validate_rows([bad])
    except ValueError as error:
        assert "下限与上限" in str(error)
    else:
        raise AssertionError("漏记窗口缺上下限应被拒绝")
    # 已确认窗口带漏记区间。
    bad2 = window("2026-11-18", "港湾路", "货栈街口", "高峰", exposure=30, count=2, lo=1, hi=5)
    try:
        validate_rows([bad2])
    except ValueError as error:
        assert "不应填写漏记" in str(error)
    else:
        raise AssertionError("已确认窗口带区间应被拒绝")
    print("ok validation rules")


if __name__ == "__main__":
    test_min_cost_flow_basic()
    test_transport_feasible_and_bounds()
    test_transport_infeasible_grand_total()
    test_legacy_ledger_unchanged()
    test_reconciliation_feasible_with_missing_window()
    test_reconciliation_infeasible_day_blocks_advice()
    test_robust_lead_and_witness()
    test_validation_rules()
    print("all reconciliation tests passed")
