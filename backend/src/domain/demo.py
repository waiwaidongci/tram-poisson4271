"""澄浦 2 号线的可复现观测台账。"""

from datetime import date, timedelta
from math import exp
from random import Random


_INTERSECTIONS = (
    ("港湾路", "货栈街口", "混行"),
    ("港湾路", "渡口街口", "混行"),
    ("科创大道", "智造街口", "独立通行"),
    ("科创大道", "东环街口", "独立通行"),
)


def _poisson(rng, mean):
    limit = exp(-mean)
    product = rng.random()
    count = 0
    while product > limit:
        product *= rng.random()
        count += 1
    return count


def demo_rows():
    rng = Random(261160)
    rows = []
    for day in range(14):
        stamp = (date(2026, 11, 7) + timedelta(days=day)).isoformat()
        for road, intersection, track in _INTERSECTIONS:
            for shift in ("平峰", "高峰"):
                peak = shift == "高峰"
                exposure = (38 if peak else 20) + rng.randrange(13)
                rate = 0.035 * exp(0.9 * peak + 1.05 * (track == "混行"))
                rows.append({
                    "date": stamp, "road": road, "intersection": intersection,
                    "shift": shift, "track": track, "exposure": exposure,
                    "count": _poisson(rng, exposure * rate),
                })
    return rows


def demo_reconciliation_rows():
    """跨系统对账演示：最后三个留出日带信号台/值班台总数，含漏记窗口与一个对不上的日期。"""
    rows = []
    rng = Random(261177)
    holdout = {date(2026, 11, 18).isoformat(), date(2026, 11, 19).isoformat(), date(2026, 11, 20).isoformat()}
    broken_day = date(2026, 11, 20).isoformat()
    missing_window = (date(2026, 11, 19).isoformat(), "港湾路", "货栈街口", "高峰")
    for day in range(14):
        stamp = (date(2026, 11, 7) + timedelta(days=day)).isoformat()
        day_ledger = []
        for road, intersection, track in _INTERSECTIONS:
            for shift in ("平峰", "高峰"):
                peak = shift == "高峰"
                exposure = (38 if peak else 20) + rng.randrange(13)
                rate = 0.035 * exp(0.9 * peak + 1.05 * (track == "混行"))
                count = _poisson(rng, exposure * rate)
                row = {
                    "date": stamp, "road": road, "intersection": intersection,
                    "shift": shift, "track": track,
                    "exposure_min": None, "exposure_max": None,
                    "signal_total": None, "duty_total": None,
                }
                if (stamp, road, intersection, shift) == missing_window:
                    row.update(exposure=None, count=None,
                               exposure_min=max(0, exposure - 6), exposure_max=exposure + 6)
                else:
                    row.update(exposure=exposure, count=count)
                day_ledger.append(row)
                rows.append(row)
        if stamp in holdout:
            signal_totals, duty_totals = {}, {}
            for row in day_ledger:
                # 漏记窗口按区间中值参与两套总数的编制，保证可对上（最后一天再人为破坏）。
                vehicles = row["exposure"] if row["exposure"] is not None else (
                    row["exposure_min"] + row["exposure_max"]) // 2
                signal_totals[row["road"]] = signal_totals.get(row["road"], 0) + vehicles
                duty_totals[row["shift"]] = duty_totals.get(row["shift"], 0) + vehicles
            for row in day_ledger:
                row["signal_total"] = signal_totals[row["road"]]
                row["duty_total"] = duty_totals[row["shift"]]
                if stamp == broken_day:
                    # 信号台多报，制造两套总数对不上的留出日。
                    row["signal_total"] = signal_totals[row["road"]] + 80
    return rows
