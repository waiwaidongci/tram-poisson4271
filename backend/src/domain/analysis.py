"""Fit a ridge Poisson rate model and review chronological holdout windows."""

from collections import defaultdict
from datetime import date
from math import exp, fsum, isclose, log, log1p, ulp

from .spec import SPEC


_MAX_ROWS = 5000
_RIDGE = 1.0
_FEATURES = ((0, 0), (0, 1), (1, 0), (1, 1))
# 旧台账（字段齐全、无外部总数）沿用的原始台账列。
_LEGACY_FIELD_KEYS = ("date", "road", "intersection", "shift", "track", "exposure", "count")


def _validate_field(field, value):
    kind = field["type"]
    label = field["label"]
    if value is None:
        if field.get("required", True):
            raise ValueError(f"{label}不能为空")
        return None
    if kind in {"int", "number"}:
        types = (int,) if kind == "int" else (int, float)
        if type(value) not in types or not field["min"] <= value <= field["max"]:
            unit = "整数" if kind == "int" else "数值"
            raise ValueError(f"{label}须为 {field['min']} 至 {field['max']} 之间的{unit}")
    elif kind == "date":
        try:
            if not isinstance(value, str) or len(value) != 10 or date.fromisoformat(value).isoformat() != value:
                raise ValueError
        except ValueError:
            raise ValueError(f"{label}须为有效的 YYYY-MM-DD 日期") from None
    elif kind == "select":
        if value not in field["options"]:
            raise ValueError(f"{label}须为{'、'.join(field['options'])}之一")
    elif not isinstance(value, str) or len(value) > field["max_length"] or not value.strip():
        raise ValueError(f"{label}须为非空文本，最多 {field['max_length']} 字符")
    return value.strip() if kind == "string" else value


def validate_rows(rows):
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= _MAX_ROWS:
        raise ValueError(f"观测台账须包含 1 至 {_MAX_ROWS} 行")
    readings = {}
    for index, row in enumerate(rows, start=1):
        try:
            if not isinstance(row, dict):
                raise ValueError("须为一条观测记录")
            clean = {field["key"]: _validate_field(field, row.get(field["key"])) for field in SPEC["fields"]}
            window = (clean["date"], clean["road"], clean["intersection"], clean["shift"])
            if window in readings:
                raise ValueError("日期、道路、路口与时段重复")
            missing = clean["exposure"] is None
            if missing:
                if clean["count"] is not None:
                    raise ValueError("漏记窗口没有实际车次，确认阻断次数须留空")
                if clean["exposure_min"] is None or clean["exposure_max"] is None:
                    raise ValueError("漏记窗口须同时填写车次下限与上限")
                if clean["exposure_min"] > clean["exposure_max"]:
                    raise ValueError("漏记车次下限不能超过上限")
            else:
                if clean["count"] is None:
                    raise ValueError("已确认窗口须填写确认阻断次数")
                if clean["exposure_min"] is not None or clean["exposure_max"] is not None:
                    raise ValueError("已确认窗口不应填写漏记车次上下限")
            readings[window] = clean
        except ValueError as error:
            raise ValueError(f"第 {index} 条观测 · {error}") from None
    ordered = [readings[key] for key in sorted(readings)]
    _validate_external_totals(ordered)
    return ordered


def _validate_external_totals(readings):
    """同一日×路段的信号台总数、同一日×时段的值班台总数须跨行一致，且同日两套须同时提供。"""
    signal = {}
    duty = {}
    days = defaultdict(lambda: {"signal": False, "duty": False})
    for row in readings:
        signal_key = (row["date"], row["road"])
        duty_key = (row["date"], row["shift"])
        if row["signal_total"] is not None:
            if signal_key in signal and signal[signal_key] != row["signal_total"]:
                raise ValueError(f"{row['date']} {row['road']} 的信号台路段总车次各行不一致")
            signal[signal_key] = row["signal_total"]
            days[row["date"]]["signal"] = True
        if row["duty_total"] is not None:
            if duty_key in duty and duty[duty_key] != row["duty_total"]:
                raise ValueError(f"{row['date']} {row['shift']} 的值班台分时总车次各行不一致")
            duty[duty_key] = row["duty_total"]
            days[row["date"]]["duty"] = True
    for day, flags in days.items():
        if flags["signal"] != flags["duty"]:
            raise ValueError(f"{day} 须同时提供信号台路段总车次与值班台分时总车次，或两者都留空")


def _features(row):
    return (int(row["shift"] == "高峰"), int(row["track"] == "混行"))


def _profile(slopes, groups, total):
    """Profile out the intercept by matching the fitted training count total."""
    first, second = slopes
    log_weights = [log_exposure + first * peak + second * shared for peak, shared, log_exposure, _ in groups]
    center = max(log_weights)
    centered = [weight - center for weight in log_weights]
    weights = [exp(weight) for weight in centered]
    weight_sum = fsum(weights)
    dominant = log_weights.index(center)
    log_weight_sum = log1p(fsum(weight for index, weight in enumerate(weights) if index != dominant))
    probabilities = [weight / weight_sum for weight in weights]
    # This differs from the profile likelihood by a data-only constant. Center
    # before multiplying by counts to preserve precision under separation.
    objective = fsum(count * (log_weight_sum - weight)
                     for (_, _, _, count), weight in zip(groups, centered))
    objective += 0.5 * _RIDGE * (first * first + second * second)
    return objective, probabilities, center + log_weight_sum


def _fit(training):
    exposure = defaultdict(int)
    counts = defaultdict(int)
    for row in training:
        features = _features(row)
        exposure[features] += row["exposure"]
        counts[features] += row["count"]
    total = sum(counts.values())
    if not total:
        raise ValueError("训练期没有阻断事件，无法估计有限的截距。请补充观测或调整留出观测日数。")
    groups = [(peak, shared, log(exposure[peak, shared]), counts[peak, shared])
              for peak, shared in _FEATURES if (peak, shared) in exposure]
    observed_peak = sum(peak * count for peak, _, _, count in groups)
    observed_shared = sum(shared * count for _, shared, _, count in groups)
    slopes = (0.0, 0.0)
    for _ in range(100):
        objective, probabilities, normalizer = _profile(slopes, groups, total)
        mean_peak = fsum(p * group[0] for p, group in zip(probabilities, groups))
        mean_shared = fsum(p * group[1] for p, group in zip(probabilities, groups))
        mean_joint = fsum(p * group[0] * group[1] for p, group in zip(probabilities, groups))
        g_peak = total * mean_peak - observed_peak + _RIDGE * slopes[0]
        g_shared = total * mean_shared - observed_shared + _RIDGE * slopes[1]
        h_peak = total * mean_peak * (1 - mean_peak) + _RIDGE
        h_shared = total * mean_shared * (1 - mean_shared) + _RIDGE
        h_joint = total * (mean_joint - mean_peak * mean_shared)
        determinant = h_peak * h_shared - h_joint * h_joint
        step_peak = (h_shared * g_peak - h_joint * g_shared) / determinant
        step_shared = (h_peak * g_shared - h_joint * g_peak) / determinant
        step_size = max(abs(step_peak), abs(step_shared))
        if step_size <= 1e-8 * (1 + max(map(abs, slopes))):
            return (log(total) - normalizer, *slopes), total / sum(exposure.values()), set(exposure)

        # Small Newton steps can improve the score below objective rounding error.
        # For larger steps, backtrack on the profile objective.
        scale = 1.0
        descent = g_peak * step_peak + g_shared * step_shared
        for _ in range(40):
            candidate = (slopes[0] - scale * step_peak, slopes[1] - scale * step_shared)
            candidate_objective, _, _ = _profile(candidate, groups, total)
            rounding = 4 * ulp(objective)
            if step_size <= 1e-5 or candidate_objective <= objective - 1e-4 * scale * descent + rounding:
                slopes = candidate
                break
            scale *= 0.5
        else:
            break
    raise ValueError("阻断模型未在迭代上限内收敛，请核对训练期的车次和阻断次数。")


def _deviance(count, expected):
    if count == 0:
        return 2 * expected
    return max(0.0, 2 * (count * (log(count) - log(expected)) - count + expected))


def _planning(windows, keys, threshold, robust_by_intersection=None):
    groups = defaultdict(list)
    for row in windows:
        groups[tuple(row[key] for key in keys)].append(row)
    plan = []
    for group, records in groups.items():
        exposure = sum(row["exposure"] for row in records)
        expected = fsum(row["predicted"] for row in records)
        rate = 100 * expected / exposure
        severity = sum(rate >= limit or isclose(rate, limit, rel_tol=1e-12)
                       for limit in (threshold, 2 * threshold))
        robust = bool(robust_by_intersection and group[-1] in robust_by_intersection)
        if robust_by_intersection is not None and keys == ("road", "intersection"):
            # 只有在每种可行分配下都领先的路口才给「优先」，其余封顶为重点。
            severity = 2 if robust else min(severity, 1)
        entry = {
            **dict(zip(keys, group)), "exposure": exposure, "predicted": expected,
            "rate": rate, "severity": ("常规", "重点", "优先")[severity],
            "action": ("按常规频次抽查", "增排路口巡查", "优先复核放行相位与混行冲突")[severity],
        }
        if robust_by_intersection is not None:
            entry["robust_lead"] = robust
        plan.append(entry)
    return sorted(plan, key=lambda row: (-row["predicted"], -row["rate"], *(row[key] for key in keys)))


def _table(title, columns, rows):
    return {"title": title, "columns": [{"key": key, "label": label} for key, label in columns], "rows": rows}


def _glm_unit_rate(row, coefficients):
    """每个车次的 GLM 预计阻断率，只取决于窗口的时段与通行方式。"""
    intercept, peak_coefficient, shared_coefficient = coefficients
    peak, shared = _features(row)
    return exp(intercept + peak_coefficient * peak + shared_coefficient * shared)


def _build_reconciliation(holdout_rows, coefficients):
    """按留出日做跨系统对账，返回每日结果、可行范围与稳健领先集合。"""
    from .reconcile import InfeasibleFlow, reconcile_day

    by_day = defaultdict(list)
    for row in holdout_rows:
        by_day[row["date"]].append(row)
    days = []
    robust_by_day = {}
    ranges_by_day = {}
    for day in sorted(by_day):
        day_rows = by_day[day]
        signal_totals = {}
        duty_totals = {}
        active = False
        for row in day_rows:
            if row.get("signal_total") is not None:
                signal_totals[row["road"]] = row["signal_total"]
                active = True
            if row.get("duty_total") is not None:
                duty_totals[row["shift"]] = row["duty_total"]
                active = True
        entry = {"date": day, "active": active}
        if active:
            try:
                detail = reconcile_day(day_rows, signal_totals, duty_totals,
                                       lambda row, c=coefficients: _glm_unit_rate(row, c))
            except InfeasibleFlow as error:
                detail = {"feasible": False, "reason": str(error),
                          "signal_totals": signal_totals, "duty_totals": duty_totals}
            entry.update(detail)
            if detail.get("feasible"):
                robust_by_day[day] = set(detail["robust_leaders"])
                ranges_by_day[day] = detail["ranges"]
        days.append(entry)
    return days, robust_by_day, ranges_by_day


def analyze(rows, normalized_params):
    readings = validate_rows(rows)
    if set(normalized_params) - {field["key"] for field in SPEC["params"]}:
        raise ValueError("存在未知运行参数")
    params = {field["key"]: _validate_field(field, normalized_params.get(field["key"], field["default"]))
              for field in SPEC["params"]}
    days = sorted({row["date"] for row in readings})
    holdout_days = params["holdout_days"]
    if len(days) <= holdout_days:
        raise ValueError("留出观测日数须少于台账中的不同日期数，以保留训练数据。")
    cutoff = days[-holdout_days]
    training = [row for row in readings if row["date"] < cutoff]
    holdout = [row for row in readings if row["date"] >= cutoff]
    if any(row.get("exposure") is None for row in training):
        raise ValueError("训练期不能包含漏记窗口，请补充实际车次或调大留出观测日数。")
    coefficients, baseline_rate, seen = _fit(training)
    intercept, peak_coefficient, shared_coefficient = coefficients
    reconciliation_mode = any(
        row.get("signal_total") is not None or row.get("duty_total") is not None
        or row.get("exposure") is None for row in holdout)
    reconciliation = robust_by_day = ranges_by_day = None
    if reconciliation_mode:
        reconciliation, robust_by_day, ranges_by_day = _build_reconciliation(holdout, coefficients)

    windows = []
    for row in holdout:
        peak, shared = _features(row)
        unit_rate = _glm_unit_rate(row, coefficients)
        confirmed = row.get("exposure") is not None
        exposure = row["exposure"] if confirmed else None
        expected = exposure * unit_rate if confirmed else None
        baseline = exposure * baseline_rate if confirmed else None
        windows.append({
            **row, "predicted": expected, "baseline": baseline,
            "unseen": (peak, shared) not in seen,
            "window_state": "已确认" if confirmed else "漏记",
            "unit_rate": unit_rate,
            "model_deviance": _deviance(row["count"], expected) if confirmed else None,
            "baseline_deviance": _deviance(row["count"], baseline) if confirmed else None,
        })
    confirmed_windows = [row for row in windows if row["model_deviance"] is not None]
    if confirmed_windows:
        mean_model = fsum(row["model_deviance"] for row in confirmed_windows) / len(confirmed_windows)
        mean_baseline = fsum(row["baseline_deviance"] for row in confirmed_windows) / len(confirmed_windows)
        better = mean_model < mean_baseline and not isclose(mean_model, mean_baseline, rel_tol=1e-12, abs_tol=1e-12)
    else:
        # 留出期全部为漏记窗口时没有观测点，无法比较 Poisson 偏差。
        mean_model = mean_baseline = None
        better = False
    unseen_count = sum(row["unseen"] for row in windows)
    threshold = params["severity_rate"]

    if reconciliation_mode:
        feasible_days = [item for item in reconciliation if item.get("feasible")]
        infeasible_days = [item for item in reconciliation if item.get("active") and not item.get("feasible")]
        all_feasible = not infeasible_days
        # 各路口在整个留出期的预计范围：对得上的日子按日 min/max 求和，漏记日不产生固定点估计。
        intersection_ranges = defaultdict(lambda: {"min": 0.0, "max": 0.0})
        for day_ranges in ranges_by_day.values():
            for intersection, span in day_ranges.items():
                intersection_ranges[intersection]["min"] += span["min"]
                intersection_ranges[intersection]["max"] += span["max"]
        # 只有每种分配都领先（每个对得上的日子均稳健领先）的路口才整体优先。
        robust_overall = None
        if all_feasible and feasible_days:
            robust_overall = set.intersection(*(set(item["robust_leaders"]) for item in feasible_days))
        elif feasible_days:
            robust_overall = set()
        # 规划情景只用对得上的日期：基准可行分配回填漏记窗口，对不上的日期不进建议。
        feasible_dates = {item["date"] for item in feasible_days}
        scenario_rows = _scenario_windows(
            [row for row in windows if row["date"] in feasible_dates],
            reconciliation, coefficients, baseline_rate)
        roads = _planning(scenario_rows, ("road",), threshold)
        intersections = _planning(scenario_rows, ("road", "intersection"), threshold,
                                  robust_overall if robust_overall is not None else {})
        return _reconciliation_result(
            windows, reconciliation, feasible_days, infeasible_days, all_feasible,
            intersection_ranges, robust_overall, roads, intersections, coefficients,
            baseline_rate, mean_model, mean_baseline, better, unseen_count,
            threshold, days, cutoff, training, holdout_days)

    # 旧台账模式：剥离对账专属的可空字段，报表与原结构逐字段一致。
    windows = [{key: row[key] for key in row if key in _LEGACY_FIELD_KEYS
                or key in {"predicted", "baseline", "unseen", "model_deviance", "baseline_deviance"}}
               for row in windows]
    roads = _planning(windows, ("road",), threshold)
    intersections = _planning(windows, ("road", "intersection"), threshold)
    focus = intersections[0]
    plan_columns = (("exposure", "情景车次"), ("predicted", "预计阻断次数"), ("rate", "预计次／百车次"),
                    ("severity", "巡查级别"), ("action", "排班动作"))
    coefficient_rows = [
        {"term": term, "coefficient": coefficient, "multiplier": exp(coefficient)}
        for term, coefficient in zip(("截距（平峰、独立通行）", "高峰（对比平峰）", "混行（对比独立通行）"), coefficients)
    ]
    note = (f"训练 {days[0]} 至 {training[-1]['date']}，留出 {cutoff} 至 {days[-1]}。"
            f"重点阈值 {threshold:g} 次／百车次，优先阈值 {2 * threshold:g} 次／百车次。")
    comparison = "GLM 留出偏差低于基线" if better else "GLM 留出偏差未低于基线"
    return {
        "mode": "legacy",
        "state": "ok" if better and not unseen_count else "warning",
        "headline": f"{comparison}，{focus['road']} · {focus['intersection']}预计工作量最高。",
        "note": note,
        "metrics": [
            {"label": "训练窗口", "value": len(training), "detail": f"{len(days) - holdout_days} 个观测日"},
            {"label": "留出窗口", "value": len(windows), "detail": f"{holdout_days} 个观测日 · {unseen_count} 个窗口的条件组合未见于训练"},
            {"label": "GLM 平均偏差", "value": mean_model, "detail": "留出期每窗口 Poisson 偏差"},
            {"label": "基线平均偏差", "value": mean_baseline, "detail": f"训练基线 {100 * baseline_rate:.4g} 次／百车次"},
            {"label": "偏差减少量", "value": mean_baseline - mean_model, "detail": "基线平均偏差减去 GLM 平均偏差"},
        ],
        "charts": [
            {"title": "留出窗口平均 Poisson 偏差", "kind": "bar", "series": [
                {"label": "GLM", "value": mean_model}, {"label": "单一阻断率", "value": mean_baseline},
            ]},
            {"title": "预计阻断次数最高的路口（前 10）", "kind": "bar", "series": [
                {"label": f"{row['road']} · {row['intersection']}", "value": row["predicted"]}
                for row in intersections[:10]
            ]},
        ],
        "tables": [
            _table("道路巡查计划", (("road", "道路"),) + plan_columns, roads),
            _table("路口巡查计划", (("road", "道路"), ("intersection", "路口")) + plan_columns, intersections),
            _table("模型系数与发生率倍数", (("term", "模型项"), ("coefficient", "系数"),
                 ("multiplier", "每车次基准率或发生率倍数")), coefficient_rows),
            _table("留出窗口核对", tuple((key, next(field["label"] for field in SPEC["fields"] if field["key"] == key))
                 for key in _LEGACY_FIELD_KEYS) + (
                ("predicted", "GLM 预计次数"), ("baseline", "基线预计次数"), ("model_deviance", "GLM 偏差"),
                ("baseline_deviance", "基线偏差"), ("unseen", "未见条件组合"),
            ), windows),
        ],
        "sections": [
            {"title": "系数口径", "text": "截距的指数为平峰、独立通行条件下的每车次发生率。"
             "其余系数取指数后表示另一项条件相同时的发生率倍数。过车车次翻倍时，预计次数也翻倍。"},
            {"title": "排班口径", "text": "巡查表复用留出期的过车车次和条件，按预计阻断次数排序。"
             "路段和路口各自按每百车次的预计发生率定级。复核时请对照留出偏差和未见条件组合。"},
        ],
    }


def _scenario_windows(windows, reconciliation, coefficients, baseline_rate):
    """用基准可行分配把漏记窗口替换成具体车次，形成同等工作量情景。"""
    from .reconcile import decode_key

    reference = {}
    for day in reconciliation:
        if day.get("feasible"):
            for text, value in day["reference_allocation"].items():
                reference[(day["date"],) + decode_key(text)] = value
    scenario = []
    for row in windows:
        if row.get("exposure") is not None:
            scenario.append(dict(row))
            continue
        key = (row["date"], row["road"], row["intersection"], row["shift"])
        exposure = reference[key]
        unit_rate = _glm_unit_rate(row, coefficients)
        scenario.append({**row, "exposure": exposure, "predicted": exposure * unit_rate,
                         "baseline": exposure * baseline_rate})
    return scenario


def _reconciliation_result(windows, reconciliation, feasible_days, infeasible_days, all_feasible,
                           intersection_ranges, robust_overall, roads, intersections, coefficients,
                           baseline_rate, mean_model, mean_baseline, better, unseen_count,
                           threshold, days, cutoff, training, holdout_days):
    coefficient_rows = [
        {"term": term, "coefficient": coefficient, "multiplier": exp(coefficient)}
        for term, coefficient in zip(("截距（平峰、独立通行）", "高峰（对比平峰）", "混行（对比独立通行）"), coefficients)
    ]
    plan_columns = (("exposure", "情景车次"), ("predicted", "预计阻断次数"), ("rate", "预计次／百车次"),
                    ("severity", "巡查级别"), ("action", "排班动作"), ("robust_lead", "每种分配都领先"))
    range_rows = []
    for (road, intersection), row in {(item["road"], item["intersection"]): item for item in intersections}.items():
        span = intersection_ranges.get(intersection, {"min": 0.0, "max": 0.0})
        range_rows.append({"road": road, "intersection": intersection,
                           "predicted_min": span["min"], "predicted_max": span["max"],
                           "robust_lead": bool(robust_overall and intersection in robust_overall)})
    range_rows.sort(key=lambda row: (not row["robust_lead"], -row["predicted_max"], row["road"], row["intersection"]))
    daily_rows = []
    for day in reconciliation:
        if not day.get("active"):
            continue
        daily_rows.append({
            "date": day["date"],
            "feasible": day.get("feasible", False),
            "reason": day.get("reason"),
            "signal_grand": fsum(day["signal_totals"].values()),
            "duty_grand": fsum(day["duty_totals"].values()),
            "signal_totals": day["signal_totals"],
            "duty_totals": day["duty_totals"],
            "robust_leaders": "、".join(day.get("robust_leaders", [])),
        })
    state = "ok" if all_feasible and better and not unseen_count else "warning"
    if infeasible_days:
        headline = f"留出期 {len(infeasible_days)} 个日期两套总数对不上，暂不出巡查建议，请先核查信号台与值班台原数。"
    elif robust_overall:
        leaders = "、".join(sorted(robust_overall))
        headline = f"两套总数全部对上，{leaders} 在每种可行分配下都领先，列为优先巡查。"
    else:
        headline = "两套总数已对上，但没有路口在每种可行分配下都领先，只给重点与常规建议。"
    note = (f"训练 {days[0]} 至 {training[-1]['date']}，留出 {cutoff} 至 {days[-1]}。"
            f"信号台按日×路段、值班台按日×时段对账；已确认车次锁定，漏记窗口在上下限内做整数分配。"
            f"重点阈值 {threshold:g} 次／百车次，优先阈值 {2 * threshold:g} 次／百车次。")
    metrics = [
        {"label": "训练窗口", "value": len(training), "detail": f"{len(days) - holdout_days} 个观测日"},
        {"label": "留出窗口", "value": len(windows), "detail": f"{holdout_days} 个观测日 · {sum(row['window_state'] == '漏记' for row in windows)} 个漏记窗口"},
        {"label": "对账通过日", "value": len(feasible_days), "detail": f"{len(infeasible_days)} 个日期对不上"},
        {"label": "稳健领先路口", "value": len(robust_overall or []), "detail": "在每种可行分配下预计阻断都领先"},
        {"label": "GLM 平均偏差", "value": mean_model,
         "detail": "仅统计已确认窗口的 Poisson 偏差" if mean_model is not None else "留出期无已确认窗口，无法计算偏差"},
        {"label": "基线平均偏差", "value": mean_baseline, "detail": f"训练基线 {100 * baseline_rate:.4g} 次／百车次"},
    ]
    range_table = _table("路口阻断次数可行范围与稳健领先", (
        ("road", "道路"), ("intersection", "路口"),
        ("predicted_min", "预计阻断下限"), ("predicted_max", "预计阻断上限"),
        ("robust_lead", "每种分配都领先")), range_rows)
    daily_table = _table("跨系统对账（按日）", (
        ("date", "观测日期"), ("feasible", "是否对得上"), ("signal_grand", "信号台合计"),
        ("duty_grand", "值班台合计"), ("robust_leaders", "稳健领先路口"), ("reason", "对不上的原因")), daily_rows)
    deviance_series = ([
        {"label": "GLM", "value": mean_model}, {"label": "单一阻断率", "value": mean_baseline}]
        if mean_model is not None else [])
    charts = [
        {"title": "留出窗口平均 Poisson 偏差", "kind": "bar", "series": deviance_series},
        {"title": "各路口预计阻断次数可行区间（对得上的日期合计）", "kind": "bar", "series": [
            {"label": f"{row['road']} · {row['intersection']}（下限）", "value": row["predicted_min"]}
            for row in range_rows[:10]]},
    ]
    return {
        "mode": "reconciliation",
        "state": state,
        "headline": headline,
        "note": note,
        "metrics": metrics,
        "charts": charts,
        "tables": [
            daily_table,
            range_table,
            _table("道路巡查计划", (("road", "道路"),) + plan_columns, roads),
            _table("路口巡查计划", (("road", "道路"), ("intersection", "路口")) + plan_columns, intersections),
            _table("模型系数与发生率倍数", (("term", "模型项"), ("coefficient", "系数"),
                 ("multiplier", "每车次基准率或发生率倍数")), coefficient_rows),
            _table("留出窗口核对", tuple((field["key"], field["label"]) for field in SPEC["fields"]) + (
                ("window_state", "窗口状态"), ("unit_rate", "每车次预计率"),
                ("predicted", "GLM 预计次数"), ("baseline", "基线预计次数"),
                ("model_deviance", "GLM 偏差"), ("baseline_deviance", "基线偏差"), ("unseen", "未见条件组合"),
            ), windows),
        ],
        "reconciliation": reconciliation,
        "all_feasible": all_feasible,
        "infeasible_dates": [row["date"] for row in infeasible_days],
        "robust_leaders": sorted(robust_overall or []),
        "sections": [
            {"title": "对账口径", "text": "每个留出日单独做有界运输：行和等于信号台路段总车次，"
             "列和等于值班台全线分时总车次；已确认窗口车次锁定不可挪动，漏记窗口在上下限内取整数。"
             "两套总数或区间约束无解的日期标记为对不上，并不产生巡查建议。"},
            {"title": "稳健领先口径", "text": "对每个路口在所有可行分配上求预计阻断次数的最小值与最大值；"
             "只有对任意其他路口、在最小差距下仍不落后的路口，才在每种分配下都领先，页面同时保留可推翻领先的具体分配。"},
        ],
    }
