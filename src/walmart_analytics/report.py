"""Turn the analysis outputs into figures, a JSON of headline numbers and a
markdown results page. Every number in the README comes from results.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, ACCENT, LINE = "#1d2633", "#6b7480", "#0053e2", "#d9dee5"
STATE_COLORS = {"CA": "#0053e2", "TX": "#ffc220", "WI": "#2a8703"}


def _style(ax, title: str):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=LINE, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_revenue(monthly: pd.DataFrame, out: Path):
    m = monthly.copy()
    m["month"] = pd.to_datetime(m["month"])
    m = m[(m["month"] > m["month"].min()) & (m["month"] < m["month"].max())]  # drop partial first and last months
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for state, g in m.groupby("state_id"):
        ax.plot(g["month"], g["revenue"] / 1e6, color=STATE_COLORS[state], lw=1.8, label=state)
    ax.set_ylabel("Revenue, $M per month", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    _style(ax, "Monthly revenue by state")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_accuracy(scores: pd.DataFrame, out: Path):
    s = scores.sort_values("wape", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = [ACCENT if m == s["wape"].idxmin() else "#9aa6b5" for m in s.index]
    ax.barh(s.index, s["wape"] * 100, color=colors)
    for y, v in enumerate(s["wape"] * 100):
        ax.text(v + 0.2, y, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_xlabel("WAPE, % (lower is better), mean of three 28-day backtests", color=MUTED, fontsize=9)
    _style(ax, "28-day forecast error by model")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=LINE, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_forecast(preds: pd.DataFrame, out: Path):
    last = preds[preds["fold"] == preds["fold"].max()]
    p = last.groupby(["model", "date"], as_index=False)[["units", "pred"]].sum()
    actual = p[p["model"] == "LightGBM"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(actual["date"], actual["units"] / 1e3, color=INK, lw=1.8, label="Actual")
    for model, color, ls in (("LightGBM", ACCENT, "-"), ("Seasonal naive", "#9aa6b5", "--")):
        g = p[p["model"] == model]
        ax.plot(g["date"], g["pred"] / 1e3, color=color, lw=1.5, ls=ls, label=model)
    ax.set_ylabel("Thousand units per day", color=MUTED, fontsize=9)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.18)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    _style(ax, "All stores, final 28 days, forecasts made 28 days ahead")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_elasticity(el: pd.DataFrame, out: Path):
    e = el.sort_values("elasticity", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.errorbar(e["elasticity"], e.index, xerr=1.96 * e["se"], fmt="o", color=ACCENT, ecolor="#9aa6b5", capsize=3)
    ax.axvline(-1, color=MUTED, lw=1, ls="--")
    ax.set_xlabel(
        "Price elasticity of unit sales, with 95% interval (left of the dashed line: elastic)", color=MUTED, fontsize=9
    )
    _style(ax, "How unit sales respond to price, by department")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=LINE, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_inventory(inv: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    v = inv["avg_daily_excess_value"] / 1e3
    colors = [ACCENT if m == v.idxmin() else "#9aa6b5" for m in v.index]
    ax.barh(v.index, v, color=colors)
    for y, (val, fill) in enumerate(zip(v, inv["unit_fill_rate"])):
        ax.text(val + 0.5, y, f"${val:.1f}K  (fill rate {fill:.1%})", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Leftover stock value per day across 10 stores, $K (out-of-sample month)", color=MUTED, fontsize=9)
    _style(ax, "Stock left over when ordering to each forecast plus safety stock")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, v.max() * 1.45)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write(
    outdir: Path, marts: dict, scores: pd.DataFrame, preds: pd.DataFrame, el: pd.DataFrame, inv: pd.DataFrame
) -> dict:
    figs = outdir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig_revenue(marts["revenue_by_month_state"], figs / "revenue_by_state.png")
    fig_accuracy(scores, figs / "forecast_accuracy.png")
    fig_forecast(preds, figs / "forecast_example.png")
    fig_elasticity(el, figs / "price_elasticity.png")
    fig_inventory(inv, figs / "inventory.png")

    best = scores["wape"].idxmin()
    naive = scores.loc["Seasonal naive", "wape"]
    prof = scores.loc["Weekday profile", "wape"]
    inv_b, inv_n = inv.loc[best], inv.loc["Seasonal naive"]
    by_year = marts["revenue_by_year"].set_index("year")
    dept_share = marts["department_mix"].set_index("dept_id")["share"]
    inelastic = el[el["elasticity"] - 1.96 * el["se"] > -1].index.tolist()
    results = {
        "days": int(by_year["days"].sum()),
        "revenue_total": float(by_year["revenue"].sum()),
        "revenue_2015": float(by_year.loc[2015, "revenue"]),
        "growth_2015_vs_2014": float(by_year.loc[2015, "revenue"] / by_year.loc[2014, "revenue"] - 1),
        "category_share": marts["category_mix"].set_index("cat_id")["share"].round(4).to_dict(),
        "concentration_share_for_80pct": float(marts["revenue_concentration"]["share"].iloc[0]),
        "weekend_index": marts["weekday_index"].set_index("weekday")["index"].round(3).to_dict(),
        "snap_uplift": marts["snap_effect"].set_index("state_id")["snap_uplift"].round(4).to_dict(),
        "forecast": {
            "best_model": best,
            "wape": scores["wape"].round(4).to_dict(),
            "accuracy_best": round(1 - float(scores.loc[best, "wape"]), 4),
            "error_cut_vs_seasonal_naive": round(1 - float(scores.loc[best, "wape"]) / naive, 4),
            "error_cut_vs_weekday_profile": round(1 - float(scores.loc[best, "wape"]) / prof, 4),
        },
        "inventory": {
            "service_target": float(inv_b["target_service"]),
            "fill_rate_best": round(float(inv_b["unit_fill_rate"]), 4),
            "fill_rate_naive": round(float(inv_n["unit_fill_rate"]), 4),
            "excess_value_per_day_best": round(float(inv_b["avg_daily_excess_value"]), 2),
            "excess_value_per_day_naive": round(float(inv_n["avg_daily_excess_value"]), 2),
            "excess_cut_vs_naive": round(
                1 - float(inv_b["avg_daily_excess_value"] / inv_n["avg_daily_excess_value"]), 4
            ),
            "safety_stock_cut_vs_naive": round(
                1 - float(inv_b["avg_daily_buffer_value"] / inv_n["avg_daily_buffer_value"]), 4
            ),
        },
        "clearly_inelastic_departments": inelastic,
        "clearly_inelastic_revenue_share": round(float(dept_share.reindex(inelastic).sum()), 4),
        "elasticity": el[["elasticity", "se", "revenue_10pct_off"]].round(4).to_dict(orient="index"),
    }
    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    tables = [
        ("Forecast backtest (WAPE, lower is better)", scores.round(4)),
        ("Safety stock test, final 28 days", inv.round(3)),
        ("Price elasticity by department", el.round(3)),
        (
            "Store scorecard, 2015 (revenue in $M)",
            marts["store_scorecard"].assign(revenue_2015=lambda d: d["revenue_2015"] / 1e6).set_index("store_id"),
        ),
        ("SNAP-day uplift in FOODS revenue", marts["snap_effect"].round(3).set_index("state_id")),
    ]
    md = ["# Results", "", "Generated by `walmart-analytics all`. Do not edit by hand.", ""]
    for title, t in tables:
        md += [f"## {title}", "", t.to_markdown(floatfmt=",.3f", intfmt=","), ""]
    (outdir / "RESULTS.md").write_text("\n".join(md))
    return results
