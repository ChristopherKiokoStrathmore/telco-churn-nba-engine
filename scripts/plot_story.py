#!/usr/bin/env python3
"""Draw the README story poster and the supporting charts from this repo.

Loads the committed scoring bundle, scores the same stratified holdout as
training, and checks every headline number against reports/metrics.json and
examples/score_response.json. The people in the poster are an illustration.
The lift, the churn count, and the action bars are measured.

Does not fit a new model.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
import numpy as np
from PIL import Image

from telco_nba.data import FEATURE_COLUMNS, TARGET_COLUMN, binary_label, load_telco
from telco_nba.metrics import classification_metrics
from telco_nba.model_io import load_bundle
from telco_nba.paths import EXAMPLE_REQUEST_PATH, EXAMPLE_RESPONSE_PATH, METRICS_PATH, REPO_ROOT
from telco_nba.pipeline import feature_frame, positive_proba, split_customers
from telco_nba.scoring import nba_summary, score_customer

ASSETS = REPO_ROOT / "assets"
FONT_DIR = Path("/tmp/fonts")
FONT_URLS = {
    "fraunces-700.ttf": "https://cdn.jsdelivr.net/fontsource/fonts/fraunces@5.2.9/latin-700-normal.ttf",
    "fraunces-600.ttf": "https://cdn.jsdelivr.net/fontsource/fonts/fraunces@5.2.9/latin-600-normal.ttf",
    "inter-700.ttf": "https://cdn.jsdelivr.net/fontsource/fonts/inter-tight@5.2.7/latin-700-normal.ttf",
    "inter-600.ttf": "https://cdn.jsdelivr.net/fontsource/fonts/inter-tight@5.2.7/latin-600-normal.ttf",
    "inter-500.ttf": "https://cdn.jsdelivr.net/fontsource/fonts/inter-tight@5.2.7/latin-500-normal.ttf",
    "inter-400.ttf": "https://cdn.jsdelivr.net/fontsource/fonts/inter-tight@5.2.7/latin-400-normal.ttf",
}

CREAM = "#F4EFE4"
CARD = "#FFFCF7"
GREEN = "#0B3D2E"
GREEN_MID = "#1B5C45"
GOLD = "#C8962E"
GOLD_SOFT = "#F6E7C4"
CHARCOAL = "#24302C"
MUTED = "#5C675F"
LINE = "#E4D8C4"
MIST = "#E7EFEA"
SAND = "#E6DCCB"
WHITE = "#FFFFFF"


def ensure_fonts() -> dict[str, Path]:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in FONT_URLS.items():
        destination = FONT_DIR / name
        if not destination.exists() or destination.stat().st_size < 1000:
            urllib.request.urlretrieve(url, destination)
        font_manager.fontManager.addfont(str(destination))
        paths[name] = destination
    return paths


def fp(paths: dict[str, Path], name: str, size: float):
    font = font_manager.FontProperties(fname=str(paths[name]))
    font.set_size(size)
    return font


def commas(value: int) -> str:
    return f"{int(value):,}"


def round_box(ax, x, y, w, h, radius, face, edge=None, lw=1.0, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge or face,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, fonts, name, size, color, ha="left", va="top", z=5):
    return ax.text(
        x,
        y,
        value,
        fontproperties=fp(fonts, f"{name}.ttf", size),
        color=color,
        ha=ha,
        va=va,
        zorder=z,
    )


def person(ax, x, y, color, scale=1.0, z=4):
    """Flat person. x, y is the top of the head in y-down coordinates."""
    s = scale
    ax.add_patch(Circle((x, y + 11 * s), 11 * s, facecolor=color, edgecolor="none", zorder=z))
    round_box(ax, x - 13 * s, y + 24 * s, 26 * s, 28 * s, 8 * s, color, z=z)
    ax.add_patch(Rectangle((x - 11 * s, y + 50 * s), 7 * s, 22 * s, facecolor=color, edgecolor="none", zorder=z))
    ax.add_patch(Rectangle((x + 4 * s, y + 50 * s), 7 * s, 22 * s, facecolor=color, edgecolor="none", zorder=z))


def telco_mark(ax, x, y, scale=1.0, z=4):
    """Small telco building with a mast. x, y is the top-left."""
    s = scale
    mast_x = x + 36 * s
    ax.plot([mast_x, mast_x], [y + 8 * s, y + 36 * s], color=GREEN, lw=3 * s, solid_capstyle="round", zorder=z)
    ax.add_patch(Circle((mast_x, y + 6 * s), 4.5 * s, facecolor=GOLD, edgecolor="none", zorder=z + 1))
    for radius, width in ((18, 1.6), (28, 1.4)):
        arc = matplotlib.patches.Arc(
            (mast_x + 2 * s, y + 10 * s),
            radius * s,
            radius * s,
            angle=0,
            theta1=300,
            theta2=60,
            color=GOLD,
            lw=width * s,
            zorder=z,
        )
        ax.add_patch(arc)
    round_box(ax, x, y + 34 * s, 72 * s, 58 * s, 6 * s, GREEN, z=z)
    ax.add_patch(Rectangle((x + 12 * s, y + 48 * s), 14 * s, 14 * s, facecolor=GOLD_SOFT, edgecolor="none", zorder=z + 1))
    ax.add_patch(Rectangle((x + 46 * s, y + 48 * s), 14 * s, 14 * s, facecolor=GOLD_SOFT, edgecolor="none", zorder=z + 1))
    ax.add_patch(Rectangle((x + 28 * s, y + 70 * s), 16 * s, 22 * s, facecolor=CREAM, edgecolor="none", zorder=z + 1))


def exit_mark(ax, x, y, scale=1.0, z=4):
    s = scale
    round_box(ax, x, y, 34 * s, 62 * s, 4 * s, CREAM, GOLD, lw=2.2 * s, z=z)
    ax.add_patch(
        Polygon(
            [(x + 42 * s, y + 24 * s), (x + 58 * s, y + 31 * s), (x + 42 * s, y + 38 * s)],
            closed=True,
            facecolor=GOLD,
            edgecolor="none",
            zorder=z,
        )
    )
    ax.plot(
        [x + 28 * s, x + 44 * s],
        [y + 31 * s, y + 31 * s],
        color=GOLD,
        lw=2.2 * s,
        solid_capstyle="round",
        zorder=z,
    )


def chevron(ax, x, y, scale=1.0, z=4):
    s = scale
    ax.add_patch(
        Polygon(
            [(x, y), (x + 16 * s, y + 14 * s), (x, y + 28 * s), (x + 5 * s, y + 14 * s)],
            closed=True,
            facecolor=GOLD,
            edgecolor="none",
            zorder=z,
        )
    )


def down_chevron(ax, x, y, z=4):
    ax.add_patch(
        Polygon(
            [(x - 7, y), (x + 7, y), (x, y + 10)],
            closed=True,
            facecolor=GOLD,
            edgecolor="none",
            zorder=z,
        )
    )


def flow_steps(ax, fonts, x, y, w, steps, compact=False):
    title_size = 15 if compact else 20
    body_size = 13 if compact else 16
    box_h = 62 if compact else 78
    gap = 16 if compact else 26
    for index, (title, body) in enumerate(steps):
        top = y + index * (box_h + gap)
        round_box(ax, x, top, w, box_h, 12, WHITE, LINE, lw=1.2, z=2)
        round_box(ax, x + 12, top + (box_h - 28) / 2, 28, 28, 14, GREEN, z=3)
        text(
            ax, x + 26, top + box_h / 2, str(index + 1), fonts, "inter-700", 14, WHITE, ha="center", va="center"
        )
        text(ax, x + 52, top + (12 if compact else 16), title, fonts, "inter-700", title_size, GREEN)
        text(ax, x + 52, top + (32 if compact else 44), body, fonts, "inter-400", body_size, MUTED)
        if index < len(steps) - 1:
            down_chevron(ax, x + w / 2, top + box_h + (gap - 10) / 2)
    return y + len(steps) * box_h + (len(steps) - 1) * gap


def action_bars(ax, fonts, x, y, w, counts, compact=False):
    rows = (
        ("Save call", counts["save_call"], GREEN),
        ("Offer", counts["offer"], GOLD),
        ("No action", counts["no_action"], SAND),
    )
    label_w = 92 if compact else 118
    value_w = 64 if compact else 78
    bar_max = w - label_w - value_w
    peak = max(count for _, count, _ in rows)
    row_h = 28 if compact else 36
    gap = 12 if compact else 16
    name_size = 13 if compact else 16
    value_size = 14 if compact else 18
    for index, (name, count, color) in enumerate(rows):
        top = y + index * (row_h + gap)
        text(ax, x, top + row_h / 2, name, fonts, "inter-600", name_size, CHARCOAL, va="center")
        bar_w = bar_max * (count / peak)
        round_box(ax, x + label_w, top, max(bar_w, 8), row_h, 6, color, z=3)
        text(
            ax,
            x + label_w + bar_max + 8,
            top + row_h / 2,
            commas(count),
            fonts,
            "inter-700",
            value_size,
            CHARCOAL,
            va="center",
        )
    return y + 3 * row_h + 2 * gap


def measured_facts() -> dict:
    bundle = load_bundle()
    metrics = json.loads(METRICS_PATH.read_text())
    frame = load_telco()
    _train, test = split_customers(frame)
    labels = binary_label(test[TARGET_COLUMN]).to_numpy()
    features = feature_frame(test)
    model_metrics = {}
    for name, pipeline in bundle["churn_models"].items():
        scores = positive_proba(pipeline, features)
        model_metrics[name] = classification_metrics(labels, scores)

    active = nba_summary(test, bundle, active_only=True)
    request = json.loads(EXAMPLE_REQUEST_PATH.read_text())
    scored = score_customer(request, bundle)
    saved_response = json.loads(EXAMPLE_RESPONSE_PATH.read_text())

    churn_yes = int((frame[TARGET_COLUMN] == "Yes").sum())
    n_rows = int(len(frame))
    rate = churn_yes / n_rows
    if f"{rate:.6f}" != f"{metrics['dataset']['churn_rate']:.6f}":
        raise SystemExit("Churn rate from the CSV does not match reports/metrics.json")
    if churn_yes != metrics["dataset"]["churn_yes"] or n_rows != metrics["dataset"]["n_rows"]:
        raise SystemExit("Churn count from the CSV does not match reports/metrics.json")

    for name, block in model_metrics.items():
        saved = metrics["churn"]["models"][name]
        for key in ("roc_auc", "pr_auc", "top_decile_lift"):
            if f"{block[key]:.6f}" != f"{saved[key]:.6f}":
                raise SystemExit(f"{name} {key} does not match reports/metrics.json")

    saved_counts = metrics["nba"]["test_active_customers"]["action_counts"]
    for key, value in active["action_counts"].items():
        if int(value) != int(saved_counts[key]):
            raise SystemExit(f"Action count {key} does not match reports/metrics.json")
    saved_offers = metrics["nba"]["test_active_customers"]["offer_counts"]
    for key, value in active["offer_counts"].items():
        if int(value) != int(saved_offers[key]):
            raise SystemExit(f"Offer count {key} does not match reports/metrics.json")

    if f"{scored['churn_probability']:.6f}" != f"{saved_response['churn_probability']:.6f}":
        raise SystemExit("Example score does not match examples/score_response.json")
    if scored["next_best_action"]["action"] != saved_response["next_best_action"]["action"]:
        raise SystemExit("Example action does not match examples/score_response.json")
    for index, reason in enumerate(saved_response["top_reasons"]):
        got = scored["top_reasons"][index]
        if got["feature"] != reason["feature"]:
            raise SystemExit("Example reasons do not match examples/score_response.json")
        if f"{got['contribution']:.6f}" != f"{reason['contribution']:.6f}":
            raise SystemExit("Example contributions do not match examples/score_response.json")

    lift = float(model_metrics["gradient_boosting"]["top_decile_lift"])
    if f"{lift:.1f}" != "2.8":
        raise SystemExit(f"Expected the served lift to round to 2.8, got {lift:.1f}")

    return {
        "metrics": metrics,
        "model_metrics": model_metrics,
        "counts": active["action_counts"],
        "offers": active["offer_counts"],
        "churn_yes": churn_yes,
        "n_rows": n_rows,
        "rate": rate,
        "lift": lift,
        "scored": scored,
        "saved_response": saved_response,
        "thresholds": bundle["nba"]["thresholds"],
    }


def new_canvas(width, height, fonts):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor(CREAM)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    ax.set_facecolor(CREAM)
    return fig, ax


def draw_problem(ax, fonts, x, y, w, h, facts, compact=False):
    round_box(ax, x, y, w, h, 18, CARD, LINE, lw=1.2, z=1)
    pad = 18 if compact else 26
    text(ax, x + pad, y + (16 if compact else 22), "THE PROBLEM", fonts, "inter-700", 13 if compact else 15, GOLD)
    text(
        ax,
        x + pad,
        y + (36 if compact else 48),
        "Customers leave",
        fonts,
        "fraunces-700",
        26 if compact else 34,
        GREEN,
    )
    scene_top = y + (78 if compact else 108)
    scene_h = 150 if compact else 230
    round_box(ax, x + pad, scene_top, w - 2 * pad, scene_h, 14, MIST, z=2)
    scale = 0.72 if compact else 1.05
    telco_mark(ax, x + pad + (16 if compact else 28), scene_top + (28 if compact else 48), scale)
    people_y = scene_top + (36 if compact else 78)
    gap = 36 if compact else 52
    base_x = x + pad + (108 if compact else 150)
    for index in range(3):
        person(ax, base_x + index * gap, people_y, GREEN, scale * 0.85)
    leave_x = base_x + 3 * gap + (8 if compact else 16)
    person(ax, leave_x, people_y, GOLD, scale * 0.85)
    person(ax, leave_x + gap, people_y, GOLD, scale * 0.85)
    exit_mark(ax, leave_x + gap + (28 if compact else 42), people_y + (8 if compact else 18), 0.85 if compact else 1.05)
    note_size = 12 if compact else 15
    text(
        ax,
        x + w / 2,
        scene_top + scene_h - (8 if compact else 12),
        "Illustration, not a headcount",
        fonts,
        "inter-500",
        note_size,
        MUTED,
        ha="center",
        va="bottom",
    )
    stat_y = scene_top + scene_h + (14 if compact else 26)
    text(ax, x + pad, stat_y, commas(facts["churn_yes"]), fonts, "fraunces-700", 40 if compact else 64, GREEN)
    text(
        ax,
        x + pad,
        stat_y + (46 if compact else 72),
        f"of {commas(facts['n_rows'])} customers churned",
        fonts,
        "inter-600",
        14 if compact else 18,
        CHARCOAL,
    )
    text(
        ax,
        x + pad,
        stat_y + (66 if compact else 100),
        f"rate {facts['rate']:.6f} on the public IBM sample",
        fonts,
        "inter-400",
        13 if compact else 16,
        MUTED,
    )


def draw_method(ax, fonts, x, y, w, h, compact=False):
    round_box(ax, x, y, w, h, 18, CARD, LINE, lw=1.2, z=1)
    pad = 18 if compact else 26
    text(ax, x + pad, y + (16 if compact else 22), "THE METHOD", fonts, "inter-700", 13 if compact else 15, GOLD)
    text(
        ax,
        x + pad,
        y + (36 if compact else 48),
        "Score, then decide",
        fonts,
        "fraunces-700",
        26 if compact else 34,
        GREEN,
    )
    steps = (
        ("Customer features", "Tenure, contract, services, bill"),
        ("Churn probability", "Gradient boosting, the served model"),
        ("CLV and add-on scores", "Kaplan-Meier proxy and lookalikes"),
        ("Next-best-action rules", "First match: call, offer, or no action"),
    )
    flow_steps(ax, fonts, x + pad, y + (78 if compact else 112), w - 2 * pad, steps, compact=compact)


def draw_result(ax, fonts, x, y, w, h, facts, compact=False):
    round_box(ax, x, y, w, h, 18, CARD, LINE, lw=1.2, z=1)
    pad = 18 if compact else 26
    counts = facts["counts"]
    text(ax, x + pad, y + (16 if compact else 22), "THE RESULT", fonts, "inter-700", 13 if compact else 15, GOLD)
    text(
        ax,
        x + pad,
        y + (36 if compact else 48),
        "Who to contact",
        fonts,
        "fraunces-700",
        26 if compact else 34,
        GREEN,
    )

    tile_y = y + (82 if compact else 118)
    tile_h = 86 if compact else 118
    tile_gap = 10 if compact else 14
    tile_w = (w - 2 * pad - tile_gap) / 2
    round_box(ax, x + pad, tile_y, tile_w, tile_h, 12, GREEN, z=2)
    round_box(ax, x + pad + tile_w + tile_gap, tile_y, tile_w, tile_h, 12, GOLD_SOFT, z=2)
    num_size = 32 if compact else 46
    text(ax, x + pad + 14, tile_y + 8, f"{facts['lift']:.1f}×", fonts, "fraunces-700", num_size, CREAM)
    text(
        ax,
        x + pad + 14,
        tile_y + (46 if compact else 62),
        "top-decile lift",
        fonts,
        "inter-500",
        12 if compact else 15,
        "#D5E6DC",
    )
    text(
        ax,
        x + pad + 14,
        tile_y + (64 if compact else 86),
        f"{facts['lift']:.6f}",
        fonts,
        "inter-600",
        12 if compact else 15,
        GOLD,
    )
    right = x + pad + tile_w + tile_gap
    text(ax, right + 12, tile_y + 8, commas(counts["save_call"]), fonts, "fraunces-700", num_size, GREEN)
    text(ax, right + 12, tile_y + (46 if compact else 62), "save calls", fonts, "inter-600", 13 if compact else 16, GREEN_MID)
    text(
        ax,
        right + 12,
        tile_y + (64 if compact else 86),
        f"of {commas(counts['n'])} active",
        fonts,
        "inter-400",
        12 if compact else 15,
        MUTED,
    )
    chart_y = tile_y + tile_h + (16 if compact else 22)
    text(
        ax,
        x + pad,
        chart_y,
        "Held-out actions, Churn No",
        fonts,
        "inter-600",
        13 if compact else 16,
        CHARCOAL,
    )
    action_bars(ax, fonts, x + pad, chart_y + (20 if compact else 28), w - 2 * pad, counts, compact=compact)
    offers = facts["offers"]
    text(
        ax,
        x + pad,
        y + h - (14 if compact else 20),
        f"Offers: {offers['OnlineSecurity']} OnlineSecurity, {offers['TechSupport']} TechSupport",
        fonts,
        "inter-500",
        12 if compact else 15,
        MUTED,
        va="bottom",
    )


def draw_hero(facts, fonts) -> Path:
    width, height = 1600, 800
    fig, ax = new_canvas(width, height, fonts)
    text(ax, 56, 34, "Who to contact, and with what offer", fonts, "fraunces-700", 40, GREEN)
    text(
        ax,
        56,
        88,
        "Churn risk scoring feeds a next-best-action rule for one customer at a time.",
        fonts,
        "inter-500",
        20,
        MUTED,
    )
    card_y, card_h, card_w = 142, 590, 476
    gap = 40
    xs = [56, 56 + card_w + gap, 56 + 2 * (card_w + gap)]
    draw_problem(ax, fonts, xs[0], card_y, card_w, card_h, facts)
    draw_method(ax, fonts, xs[1], card_y, card_w, card_h)
    draw_result(ax, fonts, xs[2], card_y, card_w, card_h, facts)
    for x in (xs[0] + card_w + 12, xs[1] + card_w + 12):
        chevron(ax, x, card_y + card_h / 2 - 14, scale=1.15)
    text(
        ax,
        56,
        752,
        "Illustration on the left. Lift and action counts are measured on the public IBM telco sample.",
        fonts,
        "inter-500",
        16,
        MUTED,
    )
    destination = ASSETS / "hero.png"
    save_figure(fig, destination, width, height)
    return destination


def draw_social(facts, fonts) -> Path:
    width, height = 1280, 640
    fig, ax = new_canvas(width, height, fonts)
    text(ax, 48, 44, "telco-churn-nba-engine", fonts, "fraunces-700", 32, GREEN)
    text(
        ax,
        48,
        90,
        "Which customers should a retention team contact, and with what offer?",
        fonts,
        "inter-500",
        18,
        CHARCOAL,
    )
    ax.plot([48, 1232], [128, 128], color=GOLD, lw=2, solid_capstyle="round", zorder=3)
    card_y, card_h, card_w = 148, 440, 376
    gap = 28
    xs = [48, 48 + card_w + gap, 48 + 2 * (card_w + gap)]
    draw_problem(ax, fonts, xs[0], card_y, card_w, card_h, facts, compact=True)
    draw_method(ax, fonts, xs[1], card_y, card_w, card_h, compact=True)
    draw_result(ax, fonts, xs[2], card_y, card_w, card_h, facts, compact=True)
    for x in (xs[0] + card_w + 6, xs[1] + card_w + 6):
        chevron(ax, x, card_y + card_h / 2 - 10, scale=0.7)
    destination = ASSETS / "social-preview.png"
    save_figure(fig, destination, width, height)
    return destination


def save_figure(fig, path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    image = Image.open(path)
    if image.size != (width, height):
        raise SystemExit(f"{path.name} is {image.size}, expected {(width, height)}")
    image.save(path, format="PNG", optimize=True)


def styled_figure(width, height, fonts):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CARD)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=13)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fp(fonts, "inter-500.ttf", 14))
    return fig, ax


def draw_holdout_chart(facts, fonts) -> Path:
    width, height = 1400, 760
    fig, ax = styled_figure(width, height, fonts)
    counts = facts["counts"]
    names = ["Save call", "Offer", "No action"]
    keys = ["save_call", "offer", "no_action"]
    values = [counts[key] for key in keys]
    colors = [GREEN, GOLD, SAND]
    positions = np.arange(len(names))[::-1]
    ax.barh(positions, values, color=colors, height=0.62)
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    for label in ax.get_yticklabels():
        label.set_fontproperties(fp(fonts, "inter-600.ttf", 18))
        label.set_color(CHARCOAL)
    ax.set_xlabel("Customers", fontproperties=fp(fonts, "inter-500.ttf", 16), color=MUTED)
    peak = max(values)
    ax.set_xlim(0, peak * 1.18)
    for pos, value in zip(positions, values):
        ax.text(
            value + peak * 0.02,
            pos,
            commas(value),
            va="center",
            fontproperties=fp(fonts, "inter-700.ttf", 18),
            color=CHARCOAL,
        )
    offers = facts["offers"]
    fig.suptitle(
        "Next-best action for held-out customers who have not churned",
        x=0.06,
        y=0.95,
        ha="left",
        fontproperties=fp(fonts, "fraunces-700.ttf", 26),
        color=GREEN,
    )
    fig.text(
        0.06,
        0.88,
        (
            f"Historical Churn No, n = {commas(counts['n'])}. "
            f"Offers: {offers['OnlineSecurity']} OnlineSecurity and {offers['TechSupport']} TechSupport. "
            "Public IBM telco sample."
        ),
        fontproperties=fp(fonts, "inter-400.ttf", 15),
        color=MUTED,
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.98, 0.82))
    destination = ASSETS / "holdout_actions.png"
    save_figure(fig, destination, width, height)
    return destination


def draw_lift_chart(facts, fonts) -> Path:
    width, height = 1400, 760
    fig, ax = styled_figure(width, height, fonts)
    order = ("dummy_prior", "logistic_regression", "gradient_boosting")
    labels = {
        "dummy_prior": "Dummy prior",
        "logistic_regression": "Logistic regression",
        "gradient_boosting": "Gradient boosting",
    }
    colors = [SAND, GREEN_MID, GOLD]
    lifts = [facts["model_metrics"][name]["top_decile_lift"] for name in order]
    rocs = [facts["model_metrics"][name]["roc_auc"] for name in order]
    positions = np.arange(len(order))
    ax.bar(positions, lifts, color=colors, width=0.66)
    ax.axhline(1.0, color=GREEN, lw=1.2, ls=(0, (3, 3)))
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[name] for name in order])
    for label in ax.get_xticklabels():
        label.set_fontproperties(fp(fonts, "inter-600.ttf", 16))
        label.set_color(CHARCOAL)
    ax.set_ylabel("Top-decile lift", fontproperties=fp(fonts, "inter-500.ttf", 16), color=MUTED)
    ax.set_ylim(0, max(lifts) * 1.28)
    for pos, lift, roc in zip(positions, lifts, rocs):
        ax.text(
            pos,
            lift + 0.08,
            f"lift {lift:.6f}\nROC-AUC {roc:.6f}",
            ha="center",
            va="bottom",
            fontproperties=fp(fonts, "inter-600.ttf", 14),
            color=CHARCOAL,
        )
    fig.suptitle(
        "Top-decile lift of the churn models on the holdout",
        x=0.06,
        y=0.95,
        ha="left",
        fontproperties=fp(fonts, "fraunces-700.ttf", 26),
        color=GREEN,
    )
    fig.text(
        0.06,
        0.88,
        "Same stratified split as training. Gradient boosting is the served model. Public IBM telco sample.",
        fontproperties=fp(fonts, "inter-400.ttf", 15),
        color=MUTED,
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.98, 0.82))
    destination = ASSETS / "churn_lift.png"
    save_figure(fig, destination, width, height)
    return destination


def draw_example_chart(facts, fonts) -> Path:
    width, height = 1400, 760
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor(CREAM)
    ax = fig.add_axes((0.07, 0.14, 0.55, 0.58))
    ax.set_facecolor(CARD)
    scored = facts["saved_response"]
    reasons = list(reversed(scored["top_reasons"]))
    names = [reason["feature"] for reason in reasons]
    values = [reason["contribution"] for reason in reasons]
    positions = np.arange(len(names))
    colors = [GREEN if value < 0 else GOLD for value in values]
    ax.barh(positions, values, color=colors, height=0.58)
    ax.axvline(0, color=GREEN, lw=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    for label in ax.get_yticklabels():
        label.set_fontproperties(fp(fonts, "inter-600.ttf", 16))
        label.set_color(CHARCOAL)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.set_xlabel(
        "Path contribution to churn log-odds",
        fontproperties=fp(fonts, "inter-500.ttf", 15),
        color=MUTED,
    )
    span = max(abs(value) for value in values)
    ax.set_xlim(-span * 1.55, span * 0.35)
    for pos, value in zip(positions, values):
        ax.text(
            value - span * 0.04,
            pos,
            f"{value:.6f}",
            ha="right",
            va="center",
            fontproperties=fp(fonts, "inter-600.ttf", 14),
            color=CHARCOAL,
        )
    thresholds = facts["thresholds"]
    probability = scored["churn_probability"]
    card = fig.add_axes((0.68, 0.18, 0.26, 0.54))
    card.set_xlim(0, 1)
    card.set_ylim(0, 1)
    card.set_axis_off()
    card.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0,rounding_size=0.04",
            facecolor=GREEN,
            edgecolor=GREEN,
            transform=card.transAxes,
        )
    )
    detail = (
        ("This customer", "inter-500.ttf", GOLD, 0.86, 15),
        (f"{probability:.6f}", "fraunces-700.ttf", CREAM, 0.68, 28),
        ("churn probability", "inter-400.ttf", "#D5E6DC", 0.54, 15),
        (f"medium cutoff {thresholds['churn_medium']:.6f}", "inter-500.ttf", CREAM, 0.36, 14),
        ("below the cutoff", "inter-400.ttf", "#D5E6DC", 0.26, 14),
        ("no_action", "fraunces-700.ttf", GOLD, 0.12, 26),
    )
    for line, font_file, color, y, size in detail:
        card.text(
            0.08,
            y,
            line,
            transform=card.transAxes,
            fontproperties=fp(fonts, font_file, size),
            color=color,
            va="center",
        )
    fig.suptitle(
        "A score below the medium cutoff returns no action",
        x=0.06,
        y=0.93,
        ha="left",
        fontproperties=fp(fonts, "fraunces-700.ttf", 26),
        color=GREEN,
    )
    fig.text(
        0.06,
        0.86,
        "Example customer 5343-SGUBI. Path contributions are not SHAP. Public IBM telco sample.",
        fontproperties=fp(fonts, "inter-400.ttf", 15),
        color=MUTED,
    )
    destination = ASSETS / "example_reasons.png"
    save_figure(fig, destination, width, height)
    return destination


def main() -> None:
    fonts = ensure_fonts()
    facts = measured_facts()
    paths = [
        draw_hero(facts, fonts),
        draw_social(facts, fonts),
        draw_holdout_chart(facts, fonts),
        draw_lift_chart(facts, fonts),
        draw_example_chart(facts, fonts),
    ]
    for path in paths:
        size = path.stat().st_size
        print(f"{path.relative_to(REPO_ROOT)} {size} bytes")
        if size > 1_000_000:
            raise SystemExit(f"{path.name} is over 1 MB")


if __name__ == "__main__":
    main()
