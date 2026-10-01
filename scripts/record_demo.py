#!/usr/bin/env python3
"""Record a short GIF of POST /score against a live local server.

Picks held-out customers from the saved models, posts their features (not the
customer id and not the churn label), and draws the real JSON fields the
server returns. Does not fit a new model.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from telco_nba.clv import clv_proxy
from telco_nba.data import FEATURE_COLUMNS, UPTAKE_TARGETS, load_telco, model_features, uptake_excluded
from telco_nba.model_io import load_bundle
from telco_nba.nba import ADDON_NAMES, decide, eligibility
from telco_nba.paths import EXAMPLE_REQUEST_PATH, METRICS_PATH, REPO_ROOT
from telco_nba.pipeline import SCORING_MODEL, feature_frame, positive_proba, split_customers
from telco_nba.scoring import nba_summary

ASSETS = REPO_ROOT / "assets"
FONT_DIR = Path("/tmp/fonts")
HOST = "127.0.0.1"
PORT = 8000
WIDTH, HEIGHT = 1120, 640

CREAM = (244, 239, 228)
GREEN = (11, 61, 46)
GOLD = (200, 150, 46)
INK = (36, 48, 44)
MUTED = (168, 195, 180)


def payload_from_row(row: pd.Series) -> dict:
    payload = {}
    for column in FEATURE_COLUMNS:
        value = row[column]
        if column in {"tenure", "SeniorCitizen"}:
            payload[column] = int(value)
        elif column in {"MonthlyCharges", "TotalCharges"}:
            payload[column] = None if pd.isna(value) else float(value)
        else:
            payload[column] = str(value)
    return payload


def choose_customers(bundle: dict) -> list[dict]:
    metrics = json.loads(METRICS_PATH.read_text())
    frame = load_telco()
    _train, test = split_customers(frame)
    active = nba_summary(test, bundle, active_only=True)
    saved = metrics["nba"]["test_active_customers"]["action_counts"]
    for key, value in active["action_counts"].items():
        if int(value) != int(saved[key]):
            raise SystemExit(f"Action count {key} drifted from reports/metrics.json")

    wanted = {"save_call": None, "offer": None}
    ranked = test.loc[test["Churn"] == "No"].sort_values("customerID").reset_index(drop=True)
    churn_model = bundle["churn_models"][SCORING_MODEL]
    probabilities = positive_proba(churn_model, feature_frame(ranked))
    values = clv_proxy(ranked["MonthlyCharges"], ranked["tenure"], bundle["retention_curve"])
    uptake_scores = {}
    for name in UPTAKE_TARGETS:
        scores = np.full(len(ranked), np.nan)
        mask = (ranked["InternetService"] != "No") & (ranked[name] == "No")
        if bool(mask.any()):
            columns = model_features(uptake_excluded(name))
            scores[mask.to_numpy()] = positive_proba(
                bundle["uptake_models"][name][SCORING_MODEL],
                ranked.loc[mask, columns],
            )
        uptake_scores[name] = scores
    thresholds = bundle["nba"]["thresholds"]
    rules = bundle["nba"]["rules_by_id"]
    for index, row in ranked.iterrows():
        if pd.isna(row["TotalCharges"]):
            continue
        flags = eligibility(row.to_dict())
        propensities = {
            name: float(uptake_scores[name][index]) if flags[name]["eligible"] else None
            for name in ADDON_NAMES
        }
        decision = decide(
            churn_probability=float(probabilities[index]),
            clv=float(values[index]),
            addon_propensities=propensities,
            eligible={name: flags[name]["eligible"] for name in ADDON_NAMES},
            thresholds=thresholds,
            rules_by_id=rules,
        )
        action = decision["action"]
        if action in wanted and wanted[action] is None:
            wanted[action] = {
                "customer_id": str(row["customerID"]),
                "historical_churn": "No",
                "payload": payload_from_row(row),
                "label": action,
            }
        if all(wanted.values()):
            break
    if not all(wanted.values()):
        raise SystemExit("Holdout did not produce both a save call and an offer")

    example = {
        "customer_id": metrics["example_customer_id"],
        "historical_churn": metrics["example_customer_historical_churn"],
        "payload": json.loads(EXAMPLE_REQUEST_PATH.read_text()),
        "label": "no_action",
        "request_path": "examples/score_request.json",
    }
    return [example, wanted["save_call"], wanted["offer"]]


def wait_for_server() -> None:
    url = f"http://{HOST}:{PORT}/openapi.json"
    for _ in range(50):
        try:
            with urllib.request.urlopen(url, timeout=0.4) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.2)
    raise SystemExit("Scoring server did not start")


def post_score(payload_path: Path) -> dict:
    result = subprocess.run(
        [
            "curl",
            "-s",
            "-X",
            "POST",
            f"http://{HOST}:{PORT}/score",
            "-H",
            "Content-Type: application/json",
            "-d",
            f"@{payload_path}",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return json.loads(result.stdout)


def scene_lines(customer: dict, body: dict) -> list[tuple[str, tuple[int, int, int]]]:
    action = body["next_best_action"]
    offer = action["offer"] if action["offer"] else "null"
    security = body["addon_propensities"]["OnlineSecurity"]
    support = body["addon_propensities"]["TechSupport"]

    def addon(block: dict) -> str:
        if not block["eligible"]:
            return "not eligible"
        return f"{block['probability']:.6f}"

    request = customer["request_path"]
    rows = [
        (
            f"# {customer['customer_id']}   historical Churn {customer['historical_churn']}   id is not sent",
            MUTED,
        ),
        (f"$ curl -s -X POST http://{HOST}:{PORT}/score \\", GOLD),
        (f"    -H 'Content-Type: application/json' -d @{request}", GOLD),
        ("", CREAM),
        (f"churn_probability    {body['churn_probability']:.6f}", CREAM),
        (f"clv_proxy            {body['clv_proxy']['value']:.6f}", CREAM),
        (f"OnlineSecurity       {addon(security)}", CREAM),
        (f"TechSupport          {addon(support)}", CREAM),
        (f"next_best_action     {action['action']}", GOLD),
        (f"offer                {offer}", CREAM),
        (f"rule                 {action['rule_id']}", CREAM),
    ]
    rationale = action["rationale"]
    words = rationale.split()
    current = "rationale            "
    for word in words:
        trial = word if current.endswith(" ") and current.strip() == "rationale" else f"{current} {word}".rstrip()
        if current == "rationale            ":
            trial = f"rationale            {word}"
        if len(trial) > 78 and current.strip() != "rationale":
            rows.append((current.rstrip(), MUTED))
            current = f"                     {word}"
        else:
            current = trial
    rows.append((current.rstrip(), MUTED))
    return rows


def render(header: str, lines: list[tuple[str, tuple[int, int, int]]], fonts) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), GREEN)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 78), fill=CREAM)
    draw.text((36, 16), "telco-churn-nba-engine", font=fonts["title"], fill=GREEN)
    draw.text((36, 48), header, font=fonts["sub"], fill=INK)
    y = 104
    for line, color in lines:
        draw.text((36, y), line, font=fonts["mono"], fill=color)
        y += 32
    return image


def save_gif(frames: list[Image.Image], durations: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    palette = frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    bundle = load_bundle()
    customers = choose_customers(bundle)
    server = subprocess.Popen(
        [
            "uvicorn",
            "telco_nba.api:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
        ],
        cwd=REPO_ROOT,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": "src"},
    )
    try:
        wait_for_server()
        scenes = []
        for customer in customers:
            if "request_path" not in customer:
                filename = {
                    "save_call": "examples/holdout_save_call.json",
                    "offer": "examples/holdout_offer.json",
                }[customer["label"]]
                (REPO_ROOT / filename).write_text(json.dumps(customer["payload"], indent=2) + "\n")
                customer["request_path"] = filename
            body = post_score(Path(customer["request_path"]))
            if body["next_best_action"]["action"] != customer["label"]:
                raise SystemExit(
                    f"{customer['customer_id']} returned {body['next_best_action']['action']}, "
                    f"expected {customer['label']}"
                )
            scenes.append((customer, body))
            print(
                customer["customer_id"],
                body["next_best_action"]["rule_id"],
                body["churn_probability"],
                body["next_best_action"].get("offer"),
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    fonts = {
        "title": ImageFont.truetype(str(FONT_DIR / "fraunces-700.ttf"), 26),
        "sub": ImageFont.truetype(str(FONT_DIR / "inter-500.ttf"), 16),
        "mono": ImageFont.truetype(str(FONT_DIR / "plex-400.ttf"), 20),
    }
    frames = []
    durations = []
    headers = {
        "no_action": "POST /score - churn stays below the offer rule",
        "save_call": "POST /score - high risk and high CLV, so place a save call",
        "offer_high_risk": "POST /score - high risk below the CLV bar, so offer an add-on",
        "offer_medium_risk": "POST /score - moderate risk, so offer an eligible add-on",
    }
    for customer, body in scenes:
        lines = scene_lines(customer, body)
        command = lines[:3]
        header = headers[body["next_best_action"]["rule_id"]]
        frames.append(render(header, command, fonts))
        durations.append(1400)
        frames.append(render(header, lines, fonts))
        durations.append(2600)
    destination = ASSETS / "demo.gif"
    save_gif(frames, durations, destination)
    total_ms = sum(durations)
    size = destination.stat().st_size
    print(f"Wrote {destination} {size} bytes {total_ms} ms {len(frames)} frames")
    if not 5000 <= total_ms <= 15000:
        raise SystemExit(f"GIF duration {total_ms} ms is outside 5-15 s")
    if size > 5_000_000:
        raise SystemExit("GIF is over 5 MB")


if __name__ == "__main__":
    main()
