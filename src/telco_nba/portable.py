"""Browser-sized export of the scoring bundle.

The JSON document stores the fitted gradient-boosting trees, the training-split
preprocessor, the Kaplan-Meier curve, and the frozen next-best-action rules.
`score_portable` walks that document with the standard library only. The web
demo ports the same steps to TypeScript so a Vercel page can score customers
without hosting the sklearn process.

Probability uses the leaf accumulation sklearn uses for `decision_function`:
the prior log-odds plus learning_rate times each tree's leaf. Path
contributions are the same decomposition as `telco_nba.explain`: each leaf
equals its root plus the splits along the path. They are not SHAP values.
"""

from __future__ import annotations

import csv
import io
import json
import math
import struct
from pathlib import Path

from telco_nba.data import FEATURE_COLUMNS, load_telco
from telco_nba.explain import original_feature
from telco_nba.model_io import load_bundle
from telco_nba.nba import ADDON_NAMES, decide, eligibility
from telco_nba.paths import CSV_PATH, REPO_ROOT
from telco_nba.pipeline import split_customers
from telco_nba.scoring import EXPLANATION_METHOD

FORMAT = "telco-nba-portable-v1"
ADDON_ORDER = ADDON_NAMES
EXAMPLE_FILES = (
    "examples/score_request.json",
    "examples/holdout_save_call.json",
    "examples/holdout_offer.json",
)
WEB_MODEL = "web/public/model/scoring_model.json"
WEB_HOLDOUT = "web/public/sample/holdout.csv"
WEB_EXAMPLES = "web/public/sample/examples.json"
WEB_PARITY = "web/data/holdout_parity.json"


def r6(value: float) -> float:
    return float(f"{float(value):.6f}")


def export_document(bundle: dict) -> dict:
    schema = bundle["feature_schema"]
    churn = export_pipeline(bundle["churn_models"]["gradient_boosting"])
    models = {"churn": churn}
    for name in ADDON_ORDER:
        models[name] = export_pipeline(bundle["uptake_models"][name]["gradient_boosting"])
    thresholds = bundle["nba"]["thresholds"]
    rules = []
    for rule in bundle["nba"]["rules"]:
        rules.append(
            {
                "id": rule["id"],
                "priority": int(rule["priority"]),
                "action": rule["action"],
                "rationale": rule["rationale"],
            }
        )
    return {
        "format": FORMAT,
        "scoring_model": "gradient_boosting",
        "explanation_method": EXPLANATION_METHOD,
        "top_reasons": int(schema.get("top_reasons", 3)),
        "feature_order": list(schema["feature_order"]),
        "inputs": export_inputs(churn, list(schema["feature_order"])),
        "thresholds": {
            "churn_high": float(thresholds["churn_high"]),
            "churn_medium": float(thresholds["churn_medium"]),
            "clv_high": float(thresholds["clv_high"]),
            "min_offer_propensity": float(thresholds["min_offer_propensity"]),
            "offer_tie_break": thresholds["offer_tie_break"],
        },
        "rules": rules,
        "retention_curve": {
            "times": [float(value) for value in bundle["retention_curve"]["times"]],
            "survival": [float(value) for value in bundle["retention_curve"]["survival"]],
        },
        "models": models,
    }


def export_pipeline(pipeline) -> dict:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    if list(pipeline.classes_) != [0, 1]:
        raise RuntimeError(f"Expected classes [0, 1], found {list(pipeline.classes_)}")
    if classifier.estimators_.shape[1] != 1:
        raise RuntimeError("Expected one tree per boosting stage")
    input_columns = [str(column) for column in preprocessor.feature_names_in_]
    numeric = []
    categorical = []
    encoded_features: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or name == "remainder":
            continue
        columns = [str(column) for column in columns]
        if name == "num":
            imputer = transformer.named_steps["imputer"]
            scaler = transformer.named_steps["scaler"]
            for index, column in enumerate(columns):
                numeric.append(
                    {
                        "name": column,
                        "median": float(imputer.statistics_[index]),
                        "mean": float(scaler.mean_[index]),
                        "scale": float(scaler.scale_[index]),
                    }
                )
                encoded_features.append(column)
        elif name == "cat":
            for index, column in enumerate(columns):
                categories = transformer.categories_[index]
                integer = all(_is_integer_category(value) for value in categories)
                labels = [
                    str(int(value)) if integer else str(value)
                    for value in categories
                ]
                categorical.append({"name": column, "integer": integer, "categories": labels})
                encoded_features.extend([column] * len(labels))
        else:
            raise RuntimeError(f"Unexpected transformer {name}")
    mapped = [
        original_feature(encoded_name, input_columns)
        for encoded_name in preprocessor.get_feature_names_out()
    ]
    if mapped != encoded_features:
        raise RuntimeError("Encoded columns do not map back to the original features")
    trees = []
    for estimator in classifier.estimators_[:, 0]:
        tree = estimator.tree_
        trees.append(
            {
                "l": [int(value) for value in tree.children_left],
                "r": [int(value) for value in tree.children_right],
                "f": [int(value) for value in tree.feature],
                "t": [float(value) for value in tree.threshold],
                "v": [float(value) for value in tree.value[:, 0, 0]],
            }
        )
    return {
        "input_columns": input_columns,
        "numeric": numeric,
        "categorical": categorical,
        "encoded_features": encoded_features,
        "learning_rate": float(classifier.learning_rate),
        "prior": float(classifier.init_.class_prior_[1]),
        "trees": trees,
    }


def export_inputs(churn: dict, feature_order: list[str]) -> list[dict]:
    by_name = {}
    for spec in churn["numeric"]:
        by_name[spec["name"]] = {
            "name": spec["name"],
            "kind": "number",
            "allow_blank": spec["name"] == "TotalCharges",
        }
    for spec in churn["categorical"]:
        by_name[spec["name"]] = {
            "name": spec["name"],
            "kind": "category",
            "integer": spec["integer"],
            "options": list(spec["categories"]),
        }
    inputs = []
    for name in feature_order:
        if name not in by_name:
            raise RuntimeError(f"Churn preprocessor is missing {name}")
        inputs.append(by_name[name])
    return inputs


def _is_integer_category(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return True
        if isinstance(value, np.floating):
            return float(value).is_integer()
    except ImportError:
        pass
    if isinstance(value, float):
        return value.is_integer()
    return False


def dumps_document(document: dict) -> str:
    return json.dumps(document, separators=(",", ":"), allow_nan=False) + "\n"


def normalize_features(document: dict, features: dict) -> dict:
    """Coerce one request into the types POST /score accepts."""
    inputs = {item["name"]: item for item in document["inputs"]}
    normalized = {}
    for name in document["feature_order"]:
        if name not in features:
            raise ValueError(f"Missing {name}")
        spec = inputs[name]
        raw = features[name]
        if spec["kind"] == "number":
            normalized[name] = _numeric(raw, name)
        else:
            key = _category_key(raw, spec)
            normalized[name] = int(key) if spec["integer"] else key
    return normalized


def _numeric(raw, name: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            return None
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if math.isnan(number):
        return None
    return number


def _category_key(raw, column: dict) -> str:
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        raise ValueError(f"{column['name']} is required")
    if column["integer"]:
        try:
            return str(int(float(str(raw).strip())))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column['name']} must be an integer code") from exc
    if isinstance(raw, str):
        return raw.strip()
    return str(raw)


def transform_row(spec: dict, features: dict) -> list[float]:
    encoded: list[float] = []
    for column in spec["numeric"]:
        number = _numeric(features.get(column["name"]), column["name"])
        if number is None:
            number = column["median"]
        scale = column["scale"] or 1.0
        encoded.append((number - column["mean"]) / scale)
    for column in spec["categorical"]:
        key = _category_key(features.get(column["name"]), column)
        encoded.extend(1.0 if key == category else 0.0 for category in column["categories"])
    if len(encoded) != len(spec["encoded_features"]):
        raise RuntimeError("Transformed width does not match the exported feature map")
    return encoded


def _sigmoid(decision: float) -> float:
    if decision >= 0:
        z = math.exp(-decision)
        return 1.0 / (1.0 + z)
    z = math.exp(decision)
    return z / (1.0 + z)


def to_float32(value: float) -> float:
    """Match the float32 cast sklearn's tree uses before a split."""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def predict_tree(spec: dict, encoded: list[float], *, float32_splits: bool) -> tuple[float, dict[str, float]]:
    """Walk one gradient-boosting model.

    sklearn's tree predicts with float32 features, so probabilities must use
    that cast. Path contributions in this repo walk the float64 preprocessor
    output, matching `explain_log_odds`. A value that sits on a threshold can
    take different branches under the two casts.
    """
    observed_values = [to_float32(value) if float32_splits else value for value in encoded]
    prior = float(spec["prior"])
    prior = min(max(prior, 1e-12), 1.0 - 1e-12)
    init = math.log(prior / (1.0 - prior))
    learning_rate = float(spec["learning_rate"])
    contributions = [0.0] * len(encoded)
    decision = init
    for tree in spec["trees"]:
        left = tree["l"]
        right = tree["r"]
        feature_index = tree["f"]
        threshold = tree["t"]
        value = tree["v"]
        node = 0
        while left[node] != -1:
            feature = feature_index[node]
            observed = observed_values[feature]
            if math.isnan(observed):
                raise ValueError("Cannot score a missing transformed feature")
            child = left[node] if observed <= threshold[node] else right[node]
            contributions[feature] += learning_rate * (value[child] - value[node])
            node = child
        decision += learning_rate * value[node]
    grouped = {name: 0.0 for name in spec["input_columns"]}
    for index, name in enumerate(spec["encoded_features"]):
        grouped[name] += contributions[index]
    return decision, grouped


def expected_remaining_months(curve: dict, tenure: float) -> float:
    times = curve["times"]
    survival = curve["survival"]
    if len(times) != len(survival) or not times:
        raise ValueError("Retention curve times and survival must be the same length")
    horizon = float(times[-1])
    if tenure >= horizon:
        return 0.0
    left = _bisect_right(times, tenure) - 1
    if left < 0:
        left = 0
    survival_now = float(survival[left])
    if survival_now <= 1e-12:
        return 0.0
    area = survival_now * (float(times[left + 1]) - tenure)
    for step in range(left + 1, len(times) - 1):
        area += float(survival[step]) * (float(times[step + 1]) - float(times[step]))
    return area / survival_now


def _bisect_right(values: list[float], target: float) -> int:
    low = 0
    high = len(values)
    while low < high:
        mid = (low + high) // 2
        if target < values[mid]:
            high = mid
        else:
            low = mid + 1
    return low


def top_reasons(contributions: dict[str, float], limit: int) -> list[dict]:
    ranked = sorted(contributions.items(), key=lambda item: (-abs(item[1]), item[0]))
    reasons = []
    for feature, contribution in ranked:
        if contribution == 0:
            continue
        reasons.append(
            {
                "feature": feature,
                "contribution": r6(contribution),
                "direction": "increases_churn_risk" if contribution > 0 else "decreases_churn_risk",
            }
        )
        if len(reasons) == limit:
            break
    return reasons


def score_portable(document: dict, features: dict) -> dict:
    if document.get("format") != FORMAT:
        raise ValueError(f"Unsupported scoring document {document.get('format')!r}")
    normalized = normalize_features(document, features)
    churn_encoded = transform_row(document["models"]["churn"], normalized)
    decision, _grouped_float32 = predict_tree(document["models"]["churn"], churn_encoded, float32_splits=True)
    _decision_float64, grouped = predict_tree(document["models"]["churn"], churn_encoded, float32_splits=False)
    probability = _sigmoid(decision)
    flags = eligibility(normalized)
    addon_scores = {}
    propensities: dict[str, float | None] = {}
    for name in ADDON_ORDER:
        if not flags[name]["eligible"]:
            addon_scores[name] = {
                "eligible": False,
                "probability": None,
                "reason": flags[name]["reason"],
            }
            propensities[name] = None
            continue
        addon_decision, _grouped = predict_tree(
            document["models"][name],
            transform_row(document["models"][name], normalized),
            float32_splits=True,
        )
        addon_probability = _sigmoid(addon_decision)
        addon_scores[name] = {"eligible": True, "probability": r6(addon_probability), "reason": None}
        propensities[name] = addon_probability
    monthly = float(normalized["MonthlyCharges"])
    remaining = expected_remaining_months(document["retention_curve"], float(normalized["tenure"]))
    monthly_r = r6(monthly)
    remaining_r = r6(remaining)
    rules_by_id = {rule["id"]: rule for rule in document["rules"]}
    action = decide(
        churn_probability=probability,
        clv=monthly * remaining,
        addon_propensities=propensities,
        eligible={name: flags[name]["eligible"] for name in ADDON_ORDER},
        thresholds=document["thresholds"],
        rules_by_id=rules_by_id,
    )
    horizon = float(document["retention_curve"]["times"][-1])
    return {
        "churn_probability": r6(probability),
        "scoring_model": document["scoring_model"],
        "top_reasons": top_reasons(grouped, int(document["top_reasons"])),
        "next_best_action": action,
        "addon_propensities": addon_scores,
        "clv_proxy": {
            "monthly_charges": monthly_r,
            "expected_remaining_months": remaining_r,
            "value": r6(monthly_r * remaining_r),
            "horizon_months": r6(horizon),
        },
        "explanation_method": document["explanation_method"],
        "_unrounded_churn_probability": probability,
    }


def public_score(document: dict, features: dict) -> dict:
    """Score payload without the unrounded helper field."""
    payload = score_portable(document, features)
    payload.pop("_unrounded_churn_probability", None)
    return payload


def _load_original_rows() -> tuple[list[str], dict[str, dict[str, str]]]:
    with CSV_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("Telco CSV is missing a header")
        fieldnames = list(reader.fieldnames)
        rows = {row["customerID"]: row for row in reader}
    return fieldnames, rows


def _holdout_table() -> tuple[list[str], list[dict[str, str]], list[dict]]:
    fieldnames, by_id = _load_original_rows()
    frame = load_telco()
    _train, test = split_customers(frame)
    missing = [customer_id for customer_id in test["customerID"] if customer_id not in by_id]
    if missing:
        raise RuntimeError(f"Holdout ids missing from the CSV: {missing[:3]}")
    raw_rows = [by_id[customer_id] for customer_id in test["customerID"]]
    typed_rows = []
    for record in test.to_dict(orient="records"):
        features = {}
        for column in FEATURE_COLUMNS:
            value = record[column]
            if column == "TotalCharges" and _is_missing(value):
                features[column] = None
            elif column == "SeniorCitizen":
                features[column] = int(value)
            elif column == "tenure":
                features[column] = int(value)
            elif column in ("MonthlyCharges", "TotalCharges"):
                features[column] = float(value)
            else:
                features[column] = value
        typed_rows.append(
            {
                "customer_id": record["customerID"],
                "historical_churn": record["Churn"],
                "features": features,
            }
        )
    return fieldnames, raw_rows, typed_rows


def _is_missing(value) -> bool:
    try:
        import math as _math

        return value is None or (isinstance(value, float) and _math.isnan(value))
    except TypeError:
        return False


def _csv_text(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _example_customers() -> list[dict]:
    frame = load_telco()
    lookup = {}
    for record in frame.to_dict(orient="records"):
        features = {}
        for column in FEATURE_COLUMNS:
            value = record[column]
            if column == "TotalCharges" and isinstance(value, float) and math.isnan(value):
                features[column] = None
            elif column == "SeniorCitizen":
                features[column] = int(value)
            elif column == "tenure":
                features[column] = int(value)
            elif column in ("MonthlyCharges", "TotalCharges"):
                features[column] = None if isinstance(value, float) and math.isnan(value) else float(value)
            else:
                features[column] = value
        lookup[json.dumps(features, sort_keys=True)] = {
            "customer_id": record["customerID"],
            "historical_churn": record["Churn"],
        }
    customers = []
    for relative in EXAMPLE_FILES:
        features = json.loads((REPO_ROOT / relative).read_text())
        found = lookup.get(json.dumps(features, sort_keys=True))
        if found is None:
            raise RuntimeError(f"Could not match {relative} to a CSV row")
        customers.append(
            {
                "source": relative,
                "customer_id": found["customer_id"],
                "historical_churn": found["historical_churn"],
                "features": features,
            }
        )
    return customers


def render_web_demo_files() -> dict[str, str]:
    """Return repo-relative path to file text for the browser demo."""
    bundle = load_bundle()
    document = json.loads(dumps_document(export_document(bundle)))
    fieldnames, raw_rows, typed_rows = _holdout_table()
    examples = _example_customers()
    parity_rows = []
    labels = []
    probabilities = []
    for row in typed_rows:
        scored = score_portable(document, row["features"])
        top_reason = scored["top_reasons"][0]
        parity_rows.append(
            {
                "customer_id": row["customer_id"],
                "historical_churn": row["historical_churn"],
                "churn_probability": scored["churn_probability"],
                "unrounded_churn_probability": scored["_unrounded_churn_probability"],
                "action": scored["next_best_action"]["action"],
                "offer": scored["next_best_action"]["offer"],
                "rule_id": scored["next_best_action"]["rule_id"],
                "top_feature": top_reason["feature"],
                "top_contribution": top_reason["contribution"],
            }
        )
        labels.append(1 if row["historical_churn"] == "Yes" else 0)
        probabilities.append(scored["_unrounded_churn_probability"])
    from telco_nba.metrics import top_decile_stats

    lift = top_decile_stats(labels, probabilities)
    example_scores = []
    for example in examples:
        example_scores.append(
            {
                **example,
                "response": public_score(document, example["features"]),
            }
        )
    parity = {
        "n": len(parity_rows),
        "lift": lift["lift"],
        "top_decile_k": lift["k"],
        "base_rate": lift["base_rate"],
        "top_decile_positive_rate": lift["top_decile_positive_rate"],
        "rows": parity_rows,
        "examples": example_scores,
    }
    return {
        WEB_MODEL: dumps_document(document),
        WEB_HOLDOUT: _csv_text(fieldnames, raw_rows),
        WEB_EXAMPLES: json.dumps({"customers": examples}, indent=2) + "\n",
        WEB_PARITY: json.dumps(parity, indent=2) + "\n",
    }


def write_web_demo_files(root: Path | None = None) -> list[Path]:
    destination_root = root or REPO_ROOT
    written = []
    for relative, text in render_web_demo_files().items():
        path = destination_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        written.append(path)
    return written
