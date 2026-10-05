import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(webRoot, "..");

const parentPairs = [
  ["examples/score_request.json", "data/examples/score_request.json"],
  ["examples/score_request.json", "public/examples/score_request.json"],
  ["examples/score_response.json", "data/examples/score_response.json"],
  ["examples/score_response.json", "public/examples/score_response.json"],
  ["examples/holdout_save_call.json", "data/examples/holdout_save_call.json"],
  ["examples/holdout_save_call.json", "public/examples/holdout_save_call.json"],
  ["examples/holdout_offer.json", "data/examples/holdout_offer.json"],
  ["examples/holdout_offer.json", "public/examples/holdout_offer.json"],
  ["reports/metrics.json", "data/metrics.json"],
  ["config/nba_rules.yaml", "data/nba_rules.yaml"],
  ["reports/figures/churn_roc_pr.png", "public/figures/churn_roc_pr.png"],
  ["assets/churn_lift.png", "public/figures/churn_lift.png"],
  ["assets/holdout_actions.png", "public/figures/holdout_actions.png"],
  ["assets/example_reasons.png", "public/figures/example_reasons.png"],
];

const internalPairs = [
  ["data/examples/score_request.json", "public/examples/score_request.json"],
  ["data/examples/score_response.json", "public/examples/score_response.json"],
  ["data/examples/holdout_save_call.json", "public/examples/holdout_save_call.json"],
  ["data/examples/holdout_offer.json", "public/examples/holdout_offer.json"],
];

function same(left, right) {
  const a = fs.readFileSync(path.join(webRoot, left));
  const b = fs.readFileSync(path.join(webRoot, right));
  if (!a.equals(b)) {
    console.error(`Drift inside web/: ${left} does not match ${right}`);
    return false;
  }
  return true;
}

let ok = true;
for (const [left, right] of internalPairs) {
  ok = same(left, right) && ok;
}

const parentMarker = path.join(repoRoot, "examples/score_request.json");
if (!fs.existsSync(parentMarker)) {
  console.log("Parent repo artifacts are not beside web/; skipped the upstream drift check.");
} else {
  for (const [from, to] of parentPairs) {
    const a = fs.readFileSync(path.join(repoRoot, from));
    const b = fs.readFileSync(path.join(webRoot, to));
    if (!a.equals(b)) {
      console.error(`Drift: web/${to} does not match ${from}`);
      ok = false;
    }
  }
}

if (!ok) {
  process.exit(1);
}

console.log("Copied demo artifacts match.");
