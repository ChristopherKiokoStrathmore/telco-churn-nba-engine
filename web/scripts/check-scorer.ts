/**
 * Compare the browser scorer with the Python export on the public holdout.
 *
 *   node --experimental-strip-types web/scripts/check-scorer.ts
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { customersFromCsv } from "../lib/customers.ts";
import { scoreCustomer, topDecileLift, type PortableModel, type ScoreResponse } from "../lib/portableScore.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

type ParityRow = {
  customer_id: string;
  historical_churn: "Yes" | "No";
  churn_probability: number;
  action: string;
  offer: string | null;
  rule_id: string;
  top_feature: string;
  top_contribution: number;
};

type ParityFile = {
  n: number;
  lift: number;
  rows: ParityRow[];
  examples: {
    source: string;
    features: Record<string, string | number | null>;
    response: ScoreResponse;
  }[];
};

function readJson<T>(relative: string): T {
  return JSON.parse(fs.readFileSync(path.join(repoRoot, relative), "utf8")) as T;
}

function sameResponse(actual: ScoreResponse, expected: ScoreResponse): boolean {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

function main(): void {
  const model = readJson<PortableModel>("web/public/model/scoring_model.json");
  const parity = readJson<ParityFile>("web/data/holdout_parity.json");
  const csv = fs.readFileSync(path.join(repoRoot, "web/public/sample/holdout.csv"), "utf8");
  const customers = customersFromCsv(csv, model);
  if (customers.length !== parity.n || customers.length !== parity.rows.length) {
    throw new Error(`Holdout length ${customers.length} does not match parity ${parity.n}`);
  }

  const labels: number[] = [];
  const scores: number[] = [];
  let mismatches = 0;
  for (let index = 0; index < customers.length; index += 1) {
    const customer = customers[index];
    const expected = parity.rows[index];
    if (customer.customerId !== expected.customer_id) {
      throw new Error(`Row ${index} id ${customer.customerId} !== ${expected.customer_id}`);
    }
    const scored = scoreCustomer(model, customer.features);
    const response = scored.response;
    labels.push(customer.label === "Yes" ? 1 : 0);
    scores.push(scored.unroundedChurnProbability);
    const probabilityOk = response.churn_probability.toFixed(6) === expected.churn_probability.toFixed(6);
    const reason = response.top_reasons[0];
    const reasonOk =
      reason !== undefined &&
      reason.feature === expected.top_feature &&
      reason.contribution.toFixed(6) === expected.top_contribution.toFixed(6);
    if (
      !probabilityOk ||
      !reasonOk ||
      response.next_best_action.action !== expected.action ||
      response.next_best_action.offer !== expected.offer ||
      response.next_best_action.rule_id !== expected.rule_id
    ) {
      mismatches += 1;
      if (mismatches <= 5) {
        console.error(
          expected.customer_id,
          "got",
          response.churn_probability,
          response.next_best_action,
          reason,
          "expected",
          expected.churn_probability,
          expected.action,
          expected.offer,
          expected.top_feature,
          expected.top_contribution,
        );
      }
    }
  }
  if (mismatches > 0) {
    throw new Error(`${mismatches} holdout rows did not match the Python scorer`);
  }

  const lift = topDecileLift(labels, scores);
  const liftText = lift.lift.toFixed(6);
  if (liftText !== "2.806733") {
    throw new Error(`Holdout lift ${liftText} did not match the committed 2.806733`);
  }
  if (lift.k !== 176) {
    throw new Error(`Expected top-decile k 176, found ${lift.k}`);
  }

  for (const example of parity.examples) {
    const scored = scoreCustomer(model, example.features);
    if (!sameResponse(scored.response, example.response)) {
      console.error(example.source);
      console.error(JSON.stringify(scored.response, null, 2));
      console.error(JSON.stringify(example.response, null, 2));
      throw new Error(`Example ${example.source} did not match the Python response`);
    }
  }

  const committed = readJson<ScoreResponse>("examples/score_response.json");
  const first = parity.examples.find((example) => example.source === "examples/score_request.json");
  if (!first || !sameResponse(first.response, committed)) {
    throw new Error("Parity example does not match examples/score_response.json");
  }

  console.log(
    `Browser scorer matched ${customers.length} holdout rows, lift ${liftText}, and ${parity.examples.length} committed examples.`,
  );
}

main();
