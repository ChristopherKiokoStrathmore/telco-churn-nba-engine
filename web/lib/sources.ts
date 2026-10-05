import fs from "node:fs";
import path from "node:path";

/**
 * Static copy of committed repo artifacts. Numbers shown in the UI are read
 * from these files and rejected if the rendered token is not in the source text.
 */

const dataDir = path.join(process.cwd(), "data");

function read(relativePath: string): string {
  return fs.readFileSync(path.join(dataDir, relativePath), "utf8");
}

export const metricsText = read("metrics.json");
export const rulesText = read("nba_rules.yaml");
export const responseText = read("examples/score_response.json");
const requestTexts = {
  score: read("examples/score_request.json"),
  save: read("examples/holdout_save_call.json"),
  offer: read("examples/holdout_offer.json"),
};

type ModelMetrics = {
  roc_auc: number;
  pr_auc: number;
  top_decile_lift: number;
  top_decile_k: number;
  base_rate: number;
};

type CountBlock = {
  action_counts: { save_call: number; offer: number; no_action: number; n: number };
  offer_counts: { OnlineSecurity: number; TechSupport: number };
};

type MetricsFile = {
  dataset: {
    name: string;
    source_url: string;
    repository: string;
    repository_license: string;
    data_license_note: string;
    sha256: string;
    n_rows: number;
    n_columns: number;
    churn_yes: number;
    churn_rate: number;
    internet_customers: number;
    blank_total_charges: number;
  };
  split: {
    seed: number;
    test_size: number;
    stratify: string;
    n_train: number;
    n_test: number;
    train_churn_rate: number;
    test_churn_rate: number;
  };
  top_decile_definition: string;
  scoring_model: string;
  churn: {
    positive_label: string;
    test_base_rate: number;
    roc_auc_rank_high_to_low: string[];
    models: Record<string, ModelMetrics>;
  };
  uptake: Record<
    string,
    {
      population: string;
      label: string;
      excluded_features: string[];
      n_train: number;
      n_test: number;
      test_base_rate: number;
      models: Record<string, ModelMetrics>;
    }
  >;
  clv: {
    definition: string;
    horizon_months: number;
    expected_remaining_months_at_tenure_0: number;
    train_clv_p50: number;
    train_clv_p75: number;
  };
  nba: {
    threshold_source: string;
    thresholds: {
      churn_high: number;
      churn_medium: number;
      clv_high: number;
      min_offer_propensity: number;
      offer_tie_break: string;
    };
    test_active_customers: CountBlock;
    test_all_customers: CountBlock;
    active_definition: string;
  };
  example_customer_id: string;
  example_customer_historical_churn: string;
  versions: {
    python: string;
    "scikit-learn": string;
    lifelines: string;
  };
};

type ScoreResponse = {
  churn_probability: number;
  scoring_model: string;
  top_reasons: { feature: string; contribution: number; direction: string }[];
  next_best_action: {
    action: string;
    offer: string | null;
    rule_id: string;
    rationale: string;
  };
  addon_propensities: Record<
    string,
    { eligible: boolean; probability: number | null; reason: string | null }
  >;
  clv_proxy: {
    monthly_charges: number;
    expected_remaining_months: number;
    value: number;
    horizon_months: number;
  };
  explanation_method: string;
};

const metrics = JSON.parse(metricsText) as MetricsFile;
const response = JSON.parse(responseText) as ScoreResponse;

const numberSources = [metricsText, responseText, rulesText, ...Object.values(requestTexts)];

function hasToken(source: string, token: string): boolean {
  const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(?<![\\d.])${escaped}(?![\\d.])`).test(source);
}

/** Render a number only when that exact token exists in a copied source file. */
export function figure(value: number, sources: string[] = numberSources): string {
  const six = value.toFixed(6);
  if (sources.some((source) => hasToken(source, six))) return six;
  const plain = String(value);
  if (sources.some((source) => hasToken(source, plain))) return plain;
  throw new Error(`Refusing to display ${value}; neither ${six} nor ${plain} is in a copied source file`);
}

function jsonField(raw: string, key: string): string {
  const match = raw.match(new RegExp(`"${key}"\\s*:\\s*(-?\\d+(?:\\.\\d+)?|null|"[^"]*")`));
  if (!match) throw new Error(`Missing JSON field ${key}`);
  const token = match[1];
  return token.startsWith('"') ? token.slice(1, -1) : token;
}

function requireText(haystack: string, needle: string, label: string): void {
  if (!haystack.includes(needle)) {
    throw new Error(`Copied ${label} is missing ${JSON.stringify(needle)}`);
  }
}

const MODEL_ORDER = ["dummy_prior", "logistic_regression", "gradient_boosting"] as const;

const MODEL_LABELS: Record<(typeof MODEL_ORDER)[number], string> = {
  dummy_prior: "Dummy prior",
  logistic_regression: "Logistic regression",
  gradient_boosting: "Gradient boosting",
};

export type ModelRow = {
  id: string;
  label: string;
  rocAuc: string;
  prAuc: string;
  lift: string;
  served: boolean;
};

function modelRows(block: Record<string, ModelMetrics>): ModelRow[] {
  return MODEL_ORDER.map((id) => {
    const row = block[id];
    if (!row) throw new Error(`Missing model ${id}`);
    return {
      id,
      label: MODEL_LABELS[id],
      rocAuc: figure(row.roc_auc),
      prAuc: figure(row.pr_auc),
      lift: figure(row.top_decile_lift),
      served: id === metrics.scoring_model,
    };
  });
}

export type RuleView = {
  priority: string;
  id: string;
  action: string;
  offer: string;
  when: string;
  rationale: string;
};

const RULES: RuleView[] = [
  {
    priority: "1",
    id: "save_call",
    action: "save_call",
    offer: "null",
    when: "churn_probability >= churn_high AND clv >= clv_high",
    rationale: "High churn risk and high CLV proxy: place a retention call.",
  },
  {
    priority: "2",
    id: "offer_high_risk",
    action: "offer",
    offer: "best_eligible",
    when: "churn_probability >= churn_high AND at least one add-on is eligible",
    rationale:
      "High churn risk with CLV below the save-call bar: offer the eligible add-on with the higher uptake propensity.",
  },
  {
    priority: "3",
    id: "offer_medium_risk",
    action: "offer",
    offer: "best_eligible_above_cutoff",
    when: "churn_probability >= churn_medium AND an eligible add-on has propensity >= min_offer_propensity",
    rationale: "Moderate churn risk and an eligible add-on whose lookalike propensity meets the policy cutoff.",
  },
  {
    priority: "4",
    id: "no_action",
    action: "no_action",
    offer: "null",
    when: "otherwise",
    rationale: "Churn risk, CLV, and offer eligibility do not match a save call or an offer.",
  },
];

for (const rule of RULES) {
  requireText(rulesText, `id: ${rule.id}`, "nba_rules.yaml");
  requireText(rulesText, `priority: ${rule.priority}`, "nba_rules.yaml");
  requireText(rulesText, `action: ${rule.action}`, "nba_rules.yaml");
  requireText(rulesText, rule.when, "nba_rules.yaml");
  requireText(rulesText, rule.rationale, "nba_rules.yaml");
  if (rule.offer === "null") requireText(rulesText, "offer: null", "nba_rules.yaml");
  else requireText(rulesText, `offer: ${rule.offer}`, "nba_rules.yaml");
}

requireText(rulesText, "scoring_model: gradient_boosting", "nba_rules.yaml");
requireText(rulesText, "churn_high_quantile: 0.75", "nba_rules.yaml");
requireText(rulesText, "churn_medium_quantile: 0.50", "nba_rules.yaml");
requireText(rulesText, "clv_high_quantile: 0.75", "nba_rules.yaml");
requireText(rulesText, "min_offer_propensity: 0.40", "nba_rules.yaml");
requireText(rulesText, "offer_tie_break: OnlineSecurity", "nba_rules.yaml");
requireText(rulesText, "First matching rule wins.", "nba_rules.yaml");

if (metrics.scoring_model !== "gradient_boosting") {
  throw new Error(`Unexpected scoring_model ${metrics.scoring_model}`);
}
if (metrics.nba.thresholds.offer_tie_break !== "OnlineSecurity") {
  throw new Error("Unexpected offer tie break");
}

const rulesById = Object.fromEntries(RULES.map((rule) => [rule.id, rule])) as Record<string, RuleView>;

if (response.scoring_model !== "gradient_boosting") throw new Error("Response scoring model drifted");
if (response.next_best_action.rule_id !== "no_action") throw new Error("Example response rule drifted");
if (response.next_best_action.action !== "no_action") throw new Error("Example response action drifted");
if (response.next_best_action.offer !== null) throw new Error("Example response offer drifted");
if (response.next_best_action.rationale !== rulesById.no_action.rationale) {
  throw new Error("Example rationale does not match nba_rules.yaml");
}

const FIELD_GROUPS: { group: string; keys: string[] }[] = [
  {
    group: "Account",
    keys: [
      "gender",
      "SeniorCitizen",
      "Partner",
      "Dependents",
      "tenure",
      "Contract",
      "PaperlessBilling",
      "PaymentMethod",
    ],
  },
  {
    group: "Services",
    keys: [
      "PhoneService",
      "MultipleLines",
      "InternetService",
      "OnlineSecurity",
      "OnlineBackup",
      "DeviceProtection",
      "TechSupport",
      "StreamingTV",
      "StreamingMovies",
    ],
  },
  {
    group: "Charges",
    keys: ["MonthlyCharges", "TotalCharges"],
  },
];

function fieldsFrom(raw: string): { group: string; items: { key: string; value: string }[] }[] {
  const parsed = JSON.parse(raw) as Record<string, unknown>;
  const actual = new Set(Object.keys(parsed));
  const grouped = new Set(FIELD_GROUPS.flatMap((group) => group.keys));
  for (const key of actual) {
    if (!grouped.has(key)) throw new Error(`Ungrouped request field ${key}`);
  }
  for (const key of grouped) {
    if (!actual.has(key)) throw new Error(`Request is missing ${key}`);
  }
  return FIELD_GROUPS.map((group) => ({
    group: group.group,
    items: group.keys.map((key) => ({ key, value: jsonField(raw, key) })),
  }));
}

function fieldMap(raw: string): Record<string, string> {
  return Object.fromEntries(fieldsFrom(raw).flatMap((group) => group.items.map((item) => [item.key, item.value])));
}

function summaryLine(fields: Record<string, string>): string {
  return [
    fields.InternetService,
    fields.Contract,
    `tenure ${fields.tenure}`,
    `MonthlyCharges ${fields.MonthlyCharges}`,
  ].join(" · ");
}

export type ReasonView = {
  feature: string;
  contribution: string;
  direction: string;
  directionLabel: string;
  barPercent: number;
};

export type ResponseView = {
  file: string;
  publicPath: string;
  text: string;
  churnProbability: string;
  scoringModel: string;
  topReasons: ReasonView[];
  action: string;
  offer: string;
  ruleId: string;
  rationale: string;
  addons: {
    name: string;
    eligible: boolean;
    probability: string;
    reason: string | null;
  }[];
  clv: {
    monthly: string;
    remaining: string;
    value: string;
    horizon: string;
  };
  explanation: string;
};

function directionLabel(direction: string): string {
  if (direction === "decreases_churn_risk") return "Decreases churn risk";
  if (direction === "increases_churn_risk") return "Increases churn risk";
  throw new Error(`Unknown contribution direction ${direction}`);
}

function committedResponse(): ResponseView {
  const magnitudes = response.top_reasons.map((reason) => Math.abs(reason.contribution));
  const maxAbs = Math.max(...magnitudes);
  return {
    file: "examples/score_response.json",
    publicPath: "/examples/score_response.json",
    text: responseText,
    churnProbability: figure(response.churn_probability, [responseText]),
    scoringModel: response.scoring_model,
    topReasons: response.top_reasons.map((reason) => ({
      feature: reason.feature,
      contribution: figure(reason.contribution, [responseText]),
      direction: reason.direction,
      directionLabel: directionLabel(reason.direction),
      barPercent: maxAbs === 0 ? 0 : (Math.abs(reason.contribution) / maxAbs) * 100,
    })),
    action: response.next_best_action.action,
    offer: response.next_best_action.offer ?? "null",
    ruleId: response.next_best_action.rule_id,
    rationale: response.next_best_action.rationale,
    addons: ["OnlineSecurity", "TechSupport"].map((name) => {
      const block = response.addon_propensities[name];
      if (!block) throw new Error(`Missing addon ${name}`);
      return {
        name,
        eligible: block.eligible,
        probability:
          block.probability === null ? "null" : figure(block.probability, [responseText]),
        reason: block.reason,
      };
    }),
    clv: {
      monthly: figure(response.clv_proxy.monthly_charges, [responseText]),
      remaining: figure(response.clv_proxy.expected_remaining_months, [responseText]),
      value: figure(response.clv_proxy.value, [responseText]),
      horizon: figure(response.clv_proxy.horizon_months, [responseText]),
    },
    explanation: response.explanation_method,
  };
}

export type DemoCase = {
  id: "no-action" | "save-call" | "offer";
  index: string;
  title: string;
  requestFile: string;
  publicRequestPath: string;
  requestText: string;
  summary: string;
  customerId: string | null;
  historicalChurn: string | null;
  fields: { group: string; items: { key: string; value: string }[] }[];
  documentedAction: string;
  documentedActionLabel: string;
  gapNote: string | null;
  relatedRules: RuleView[];
  response: ResponseView | null;
};

const scoreFields = fieldMap(requestTexts.score);
const saveFields = fieldMap(requestTexts.save);
const offerFields = fieldMap(requestTexts.offer);

const committed = committedResponse();

if (committed.churnProbability !== "0.122990") {
  throw new Error("score_response.json churn probability is not the committed 0.122990");
}
if (scoreFields.MonthlyCharges !== committed.clv.monthly) {
  throw new Error("Request MonthlyCharges does not match the committed CLV monthly charge");
}

const cases: DemoCase[] = [
  {
    id: "no-action",
    index: "01",
    title: "No action",
    requestFile: "examples/score_request.json",
    publicRequestPath: "/examples/score_request.json",
    requestText: requestTexts.score,
    summary: summaryLine(scoreFields),
    customerId: metrics.example_customer_id,
    historicalChurn: metrics.example_customer_historical_churn,
    fields: fieldsFrom(requestTexts.score),
    documentedAction: "no_action",
    documentedActionLabel: "No action",
    gapNote: null,
    relatedRules: [rulesById.no_action],
    response: committed,
  },
  {
    id: "save-call",
    index: "02",
    title: "Save call",
    requestFile: "examples/holdout_save_call.json",
    publicRequestPath: "/examples/holdout_save_call.json",
    requestText: requestTexts.save,
    summary: summaryLine(saveFields),
    customerId: null,
    historicalChurn: null,
    fields: fieldsFrom(requestTexts.save),
    documentedAction: "save_call",
    documentedActionLabel: "Save call",
    gapNote:
      "The README demo posts this file second and describes the clip as returning a save call. A response JSON is not committed for it, so this panel does not show a churn probability, CLV proxy, path contributions, or offer.",
    relatedRules: [rulesById.save_call],
    response: null,
  },
  {
    id: "offer",
    index: "03",
    title: "Add-on offer",
    requestFile: "examples/holdout_offer.json",
    publicRequestPath: "/examples/holdout_offer.json",
    requestText: requestTexts.offer,
    summary: summaryLine(offerFields),
    customerId: null,
    historicalChurn: null,
    fields: fieldsFrom(requestTexts.offer),
    documentedAction: "offer",
    documentedActionLabel: "Add-on offer",
    gapNote:
      "The README demo posts this file third and describes the clip as returning an add-on offer. A response JSON is not committed for it. The files do not say whether the match was offer_high_risk or offer_medium_risk, or whether the offer was OnlineSecurity or TechSupport.",
    relatedRules: [rulesById.offer_high_risk, rulesById.offer_medium_risk],
    response: null,
  },
];

function countView(block: CountBlock) {
  return {
    saveCall: figure(block.action_counts.save_call),
    offer: figure(block.action_counts.offer),
    noAction: figure(block.action_counts.no_action),
    n: figure(block.action_counts.n),
    onlineSecurity: figure(block.offer_counts.OnlineSecurity),
    techSupport: figure(block.offer_counts.TechSupport),
  };
}

const churnModels = modelRows(metrics.churn.models);
const servedRow = churnModels.find((row) => row.id === "gradient_boosting");
const logisticRow = churnModels.find((row) => row.id === "logistic_regression");
if (!servedRow || !logisticRow) throw new Error("Churn model rows missing");
if (servedRow.rocAuc !== "0.846001" || servedRow.prAuc !== "0.656070" || servedRow.lift !== "2.806733") {
  throw new Error("Served churn metrics are not the committed ROC-AUC, PR-AUC, and top-decile lift");
}

const rocRank = metrics.churn.roc_auc_rank_high_to_low.map((id) => {
  if (!(id in MODEL_LABELS)) throw new Error(`Unknown rank id ${id}`);
  return MODEL_LABELS[id as keyof typeof MODEL_LABELS];
});

const uptakeOrder = ["OnlineSecurity", "TechSupport"] as const;

export const githubUrl = "https://github.com/ChristopherKiokoStrathmore/telco-churn-nba-engine";

export const demo = {
  projectTitle: "Telco churn, propensity, and next-best-action",
  githubUrl,
  modelCardUrl: "https://github.com/ChristopherKiokoStrathmore/responsible-ai-pack",
  served: servedRow,
  logistic: logisticRow,
  churnModels,
  rocRank,
  positiveLabel: metrics.churn.positive_label,
  topDecileDefinition: metrics.top_decile_definition,
  topDecileK: figure(metrics.churn.models.gradient_boosting.top_decile_k),
  dataset: {
    name: metrics.dataset.name,
    sourceUrl: metrics.dataset.source_url,
    repository: metrics.dataset.repository,
    repositoryLicense: metrics.dataset.repository_license,
    dataLicenseNote: metrics.dataset.data_license_note,
    sha256: metrics.dataset.sha256,
    nRows: figure(metrics.dataset.n_rows),
    nColumns: figure(metrics.dataset.n_columns),
    churnYes: figure(metrics.dataset.churn_yes),
    churnRate: figure(metrics.dataset.churn_rate),
    internetCustomers: figure(metrics.dataset.internet_customers),
    blankTotalCharges: figure(metrics.dataset.blank_total_charges),
  },
  split: {
    seed: figure(metrics.split.seed),
    testSize: figure(metrics.split.test_size),
    stratify: metrics.split.stratify,
    nTrain: figure(metrics.split.n_train),
    nTest: figure(metrics.split.n_test),
    trainChurnRate: figure(metrics.split.train_churn_rate),
    testChurnRate: figure(metrics.split.test_churn_rate),
    testBaseRate: figure(metrics.churn.test_base_rate),
  },
  thresholds: {
    churnHigh: figure(metrics.nba.thresholds.churn_high),
    churnMedium: figure(metrics.nba.thresholds.churn_medium),
    clvHigh: figure(metrics.nba.thresholds.clv_high),
    minOfferPropensity: figure(metrics.nba.thresholds.min_offer_propensity),
    offerTieBreak: metrics.nba.thresholds.offer_tie_break,
    source: metrics.nba.threshold_source,
    quantiles: {
      churnHigh: "0.75",
      churnMedium: "0.50",
      clvHigh: "0.75",
    },
  },
  rules: RULES,
  active: countView(metrics.nba.test_active_customers),
  allCustomers: countView(metrics.nba.test_all_customers),
  activeDefinition: metrics.nba.active_definition,
  clv: {
    definition: metrics.clv.definition,
    horizon: figure(metrics.clv.horizon_months),
    remainingAtZero: figure(metrics.clv.expected_remaining_months_at_tenure_0),
    p50: figure(metrics.clv.train_clv_p50),
    p75: figure(metrics.clv.train_clv_p75),
  },
  uptake: uptakeOrder.map((name) => {
    const block = metrics.uptake[name];
    if (!block) throw new Error(`Missing uptake ${name}`);
    return {
      name,
      population: block.population,
      label: block.label,
      excluded: block.excluded_features.join(", "),
      nTrain: figure(block.n_train),
      nTest: figure(block.n_test),
      baseRate: figure(block.test_base_rate),
      k: figure(block.models.gradient_boosting.top_decile_k),
      rows: modelRows(block.models),
    };
  }),
  trainingEnvironment: {
    python: metrics.versions.python,
    sklearn: metrics.versions["scikit-learn"],
    lifelines: metrics.versions.lifelines,
  },
  cases,
  exampleCustomerId: metrics.example_customer_id,
  exampleHistoricalChurn: metrics.example_customer_historical_churn,
};

if (demo.thresholds.minOfferPropensity !== "0.400000") {
  throw new Error("min_offer_propensity token drifted");
}
if (demo.clv.p75 !== demo.thresholds.clvHigh) {
  throw new Error("CLV p75 and clv_high diverged in the copied metrics");
}
if (demo.active.saveCall !== "108" || demo.active.offer !== "121" || demo.active.noAction !== "1065") {
  throw new Error("Active holdout action counts drifted");
}
