/**
 * Browser port of telco_nba.portable.score_portable.
 *
 * Probabilities follow sklearn's float32 tree comparisons. Path contributions
 * follow the float64 walk in telco_nba.explain. They are not SHAP values.
 */

export const FORMAT = "telco-nba-portable-v1";
const ADDONS = ["OnlineSecurity", "TechSupport"] as const;

export type AddonName = (typeof ADDONS)[number];

export type TreeNode = {
  l: number[];
  r: number[];
  f: number[];
  t: number[];
  v: number[];
};

export type NumericColumn = {
  name: string;
  median: number;
  mean: number;
  scale: number;
};

export type CategoryColumn = {
  name: string;
  integer: boolean;
  categories: string[];
};

export type ModelSpec = {
  input_columns: string[];
  numeric: NumericColumn[];
  categorical: CategoryColumn[];
  encoded_features: string[];
  learning_rate: number;
  prior: number;
  trees: TreeNode[];
};

export type InputSpec = {
  name: string;
  kind: "number" | "category";
  allow_blank?: boolean;
  integer?: boolean;
  options?: string[];
};

export type RuleSpec = {
  id: string;
  priority: number;
  action: string;
  rationale: string;
};

export type PortableModel = {
  format: string;
  scoring_model: "gradient_boosting";
  explanation_method: string;
  top_reasons: number;
  feature_order: string[];
  inputs: InputSpec[];
  thresholds: {
    churn_high: number;
    churn_medium: number;
    clv_high: number;
    min_offer_propensity: number;
    offer_tie_break: string;
  };
  rules: RuleSpec[];
  retention_curve: { times: number[]; survival: number[] };
  models: {
    churn: ModelSpec;
    OnlineSecurity: ModelSpec;
    TechSupport: ModelSpec;
  };
};

export type FeatureValue = string | number | null;
export type FeatureMap = Record<string, FeatureValue>;

export type Reason = {
  feature: string;
  contribution: number;
  direction: "increases_churn_risk" | "decreases_churn_risk";
};

export type ScoreResponse = {
  churn_probability: number;
  scoring_model: "gradient_boosting";
  top_reasons: Reason[];
  next_best_action: {
    action: "save_call" | "offer" | "no_action";
    offer: AddonName | null;
    rule_id: string;
    rationale: string;
  };
  addon_propensities: Record<
    AddonName,
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

export type ScoreResult = {
  response: ScoreResponse;
  unroundedChurnProbability: number;
  unroundedClv: number;
};

const float32 = new Float32Array(1);

function toFloat32(value: number): number {
  float32[0] = value;
  return float32[0];
}

/** Six-decimal rounding, half to even, matching Python's format specifier. */
export function r6(value: number): number {
  if (!Number.isFinite(value)) {
    throw new Error("Cannot round a non-finite score");
  }
  const formatted = new Intl.NumberFormat("en-US", {
    useGrouping: false,
    minimumFractionDigits: 6,
    maximumFractionDigits: 6,
    roundingMode: "halfEven",
  }).format(value);
  return Number(formatted);
}

export function format6(value: number): string {
  return r6(value).toFixed(6);
}

function numeric(raw: FeatureValue, name: string): number | null {
  if (raw === null || raw === undefined) return null;
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (trimmed === "") return null;
    raw = trimmed;
  }
  const number = typeof raw === "number" ? raw : Number(raw);
  if (!Number.isFinite(number)) {
    throw new Error(`${name} must be a number`);
  }
  return number;
}

function categoryKey(raw: FeatureValue, column: { name: string; integer: boolean }): string {
  if (raw === null || raw === undefined || (typeof raw === "string" && raw.trim() === "")) {
    throw new Error(`${column.name} is required`);
  }
  if (column.integer) {
    const number = typeof raw === "number" ? raw : Number(String(raw).trim());
    if (!Number.isFinite(number)) {
      throw new Error(`${column.name} must be an integer code`);
    }
    return String(Math.trunc(number));
  }
  return typeof raw === "string" ? raw.trim() : String(raw);
}

export function normalizeFeatures(model: PortableModel, features: FeatureMap): FeatureMap {
  const inputs = new Map(model.inputs.map((item) => [item.name, item]));
  const normalized: FeatureMap = {};
  for (const name of model.feature_order) {
    if (!Object.prototype.hasOwnProperty.call(features, name)) {
      throw new Error(`Missing ${name}`);
    }
    const spec = inputs.get(name);
    if (!spec) throw new Error(`Missing input spec for ${name}`);
    const raw = features[name];
    if (spec.kind === "number") {
      normalized[name] = numeric(raw, name);
    } else {
      const key = categoryKey(raw, { name, integer: Boolean(spec.integer) });
      normalized[name] = spec.integer ? Number(key) : key;
    }
  }
  return normalized;
}

function transformRow(spec: ModelSpec, features: FeatureMap): number[] {
  const encoded: number[] = [];
  for (const column of spec.numeric) {
    let number = numeric(features[column.name] ?? null, column.name);
    if (number === null) number = column.median;
    const scale = column.scale || 1;
    encoded.push((number - column.mean) / scale);
  }
  for (const column of spec.categorical) {
    const key = categoryKey(features[column.name] ?? null, column);
    for (const category of column.categories) {
      encoded.push(key === category ? 1 : 0);
    }
  }
  if (encoded.length !== spec.encoded_features.length) {
    throw new Error("Transformed width does not match the exported feature map");
  }
  return encoded;
}

function sigmoid(decision: number): number {
  if (decision >= 0) {
    const z = Math.exp(-decision);
    return 1 / (1 + z);
  }
  const z = Math.exp(decision);
  return z / (1 + z);
}

function predictTree(
  spec: ModelSpec,
  encoded: number[],
  float32Splits: boolean,
): { decision: number; grouped: Map<string, number> } {
  const observed = float32Splits ? encoded.map(toFloat32) : encoded.slice();
  let prior = spec.prior;
  if (prior < 1e-12) prior = 1e-12;
  else if (prior > 1 - 1e-12) prior = 1 - 1e-12;
  const init = Math.log(prior / (1 - prior));
  const learningRate = spec.learning_rate;
  const contributions = new Array<number>(encoded.length).fill(0);
  let decision = init;
  for (const tree of spec.trees) {
    let node = 0;
    while (tree.l[node] !== -1) {
      const feature = tree.f[node];
      const value = observed[feature];
      if (Number.isNaN(value)) {
        throw new Error("Cannot score a missing transformed feature");
      }
      const child = value <= tree.t[node] ? tree.l[node] : tree.r[node];
      contributions[feature] += learningRate * (tree.v[child] - tree.v[node]);
      node = child;
    }
    decision += learningRate * tree.v[node];
  }
  const grouped = new Map<string, number>();
  for (const name of spec.input_columns) grouped.set(name, 0);
  spec.encoded_features.forEach((name, index) => {
    grouped.set(name, (grouped.get(name) ?? 0) + contributions[index]);
  });
  return { decision, grouped };
}

function bisectRight(values: number[], target: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (target < values[mid]) high = mid;
    else low = mid + 1;
  }
  return low;
}

export function expectedRemainingMonths(curve: PortableModel["retention_curve"], tenure: number): number {
  const { times, survival } = curve;
  if (times.length !== survival.length || times.length === 0) {
    throw new Error("Retention curve times and survival must be the same length");
  }
  const horizon = times[times.length - 1];
  if (tenure >= horizon) return 0;
  let left = bisectRight(times, tenure) - 1;
  if (left < 0) left = 0;
  const survivalNow = survival[left];
  if (survivalNow <= 1e-12) return 0;
  let area = survivalNow * (times[left + 1] - tenure);
  for (let step = left + 1; step < times.length - 1; step += 1) {
    area += survival[step] * (times[step + 1] - times[step]);
  }
  return area / survivalNow;
}

function topReasons(contributions: Map<string, number>, limit: number): Reason[] {
  const ranked = [...contributions.entries()].sort((a, b) => {
    const diff = Math.abs(b[1]) - Math.abs(a[1]);
    if (diff !== 0) return diff;
    if (a[0] < b[0]) return -1;
    if (a[0] > b[0]) return 1;
    return 0;
  });
  const reasons: Reason[] = [];
  for (const [feature, contribution] of ranked) {
    if (contribution === 0) continue;
    reasons.push({
      feature,
      contribution: r6(contribution),
      direction: contribution > 0 ? "increases_churn_risk" : "decreases_churn_risk",
    });
    if (reasons.length === limit) break;
  }
  return reasons;
}

function pythonRepr(value: FeatureValue): string {
  if (typeof value === "string") {
    return `'${value.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
  }
  return String(value);
}

function eligibility(customer: FeatureMap): Record<AddonName, { eligible: boolean; reason: string | null }> {
  const internet = customer.InternetService !== "No";
  const result = {} as Record<AddonName, { eligible: boolean; reason: string | null }>;
  for (const name of ADDONS) {
    const holding = customer[name];
    if (!internet) {
      result[name] = { eligible: false, reason: "No internet service, so the add-on is not available." };
    } else if (holding === "Yes") {
      result[name] = { eligible: false, reason: `Customer already has ${name}.` };
    } else if (holding === "No") {
      result[name] = { eligible: true, reason: null };
    } else {
      result[name] = {
        eligible: false,
        reason: `${name} is ${pythonRepr(holding ?? null)}, so the customer is not an eligible cross-sell.`,
      };
    }
  }
  return result;
}

function bestOffer(
  offers: { name: AddonName; propensity: number }[],
  tieBreak: string,
): AddonName | null {
  if (offers.length === 0) return null;
  let best = offers[0];
  for (const offer of offers.slice(1)) {
    const bestTie = best.name === tieBreak ? 1 : 0;
    const offerTie = offer.name === tieBreak ? 1 : 0;
    if (offer.propensity > best.propensity || (offer.propensity === best.propensity && offerTie > bestTie)) {
      best = offer;
    }
  }
  return best.name;
}

function ruleAction(rule: RuleSpec, offer: AddonName | null): ScoreResponse["next_best_action"] {
  return {
    action: rule.action as ScoreResponse["next_best_action"]["action"],
    offer,
    rule_id: rule.id,
    rationale: rule.rationale,
  };
}

function decide(
  model: PortableModel,
  churnProbability: number,
  clv: number,
  propensities: Record<AddonName, number | null>,
  eligible: Record<AddonName, boolean>,
): ScoreResponse["next_best_action"] {
  const thresholds = model.thresholds;
  const rules = new Map(model.rules.map((rule) => [rule.id, rule]));
  const rule = (id: string) => {
    const found = rules.get(id);
    if (!found) throw new Error(`Missing rule ${id}`);
    return found;
  };
  const available = ADDONS.filter((name) => eligible[name] && propensities[name] !== null).map((name) => ({
    name,
    propensity: propensities[name] as number,
  }));
  if (churnProbability >= thresholds.churn_high && clv >= thresholds.clv_high) {
    return ruleAction(rule("save_call"), null);
  }
  if (churnProbability >= thresholds.churn_high) {
    const offer = bestOffer(available, thresholds.offer_tie_break);
    if (offer !== null) return ruleAction(rule("offer_high_risk"), offer);
  }
  if (churnProbability >= thresholds.churn_medium) {
    const qualified = available.filter((offer) => offer.propensity >= thresholds.min_offer_propensity);
    const offer = bestOffer(qualified, thresholds.offer_tie_break);
    if (offer !== null) return ruleAction(rule("offer_medium_risk"), offer);
  }
  return ruleAction(rule("no_action"), null);
}

export function scoreCustomer(model: PortableModel, features: FeatureMap): ScoreResult {
  if (model.format !== FORMAT) {
    throw new Error(`Unsupported scoring document ${model.format}`);
  }
  const normalized = normalizeFeatures(model, features);
  const churnEncoded = transformRow(model.models.churn, normalized);
  const probability = sigmoid(predictTree(model.models.churn, churnEncoded, true).decision);
  const grouped = predictTree(model.models.churn, churnEncoded, false).grouped;
  const flags = eligibility(normalized);
  const addonScores = {} as ScoreResponse["addon_propensities"];
  const propensities = {} as Record<AddonName, number | null>;
  for (const name of ADDONS) {
    if (!flags[name].eligible) {
      addonScores[name] = { eligible: false, probability: null, reason: flags[name].reason };
      propensities[name] = null;
      continue;
    }
    const addonProbability = sigmoid(
      predictTree(model.models[name], transformRow(model.models[name], normalized), true).decision,
    );
    addonScores[name] = { eligible: true, probability: r6(addonProbability), reason: null };
    propensities[name] = addonProbability;
  }
  const monthly = numeric(normalized.MonthlyCharges ?? null, "MonthlyCharges");
  const tenure = numeric(normalized.tenure ?? null, "tenure");
  if (monthly === null || tenure === null) {
    throw new Error("MonthlyCharges and tenure are required");
  }
  const remaining = expectedRemainingMonths(model.retention_curve, tenure);
  const monthlyR = r6(monthly);
  const remainingR = r6(remaining);
  const horizon = model.retention_curve.times[model.retention_curve.times.length - 1];
  return {
    unroundedChurnProbability: probability,
    unroundedClv: monthly * remaining,
    response: {
      churn_probability: r6(probability),
      scoring_model: model.scoring_model,
      top_reasons: topReasons(grouped, model.top_reasons),
      next_best_action: decide(model, probability, monthly * remaining, propensities, {
        OnlineSecurity: flags.OnlineSecurity.eligible,
        TechSupport: flags.TechSupport.eligible,
      }),
      addon_propensities: addonScores,
      clv_proxy: {
        monthly_charges: monthlyR,
        expected_remaining_months: remainingR,
        value: r6(monthlyR * remainingR),
        horizon_months: r6(horizon),
      },
      explanation_method: model.explanation_method,
    },
  };
}

export type LiftStats = {
  k: number;
  lift: number;
  baseRate: number;
  topRate: number;
};

export function topDecileLift(labels: number[], scores: number[]): LiftStats {
  if (labels.length !== scores.length || labels.length < 10) {
    throw new Error("top decile needs at least 10 rows");
  }
  if (scores.some((score) => !Number.isFinite(score))) {
    throw new Error("scores must be finite");
  }
  const k = Math.floor(labels.length / 10);
  const order = scores.map((score, index) => ({ score, index }));
  order.sort((a, b) => b.score - a.score || a.index - b.index);
  let taken = 0;
  let weighted = 0;
  let index = 0;
  while (index < order.length && taken < k) {
    let end = index + 1;
    while (end < order.length && order[end].score === order[index].score) end += 1;
    let positives = 0;
    for (let cursor = index; cursor < end; cursor += 1) positives += labels[order[cursor].index];
    const size = end - index;
    const need = k - taken;
    if (size <= need) {
      weighted += positives;
      taken += size;
    } else {
      weighted += (positives / size) * need;
      taken += need;
    }
    index = end;
  }
  const topRate = weighted / k;
  const baseRate = labels.reduce((sum, label) => sum + label, 0) / labels.length;
  return {
    k,
    topRate,
    baseRate,
    lift: baseRate <= 0 ? Number.NaN : topRate / baseRate,
  };
}
