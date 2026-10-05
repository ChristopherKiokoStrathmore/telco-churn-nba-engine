"use client";

import { useEffect, useMemo, useState } from "react";
import { customersFromCsv, type ParsedCustomer } from "@/lib/customers";
import {
  format6,
  scoreCustomer,
  topDecileLift,
  type FeatureMap,
  type InputSpec,
  type PortableModel,
  type ScoreResponse,
} from "@/lib/portableScore";

type BatchSource = "examples" | "holdout" | "upload";

type ScoredRow = {
  key: string;
  customerId: string | null;
  label: "Yes" | "No" | null;
  features: FeatureMap;
  response: ScoreResponse;
  unrounded: number;
  unroundedClv: number;
};

type ExampleCustomer = {
  source: string;
  customer_id: string;
  historical_churn: "Yes" | "No";
  features: FeatureMap;
};

const PAGE_SIZE = 12;

const LABELS: Record<string, string> = {
  gender: "Gender",
  SeniorCitizen: "Senior citizen",
  Partner: "Partner",
  Dependents: "Dependents",
  tenure: "Tenure (months)",
  PhoneService: "Phone service",
  MultipleLines: "Multiple lines",
  InternetService: "Internet service",
  OnlineSecurity: "Online security",
  OnlineBackup: "Online backup",
  DeviceProtection: "Device protection",
  TechSupport: "Tech support",
  StreamingTV: "Streaming TV",
  StreamingMovies: "Streaming movies",
  Contract: "Contract",
  PaperlessBilling: "Paperless billing",
  PaymentMethod: "Payment method",
  MonthlyCharges: "Monthly charges",
  TotalCharges: "Total charges",
};

function actionTitle(action: string, offer: string | null): string {
  if (action === "save_call") return "Save call";
  if (action === "no_action") return "No action";
  if (action === "offer") return offer ? `Offer ${offer}` : "Offer";
  return action;
}

function directionLabel(direction: string): string {
  if (direction === "decreases_churn_risk") return "Decreases churn risk";
  if (direction === "increases_churn_risk") return "Increases churn risk";
  return direction;
}

function draftFrom(features: FeatureMap, inputs: InputSpec[]): Record<string, string> {
  const draft: Record<string, string> = {};
  for (const input of inputs) {
    const value = features[input.name];
    draft[input.name] = value === null || value === undefined ? "" : String(value);
  }
  return draft;
}

function featuresFromDraft(draft: Record<string, string>, inputs: InputSpec[]): FeatureMap {
  const features: FeatureMap = {};
  for (const input of inputs) {
    features[input.name] = draft[input.name] ?? "";
  }
  return features;
}

function scoreParsed(model: PortableModel, customer: ParsedCustomer, index: number): ScoredRow {
  const scored = scoreCustomer(model, customer.features);
  return {
    key: `${customer.customerId ?? "row"}-${index}`,
    customerId: customer.customerId,
    label: customer.label,
    features: customer.features,
    response: scored.response,
    unrounded: scored.unroundedChurnProbability,
    unroundedClv: scored.unroundedClv,
  };
}

async function loadJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load ${url} (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function LiveDemo({
  publishedLift,
  publishedActive,
}: {
  publishedLift: string;
  publishedActive: { save: string; offer: string; none: string; n: string };
}) {
  const [model, setModel] = useState<PortableModel | null>(null);
  const [examples, setExamples] = useState<ExampleCustomer[]>([]);
  const [rows, setRows] = useState<ScoredRow[]>([]);
  const [source, setSource] = useState<BatchSource | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [draftScore, setDraftScore] = useState<ScoredRow | null>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [filter, setFilter] = useState<"all" | "save_call" | "offer" | "no_action">("all");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const [status, setStatus] = useState("Loading the scoring bundle…");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [nextModel, exampleFile] = await Promise.all([
          loadJson<PortableModel>("/model/scoring_model.json"),
          loadJson<{ customers: ExampleCustomer[] }>("/sample/examples.json"),
        ]);
        if (cancelled) return;
        const scored = exampleFile.customers.map((customer, index) =>
          scoreParsed(
            nextModel,
            {
              customerId: customer.customer_id,
              label: customer.historical_churn,
              features: customer.features,
            },
            index,
          ),
        );
        const top = scored.slice().sort((a, b) => b.unrounded - a.unrounded)[0];
        setModel(nextModel);
        setExamples(exampleFile.customers);
        setRows(scored);
        setSource("examples");
        if (top) setDraft(draftFrom(top.features, nextModel.inputs));
        setSelectedKey(top?.key ?? null);
        setStatus("Scored the three committed customers.");
        setBusy(false);
      } catch (cause) {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : "Could not load the demo.");
        setStatus("");
        setBusy(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return rows
      .filter((row) => (filter === "all" ? true : row.response.next_best_action.action === filter))
      .filter((row) => (needle === "" ? true : (row.customerId ?? "").toLowerCase().includes(needle)))
      .slice()
      .sort((a, b) => b.unrounded - a.unrounded || (a.customerId ?? "").localeCompare(b.customerId ?? ""));
  }, [rows, filter, query]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const visible = filtered.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE);
  const selected = rows.find((row) => row.key === selectedKey) ?? null;
  const detail = draftScore ?? selected;

  const lift = useMemo(() => {
    const labeled = rows.filter((row) => row.label === "Yes" || row.label === "No");
    if (labeled.length < 10 || labeled.length !== rows.length) return null;
    return topDecileLift(
      labeled.map((row) => (row.label === "Yes" ? 1 : 0)),
      labeled.map((row) => row.unrounded),
    );
  }, [rows]);

  const counts = useMemo(() => {
    const tally = { save_call: 0, offer: 0, no_action: 0 };
    for (const row of rows) {
      const action = row.response.next_best_action.action;
      if (action === "save_call" || action === "offer" || action === "no_action") tally[action] += 1;
    }
    return tally;
  }, [rows]);

  const activeCounts = useMemo(() => {
    const tally = { save_call: 0, offer: 0, no_action: 0, n: 0 };
    for (const row of rows) {
      if (row.label !== "No") continue;
      tally.n += 1;
      const action = row.response.next_best_action.action;
      if (action === "save_call" || action === "offer" || action === "no_action") tally[action] += 1;
    }
    return tally;
  }, [rows]);

  const meanProbability = rows.length
    ? rows.reduce((sum, row) => sum + row.unrounded, 0) / rows.length
    : null;

  async function scoreExamples() {
    if (!model) return;
    setBusy(true);
    setError(null);
    setDraftScore(null);
    await new Promise((resolve) => setTimeout(resolve, 20));
    try {
      const scored = examples.map((customer, index) =>
        scoreParsed(
          model,
          { customerId: customer.customer_id, label: customer.historical_churn, features: customer.features },
          index,
        ),
      );
      const top = scored.slice().sort((a, b) => b.unrounded - a.unrounded)[0];
      setRows(scored);
      setSource("examples");
      setFilter("all");
      setQuery("");
      setPage(0);
      if (top) setDraft(draftFrom(top.features, model.inputs));
      setSelectedKey(top?.key ?? null);
      setStatus("Scored the three committed customers.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Scoring failed.");
    } finally {
      setBusy(false);
    }
  }

  async function scoreHoldout() {
    if (!model) return;
    setBusy(true);
    setError(null);
    setDraftScore(null);
    setStatus("Loading the held-out sample…");
    try {
      const response = await fetch("/sample/holdout.csv");
      if (!response.ok) throw new Error(`Could not load the holdout sample (${response.status})`);
      const text = await response.text();
      const parsed = customersFromCsv(text, model);
      setStatus(`Scoring ${parsed.length} held-out customers…`);
      await new Promise((resolve) => setTimeout(resolve, 20));
      const scored = parsed.map((customer, index) => scoreParsed(model, customer, index));
      setRows(scored);
      setSource("holdout");
      setFilter("all");
      setQuery("");
      setPage(0);
      setSelectedKey(scored.slice().sort((a, b) => b.unrounded - a.unrounded)[0]?.key ?? null);
      setStatus(`Scored ${scored.length} held-out customers.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Scoring failed.");
      setStatus("");
    } finally {
      setBusy(false);
    }
  }

  async function scoreUpload(file: File) {
    if (!model) return;
    setBusy(true);
    setError(null);
    setDraftScore(null);
    setStatus(`Reading ${file.name}…`);
    try {
      const text = await file.text();
      const parsed = customersFromCsv(text, model);
      setStatus(`Scoring ${parsed.length} uploaded customers…`);
      await new Promise((resolve) => setTimeout(resolve, 20));
      const scored = parsed.map((customer, index) => scoreParsed(model, customer, index));
      setRows(scored);
      setSource("upload");
      setFilter("all");
      setQuery("");
      setPage(0);
      setSelectedKey(scored.slice().sort((a, b) => b.unrounded - a.unrounded)[0]?.key ?? null);
      if (scored[0]) setDraft(draftFrom(scored[0].features, model.inputs));
      setStatus(`Scored ${scored.length} customers from ${file.name}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not read that CSV.");
      setStatus("");
    } finally {
      setBusy(false);
    }
  }

  function scoreDraft() {
    if (!model) return;
    setError(null);
    try {
      const features = featuresFromDraft(draft, model.inputs);
      const scored = scoreCustomer(model, features);
      setDraftScore({
        key: "draft",
        customerId: null,
        label: null,
        features,
        response: scored.response,
        unrounded: scored.unroundedChurnProbability,
        unroundedClv: scored.unroundedClv,
      });
      setStatus("Scored the customer in the form.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Scoring failed.");
    }
  }

  const bins = useMemo(() => {
    const countsByBin = new Array<number>(10).fill(0);
    for (const row of rows) {
      const index = Math.min(9, Math.floor(row.unrounded * 10));
      countsByBin[index] += 1;
    }
    const max = Math.max(1, ...countsByBin);
    return countsByBin.map((count) => ({ count, height: (count / max) * 100 }));
  }, [rows]);

  return (
    <div className="demo">
      <div className="toolbar" role="group" aria-label="Scoring sources">
        <button className="button primary" type="button" onClick={() => void scoreExamples()} disabled={!model || busy}>
          Score 3 committed customers
        </button>
        <button className="button" type="button" onClick={() => void scoreHoldout()} disabled={!model || busy}>
          Score held-out sample
        </button>
        <label className="button file-button">
          Upload CSV
          <input
            className="sr-only"
            type="file"
            accept=".csv,text/csv"
            disabled={!model || busy}
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) void scoreUpload(file);
            }}
          />
        </label>
        <a className="text-link" href="/sample/holdout.csv">
          Download held-out CSV
        </a>
      </div>
      <p className="note" role="status">
        {status}
        {source === "holdout" ? " Historical Churn is used only for lift." : ""}
        {source === "upload" ? " Uploaded rows stay in this browser." : ""}
      </p>
      {error ? <p className="banner">{error}</p> : null}

      {rows.length > 0 ? (
        <>
          <div className="stat-row" aria-label="Batch summary">
            <p className="stat">
              <b>{rows.length}</b>
              <span>Customers scored</span>
            </p>
            <p className="stat">
              <b>{meanProbability === null ? "—" : format6(meanProbability)}</b>
              <span>Mean churn probability</span>
            </p>
            <p className="stat">
              <b>{lift ? lift.lift.toFixed(6) : "—"}</b>
              <span>Top-decile lift</span>
            </p>
          </div>
          {lift ? (
            <p className="source-line">
              Top decile uses k = floor(n / 10) = {lift.k}. Base rate {format6(lift.baseRate)}. Churn rate in the
              top decile {format6(lift.topRate)}. Lift {lift.lift.toFixed(6)}.
              {source === "holdout" && lift.lift.toFixed(6) === publishedLift
                ? ` This matches the committed holdout lift ${publishedLift}.`
                : " The label is not a model input."}
            </p>
          ) : (
            <p className="source-line">
              {rows.length < 10
                ? "Top-decile lift needs at least 10 rows with a historical Churn label. Score the held-out sample to compare with the briefing."
                : "Top-decile lift needs a historical Churn label on every scored row. The label is not a model input."}
            </p>
          )}

          <div className="mix" aria-hidden="true">
            <span className="mix-save" style={{ flex: counts.save_call }} />
            <span className="mix-offer" style={{ flex: counts.offer }} />
            <span className="mix-none" style={{ flex: counts.no_action }} />
          </div>
          <ul className="mix-legend">
            <li>Save call {counts.save_call}</li>
            <li>Offer {counts.offer}</li>
            <li>No action {counts.no_action}</li>
          </ul>
          {activeCounts.n > 0 ? (
            <p className="source-line">
              Among rows with historical Churn No: {activeCounts.save_call} save calls, {activeCounts.offer} offers,{" "}
              {activeCounts.no_action} no-action outcomes (n = {activeCounts.n}).
              {String(activeCounts.save_call) === publishedActive.save &&
              String(activeCounts.offer) === publishedActive.offer &&
              String(activeCounts.no_action) === publishedActive.none &&
              String(activeCounts.n) === publishedActive.n
                ? " That matches the committed briefing."
                : " The briefing counts that same population separately from customers already labeled Churn Yes."}
            </p>
          ) : null}

          <h2>Churn probability</h2>
          <div className="hist" aria-hidden="true">
            {bins.map((bin, index) => (
              <div key={index} className="hist-col">
                <span style={{ height: `${bin.height}%` }} />
              </div>
            ))}
          </div>
          <p className="note">Ten bins from 0 to 1. Bar height is the count in that bin.</p>

          <div className="filters">
            <label>
              Action
              <select
                value={filter}
                onChange={(event) => {
                  setFilter(event.target.value as typeof filter);
                  setPage(0);
                }}
              >
                <option value="all">All</option>
                <option value="save_call">Save call</option>
                <option value="offer">Offer</option>
                <option value="no_action">No action</option>
              </select>
            </label>
            <label>
              Customer id
              <input
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value);
                  setPage(0);
                }}
                placeholder="5343-SGUBI"
                autoComplete="off"
              />
            </label>
          </div>

          <div className="split demo-split">
            <div>
              <div className="table-wrap">
                <table className="demo-table">
                  <caption>
                    {filtered.length === 0
                      ? "No customers match."
                      : `Showing ${safePage * PAGE_SIZE + 1}–${Math.min(filtered.length, (safePage + 1) * PAGE_SIZE)} of ${filtered.length}, highest churn probability first.`}
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Customer</th>
                      <th scope="col">Churn</th>
                      <th scope="col" className="num">
                        Probability
                      </th>
                      <th scope="col">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visible.map((row) => {
                      const current = row.key === selectedKey && draftScore === null;
                      return (
                        <tr key={row.key} className={current ? "is-selected" : undefined}>
                          <th scope="row">
                            <button
                              type="button"
                              className="row-button"
                              aria-current={current ? "true" : undefined}
                              onClick={() => {
                                setSelectedKey(row.key);
                                setDraftScore(null);
                                if (model) setDraft(draftFrom(row.features, model.inputs));
                              }}
                            >
                              {row.customerId ?? "Uploaded row"}
                            </button>
                          </th>
                          <td>{row.label ?? "—"}</td>
                          <td className="num mono">{format6(row.response.churn_probability)}</td>
                          <td>{actionTitle(row.response.next_best_action.action, row.response.next_best_action.offer)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="pager">
                <button type="button" className="button" disabled={safePage === 0} onClick={() => setPage(safePage - 1)}>
                  Previous
                </button>
                <span>
                  Page {safePage + 1} of {pageCount}
                </span>
                <button
                  type="button"
                  className="button"
                  disabled={safePage >= pageCount - 1}
                  onClick={() => setPage(safePage + 1)}
                >
                  Next
                </button>
              </div>
            </div>
            <ScoreDetail row={detail} model={model} edited={draftScore !== null} />
          </div>
        </>
      ) : null}

      {model ? (
        <form
          className="editor"
          onSubmit={(event) => {
            event.preventDefault();
            scoreDraft();
          }}
        >
          <h2>Score one customer</h2>
          <p className="note">
            Change any field and score again. Blank total charges are median-imputed inside the pipeline. The id
            and the historical label are not inputs.
          </p>
          <div className="editor-grid">
            {model.inputs.map((input) => (
              <label key={input.name}>
                {LABELS[input.name] ?? input.name}
                {input.kind === "category" ? (
                  <select
                    value={draft[input.name] ?? ""}
                    onChange={(event) => setDraft((current) => ({ ...current, [input.name]: event.target.value }))}
                  >
                    {(input.options ?? []).map((option) => (
                      <option key={option} value={option}>
                        {input.name === "SeniorCitizen" ? `${option} — ${option === "1" ? "Yes" : "No"}` : option}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    inputMode="decimal"
                    value={draft[input.name] ?? ""}
                    placeholder={input.allow_blank ? "Blank imputes the training median" : undefined}
                    onChange={(event) => setDraft((current) => ({ ...current, [input.name]: event.target.value }))}
                  />
                )}
              </label>
            ))}
          </div>
          <button className="button primary" type="submit" disabled={busy}>
            Score this customer
          </button>
        </form>
      ) : null}
    </div>
  );
}

function ScoreDetail({
  row,
  model,
  edited,
}: {
  row: ScoredRow | null;
  model: PortableModel | null;
  edited: boolean;
}) {
  if (!row || !model) {
    return (
      <section aria-label="Score detail">
        <p className="note">Score a customer to see the next action.</p>
      </section>
    );
  }
  const body = row.response;
  const reasons = body.top_reasons;
  const maxAbs = Math.max(...reasons.map((reason) => Math.abs(reason.contribution)), 0);
  const high = model.thresholds.churn_high;
  const medium = model.thresholds.churn_medium;
  return (
    <section aria-labelledby="score-detail-heading">
      <h2 id="score-detail-heading">{edited ? "Edited customer" : (row.customerId ?? "Selected customer")}</h2>
      {edited ? (
        <p className="note">This score uses the form. It is separate from the batch until you pick a row again.</p>
      ) : row.label ? (
        <p className="note">Historical Churn {row.label}. That label was not sent to the model.</p>
      ) : null}
      <p className={`stamp stamp-${body.next_best_action.action}`}>
        {actionTitle(body.next_best_action.action, body.next_best_action.offer)}
      </p>
      <p className="kicker">Churn probability</p>
      <p className="probability">{format6(body.churn_probability)}</p>
      <p>
        {row.unrounded >= high ? "At or above" : "Below"} the high cutoff {format6(high)}.{" "}
        {row.unrounded >= medium ? "At or above" : "Below"} the medium cutoff {format6(medium)}.
      </p>
      <p>
        Rule <span className="mono">{body.next_best_action.rule_id}</span>. Offer{" "}
        <span className="mono">{body.next_best_action.offer ?? "null"}</span>.
      </p>
      <p>{body.next_best_action.rationale}</p>
      <h3>Top reasons</h3>
      <p className="note">{body.explanation_method}</p>
      {reasons.map((reason) => (
        <div
          key={reason.feature}
          className={reason.direction === "increases_churn_risk" ? "reason increases" : "reason"}
        >
          <b>{reason.feature}</b>
          <div className="bar" aria-hidden="true">
            <span style={{ width: `${maxAbs === 0 ? 0 : (Math.abs(reason.contribution) / maxAbs) * 100}%` }} />
          </div>
          <span className="contribution mono">
            {format6(reason.contribution)}
            <span className="note"> {directionLabel(reason.direction)}</span>
          </span>
        </div>
      ))}
      <h3>Add-on propensities</h3>
      <p className="note">Lookalike probability of current holding, not of accepting an offer.</p>
      <table className="fields">
        <tbody>
          {(["OnlineSecurity", "TechSupport"] as const).map((name) => {
            const addon = body.addon_propensities[name];
            return (
              <tr key={name}>
                <th scope="row">{name}</th>
                <td>
                  {addon.probability === null ? "null" : format6(addon.probability)} ·{" "}
                  {addon.eligible ? "eligible" : "not eligible"}
                  {addon.reason ? ` · ${addon.reason}` : ""}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <h3>CLV proxy</h3>
      <p>
        Value <span className="mono">{format6(body.clv_proxy.value)}</span>
        {row.unroundedClv >= model.thresholds.clv_high ? " at or above" : " below"} the high-CLV cutoff{" "}
        {format6(model.thresholds.clv_high)}.
      </p>
      <p className="note">
        Monthly charges {format6(body.clv_proxy.monthly_charges)} × expected remaining months{" "}
        {format6(body.clv_proxy.expected_remaining_months)}. Horizon {format6(body.clv_proxy.horizon_months)}{" "}
        months. No margin, discount rate, or causal save effect.
      </p>
    </section>
  );
}
