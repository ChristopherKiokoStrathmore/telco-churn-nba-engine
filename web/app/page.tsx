import Link from "next/link";
import { ModelTable } from "@/components/ModelTable";
import { demo } from "@/lib/sources";

export default function HomePage() {
  return (
    <>
      <p className="kicker">Portfolio · CRISP-DM</p>
      <h1>{demo.projectTitle}</h1>
      <p className="prose">
        Which customers should a retention team contact, and with what offer? This repository builds a churn
        model, add-on propensity models, a CLV proxy, and a readable next-best-action rule table, served one
        customer at a time through a FastAPI POST /score endpoint. The training table is {demo.dataset.name}:{" "}
        {demo.dataset.nRows} rows and {demo.dataset.nColumns} columns, IBM&apos;s public US telco sample. The CSV
        has no country column. It is not Kenyan operator data.
      </p>
      <div className="stat-row" aria-label="Served gradient boosting holdout metrics">
        <p className="stat">
          <b>{demo.served.rocAuc}</b>
          <span>ROC-AUC</span>
        </p>
        <p className="stat">
          <b>{demo.served.prAuc}</b>
          <span>PR-AUC</span>
        </p>
        <p className="stat">
          <b>{demo.served.lift}</b>
          <span>Top-decile lift</span>
        </p>
      </div>
      <p className="source-line">
        Gradient boosting, the served model, on the held-out test set of {demo.split.nTest} customers. Seed{" "}
        {demo.split.seed}, stratified test size {demo.split.testSize}. The top 10% by this score churn at{" "}
        {demo.served.lift} times the base rate. Source: reports/metrics.json.
      </p>
      <p className="prose">
        Logistic regression leads ROC-AUC on this split ({demo.logistic.rocAuc}). Gradient boosting leads PR-AUC
        ({demo.served.prAuc}) and top-decile lift ({demo.served.lift}). POST /score still serves gradient
        boosting, because config/nba_rules.yaml fixes scoring_model before the test table. Ranked by ROC-AUC:{" "}
        {demo.rocRank.join(", ")}.
      </p>
      <ModelTable
        caption="Held-out churn metrics. The served row is gradient boosting."
        rows={demo.churnModels}
      />
      <p className="prose">
        The live demo scores customers in the browser: the three committed requests, the held-out sample, or a
        CSV you choose. This page and the score desk keep the written briefing.
      </p>
      <div className="links">
        <Link href="/demo">Open the live demo</Link>
        <Link href="/score">Open the score desk</Link>
        <Link href="/metrics">Read the holdout notes</Link>
      </div>
      <h2>What one score returns</h2>
      <ul className="limits">
        <li>Churn probability from the gradient boosting pipeline.</li>
        <li>Three path contributions. They sum with the intercept to the churn log-odds. They are not SHAP values.</li>
        <li>A CLV proxy: monthly charges times restricted mean remaining tenure.</li>
        <li>Add-on propensities for OnlineSecurity and TechSupport, which are current-holding lookalikes.</li>
        <li>A next-best action from the rule table: save call, offer, or no action.</li>
      </ul>
      <h2>Sample</h2>
      <dl className="spec">
        <dt>Dataset</dt>
        <dd>{demo.dataset.name}</dd>
        <dt>Rows</dt>
        <dd>
          {demo.dataset.nRows}, of which {demo.dataset.churnYes} have Churn = Yes (rate {demo.dataset.churnRate})
        </dd>
        <dt>Split</dt>
        <dd>
          Seed {demo.split.seed}, test size {demo.split.testSize}, stratified on {demo.split.stratify}. Train{" "}
          {demo.split.nTrain}, test {demo.split.nTest}.
        </dd>
        <dt>Source</dt>
        <dd>
          <a href={demo.dataset.sourceUrl}>{demo.dataset.sourceUrl}</a>
        </dd>
        <dt>SHA-256</dt>
        <dd className="mono break">{demo.dataset.sha256}</dd>
        <dt>License note</dt>
        <dd>{demo.dataset.dataLicenseNote}</dd>
      </dl>
      <h2>Scope</h2>
      <ul className="limits">
        <li>The metrics describe this public sample. They say nothing about any operator&apos;s customers.</li>
        <li>Uptake scores are probabilities of current holding, not of accepting an offer, and not uplift.</li>
        <li>The CLV proxy ignores margin, discounting, and whether a save call changes survival.</li>
        <li>The next-best-action table is a policy. It does not estimate the causal effect of a call or an offer.</li>
        <li>There is one stratified holdout and no hyperparameter search.</li>
      </ul>
    </>
  );
}
