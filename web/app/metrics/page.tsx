import type { Metadata } from "next";
import { ModelTable } from "@/components/ModelTable";
import { demo } from "@/lib/sources";

export const metadata: Metadata = {
  title: "Metrics",
  description:
    "Held-out ROC-AUC, PR-AUC, top-decile lift, and the next-best-action rule table from the committed metrics and rules.",
};

export default function MetricsPage() {
  return (
    <>
      <p className="kicker">Held-out test set</p>
      <h1>Metrics</h1>
      <p className="prose">
        Positive class is Churn = {demo.positiveLabel}. ROC-AUC, PR-AUC, and top-decile lift are on the held-out
        rows: {demo.split.nTest} customers, seed {demo.split.seed}, stratified test size {demo.split.testSize}.
        The test base rate is {demo.split.testBaseRate}. Train rows: {demo.split.nTrain}, train churn rate{" "}
        {demo.split.trainChurnRate}.
      </p>
      <h2>ROC and precision-recall</h2>
      <p>
        Top-decile definition, copied from reports/metrics.json: {demo.topDecileDefinition}. For churn, k is{" "}
        {demo.topDecileK}. The dummy prior&apos;s PR-AUC equals the base rate {demo.split.testBaseRate}, and its
        lift is {demo.churnModels[0]?.lift}.
      </p>
      <p>
        Logistic regression leads ROC-AUC on this split ({demo.logistic.rocAuc} against gradient boosting{" "}
        {demo.served.rocAuc}). Gradient boosting leads PR-AUC ({demo.served.prAuc}) and top-decile lift (
        {demo.served.lift}). Both clear the dummy prior on all three metrics. POST /score still uses gradient
        boosting, because the rules file fixes scoring_model before anyone looks at the test table. The stored
        ROC-AUC rank, high to low, is {demo.rocRank.join(", ")}.
      </p>
      <ModelTable caption="Churn models on the held-out split." rows={demo.churnModels} />
      <figure className="plate">
        <img
          src="/figures/churn_roc_pr.png"
          width={1175}
          height={616}
          alt="Held-out ROC and precision-recall curves for the served gradient boosting churn model, with logistic regression and a dummy prior on the same split."
        />
        <figcaption>
          Committed figure from reports/figures/churn_roc_pr.png. The curves are drawn from the scoring bundle
          and this same holdout.
        </figcaption>
      </figure>
      <figure className="plate">
        <img
          src="/figures/churn_lift.png"
          width={1400}
          height={760}
          alt="Top-decile lift for the dummy prior, logistic regression, and gradient boosting on the held-out split."
        />
        <figcaption>
          Gradient boosting, the served model, has top-decile lift {demo.served.lift}. Logistic regression leads
          ROC-AUC on this split.
        </figcaption>
      </figure>
      <h2>CLV proxy</h2>
      <p>{demo.clv.definition}</p>
      <dl className="spec">
        <dt>Horizon</dt>
        <dd>{demo.clv.horizon} months. The integral stops there and is not extrapolated.</dd>
        <dt>At tenure 0</dt>
        <dd>Expected remaining time {demo.clv.remainingAtZero} months.</dd>
        <dt>Training median</dt>
        <dd>{demo.clv.p50}</dd>
        <dt>Training 75th</dt>
        <dd>
          {demo.clv.p75}, which is the frozen <span className="mono">clv_high</span> cutoff.
        </dd>
      </dl>
      <p className="note">There is no margin, discount rate, or causal save effect in this number.</p>
      <h2>Next-best-action rules</h2>
      <p>{demo.thresholds.source}</p>
      <div className="table-wrap">
        <table>
          <caption>Frozen cutoffs from the training run, copied from reports/metrics.json.</caption>
          <thead>
            <tr>
              <th scope="col">Cutoff</th>
              <th scope="col" className="num">
                Value
              </th>
              <th scope="col">Meaning</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">churn_high</th>
              <td className="num">{demo.thresholds.churnHigh}</td>
              <td>{demo.thresholds.quantiles.churnHigh} quantile of out-of-fold training churn probabilities</td>
            </tr>
            <tr>
              <th scope="row">churn_medium</th>
              <td className="num">{demo.thresholds.churnMedium}</td>
              <td>{demo.thresholds.quantiles.churnMedium} quantile of those same probabilities</td>
            </tr>
            <tr>
              <th scope="row">clv_high</th>
              <td className="num">{demo.thresholds.clvHigh}</td>
              <td>{demo.thresholds.quantiles.clvHigh} quantile of the training CLV proxy</td>
            </tr>
            <tr>
              <th scope="row">min_offer_propensity</th>
              <td className="num">{demo.thresholds.minOfferPropensity}</td>
              <td>Fixed policy constant in the rules file (written there as 0.40), not an estimated uplift</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="scroll-hint">Scroll sideways for the full table.</p>
      <div className="table-wrap">
        <table>
          <caption>Rules from config/nba_rules.yaml. First match wins. A tie goes to {demo.thresholds.offerTieBreak}.</caption>
          <thead>
            <tr>
              <th scope="col">Priority</th>
              <th scope="col">Rule id</th>
              <th scope="col">When</th>
              <th scope="col">Action</th>
              <th scope="col">Rationale</th>
            </tr>
          </thead>
          <tbody>
            {demo.rules.map((rule) => (
              <tr key={rule.id}>
                <td className="num">{rule.priority}</td>
                <th scope="row">{rule.id}</th>
                <td>{rule.when}</td>
                <td>
                  {rule.action}
                  {rule.offer === "null" ? "" : ` · offer ${rule.offer}`}
                </td>
                <td>{rule.rationale}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="scroll-hint">Scroll sideways for the full table.</p>
      <p>
        Eligible means the customer has internet service and does not already hold that add-on. A customer with
        no internet service is not scored by the uptake models.
      </p>
      <h2>Held-out actions</h2>
      <p>{demo.activeDefinition}</p>
      <div className="table-wrap">
        <table>
          <caption>Action counts. Offer columns count OnlineSecurity and TechSupport inside the offer total.</caption>
          <thead>
            <tr>
              <th scope="col">Slice</th>
              <th scope="col" className="num">
                Save call
              </th>
              <th scope="col" className="num">
                Offer
              </th>
              <th scope="col" className="num">
                No action
              </th>
              <th scope="col" className="num">
                n
              </th>
              <th scope="col" className="num">
                OnlineSecurity
              </th>
              <th scope="col" className="num">
                TechSupport
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">Historical Churn No</th>
              <td className="num">{demo.active.saveCall}</td>
              <td className="num">{demo.active.offer}</td>
              <td className="num">{demo.active.noAction}</td>
              <td className="num">{demo.active.n}</td>
              <td className="num">{demo.active.onlineSecurity}</td>
              <td className="num">{demo.active.techSupport}</td>
            </tr>
            <tr>
              <th scope="row">Every held-out customer</th>
              <td className="num">{demo.allCustomers.saveCall}</td>
              <td className="num">{demo.allCustomers.offer}</td>
              <td className="num">{demo.allCustomers.noAction}</td>
              <td className="num">{demo.allCustomers.n}</td>
              <td className="num">{demo.allCustomers.onlineSecurity}</td>
              <td className="num">{demo.allCustomers.techSupport}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="scroll-hint">Scroll sideways for the full table.</p>
      <figure className="plate">
        <img
          src="/figures/holdout_actions.png"
          width={1400}
          height={760}
          alt="Held-out next-best actions for customers whose historical Churn label is No."
        />
        <figcaption>
          The bars are those same held-out counts: {demo.active.saveCall} save calls, {demo.active.offer} offers,
          and {demo.active.noAction} no-action outcomes.
        </figcaption>
      </figure>
      <h2>Add-on holding, not campaign response</h2>
      <p>
        The uptake models predict whether an internet customer currently holds OnlineSecurity or TechSupport.
        They are not models of campaign response, and they are not uplift. The API uses the gradient boosting
        uptake models because that family is the fixed scoring model. Monthly charges and total charges are
        excluded so the bill does not reconstruct the holding.
      </p>
      {demo.uptake.map((block) => (
        <section key={block.name}>
          <h3>{block.name}</h3>
          <p>{block.label}</p>
          <p className="note">
            {block.population} Train {block.nTrain}, test {block.nTest}, test base rate {block.baseRate},
            top-decile k {block.k}. Excluded features: {block.excluded}.
          </p>
          <ModelTable caption={`${block.name} holding models on the internet-customer holdout.`} rows={block.rows} />
        </section>
      ))}
      <h2>Training environment</h2>
      <p>
        Recorded in the metrics file: Python {demo.trainingEnvironment.python}, scikit-learn{" "}
        {demo.trainingEnvironment.sklearn}, lifelines {demo.trainingEnvironment.lifelines}. Path contributions
        for the served churn model are not SHAP. The model card and SHAP write-up live in{" "}
        <a href={demo.modelCardUrl}>responsible-ai-pack</a>.
      </p>
    </>
  );
}
