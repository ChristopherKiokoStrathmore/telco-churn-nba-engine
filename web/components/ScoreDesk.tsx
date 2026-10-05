import type { DemoCase } from "@/lib/sources";
import { demo } from "@/lib/sources";

function RequestFields({ item }: { item: DemoCase }) {
  return (
    <section aria-labelledby={`request-${item.id}`}>
      <h3 id={`request-${item.id}`}>Request</h3>
      {item.fields.map((group) => (
        <div key={group.group}>
          <p className="kicker">{group.group}</p>
          <table className="fields">
            <tbody>
              {group.items.map((field) => (
                <tr key={field.key}>
                  <th scope="row">{field.key}</th>
                  <td>{field.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      <details>
        <summary>Committed request JSON</summary>
        <pre>{item.requestText}</pre>
      </details>
      <p>
        <a href={item.publicRequestPath}>{item.requestFile}</a>
      </p>
    </section>
  );
}

function ResponseSlip({ item }: { item: DemoCase }) {
  const body = item.response;
  if (!body) {
    return (
      <section className="gap" aria-labelledby={`result-${item.id}`}>
        <h3 id={`result-${item.id}`}>Documented result</h3>
        <p className="stamp">{item.documentedActionLabel}</p>
        <p>
          Action named for this scene: <span className="mono">{item.documentedAction}</span>
        </p>
        <p>{item.gapNote}</p>
        <h3>Rule text</h3>
        <p className="note">
          Copied from config/nba_rules.yaml. This is the policy text, not a measured score for this request.
        </p>
        <ul className="rule-list">
          {item.relatedRules.map((rule) => (
            <li key={rule.id}>
              <p className="kicker">
                Priority {rule.priority} · {rule.id}
              </p>
              <p className="when">
                When <span className="mono">{rule.when}</span>
              </p>
              <p>{rule.rationale}</p>
              <p className="note">
                Action <span className="mono">{rule.action}</span>, offer <span className="mono">{rule.offer}</span>
              </p>
            </li>
          ))}
        </ul>
      </section>
    );
  }

  const security = body.addons.find((addon) => addon.name === "OnlineSecurity");

  return (
    <section aria-labelledby={`result-${item.id}`}>
      <h3 id={`result-${item.id}`}>Committed response</h3>
      <p className="stamp">{item.documentedActionLabel}</p>
      <p className="kicker">Churn probability</p>
      <p className="probability">{body.churnProbability}</p>
      <p>
        Scoring model <span className="mono">{body.scoringModel}</span>. Rule{" "}
        <span className="mono">{body.ruleId}</span>. Offer <span className="mono">{body.offer}</span>.
      </p>
      <p>{body.rationale}</p>
      <p>
        README, on this response: churn probability {body.churnProbability} is below churn_medium{" "}
        {demo.thresholds.churnMedium}, so the rule is no_action, even though the OnlineSecurity lookalike
        probability is {security?.probability}, above the offer cutoff {demo.thresholds.minOfferPropensity}.
      </p>
      <h3>Top reasons</h3>
      <p className="note">{body.explanation}</p>
      {body.topReasons.map((reason) => (
        <div
          key={reason.feature}
          className={reason.direction === "increases_churn_risk" ? "reason increases" : "reason"}
        >
          <b>{reason.feature}</b>
          <div className="bar" aria-hidden="true">
            <span style={{ width: `${reason.barPercent}%` }} />
          </div>
          <span className="contribution mono">
            {reason.contribution}
            <span className="note"> {reason.directionLabel}</span>
          </span>
        </div>
      ))}
      <h3>Add-on propensities</h3>
      <p className="note">
        Lookalike probability of current holding. Eligible means the customer has internet service and does not
        already hold that add-on.
      </p>
      <table className="fields">
        <tbody>
          {body.addons.map((addon) => (
            <tr key={addon.name}>
              <th scope="row">{addon.name}</th>
              <td>
                {addon.probability} · {addon.eligible ? "eligible" : "not eligible"}
                {addon.reason ? ` · ${addon.reason}` : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3>CLV proxy</h3>
      <p>
        Value <span className="mono">{body.clv.value}</span>
      </p>
      <p className="note">
        Monthly charges {body.clv.monthly} × expected remaining months {body.clv.remaining}. Horizon{" "}
        {body.clv.horizon} months. The README states that the value is the six-decimal product of those two
        displayed figures.
      </p>
      <h3>Frozen cutoffs</h3>
      <table className="fields">
        <caption className="note">From reports/metrics.json. Cutoffs are not chosen on the test set.</caption>
        <tbody>
          <tr>
            <th scope="row">churn_high</th>
            <td>{demo.thresholds.churnHigh}</td>
          </tr>
          <tr>
            <th scope="row">churn_medium</th>
            <td>{demo.thresholds.churnMedium}</td>
          </tr>
          <tr>
            <th scope="row">clv_high</th>
            <td>{demo.thresholds.clvHigh}</td>
          </tr>
          <tr>
            <th scope="row">min_offer_propensity</th>
            <td>{demo.thresholds.minOfferPropensity}</td>
          </tr>
        </tbody>
      </table>
      <figure className="plate">
        <img
          src="/figures/example_reasons.png"
          width={1400}
          height={760}
          alt="Path contributions for the example score. The churn probability sits below the medium cutoff, so the action is no_action."
        />
        <figcaption>
          Committed chart for customer {demo.exampleCustomerId}. Contract, StreamingMovies, and MonthlyCharges
          lower this customer&apos;s churn log-odds. The contributions are not SHAP values.
        </figcaption>
      </figure>
      <details>
        <summary>Committed response JSON</summary>
        <pre>{body.text}</pre>
      </details>
      <p>
        <a href={body.publicPath}>{body.file}</a>
      </p>
    </section>
  );
}

export function ScoreDesk({ cases }: { cases: DemoCase[] }) {
  return (
    <fieldset className="desk">
      <legend className="sr-only">Committed holdout examples</legend>
      <input className="picker-input" type="radio" name="holdout-customer" id="case-no-action" defaultChecked />
      <input className="picker-input" type="radio" name="holdout-customer" id="case-save-call" />
      <input className="picker-input" type="radio" name="holdout-customer" id="case-offer" />
      <div className="picker">
        {cases.map((item) => (
          <label key={item.id} className="picker-card" htmlFor={`case-${item.id}`}>
            <span className="picker-index">{item.index}</span>
            <span className="picker-title">{item.title}</span>
            <span className="picker-file">{item.requestFile}</span>
            <span className="picker-summary">{item.summary}</span>
          </label>
        ))}
      </div>
      {cases.map((item) => (
        <article key={item.id} className={`panel panel-${item.id}`} aria-labelledby={`heading-${item.id}`}>
          <h2 id={`heading-${item.id}`}>{item.title}</h2>
          <p className="note">{item.summary}</p>
          {item.customerId ? (
            <p>
              Customer id <span className="mono">{item.customerId}</span>, historical Churn{" "}
              {item.historicalChurn}. The id and the label are not sent to the model. The request is the first
              held-out test customer with a numeric TotalCharges.
            </p>
          ) : (
            <p>This request file does not include a customer id. The id is not part of POST /score.</p>
          )}
          <p className="note">Local command from the README. This page does not send it.</p>
          <pre>{`curl -s -X POST http://localhost:8000/score \\
  -H 'Content-Type: application/json' \\
  -d @${item.requestFile}`}</pre>
          <div className="split">
            <RequestFields item={item} />
            <ResponseSlip item={item} />
          </div>
        </article>
      ))}
    </fieldset>
  );
}
