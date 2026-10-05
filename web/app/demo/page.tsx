import type { Metadata } from "next";
import Link from "next/link";
import { LiveDemo } from "@/components/LiveDemo";
import { demo } from "@/lib/sources";

export const metadata: Metadata = {
  title: "Demo",
  description:
    "Score IBM telco sample customers in the browser: churn probability, path contributions, a CLV proxy, and the next-best action.",
};

export default function DemoPage() {
  return (
    <>
      <p className="kicker">Live demo</p>
      <h1>Score customers</h1>
      <p className="prose">
        Score the three committed requests, the held-out public sample, or a CSV with the same columns.
        The page runs the exported gradient boosting bundle in this browser: churn probability, path contributions,
        the CLV proxy, add-on propensities, and the next-best-action rules. Rows are not posted to a server. The
        written briefing stays on the <Link href="/">home page</Link>.
      </p>
      <p className="banner">
        Scoring uses the public sample and the committed model export. It does not call the FastAPI process.
      </p>
      <noscript>
        <p className="banner">This demo scores in the browser and needs JavaScript enabled.</p>
      </noscript>
      <LiveDemo
        publishedLift={demo.served.lift}
        publishedActive={{
          save: demo.active.saveCall,
          offer: demo.active.offer,
          none: demo.active.noAction,
          n: demo.active.n,
        }}
      />
    </>
  );
}
