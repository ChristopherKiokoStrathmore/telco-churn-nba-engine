import type { Metadata } from "next";
import { ScoreDesk } from "@/components/ScoreDesk";
import { demo } from "@/lib/sources";

export const metadata: Metadata = {
  title: "Score",
  description:
    "Interactive view of the committed POST /score examples: no action, a save call, and an add-on offer.",
};

export default function ScorePage() {
  return (
    <>
      <p className="kicker">POST /score</p>
      <h1>Score</h1>
      <p className="banner">Demo uses committed holdout examples — not a live model server.</p>
      <p className="prose">
        Pick one of the three request bodies committed in the repository.{" "}
        <span className="mono">examples/score_response.json</span> is the response for the first. The save-call
        and offer files are the other two requests the README demo posts. Those two do not have committed
        response JSON, so their panels show the documented action and the rule text, and they leave probability,
        CLV, and reasons blank.
      </p>
      <ScoreDesk cases={demo.cases} />
    </>
  );
}
