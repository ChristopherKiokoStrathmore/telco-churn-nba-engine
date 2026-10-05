import type { Metadata } from "next";
import Link from "next/link";
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
      <p className="banner">
        This desk shows the committed examples. <Link href="/demo">Score customers in the live demo</Link>.
      </p>
      <p className="prose">
        Pick one of the three request bodies committed in the repository. The live demo scores all three.{" "}
        <span className="mono">examples/score_response.json</span> is the committed response for the first. The
        save-call and offer files do not have committed response JSON, so their panels show the documented action
        and the rule text, and they leave probability, CLV, and reasons blank.
      </p>
      <ScoreDesk cases={demo.cases} />
    </>
  );
}
