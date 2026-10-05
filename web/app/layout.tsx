import type { Metadata } from "next";
import { Fraunces, IBM_Plex_Mono, Source_Sans_3 } from "next/font/google";
import Link from "next/link";
import { Nav } from "@/components/Nav";
import { demo } from "@/lib/sources";
import "./globals.css";

const display = Fraunces({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-display",
  weight: ["500", "600", "700"],
  style: ["normal", "italic"],
});

const sans = Source_Sans_3({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans",
  weight: ["400", "600", "700"],
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-mono",
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: {
    default: demo.projectTitle,
    template: "%s · Telco churn NBA",
  },
  description:
    "Holdout briefing for a churn model, a CLV proxy, and next-best-action rules on IBM's public telco sample. Scores shown here are committed examples, not a live model server.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${sans.variable} ${mono.variable}`}>
      <body>
        <a className="skip" href="#content">
          Skip to content
        </a>
        <div className="signal-bar" aria-hidden="true" />
        <div className="frame">
          <header>
            <div className="mast">
              <div>
                <Link className="wordmark" href="/">
                  telco-churn-nba-engine
                </Link>
                <p className="mast-kicker">Holdout briefing · IBM public sample</p>
              </div>
              <a className="github" href={demo.githubUrl} target="_blank" rel="noopener noreferrer">
                GitHub
                <span className="sr-only"> (opens in a new tab)</span>
              </a>
            </div>
            <Nav />
          </header>
          <main id="content">{children}</main>
          <footer className="footer">
            <p>
              Seed {demo.split.seed} · held-out n = {demo.split.nTest} · figures from the committed metrics,
              examples, and rule file.
            </p>
            <p>
              <a href={demo.githubUrl} target="_blank" rel="noopener noreferrer">
                ChristopherKiokoStrathmore/telco-churn-nba-engine
                <span className="sr-only"> (opens in a new tab)</span>
              </a>
            </p>
          </footer>
        </div>
      </body>
    </html>
  );
}
