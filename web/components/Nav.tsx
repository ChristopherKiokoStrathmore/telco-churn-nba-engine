"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Briefing" },
  { href: "/demo", label: "Demo" },
  { href: "/score", label: "Score" },
  { href: "/metrics", label: "Metrics" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <nav className="nav" aria-label="Pages">
      {LINKS.map((link) => {
        const current = link.href === "/" ? pathname === "/" || pathname === "/briefing" : pathname === link.href;
        return (
          <Link key={link.href} href={link.href} aria-current={current ? "page" : undefined}>
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}
