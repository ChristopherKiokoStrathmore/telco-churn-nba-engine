import type { ModelRow } from "@/lib/sources";

export function ModelTable({ caption, rows }: { caption: string; rows: ModelRow[] }) {
  return (
    <>
      <div className="table-wrap">
        <table>
          <caption>{caption}</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col" className="num">
                ROC-AUC
              </th>
              <th scope="col" className="num">
                PR-AUC
              </th>
              <th scope="col" className="num">
                Top-decile lift
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} className={row.served ? "served" : undefined}>
                <th scope="row">
                  {row.label}
                  {row.served ? " (served)" : ""}
                </th>
                <td className="num">{row.rocAuc}</td>
                <td className="num">{row.prAuc}</td>
                <td className="num">{row.lift}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="scroll-hint">Scroll sideways for the full table.</p>
    </>
  );
}
