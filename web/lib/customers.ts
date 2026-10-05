import type { FeatureMap, PortableModel } from "./portableScore";

export type ParsedCustomer = {
  customerId: string | null;
  label: "Yes" | "No" | null;
  features: FeatureMap;
};

export function parseCsv(text: string): string[][] {
  const source = text.replace(/^\uFEFF/, "");
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (inQuotes) {
      if (character === '"') {
        if (source[index + 1] === '"') {
          field += '"';
          index += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += character;
      }
      continue;
    }
    if (character === '"') {
      inQuotes = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n" || character === "\r") {
      if (character === "\r" && source[index + 1] === "\n") index += 1;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((cells) => cells.some((cell) => cell.trim() !== ""));
}

export function customersFromCsv(text: string, model: PortableModel): ParsedCustomer[] {
  const table = parseCsv(text);
  if (table.length < 2) {
    throw new Error("The CSV needs a header and at least one customer row.");
  }
  const header = table[0].map((cell) => cell.trim());
  const missing = model.feature_order.filter((name) => !header.includes(name));
  if (missing.length > 0) {
    throw new Error(`The CSV is missing ${missing.join(", ")}.`);
  }
  const idIndex = header.indexOf("customerID");
  const labelIndex = header.indexOf("Churn");
  if (table.length - 1 > 10000) {
    throw new Error("Upload at most 10,000 rows. The public sample has 7,043.");
  }
  return table.slice(1).map((cells, rowIndex) => {
    const features: FeatureMap = {};
    for (const name of model.feature_order) {
      const value = cells[header.indexOf(name)] ?? "";
      features[name] = value;
    }
    let label: "Yes" | "No" | null = null;
    if (labelIndex >= 0) {
      const raw = (cells[labelIndex] ?? "").trim();
      if (raw === "Yes" || raw === "No") label = raw;
      else if (raw !== "") {
        throw new Error(`Row ${rowIndex + 2} has Churn ${JSON.stringify(raw)}. Use Yes or No.`);
      }
    }
    const customerId = idIndex >= 0 ? (cells[idIndex] ?? "").trim() : "";
    return { customerId: customerId || null, label, features };
  });
}
