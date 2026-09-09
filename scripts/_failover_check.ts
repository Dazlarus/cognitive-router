import Database from "better-sqlite3";

const db = new Database("data/cognitive-router.db", { readonly: true });
const cols = db.prepare("PRAGMA table_info(call_outcomes)").all().map((c: any) => c.name);
console.log("columns:", cols.join(","));
const tsCol = cols.includes("timestamp") ? "timestamp" : cols.includes("created_at") ? "created_at" : cols[0];
const rows = db
  .prepare(`SELECT * FROM call_outcomes WHERE ${tsCol} > ? ORDER BY ${tsCol}`)
  .all("2026-09-08T04:15:00")
  .filter((r: any) => r.timestamp < "2026-09-08T04:50:40" || r.provider === "zai");
for (const r of rows) console.log(JSON.stringify(r));
