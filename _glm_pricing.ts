import Database from 'better-sqlite3';
const db = new Database('data/cognitive-router.db');
const cols = db.prepare('PRAGMA table_info(catalog_pricing)').all();
console.log('columns:', cols.map((c: any) => c.name));
const rows = db.prepare(
  "SELECT key, input_per_m, output_per_m FROM catalog_pricing WHERE key LIKE '%glm%' ORDER BY key"
).all();
console.log(JSON.stringify(rows, null, 2));
db.close();