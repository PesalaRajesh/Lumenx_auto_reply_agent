"""scripts/inspect_db.py — quick DB inspection."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

db_path = "data/agent.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

print("=" * 60)
print(f"Database: {db_path}")
print("=" * 60)

tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(f"\nTables: {tables}")

prod_count = c.execute("SELECT COUNT(*) FROM products_cache").fetchone()[0]
token_count = c.execute("SELECT COUNT(*) FROM token_log").fetchone()[0]
print(f"Products cached:    {prod_count}")
print(f"Token log entries:  {token_count}")

row = c.execute("SELECT SUM(input_tokens), SUM(output_tokens), SUM(cost_usd) FROM token_log").fetchone()
total_in, total_out, total_cost = row[0] or 0, row[1] or 0, row[2] or 0.0
print(f"\nTotal tokens used:  {total_in:,} in  /  {total_out:,} out")
print(f"Total bootstrap cost: ${total_cost:.5f}")

print("\nBreakdown by step + model:")
print(f"  {'Step':18s} {'Model':22s} {'Input':>8s} {'Output':>8s} {'Cost':>10s} {'Calls':>7s}")
print(f"  {'-'*18} {'-'*22} {'-'*8} {'-'*8} {'-'*10} {'-'*7}")
for row in c.execute(
    "SELECT step, model, SUM(input_tokens), SUM(output_tokens), SUM(cost_usd), COUNT(*) "
    "FROM token_log GROUP BY step, model ORDER BY SUM(cost_usd) DESC"
):
    print(f"  {row[0]:18s} {row[1]:22s} {row[2]:>8} {row[3]:>8} ${row[4]:>8.5f} {row[5]:>7}")

print("\nSample products cached:")
for row in c.execute("SELECT id, name FROM products_cache LIMIT 5"):
    print(f"  - {row[0]:20s} {row[1]}")
print(f"  ... ({prod_count - 5} more)" if prod_count > 5 else "")

conn.close()
