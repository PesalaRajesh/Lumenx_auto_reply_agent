"""Delete bootstrap drafts so we can re-run with new system prompt."""
import sqlite3
c = sqlite3.connect("data/agent.db")
n = c.execute("DELETE FROM drafts WHERE status='bootstrap'").rowcount
c.commit()
print(f"Cleared {n} bootstrap drafts")
c.close()
