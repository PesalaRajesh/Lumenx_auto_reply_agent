#!/bin/sh
# Start the inbox poller in the background (shares the same SQLite DB)
python scripts/poll_inbox.py &

# Start uvicorn in the foreground (keeps the container alive)
exec uvicorn dashboard.main:app --host 0.0.0.0 --port ${PORT:-8080}
