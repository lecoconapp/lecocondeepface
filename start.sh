#!/usr/bin/env bash
pkill -f gunicorn 2>/dev/null || true
sleep 2
cd /opt/lecocon-deepface
setsid ./venv/bin/gunicorn --workers 1 --threads 1 --timeout 120 -b 0.0.0.0:8000 app:app > server.log 2>&1 < /dev/null &
echo "started pid $!"
