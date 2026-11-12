#!/bin/sh
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
    python3 run_dashboard.py
else
    echo "Python 3 is not installed. Open OPEN_DASHBOARD.html directly instead."
fi
printf '\nPress Enter to close this window...'
read answer
