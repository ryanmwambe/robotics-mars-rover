#!/usr/bin/env bash
# Run traffic cone detection (default stream on port 5000).
cd "$(dirname "$0")"
exec ./venv/bin/python traffic_cone.py "$@"
