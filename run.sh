#!/usr/bin/env bash
# Run the Mars Rover web stream with the project venv.
cd "$(dirname "$0")"
exec ./venv/bin/python app.py
