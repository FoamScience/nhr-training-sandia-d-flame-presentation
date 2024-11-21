#!/usr/bin/bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm i -s html-inject-meta
manim-present
./node_modules/html-inject-meta/cli.js < YamlPresentation.html  > index.html
deactivate
