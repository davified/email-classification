#!/usr/bin/env bash

set -e

brew bundle

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

project_name=$(basename $(pwd))
python -m ipykernel install --user --name=${project_name}

echo "Done!"
