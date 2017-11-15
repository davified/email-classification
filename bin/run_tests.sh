#!/usr/bin/env bash
# Note: call this from the project directory, using bin/run_tests.sh

set -e

PROJ_DIR="$(pwd)"

echo "Running unit tests with nose"
source .venv/bin/activate
nosetests -w "${PROJ_DIR}/app/tests"
