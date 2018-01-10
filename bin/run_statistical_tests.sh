#!/usr/bin/env bash
# Note: call this from the project directory, using bin/run_unit_tests.sh

set -e

PROJ_DIR="$(pwd)"

echo "Running statistical tests with nose"
source ${PROJ_DIR}/.venv/bin/activate
nosetests -w "${PROJ_DIR}/app" -a 'statistical_test'
