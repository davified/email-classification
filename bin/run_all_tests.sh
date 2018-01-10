#!/usr/bin/env bash
# Note: call this from the project directory, using bin/run_unit_tests.sh

set -e

PROJ_DIR="$(pwd)"

source ${PROJ_DIR}/.venv/bin/activate

${PROJ_DIR}/bin/run_unit_tests.sh
${PROJ_DIR}/bin/run_statistical_tests.sh
