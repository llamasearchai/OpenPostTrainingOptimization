#!/bin/bash
# Healthcheck script for OpenPostTrainingOptimizations
# Author: Nik Jois

set -e

# Check if the service is running
if command -v opt &> /dev/null; then
    opt status --device cpu > /dev/null 2>&1
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Healthcheck passed"
        exit 0
    else
        echo "Healthcheck failed: opt status returned $exit_code"
        exit 1
    fi
else
    echo "Healthcheck failed: opt command not found"
    exit 1
fi

