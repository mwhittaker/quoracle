#! /usr/bin/env bash

set -euo pipefail

main() {
    python -m unittest
    python -m examples.case_study
    python -m examples.paper
    python -m examples.plot_load_distribution --output "/tmp"
    python -m examples.plot_node_loads --output "/tmp/load_nodes.pdf"
    python -m examples.plot_workload_distribution \
        --output "/tmp/workload_distribution.pdf"
    python -m examples.tutorial
}

main
