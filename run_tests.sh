#! /usr/bin/env bash

set -euo pipefail

main() {
    python3 -m unittest
    python3 -m examples.case_study
    python3 -m examples.paper
    python3 -m examples.plot_load_distribution --output "/tmp"
    python3 -m examples.plot_node_loads --output "/tmp/load_nodes.pdf"
    python3 -m examples.plot_workload_distribution \
        --output "/tmp/workload_distribution.pdf"
    python3 -m examples.tutorial
}

main
