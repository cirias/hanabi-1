#! /usr/bin/env bash

set -euo pipefail

main() {
    python heuristic_policy.py
    python heuristic_simple_policy.py
    python keyboard_policy.py
    python random_policy.py
}

main
