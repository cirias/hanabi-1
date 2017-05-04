#! /usr/bin/env bash

set -euo pipefail

main() {
    python cem_self.py \
        MiniHanabiSelf-v0 \
        pickled_policies/mini_cem_self.pickle \
        logs/mini_cem_self

    python cma_es_self.py \
        MiniHanabiSelf-v0 \
        pickled_policies/mini_cma_es_self.pickle \
        logs/mini_cma_es_self

    python trpo_self.py \
        MiniHanabiSelf-v0 \
        pickled_policies/mini_trpo_self.pickle \
        logs/mini_trpo_self \
        --n_itr=1000

    ./run_self.py -n 1000 \
        MiniHanabiSelf-v0 \
        pickled_policies/mini_cem_self.pickle \
        --output=logs/mini_cem_self.txt

    ./run_self.py -n 1000 \
        MiniHanabiSelf-v0 \
        pickled_policies/mini_cma_es_self.pickle \
        --output=logs/mini_cma_es_self.txt

    ./run_self.py -n 1000 \
        MiniHanabiSelf-v0 \
        pickled_policies/mini_trpo_self.pickle \
        --output=logs/mini_trpo_self.txt
}

main
