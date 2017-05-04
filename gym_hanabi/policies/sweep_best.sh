#! /usr/bin/env bash

set -euo pipefail

copyresult() {
    mkdir -p /nscratch/sagark/hanabilogs/$HOSTNAME/pickled
    cp -r logs/* /nscratch/sagark/hanabilogs/$HOSTNAME/
    cp -r pickled_policies/* /nscratch/sagark/hanabilogs/$HOSTNAME/pickled
}

main() {
    N_ITR=2000
    N_REPEATS=16
    N_SIMS=1000

    # Self
    if [ "$HOSTNAME" = f1 ]; then
    python trpo_self.py \
        MiniHanabiSelf-v0 \
        pickled_policies/mini_trpo_self_best.pickle \
        logs/mini_trpo_self_best \
        --n_itr=$N_ITR
    ./run_self.py -n $N_SIMS \
        MiniHanabiSelf-v0 \
        pickled_policies/mini_trpo_self_best.pickle \
        --output=logs/mini_trpo_self_best.txt
    fi

    if [ "$HOSTNAME" = f2 ]; then
    python trpo_self.py \
        MediumHanabiSelf-v0 \
        pickled_policies/medium_trpo_self_best.pickle \
        logs/medium_trpo_self_best \
        --n_itr=$N_ITR
    ./run_self.py -n $N_SIMS \
        MediumHanabiSelf-v0 \
        pickled_policies/medium_trpo_self_best.pickle \
        --output=logs/medium_trpo_self_best.txt
    fi

    if [ "$HOSTNAME" = f3 ]; then
    python trpo_self.py \
        HanabiSelf-v0 \
        pickled_policies/trpo_self_best.pickle \
        logs/trpo_self_best \
        --n_itr=$N_ITR
    ./run_self.py -n $N_SIMS \
        HanabiSelf-v0 \
        pickled_policies/trpo_self_best.pickle \
        --output=logs/trpo_self_best.txt
    fi

    # Staggered
    if [ "$HOSTNAME" = f4 ]; then
    python trpo_staggered.py \
        MiniHanabiAi-v0 \
        pickled_policies/mini_trpo_staggered_best.pickle \
        logs/mini_trpo_staggered_best \
        --n_itr=$(($N_ITR / $N_REPEATS)) \
        --n_repeats=20
    ./run_self.py -n $N_SIMS \
        MiniHanabiSelf-v0 \
        pickled_policies/mini_trpo_staggered_best.pickle \
        --output=logs/mini_trpo_staggered_best.txt

    fi

    if [ "$HOSTNAME" = f5 ]; then
    python trpo_staggered.py \
        MediumHanabiAi-v0 \
        pickled_policies/medium_trpo_staggered_best.pickle \
        logs/medium_trpo_staggered_best \
        --n_itr=$(($N_ITR / $N_REPEATS)) \
        --n_repeats=20
    ./run_self.py -n $N_SIMS \
        MediumHanabiSelf-v0 \
        pickled_policies/medium_trpo_staggered_best.pickle \
        --output=logs/medium_trpo_staggered_best.txt
    fi

    if [ "$HOSTNAME" = f6 ]; then
    python trpo_staggered.py \
        HanabiAi-v0 \
        pickled_policies/trpo_staggered_best.pickle \
        logs/trpo_staggered_best \
        --n_itr=$(($N_ITR / $N_REPEATS)) \
        --n_repeats=$N_REPEATS
    ./run_self.py -n $N_SIMS \
        HanabiSelf-v0 \
        pickled_policies/trpo_staggered_best.pickle \
        --output=logs/trpo_staggered_best.txt
    fi

    # AI
    if [ "$HOSTNAME" = f7 ]; then
    python trpo_ai.py \
        MiniHanabiAi-v0 \
        pickled_policies/MiniHeuristicPolicy.pickle \
        pickled_policies/mini_trpo_ai_best.pickle \
        logs/mini_trpo_ai_best \
        --n_itr=$N_ITR
    ./run_ai.py -n $N_SIMS \
        MiniHanabiAi-v0 \
        pickled_policies/mini_trpo_ai_best.pickle \
        pickled_policies/MiniHeuristicPolicy.pickle \
        --output=logs/mini_trpo_ai_best.txt
    fi

    if [ "$HOSTNAME" = f8 ]; then
    python trpo_ai.py \
        MediumHanabiAi-v0 \
        pickled_policies/MediumHeuristicPolicy.pickle \
        pickled_policies/medium_trpo_ai_best.pickle \
        logs/medium_trpo_ai_best \
        --n_itr=$N_ITR
    ./run_ai.py -n $N_SIMS \
        MediumHanabiAi-v0 \
        pickled_policies/medium_trpo_ai_best.pickle \
        pickled_policies/MediumHeuristicPolicy.pickle \
        --output=logs/medium_trpo_ai_best.txt
    fi

    if [ "$HOSTNAME" = f9 ]; then
    python trpo_ai.py \
        HanabiAi-v0 \
        pickled_policies/HeuristicPolicy.pickle \
        pickled_policies/trpo_ai_best.pickle \
        logs/trpo_ai_best \
        --n_itr=$N_ITR
    ./run_ai.py -n $N_SIMS \
        HanabiAi-v0 \
        pickled_policies/trpo_ai_best.pickle \
        pickled_policies/HeuristicPolicy.pickle \
        --output=logs/trpo_ai_best.txt
    fi

    copyresult
}

main
