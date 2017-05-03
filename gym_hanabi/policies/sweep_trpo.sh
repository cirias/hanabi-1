#! /usr/bin/env bash

set -euo pipefail

sweep_self() {
    readonly hidden_sizes=("(8,8)" "(16,16)" "(32,32)" "(64,64)" "(64,64,64)")
    for hidden_size in "${hidden_sizes[@]}"; do
        name="mini_trpo_self_hidden_size=$hidden_size"
        python trpo_self.py \
            --hidden_sizes="$hidden_size" \
            MiniHanabiSelf-v0 \
            "pickled_policies/${name}.pickle" \
            "logs/$name"
    done

    readonly batch_sizes=(1000 4000 10000)
    for batch_size in "${batch_sizes[@]}"; do
        name="mini_trpo_self_batch_size=$batch_size"
        python trpo_self.py \
            --batch_size="$batch_size" \
            MiniHanabiSelf-v0 \
            "pickled_policies/${name}.pickle" \
            "logs/$name"
    done

    readonly discounts=(1 0.999 0.99)
    for discount in "${discounts[@]}"; do
        name="mini_trpo_self_discount=$discount"
        python trpo_self.py \
            --discount="$discount" \
            MiniHanabiSelf-v0 \
            "pickled_policies/${name}.pickle" \
            "logs/$name"
    done

    readonly step_sizes=(1 0.1 0.01 0.001)
    for step_size in "${step_sizes[@]}"; do
        name="mini_trpo_self_step_size=$step_size"
        python trpo_self.py \
            --step_size="$step_size" \
            MiniHanabiSelf-v0 \
            "pickled_policies/${name}.pickle" \
            "logs/$name"
    done

    readonly rewards=("" LinearReward SquaredReward SkewedReward)
    for reward in "${rewards[@]}"; do
        name="mini_trpo_${reward}_self"
        python trpo_self.py \
            MiniHanabi${reward}Self-v0 \
            "pickled_policies/${name}.pickle" \
            "logs/$name"
    done
}

sweep_ai() {
    readonly hidden_sizes=("(8,8)" "(16,16)" "(32,32)" "(64,64)" "(64,64,64)")
    for hidden_size in "${hidden_sizes[@]}"; do
        name="mini_trpo_self_hidden_size=$hidden_size"
        python trpo_ai.py \
            --hidden_sizes="$hidden_size" \
            MiniHanabiAi-v0 \
            "pickled_policies/MiniHeuristicPolicy.pickle" \
            "pickled_policies/${name}.pickle" \
            "logs/$name"
    done

    readonly batch_sizes=(1000 4000 10000)
    for batch_size in "${batch_sizes[@]}"; do
        name="mini_trpo_ai_batch_size=$batch_size"
        python trpo_ai.py \
            --batch_size="$batch_size" \
            MiniHanabiAi-v0 \
            "pickled_policies/MiniHeuristicPolicy.pickle" \
            "pickled_policies/${name}.pickle" \
            "logs/$name"
    done

    readonly discounts=(1 0.999 0.99)
    for discount in "${discounts[@]}"; do
        name="mini_trpo_ai_discount=$discount"
        python trpo_ai.py \
            --discount="$discount" \
            MiniHanabiAi-v0 \
            "pickled_policies/MiniHeuristicPolicy.pickle" \
            "pickled_policies/${name}.pickle" \
            "logs/$name"
    done

    readonly step_sizes=(1 0.1 0.01 0.001)
    for step_size in "${step_sizes[@]}"; do
        name="mini_trpo_ai_step_size=$step_size"
        python trpo_ai.py \
            --step_size="$step_size" \
            MiniHanabiAi-v0 \
            "pickled_policies/MiniHeuristicPolicy.pickle" \
            "pickled_policies/${name}.pickle" \
            "logs/$name"
    done

    readonly rewards=("" LinearReward SquaredReward SkewedReward)
    for reward in "${rewards[@]}"; do
        name="mini_trpo_${reward}_ai"
        python trpo_ai.py \
            MiniHanabi${reward}Ai-v0 \
            "pickled_policies/MiniHeuristic${reward}Policy.pickle" \
            "pickled_policies/${name}.pickle" \
            "logs/$name"
    done
}

main() {
    sweep_self
    sweep_ai
}

main
