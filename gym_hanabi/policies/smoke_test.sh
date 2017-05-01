#! /usr/bin/env bash

set -euo pipefail

run_self() {
    local readonly pp=pickled_policies
    local readonly policy="$1"

    ./run_self.py HanabiSelf-v0                  "${pp}/${policy}Policy.pickle"
    ./run_self.py MediumHanabiSelf-v0            "${pp}/Medium${policy}Policy.pickle"
    ./run_self.py MiniHanabiSelf-v0              "${pp}/Mini${policy}Policy.pickle"
    ./run_self.py MiniHanabiLotsOfInfoSelf-v0    "${pp}/Mini${policy}LotsOfInfoPolicy.pickle"
    ./run_self.py MiniHanabiLinearRewardSelf-v0  "${pp}/Mini${policy}LinearRewardPolicy.pickle"
    ./run_self.py MiniHanabiSquaredRewardSelf-v0 "${pp}/Mini${policy}SquaredRewardPolicy.pickle"
    ./run_self.py MiniHanabiSkewedRewardSelf-v0  "${pp}/Mini${policy}SkewedRewardPolicy.pickle"
}

run_ai() {
    local readonly pp=pickled_policies
    local readonly policy="$1"

    ./run_ai.py HanabiAi-v0                  "${pp}/${policy}Policy.pickle"{,}
    ./run_ai.py MediumHanabiAi-v0            "${pp}/Medium${policy}Policy.pickle"{,}
    ./run_ai.py MiniHanabiAi-v0              "${pp}/Mini${policy}Policy.pickle"{,}
    ./run_ai.py MiniHanabiLotsOfInfoAi-v0    "${pp}/Mini${policy}LotsOfInfoPolicy.pickle"{,}
    ./run_ai.py MiniHanabiLinearRewardAi-v0  "${pp}/Mini${policy}LinearRewardPolicy.pickle"{,}
    ./run_ai.py MiniHanabiSquaredRewardAi-v0 "${pp}/Mini${policy}SquaredRewardPolicy.pickle"{,}
    ./run_ai.py MiniHanabiSkewedRewardAi-v0  "${pp}/Mini${policy}SkewedRewardPolicy.pickle"{,}
}

main() {
    run_self Heuristic
    run_self HeuristicSimple
    run_self Random

    run_ai Heuristic
    run_ai HeuristicSimple
    run_ai Random
}

main
