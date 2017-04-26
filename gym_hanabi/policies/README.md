# Policies

## Overview
A `Policy` is a class which implements a `get_action` method. It looks
something like this:

```python
class MyPolicy(object):
    def get_action(self, observation):
        return (42, )
```

`get_action` takes in an observation from some observation space and produces
an action wrapped in a tuple. This directory implements three basic policies:

1. the [`KeyboardPolicy`](keyboard_policy.py) reads actions from stdin;
2. the [`RandomPolicy`](random_policy.py) acts randomly; and
3. the [`HeuristicPolicy`](heuristic_policy.py) is a heuristic based Hanabi AI.

If you run [`pickle.sh`](pickle.sh), a pickled version of the three basic
policies will be put in the [`pickled_policies`](pickled_policies) directories.
We can then play these pickled policies against each other:

```bash
# Play a single game of Hanabi against ourselves.
$ ./run_self.py -r -n 1 pickled_policies/KeyboardPolicy.pickle /tmp/keyboard

# Play a Hanabi AI against itself 100 times; don't render the game.
$ ./run_self.py -n 100 pickled_policies/RandomPolicy.pickle /tmp/random
$ ./run_self.py -n 100 pickled_policies/HeuristicPolicy.pickle /tmp/heuristic

# Play one game of Hanabi against an AI.
$ ./run_ai.py -r -n 1 \
    pickled_policies/{KeyboardPolicy,RandomPolicy}.pickle \
    /tmp/keyboard_vs_random
$ ./run_ai.py -r -n 1 \
    pickled_policies/{KeyboardPolicy,HeuristicPolicy}.pickle \
    /tmp/keyboard_vs_heuristic

# Play two Hanabi AIs against each other 100 times.
$ ./run_ai.py -n 100 \
    pickled_policies/{RandomPolicy,HeuristicPolicy}.pickle \
    /tmp/random_vs_heuristic
```

In addition to the basic policies, we can also learn policies using rllab! In
particular, we can pickle and use any [rllab policy][rllab_policy]. For
example, run [`trpo.py`](trpo.py) (instructions below) to generate
`pickled_policies/trpo.pickle`. Then, we can play the AI against itself:

```
$ ./run_self -n 100 pickled_policies/trpo.py /tmp/trpo
```

## Learning policies with rllab
1. Install [conda][conda_install]. There are two versions of the conda
   installer: 3.6 and 2.7. The Python 3.6 version seems to work fine, though
   rllab suggests you use the Python 2.7 version (even though rllab itself is
   written in Python 3).
2. Clone [the rllab repository][rllab_repo] and follow the [rllab installation
   instructions][rllab_install] to install rllab. For example, if you're on
   linux, you'd run `setup_linux.sh`.
3. Start the `rllab3` conda environment by running `source activate rllab3`.
4. Add `gym_hanabi` and `rllab` to your `PYTHONPATH` (e.g. `export
   PYTHONPATH="$HOME/gym_hanabi:$HOME/rllab:$PYTHONPATH"`).
5. Write a script which uses rllab to learn and pickle a policy. For example,
   take a look at [`trpo.py`](trpo.py) which you can run with `python trpo.py`.

[conda_install]: https://www.continuum.io/downloads
[rllab_install]: https://rllab.readthedocs.io/en/latest/user/installation.html
[rllab_policy]: https://github.com/openai/rllab/blob/master/rllab/policies/base.py
[rllab_repo]: https://github.com/openai/rllab
