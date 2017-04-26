# Running rllab

1. Install [conda][conda_install]. The Python 3.6 version works, though rllab
   suggests the Python 2.7 version (even though rllab itself is written in
   Python 3).
2. Clone rllab and follow the [rllab installation instructions][rllab_install]
   to install rllab.
3. Start the `rllab3` conda environment by running `source activate rllab3`.
4. Add `gym_hanabi` and `rllab` to your `PYTHONPATH` (e.g. `export
   PYTHONPATH="$HOME/gym_hanabi:$HOME/rllab:$PYTHONPATH"`).
4. Run `python trpo.py`. This script will print out a bunch of stuff and store
   things in the `data/local/experiment` directory inside your rllab clone.

[conda_install]: https://www.continuum.io/downloads
[rllab_install]: https://rllab.readthedocs.io/en/latest/user/installation.html
