# OpenAI Gym Hanabi

This repository implements Hanabi as an OpenAI Gym environment. See
[here](https://github.com/openai/gym/blob/master/gym/envs/README.md) for
information on the structure of this repository.

## TODO
- [ ] Experiment with different loss functions.
- [ ] Experiment with different algorithms.
- [ ] Tune various hyperparameters.
- [ ] Get the code running on a cluster so that we can try various algorithms
      with various hyperparameters and train for a long time.
- [ ] Train against the hard-coded policy.
- [ ] Figure out how to store policies for later use.
- [ ] Play the random policy against itself to see how well it does.
- [ ] Benchmark to make sure that our Hanabi implementation isn't slow.
- [ ] Try reducing the number of cards and colors.
- [ ] Try playing with large hands with lots of turns after drawing the last
      card.
