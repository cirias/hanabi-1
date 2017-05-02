from gym_hanabi.envs import hanabi

class HeuristicSimplePolicy(object):
    """
    Simpler Heuristic policy for mini-hanabi (probably won't do well on larger
    hanabi)
    """

    def __init__(self, env):
        self.env = env
        self.config = self.env.config
        self.reward = self.env.reward
        self.spaces = self.env.spaces

        err = "HeuristicSimplePolicy only supports 2 players."
        assert self.config.num_players == 2, err

    def get_move(self, observation_sample):
        observation = self.spaces.sample_to_observation(observation_sample)
        numcolors = len(self.config.colors)
        numnumbers = len(self.config.card_counts)
        skips = [True for x in range(numnumbers)]
        skips[0] = False

        def count_num_played(num):
            count = 0
            for color in self.config.colors:
                if observation.played_cards[color] >= num:
                    count += 1
            return count

        for CARDCOUNT in range(1, numnumbers+1):
            # 1) walk through their cards and their info. if there's a card
            # with number = CARDCOUNT that they don't know about, tell them
            them = observation.players[0]
            them.cards = [card for card in them.cards if card is not None]
            them.info = [info for info in them.info if info is not None]
            for card, cardinfo in zip(them.cards, them.info):
                if card.number == CARDCOUNT:
                    if cardinfo.number is None and observation.num_tokens > 0:
                        return hanabi.InformNumberMove(card.number, 0)

            # 3) if we can't give info about num=CARDCOUNT cards, see if we can place any of them
            for cardind, cardinfo in enumerate(observation.your_info):
                if cardinfo.number == CARDCOUNT:

                    if (count_num_played(CARDCOUNT) == numcolors) and not skips[CARDCOUNT-1]:
                        # all of num=CARDCOUNT are already played, so discard any of num that are left
                        return hanabi.DiscardMove(cardind)
                    # b) TODO check color (but we don't give color info currently)
                    # just try playing it
                    return hanabi.PlayMove(cardind)

        # at this point, we have no info, random play (random discard is worse)
        return hanabi.PlayMove(0)

    def get_action(self, observation):
        return (self.spaces.action_to_sample(self.get_move(observation)), )
