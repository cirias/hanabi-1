from gym_hanabi.envs import hanabi_env
import pickle


class HeuristicSimplePolicy(object):
    """ Simpler Heuristic policy for mini-hanabi (probably won't do well on
    larger hanabi) """

    def __init__(self, config):
        self.config = config

    def get_move(self, observation):
        observation = hanabi_env.GameStateObservation(self.config, observation)
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
            for card, cardinfo in zip(observation.them.cards, observation.them.info):
                if card.number == CARDCOUNT:
                    print(card, cardinfo)
                    if cardinfo.number is None and observation.num_tokens > 0:
                        print("informing about a", CARDCOUNT)
                        return hanabi_env.InformNumberMove(card.number)

            # 2) color? skip for now because ai doesn't appear to learn to use color info

            # 3) if we can't give info about num=CARDCOUNT cards, see if we can place any of them
            for cardind, cardinfo in enumerate(observation.you.info):
                if cardinfo.number == CARDCOUNT:
                    print("played cards keys:", observation.played_cards.keys())

                    if (count_num_played(CARDCOUNT) == numcolors) and not skips[CARDCOUNT-1]:
                        # all of num=CARDCOUNT are already played, so discard any of num that are left
                        print("discarding a", CARDCOUNT)
                        return hanabi_env.DiscardMove(cardind)
                    # b) TODO check color (but we don't give color info currently)
                    else:
                        # just try playing it
                        print("playing a", CARDCOUNT)
                        return hanabi_env.PlayMove(cardind)

        # at this point, we have no info, random play or random discard? - 
        # TODO: don't randomly guess if there's only one fuse
#        if observation.num_fuses > 0:  #change back to 1 to do what the comment says
        return hanabi_env.PlayMove(0)
#        else:
#        return hanabi_env.DiscardMove(0)

    def get_action(self, observation):
        return (hanabi_env.move_to_sample(self.config, self.get_move(observation)), )

if __name__ == "__main__":
    configs_and_names = [
        (hanabi_env.HANABI_CONFIG, "HeuristicSimplePolicy"),
        (hanabi_env.MEDIUM_HANABI_CONFIG, "MediumHeuristicSimplePolicy"),
        (hanabi_env.MINI_HANABI_CONFIG, "MiniHeuristicSimplePolicy"),
        (hanabi_env.MINI_HANABI_LOTSOFINFO_CONFIG, "MiniHeuristicSimpleLotsOfInfoPolicy"),
        (hanabi_env.MINI_HANABI_LOTSOFTURNS_CONFIG, "MiniHeuristicSimpleLotsOfTurnsPolicy"),
        (hanabi_env.MINI_HANABI_LINEAR_REWARD_CONFIG, "MiniHeuristicSimpleLinearRewardPolicy"),
        (hanabi_env.MINI_HANABI_SQUARED_REWARD_CONFIG, "MiniHeuristicSimpleSquaredRewardPolicy"),
        (hanabi_env.MINI_HANABI_SKEWED_REWARD_CONFIG, "MiniHeuristicSimpleSkewedRewardPolicy"),
    ]

    for config, name in configs_and_names:
        with open("pickled_policies/{}.pickle".format(name), "wb") as f:
            pickle.dump(HeuristicSimplePolicy(config), f)
