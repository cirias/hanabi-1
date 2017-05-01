import collections

import gym
import gym.spaces
import gym_hanabi
from gym_hanabi.envs import hanabi
from overrides import overrides

class Spaces(object):
    def observation_space(self):
        raise NotImplementedError()

    def observation_to_sample(self, observation):
        raise NotImplementedError()

    def sample_to_observation(self, sample):
        raise NotImplementedError()

    def action_space(self):
        raise NotImplementedError()

    def action_to_sample(self, move):
        raise NotImplementedError()

    def sample_to_action(self, sample):
        raise NotImplementedError()


class NestedSpaces(Spaces):
    def __init__(self, config):
        self.config = config

    def color_to_sample(self, color):
        assert color in self.config.colors, color
        return self.config.colors.index(color)

    def sample_to_color(self, sample):
        assert 0 <= sample < len(self.config.colors), sample
        return self.config.colors[sample]

    def number_to_sample(self, number):
        assert 1 <= number <= len(self.config.card_counts), number
        return number - 1

    def sample_to_number(self, sample):
        assert 0 <= sample < len(self.config.card_counts), sample
        return sample + 1

    def card_space(self):
        """
        Consider a config with the colors [red, green] and numbers [1, 2]. The
        card space encodes all possible combinations of color and number (e.g.
        1R -> (0,0), 2G -> (1,1)). It also encodes the possibility of a missing
        card, which is represented in Python as None (e.g. None -> (2,2)).
        """
        return gym.spaces.Tuple((
            gym.spaces.Discrete(len(self.config.colors) + 1),     # Color
            gym.spaces.Discrete(len(self.config.card_counts) + 1) # Number
        ))

    def card_to_sample(self, card):
        if card is None:
            return (len(self.config.colors), len(self.config.card_counts))
        else:
            color_sample = self.color_to_sample(card.color)
            number_sample = self.number_to_sample(card.number)
            return (color_sample, number_sample)

    def sample_to_card(self, sample):
        assert len(sample) == 2, sample
        if sample == self.card_to_sample(None):
            return None
        else:
            color = self.sample_to_color(sample[0])
            number = self.sample_to_number(sample[1])
            return hanabi.Card(color, number)

    def unique_cards(self):
        """
        Returns a list of all the unique cards specified by `config`. The cards
        are ordered first by color and then by number.
        """
        num_numbers = len(self.config.card_counts)
        return [hanabi.Card(c, n)
                    for c in self.config.colors
                    for n in range(1, num_numbers + 1)]

    def discarded_cards_space(self):
        """
        The discarded cards space specifies the number of each unique card that
        has been discarded. For example, imagine a config with colors [red,
        blue] and numbers [1, 2]. `unique_cards` returns [1R, 2R, 1B, 2B]. Our
        discarded cards space has four numbers; the first is the number of
        discarded 1R's, the second is the number of discarded 2R's, and so on.
        """
        count = lambda number: self.config.card_counts[number - 1]
        spaces = [gym.spaces.Discrete(count(c.number) + 1)
                      for c in self.unique_cards()]
        return gym.spaces.Tuple(tuple(spaces))

    def played_cards_space(self):
        """
        The played cards space is similar to the discarded cards space, except
        that a card can only be played at most once.
        """
        num_cards = len(self.unique_cards())
        return gym.spaces.Tuple(tuple([gym.spaces.Discrete(2)] * num_cards))

    def cards_to_sample(self, cards):
        assert all(card in self.unique_cards() for card in cards), cards
        return tuple([cards.count(card) for card in self.unique_cards()])

    def sample_to_cards(self, sample):
        ucs = self.unique_cards()
        assert len(sample) == len(ucs)
        return [c for (card, count) in zip(ucs, sample) for c in [card] * count]

    def information_space(self):
        """
        Consider a config with colors [red, blue] and numbers [1, 2]. Information
        can be
          - any combination of the colors and numbers (e.g. 1R, 2B);
          - any combination of an unknown color and a known number (e.g. 1?, 2?);
          - any combination of an unknown number and a known color (e.g. ?R, ?B);
          - completely unkown (e.g. ??, ??); or
          - absent.
        """
        return gym.spaces.Tuple((
            gym.spaces.Discrete(len(self.config.colors) + 2),     # Color
            gym.spaces.Discrete(len(self.config.card_counts) + 2) # Number
        ))

    def information_to_sample(self, info):
        if info is None:
            return (len(self.config.colors) + 1,
                    len(self.config.card_counts) + 1)
        else:
            if info.color:
                color_sample = self.color_to_sample(info.color)
            else:
                color_sample = len(self.config.colors)

            if info.number:
                number_sample = self.number_to_sample(info.number)
            else:
                number_sample = len(self.config.card_counts)

            return (color_sample, number_sample)

    def sample_to_information(self, s):
        if s == self.information_to_sample(None):
            return None
        else:
            color = (None if s[0] == len(self.config.colors)
                          else self.sample_to_color(s[0]))
            number = (None if s[1] == len(self.config.card_counts)
                           else self.sample_to_number(s[1]))
            return hanabi.Information(color, number)

    @overrides
    def observation_space(self):
        c = self.config
        return gym.spaces.Tuple((
            # Tokens
            gym.spaces.Discrete(self.config.max_tokens),
            # Fuses
            gym.spaces.Discrete(self.config.max_fuses),
            # Discarded cards
            self.discarded_cards_space(),
            # Played cards
            self.played_cards_space(),
            # Their cards
            gym.spaces.Tuple([self.card_space()] * c.hand_size),
            # Their info
            gym.spaces.Tuple([self.information_space()] * c.hand_size),
            # Your info
            gym.spaces.Tuple([self.information_space()] * c.hand_size)
        ))

    @overrides
    def observation_to_sample(self, obs):
        played_cards = []
        for color, count in obs.played_cards.items():
            numbers = range(1, count + 1)
            played_cards += [hanabi.Card(color, number) for number in numbers]

        return (
            obs.num_tokens - 1,
            obs.num_fuses - 1,
            self.cards_to_sample(obs.discarded_cards),
            self.cards_to_sample(played_cards),
            tuple(self.card_to_sample(card) for card in obs.them.cards),
            tuple(self.information_to_sample(info) for info in obs.them.info),
            tuple(self.information_to_sample(info) for info in obs.you.info)
        )

    @overrides
    def sample_to_observation(self, sample):
        (sample_num_tokens,
         sample_num_fuses,
         sample_discarded_cards,
         sample_played_cards,
         sample_their_cards,
         sample_their_info,
         sample_your_info) = sample

        num_tokens = sample_num_tokens + 1
        num_fuses = sample_num_fuses + 1

        discarded_cards = self.sample_to_cards(sample_discarded_cards)
        played_cards = collections.defaultdict(int)
        for color, number in self.sample_to_cards(sample_played_cards):
            if number > played_cards[color]:
                played_cards[color] = number

        # Filter out cards that don't exist, for hands that are smaller than
        # the maximum hand size.
        them = hanabi.Hand(
            [self.sample_to_card(sample) for sample in sample_their_cards],
            [self.sample_to_information(sample) for sample in sample_their_info])
        them.cards = [card for card in them.cards if card is not None]
        them.info = [info for info in them.info if info is not None]
        assert len(them.cards) == len(them.info), them

        you = hanabi.Hand(
            [None] * self.config.hand_size,
            [self.sample_to_information(sample) for sample in sample_your_info])
        you.info = [info for info in you.info if info is not None]
        you.cards = [None] * len(you.info)

        return hanabi.Observation(num_tokens, num_fuses, discarded_cards,
                                  played_cards, them, you)

    def moves(self):
        c = self.config
        num_numbers = len(c.card_counts)
        return (
            [hanabi.InformColorMove(c) for c in c.colors] +
            [hanabi.InformNumberMove(i) for i in range(1, num_numbers + 1)] +
            [hanabi.DiscardMove(i) for i in range(c.hand_size)] +
            [hanabi.PlayMove(i) for i in range(c.hand_size)]
        )

    @overrides
    def action_space(self):
        return gym.spaces.Discrete(len(self.moves()))

    @overrides
    def action_to_sample(self, move):
        c = self.config
        num_numbers = len(c.card_counts)
        offsets = [0, len(c.colors), num_numbers, c.hand_size]
        if isinstance(move, hanabi.InformColorMove):
            return sum(offsets[:1]) + c.colors.index(move.color)
        elif isinstance(move, hanabi.InformNumberMove):
            return sum(offsets[:2]) + range(1, num_numbers + 1).index(move.number)
        elif isinstance(move, hanabi.DiscardMove):
            return sum(offsets[:3]) + range(c.hand_size).index(move.index)
        elif isinstance(move, hanabi.PlayMove):
            return sum(offsets[:4]) + range(c.hand_size).index(move.index)
        else:
            raise ValueError("Unexpected move {}.".format(move))

    @overrides
    def sample_to_action(self, sample):
        assert 0 <= sample < len(self.moves()), sample
        return self.moves()[sample]


class FlattenedSpaces(Spaces):
    pass
