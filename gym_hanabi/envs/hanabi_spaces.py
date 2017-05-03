import collections

import gym
import gym.spaces
import gym_hanabi
from gym_hanabi.envs import hanabi
from overrides import overrides

class Spaces(object):
    def __init__(self, config):
        self.config = config

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

    def sample_to_action(self, sample, cards):
        raise NotImplementedError()


class NestedSpaces(Spaces):
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
        other_players = [(
            # Their cards
            gym.spaces.Tuple([self.card_space()] * c.hand_size),
            # Their info
            gym.spaces.Tuple([self.information_space()] * c.hand_size),
            ) for _ in range(self.config.num_players - 1)]
        other_players = tuple(space for player in other_players for space in player)
        return gym.spaces.Tuple((
            # Tokens
            gym.spaces.Discrete(self.config.max_tokens),
            # Fuses
            gym.spaces.Discrete(self.config.max_fuses),
            # Discarded cards
            self.discarded_cards_space(),
            # Played cards
            self.played_cards_space(),
            # Your info
            gym.spaces.Tuple([self.information_space()] * c.hand_size)
        ) + other_players)

    @overrides
    def observation_to_sample(self, obs):
        played_cards = []
        for color, count in obs.played_cards.items():
            numbers = range(1, count + 1)
            played_cards += [hanabi.Card(color, number) for number in numbers]

        players = []
        for player in obs.players:
            players.append(tuple(self.card_to_sample(card) for card in player.cards))
            players.append(tuple(self.information_to_sample(info) for info in player.info))

        return (
            obs.num_tokens - 1,
            obs.num_fuses - 1,
            self.cards_to_sample(obs.discarded_cards),
            self.cards_to_sample(played_cards),
            tuple(self.information_to_sample(i) for i in obs.your_info),
        ) + tuple(players)

    @overrides
    def sample_to_observation(self, sample):
        (sample_num_tokens,
         sample_num_fuses,
         sample_discarded_cards,
         sample_played_cards,
         sample_your_info,
         *sample_players) = sample

        num_tokens = sample_num_tokens + 1
        num_fuses = sample_num_fuses + 1

        discarded_cards = self.sample_to_cards(sample_discarded_cards)
        played_cards = collections.defaultdict(int)
        for color, number in self.sample_to_cards(sample_played_cards):
            if number > played_cards[color]:
                played_cards[color] = number

        your_info = [self.sample_to_information(s) for s in sample_your_info]

        # Players hands' are organized like [cards, info, cards, info, ...]
        assert len(sample_players) % 2 == 0
        players = []
        while len(sample_players) != 0:
            sample_cards = sample_players.pop(0)
            sample_info = sample_players.pop(0)
            player = hanabi.Hand(
                [self.sample_to_card(s) for s in sample_cards],
                [self.sample_to_information(s) for s in sample_info])
            assert len(player.cards) == len(player.info), player
            players.append(player)

        return hanabi.Observation(num_tokens, num_fuses, discarded_cards,
                                  played_cards, your_info, players)

    def moves(self):
        c = self.config
        num_numbers = len(c.card_counts)
        return (
            [hanabi.InformColorMove(color, p) for p in range(c.num_players - 1)
                for color in c.colors] +
            [hanabi.InformNumberMove(i, p) for p in range(c.num_players - 1)
                for i in range(1, num_numbers + 1)] +
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
        offsets = [0, len(c.colors) * (c.num_players - 1), num_numbers * (c.num_players - 1), c.hand_size]
        if isinstance(move, hanabi.InformColorMove):
            return sum(offsets[:1]) + len(c.colors) * move.player + c.colors.index(move.color)
        elif isinstance(move, hanabi.InformNumberMove):
            return sum(offsets[:2]) + num_numbers * move.player + (move.number - 1)
        elif isinstance(move, hanabi.DiscardMove):
            return sum(offsets[:3]) + range(c.hand_size).index(move.index)
        elif isinstance(move, hanabi.PlayMove):
            return sum(offsets[:4]) + range(c.hand_size).index(move.index)
        else:
            raise ValueError("Unexpected move {}.".format(move))

    @overrides
    def sample_to_action(self, sample, cards):
        assert 0 <= sample < len(self.moves()), sample
        return self.moves()[sample]


class FlattenedSpaces(Spaces):
    def information_to_sample(self, information):
        card_colors = len(self.config.colors) + 1
        card_counts = len(self.config.card_counts) + 1
        sample = [0] * (card_colors * card_counts)
        for info in information:
            if info is None:
                continue

            if info.number is None:
                info_number = card_counts - 1
            else:
                info_number = info.number - 1

            if info.color is None:
                info_color = card_colors - 1
            else:
                info_color = self.config.colors.index(info.color)

            index = (info_color * card_counts) + info_number
            sample[index] += 1
        return tuple(sample)

    def get_information_vector(self):
        colors = self.config.colors + [None]
        counts = list(range(1, len(self.config.card_counts) + 1)) + [None]
        return [hanabi.Information(color, count) for color in colors for count in counts]

    def sample_to_information(self, sample):
        assert len(sample) == len(self.get_information_vector()), sample
        information = []
        for info, s in zip(self.get_information_vector(), sample):
            if s > 0:
                information += [info] * s
        return information

    @overrides
    def observation_space(self):
        assert self.config.hand_size >= 1, self.config
        assert self.config.hand_size >= max(self.config.card_counts), self.config

        vector_size = len(self.get_information_vector())
        discrete = gym.spaces.Discrete(self.config.hand_size + 1)
        card_space = gym.spaces.Tuple([discrete] * vector_size)

        # [(their cards, their info), ...]
        other_players = [(card_space, card_space)] * (self.config.num_players - 1)
        # [their cards, their info, ...]
        other_players = tuple(space for player in other_players for space in player)

        return gym.spaces.Tuple((
            gym.spaces.Discrete(self.config.max_tokens + 1), # Tokens
            gym.spaces.Discrete(self.config.max_fuses),      # Fuses
            card_space,                                      # Discarded cards
            card_space,                                      # Played cards
            card_space,                                      # Your info
        ) + other_players)

    @overrides
    def observation_to_sample(self, obs):
        played_cards = []
        for color, count in obs.played_cards.items():
            numbers = range(1, count + 1)
            played_cards += [hanabi.Card(color, number) for number in numbers]

        players = []
        for player in obs.players:
            players.append(self.information_to_sample(player.cards))
            players.append(self.information_to_sample(player.info))

        return (
            obs.num_tokens - 1,
            obs.num_fuses - 1,
            self.information_to_sample(obs.discarded_cards),
            self.information_to_sample(played_cards),
            self.information_to_sample(obs.your_info)
        ) + tuple(players)

    def sample_to_observation(self, sample):
        raise NotImplementedError()

    def moves(self):
        num_numbers = len(self.config.card_counts)
        card_colors = len(self.config.colors) + 1
        card_counts = len(self.config.card_counts) + 1
        num_cards = card_colors * card_counts
        return ([hanabi.InformColorMove(color, p)
                   for p in range(self.config.num_players - 1)
                   for color in self.config.colors] +
                [hanabi.InformNumberMove(i, p)
                   for p in range(self.config.num_players - 1)
                   for i in range(1, num_numbers + 1)] +
                [hanabi.DiscardMove(i) for i in range(num_cards)] +
                [hanabi.PlayMove(i) for i in range(num_cards)])

    @overrides
    def action_space(self):
        return gym.spaces.Discrete(len(self.moves()))

    @overrides
    def action_to_sample(self, move):
        c = self.config
        num_numbers = len(self.config.card_counts)
        card_colors = len(self.config.colors) + 1
        card_counts = len(self.config.card_counts) + 1
        num_cards = card_colors * card_counts

        offsets = [
            0,
            len(c.colors) * (c.num_players - 1),
            num_numbers * (c.num_players - 1),
            num_cards
        ]
        if isinstance(move, hanabi.InformColorMove):
            offset = sum(offsets[:1])
            return offset + len(c.colors) * move.player + c.colors.index(move.color)
        elif isinstance(move, hanabi.InformNumberMove):
            offset = sum(offsets[:2])
            return offset + num_numbers * move.player + (move.number - 1)
        elif isinstance(move, hanabi.DiscardMove):
            offset = sum(offsets[:3])
            return offset + move.index
        elif isinstance(move, hanabi.PlayMove):
            offset = sum(offsets[:4])
            return offset + move.index
        else:
            raise ValueError("Unexpected move {}.".format(move))

    def find_matching_card(self, info, cards):
        for i, my_card in enumerate(cards):
            color_matches = info.color is None or info.color == my_card.color
            number_matches = info.number is None or info.number == my_card.number
            if color_matches and number_matches:
                return i

        # If we can't find a matching card, return the end fo the list, an
        # illegal index.
        return len(cards)

    @overrides
    def sample_to_action(self, sample, cards):
        assert 0 <= sample < len(self.moves()), sample
        move = self.moves()[sample]
        if isinstance(move, hanabi.DiscardMove):
            info = self.get_information_vector()[move.index]
            move = hanabi.DiscardMove(index=self.find_matching_card(info, cards))
        elif isinstance(move, hanabi.PlayMove):
            info = self.get_information_vector()[move.index]
            move = hanabi.PlayMove(index=self.find_matching_card(info, cards))

        return move
